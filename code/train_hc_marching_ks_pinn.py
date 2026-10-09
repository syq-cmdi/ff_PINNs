#!/usr/bin/env python3
"""Causal hard-constrained time-marching PINN for the corrected KS problem.

Long-time global PINNs can minimize an average residual while losing the
initial-value trajectory.  This script therefore advances through short,
non-overlapping windows.  Each window has exact periodic Fourier features and
an exact left-end state represented by a truncated Fourier series.  The next
window receives only the preceding network prediction; reference data are used
for evaluation, never as training observations.
"""

from __future__ import annotations

import torch

import argparse
import copy
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from torch import nn

from train_hc_causal_ks_pinn import (
    field_metrics,
    grad,
    ks_residual,
    seed_everything,
    select_device,
    synchronize,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "ks_reference_corrected.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "2026_recalculation" / "pinn_marching"


class FourierState(nn.Module):
    """Differentiable real Fourier interpolant of a periodic grid state."""

    def __init__(self, values: np.ndarray, modes: int, x_left: float, length: float) -> None:
        super().__init__()
        values = np.asarray(values, dtype=np.float64)
        coefficients = np.fft.rfft(values) / values.size
        retained = min(modes, coefficients.size - 1)
        self.x_left = float(x_left)
        self.length = float(length)
        self.retained = int(retained)
        self.register_buffer(
            "constant", torch.tensor(float(coefficients[0].real), dtype=torch.float32)
        )
        positive = coefficients[1 : retained + 1]
        self.register_buffer(
            "cosine", torch.tensor(2.0 * positive.real, dtype=torch.float32)
        )
        self.register_buffer(
            "sine", torch.tensor(-2.0 * positive.imag, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = 2.0 * math.pi * (x - self.x_left) / self.length
        modes = torch.arange(
            1, self.retained + 1, device=x.device, dtype=x.dtype
        ).reshape(1, -1)
        return self.constant + torch.sum(
            self.cosine.reshape(1, -1) * torch.cos(theta * modes)
            + self.sine.reshape(1, -1) * torch.sin(theta * modes),
            dim=1,
            keepdim=True,
        )


class WindowPINN(nn.Module):
    def __init__(
        self,
        initial_state: FourierState,
        t_left: float,
        t_right: float,
        x_left: float,
        length: float,
        harmonics: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.initial_state = initial_state
        self.t_left = float(t_left)
        self.t_right = float(t_right)
        self.x_left = float(x_left)
        self.length = float(length)
        self.harmonics = int(harmonics)
        layers: list[nn.Module] = [nn.Linear(2 + 2 * harmonics, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        # A zero final map starts each window at its transferred left state and
        # avoids injecting unresolved high-wavenumber fourth derivatives.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = (t - self.t_left) / (self.t_right - self.t_left)
        theta = 2.0 * math.pi * (x - self.x_left) / self.length
        modes = torch.arange(
            1, self.harmonics + 1, device=x.device, dtype=x.dtype
        ).reshape(1, -1)
        return torch.cat(
            (
                2.0 * s - 1.0,
                torch.ones_like(s),
                torch.cos(theta * modes),
                torch.sin(theta * modes),
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        elapsed = t - self.t_left
        return self.initial_state(x) + elapsed * self.network(self.features(x, t))


@dataclass(frozen=True)
class MarchingConfig:
    seed: int
    windows: int
    adam_steps_per_window: int
    lbfgs_iterations_per_window: int
    collocation: int
    validation_collocation: int
    harmonics: int
    state_modes: int
    state_grid: int
    width: int
    depth: int
    learning_rate: float
    causal_bins: int
    causal_epsilon: float
    attention_exponent: float
    gradient_clip: float


def sample_window(
    count: int,
    bins: int,
    x_left: float,
    length: float,
    t_left: float,
    t_right: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_bin = math.ceil(count / bins)
    ids = torch.arange(bins, device=device).repeat_interleave(per_bin)[:count]
    t = t_left + (t_right - t_left) * (
        ids.reshape(-1, 1) + torch.rand((count, 1), device=device)
    ) / bins
    x = x_left + length * torch.rand((count, 1), device=device)
    return x.requires_grad_(True), t.requires_grad_(True)


def weighted_residual_loss(
    residual: torch.Tensor, bins: int, epsilon: float, exponent: float
) -> tuple[torch.Tensor, list[float]]:
    r2 = residual.square().reshape(-1)
    normalized = r2 / (r2.mean().detach() + 1e-12)
    attention = torch.clamp(normalized.pow(exponent), 0.2, 5.0).detach()
    usable = (r2.numel() // bins) * bins
    losses = (attention[:usable] * r2[:usable]).reshape(bins, -1).mean(dim=1)
    prefix = torch.cumsum(
        torch.cat((torch.zeros(1, device=r2.device), losses[:-1].detach())), dim=0
    )
    weights = torch.clamp(torch.exp(-epsilon * prefix), min=1e-3)
    return torch.mean(weights.detach() * losses), [float(v) for v in weights.detach().cpu()]


@torch.no_grad()
def predict_points(
    model: nn.Module, x: np.ndarray, t: np.ndarray, device: torch.device, batch: int = 8192
) -> np.ndarray:
    flat_x = np.asarray(x).ravel()
    flat_t = np.asarray(t).ravel()
    values = []
    for start in range(0, flat_x.size, batch):
        stop = min(flat_x.size, start + batch)
        xb = torch.as_tensor(flat_x[start:stop, None], dtype=torch.float32, device=device)
        tb = torch.as_tensor(flat_t[start:stop, None], dtype=torch.float32, device=device)
        values.append(model(xb, tb).detach().cpu().numpy().ravel())
    return np.concatenate(values).astype(np.float64)


def validation_residual(
    model: nn.Module,
    count: int,
    x_left: float,
    length: float,
    t_left: float,
    t_right: float,
    device: torch.device,
) -> float:
    x, t = sample_window(count, 4, x_left, length, t_left, t_right, device)
    residual, _ = ks_residual(model, x, t)
    return float(torch.sqrt(torch.mean(residual.square())).detach().cpu())


def train_window(
    model: WindowPINN,
    config: MarchingConfig,
    index: int,
    x_left: float,
    length: float,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.adam_steps_per_window, eta_min=config.learning_rate * 0.1
    )
    history = []
    start = time.perf_counter()
    for step in range(config.adam_steps_per_window):
        x, t = sample_window(
            config.collocation,
            config.causal_bins,
            x_left,
            length,
            model.t_left,
            model.t_right,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        residual, _ = ks_residual(model, x, t)
        loss, causal_weights = weighted_residual_loss(
            residual,
            config.causal_bins,
            config.causal_epsilon,
            config.attention_exponent,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        scheduler.step()
        if step % 250 == 0 or step + 1 == config.adam_steps_per_window:
            history.append(
                {
                    "optimizer": "Adam",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(
                        torch.as_tensor(gradient_norm).detach().cpu()
                    ),
                    "causal_weights": causal_weights,
                }
            )

    lbfgs_status = "not_requested"
    lbfgs_closures = 0
    if config.lbfgs_iterations_per_window > 0:
        fixed_x, fixed_t = sample_window(
            config.validation_collocation,
            config.causal_bins,
            x_left,
            length,
            model.t_left,
            model.t_right,
            device,
        )
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_iterations_per_window,
            max_eval=int(config.lbfgs_iterations_per_window * 1.25),
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal lbfgs_closures
            lbfgs_closures += 1
            lbfgs.zero_grad(set_to_none=True)
            residual, _ = ks_residual(model, fixed_x, fixed_t)
            # A fixed, smooth least-squares objective is required by the
            # quasi-Newton line search; detached adaptive weights would change
            # the objective between closure evaluations.
            loss = torch.mean(residual.square())
            loss.backward()
            return loss

        try:
            final_lbfgs = lbfgs.step(closure)
            lbfgs_status = "completed"
            history.append(
                {
                    "optimizer": "LBFGS",
                    "step": config.lbfgs_iterations_per_window,
                    "loss": float(torch.as_tensor(final_lbfgs).detach().cpu()),
                    "closures": lbfgs_closures,
                }
            )
        except (RuntimeError, NotImplementedError) as exc:
            lbfgs_status = f"failed: {type(exc).__name__}: {exc}"
    synchronize(device)
    elapsed = time.perf_counter() - start
    diagnostic = {
        "window": index,
        "t_left": model.t_left,
        "t_right": model.t_right,
        "seconds": elapsed,
        "validation_residual_rms": validation_residual(
            model,
            config.validation_collocation,
            x_left,
            length,
            model.t_left,
            model.t_right,
            device,
        ),
        "lbfgs_status": lbfgs_status,
        "lbfgs_closures": lbfgs_closures,
    }
    return history, diagnostic


def piecewise_float64_audit(
    models: list[WindowPINN],
    x_left: float,
    length: float,
    final_time: float,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    residuals = []
    jumps_by_order = [[] for _ in range(4)]
    for model32 in models:
        model = copy.deepcopy(model32).to("cpu").to(torch.float64).eval()
        x = (
            x_left + length * torch.rand((128, 1), generator=generator, dtype=torch.float64)
        ).requires_grad_(True)
        t = (
            model.t_left
            + (model.t_right - model.t_left)
            * torch.rand((128, 1), generator=generator, dtype=torch.float64)
        ).requires_grad_(True)
        residual, _ = ks_residual(model, x, t)
        residuals.append(residual.detach().numpy().ravel())

        tb = torch.linspace(model.t_left, model.t_right, 9, dtype=torch.float64).reshape(-1, 1)
        xl = torch.full_like(tb, x_left, requires_grad=True)
        xr = torch.full_like(tb, x_left + length, requires_grad=True)
        left = model(xl, tb)
        right = model(xr, tb)
        for order in range(4):
            jumps_by_order[order].append((left - right).detach().numpy().ravel())
            if order < 3:
                left = grad(left, xl)
                right = grad(right, xr)
    array = np.concatenate(residuals)
    return {
        "cpu_float64_unseen_residual": {
            "points": int(array.size),
            "rms": float(np.sqrt(np.mean(array**2))),
            "mean_absolute": float(np.mean(np.abs(array))),
            "q95_absolute": float(np.quantile(np.abs(array), 0.95)),
            "maximum_absolute": float(np.max(np.abs(array))),
        },
        "cpu_float64_periodic_jumps": [
            {
                "derivative_order": order,
                "rms": float(np.sqrt(np.mean(np.concatenate(values) ** 2))),
                "maximum_absolute": float(np.max(np.abs(np.concatenate(values)))),
            }
            for order, values in enumerate(jumps_by_order)
        ],
    }


def energy_budget_audit(
    field: np.ndarray, times: np.ndarray, length: float
) -> dict[str, float]:
    """Check the KS energy identity on the stitched output grid."""
    n = field.shape[1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    spectrum = np.fft.fft(field, axis=1)
    ux = np.fft.ifft(1j * k[None, :] * spectrum, axis=1).real
    uxx = np.fft.ifft(-(k[None, :] ** 2) * spectrum, axis=1).real
    energy = 0.5 * np.mean(field**2, axis=1)
    denergy = savgol_filter(
        energy,
        window_length=7,
        polyorder=5,
        deriv=1,
        delta=float(times[1] - times[0]),
        mode="interp",
    )
    rhs = np.mean(ux**2, axis=1) - np.mean(uxx**2, axis=1)
    residual = denergy - rhs
    core = slice(3, -3)
    return {
        "energy_balance_relative_rms": float(
            np.linalg.norm(residual[core]) / np.linalg.norm(rhs[core])
        ),
        "energy_balance_absolute_rms": float(
            np.sqrt(np.mean(residual[core] ** 2))
        ),
    }


def run_seed(
    config: MarchingConfig,
    reference: dict[str, np.ndarray],
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(config.seed)
    # Exact reconstructed geometry; the archive grid is the same to roundoff.
    x_left = -3.0 * math.pi
    length = 6.0 * math.pi
    final_time = 15.0
    state_x = np.linspace(x_left, x_left + length, config.state_grid, endpoint=False)
    initial_values = -np.sin(state_x / 3.0)
    initial_state = FourierState(initial_values, config.state_modes, x_left, length)
    window_edges = np.linspace(0.0, final_time, config.windows + 1)
    prediction = np.full_like(reference["u_archive"], np.nan, dtype=np.float64)
    models_cpu: list[WindowPINN] = []
    diagnostics: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    interface_jumps = []
    total_start = time.perf_counter()

    for index in range(config.windows):
        model = WindowPINN(
            initial_state,
            window_edges[index],
            window_edges[index + 1],
            x_left,
            length,
            config.harmonics,
            config.width,
            config.depth,
        ).to(device)
        window_history, diagnostic = train_window(
            model, config, index, x_left, length, device
        )
        history.extend({"window": index, **record} for record in window_history)

        mask = (reference["t"] >= window_edges[index] - 1e-12) & (
            reference["t"] <= window_edges[index + 1] + 1e-12
        )
        times = reference["t"][mask]
        xx, tt = np.meshgrid(reference["x"], times)
        window_prediction = predict_points(model, xx, tt, device).reshape(times.size, -1)
        prediction[mask] = window_prediction
        local_target = reference["u_archive"][mask]
        diagnostic["reference_global_relative_l2_evaluation_only"] = float(
            np.linalg.norm(window_prediction - local_target) / np.linalg.norm(local_target)
        )

        final_t = np.full(config.state_grid, window_edges[index + 1])
        final_values = predict_points(model, state_x, final_t, device)
        next_state = FourierState(final_values, config.state_modes, x_left, length)
        with torch.no_grad():
            xt = torch.as_tensor(state_x[:, None], dtype=torch.float32, device=device)
            reconstructed = next_state.to(device)(xt).detach().cpu().numpy().ravel()
        interface_jumps.append(
            {
                "window": index,
                "relative_fourier_transfer_error": float(
                    np.linalg.norm(reconstructed - final_values) / np.linalg.norm(final_values)
                ),
                "maximum_absolute_fourier_transfer_error": float(
                    np.max(np.abs(reconstructed - final_values))
                ),
            }
        )
        diagnostics.append(diagnostic)
        print(
            f"seed={config.seed} window={index + 1:02d}/{config.windows} "
            f"T=[{window_edges[index]:.2f},{window_edges[index+1]:.2f}] "
            f"res={diagnostic['validation_residual_rms']:.3e} "
            f"local_L2={diagnostic['reference_global_relative_l2_evaluation_only']:.3e} "
            f"{diagnostic['seconds']:.1f}s"
        )
        model_cpu = copy.deepcopy(model).to("cpu")
        models_cpu.append(model_cpu)
        torch.save(
            {
                "state_dict": model_cpu.state_dict(),
                "window": [window_edges[index], window_edges[index + 1]],
                "config": asdict(config),
            },
            output / f"hc_marching_seed{config.seed}_window{index:02d}.pt",
        )
        initial_state = next_state.to("cpu")
        if device.type == "mps":
            torch.mps.empty_cache()

    if np.isnan(prediction).any():
        raise RuntimeError("piecewise prediction contains unfilled times")
    total_seconds = time.perf_counter() - total_start
    metrics = field_metrics(prediction, reference["u_archive"])
    audit = piecewise_float64_audit(
        models_cpu, x_left, length, final_time, seed=config.seed + 20000
    )
    result: dict[str, object] = {
        "config": {"method": "hc_marching", **asdict(config)},
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "parameter_count_per_window": sum(p.numel() for p in models_cpu[0].parameters()),
        "parameter_count_all_windows": sum(
            sum(p.numel() for p in model.parameters()) for model in models_cpu
        ),
        "training_seconds": total_seconds,
        "initial_condition_relative_l2": float(
            np.linalg.norm(prediction[0] + np.sin(reference["x"] / 3.0))
            / np.linalg.norm(np.sin(reference["x"] / 3.0))
        ),
        "left_state_constraint": (
            "exact with respect to the supplied float32 Fourier state in every window; "
            "the reported initial error includes float32 Fourier coefficient quantization"
        ),
        "diagnostics": diagnostics,
        "interface_transfers": interface_jumps,
        "history": history,
        **metrics,
        **audit,
        **energy_budget_audit(prediction, reference["t"], length),
    }
    stem = f"hc_marching_seed{config.seed}"
    np.savez_compressed(output / f"{stem}_prediction.npz", x=reference["x"], t=reference["t"], u=prediction)
    (output / f"{stem}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def summarize(results: list[dict[str, object]], output: Path) -> None:
    locked_configs = []
    for result in results:
        config = dict(result["config"])
        config.pop("seed", None)
        locked_configs.append(json.dumps(config, sort_keys=True))
    if len(set(locked_configs)) != 1:
        raise RuntimeError("refusing to aggregate marching runs with different configurations")
    reference_directory = (
        ROOT / "results" / "2026_recalculation" / "reference"
    )
    runtime_audit_path = reference_directory / "etdrk4_runtime_audit.json"
    if runtime_audit_path.is_file():
        runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
        runtime_config = runtime_audit.get("configuration", {})
        if (
            runtime_audit.get("method")
            != "Fourier pseudospectral ETDRK4, 2/3 dealiased"
            or int(runtime_config.get("N", -1)) != 512
            or not np.isclose(float(runtime_config.get("dt", np.nan)), 0.0046875)
            or int(runtime_config.get("timed_repeats", 0)) < 5
        ):
            raise RuntimeError("repeated ETDRK4 runtime audit has the wrong configuration")
        etdrk4_seconds = float(runtime_audit["wall_seconds_median"])
        etdrk4_runtime_source = "repeated_runtime_audit_median"
    else:
        with (reference_directory / "etdrk4_convergence.csv").open(
            encoding="utf-8"
        ) as stream:
            convergence_rows = list(csv.DictReader(stream))
        canonical = next(
            row
            for row in convergence_rows
            if row["N"] == "512" and np.isclose(float(row["dt"]), 0.0046875)
        )
        etdrk4_seconds = float(canonical["wall_seconds"])
        etdrk4_runtime_source = "convergence_csv_single_run_fallback"
    if not np.isfinite(etdrk4_seconds) or etdrk4_seconds <= 0.0:
        raise RuntimeError("canonical ETDRK4 runtime must be positive and finite")
    keys = (
        "global_relative_l2",
        "median_time_relative_l2",
        "maximum_time_relative_l2",
        "final_time_relative_l2",
        "median_phase_invariant_relative_l2",
        "training_seconds",
    )
    rows = []
    for result in results:
        rows.append(
            {
                "method": "hc_marching",
                "seed": result["config"]["seed"],
                **{key: result[key] for key in keys},
                "residual_rms_cpu_float64": result["cpu_float64_unseen_residual"]["rms"],
                "residual_q95_cpu_float64": result["cpu_float64_unseen_residual"]["q95_absolute"],
                "periodic_u_jump_rms_cpu_float64": result["cpu_float64_periodic_jumps"][0]["rms"],
                "periodic_uxxx_jump_rms_cpu_float64": result["cpu_float64_periodic_jumps"][3]["rms"],
                "maximum_interface_transfer_relative_error": max(
                    item["relative_fourier_transfer_error"]
                    for item in result["interface_transfers"][:-1]
                ),
                "predicted_mean_drift": result["predicted_mean_drift"],
                "energy_balance_relative_rms": result.get("energy_balance_relative_rms", float("nan")),
                "initial_condition_relative_l2": result["initial_condition_relative_l2"],
                "training_to_etdrk4_runtime_ratio": result["training_seconds"] / etdrk4_seconds,
            }
        )
    with (output / "hc_marching_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate_keys = keys + (
        "residual_rms_cpu_float64",
        "residual_q95_cpu_float64",
        "maximum_interface_transfer_relative_error",
        "predicted_mean_drift",
        "energy_balance_relative_rms",
        "initial_condition_relative_l2",
        "training_to_etdrk4_runtime_ratio",
    )
    aggregate = {
        "runs": len(rows),
        "locked_config": json.loads(locked_configs[0]),
        "canonical_etdrk4_seconds": etdrk4_seconds,
        "canonical_etdrk4_runtime_source": etdrk4_runtime_source,
        **{
            key: {
                "median": float(np.median([row[key] for row in rows])),
                "q25": float(np.quantile([row[key] for row in rows], 0.25)),
                "q75": float(np.quantile([row[key] for row in rows], 0.75)),
            }
            for key in aggregate_keys
        },
    }
    (output / "hc_marching_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )


def make_figure(results: list[dict[str, object]], reference: dict[str, np.ndarray], output: Path) -> None:
    ordered = sorted(results, key=lambda result: result["global_relative_l2"])
    best = ordered[len(ordered) // 2]
    seed = best["config"]["seed"]
    prediction = np.load(output / f"hc_marching_seed{seed}_prediction.npz")["u"]
    target = reference["u_archive"]
    x, t = reference["x"], reference["t"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    image = axes[0, 0].pcolormesh(x, t, prediction, cmap="RdBu_r", shading="auto")
    axes[0, 0].set(xlabel=r"$X$", ylabel=r"$T$", title=f"(a) HC-marching PINN, seed {seed}")
    fig.colorbar(image, ax=axes[0, 0], label=r"$u_\theta$")
    error = np.abs(prediction - target)
    image = axes[0, 1].pcolormesh(x, t, error, cmap="magma", shading="auto")
    axes[0, 1].set(xlabel=r"$X$", ylabel=r"$T$", title="(b) Absolute error")
    fig.colorbar(image, ax=axes[0, 1], label=r"$|u_\theta-u_{\mathrm{archive}}|$")
    for result in results:
        line = axes[1, 0].semilogy(
            t, result["time_relative_l2"], label=f"seed {result['config']['seed']}"
        )[0]
        if "time_phase_invariant_relative_l2" in result:
            axes[1, 0].semilogy(
                t,
                result["time_phase_invariant_relative_l2"],
                ls="--",
                lw=0.9,
                alpha=0.8,
                color=line.get_color(),
            )
    axes[1, 0].set(xlabel=r"$T$", ylabel="relative $L_2$", title="(c) Error accumulation")
    axes[1, 0].grid(True, which="both", alpha=0.22)
    axes[1, 0].legend(frameon=False)
    diagnostics = best["diagnostics"]
    centers = [(item["t_left"] + item["t_right"]) / 2 for item in diagnostics]
    axes[1, 1].semilogy(
        centers,
        [item["validation_residual_rms"] for item in diagnostics],
        "o-",
        label="unseen residual RMS",
    )
    axes[1, 1].semilogy(
        centers,
        [item["reference_global_relative_l2_evaluation_only"] for item in diagnostics],
        "s-",
        label="window relative $L_2$",
    )
    axes[1, 1].semilogy(
        centers[:-1],
        [
            item["relative_fourier_transfer_error"]
            for item in best["interface_transfers"][:-1]
        ],
        "^-",
        label="Fourier transfer error",
    )
    axes[1, 1].set(xlabel=r"window-center $T$", title="(d) Window diagnostics")
    axes[1, 1].grid(True, which="both", alpha=0.22)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "figure_hc_marching.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_seeds(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--adam-steps", type=int, default=750)
    parser.add_argument("--lbfgs-iterations", type=int, default=100)
    parser.add_argument("--collocation", type=int, default=512)
    parser.add_argument("--validation-collocation", type=int, default=1024)
    parser.add_argument("--harmonics", type=int, default=12)
    parser.add_argument("--state-modes", type=int, default=32)
    parser.add_argument("--state-grid", type=int, default=512)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-summary", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    torch.set_float32_matmul_precision("highest")
    archive = np.load(args.reference)
    reference = {key: np.asarray(archive[key]) for key in archive.files}
    print(f"device={device}; torch={torch.__version__}")
    results = []
    if args.resume_summary:
        for path in sorted(args.output.glob("hc_marching_seed*_metrics.json")):
            results.append(json.loads(path.read_text(encoding="utf-8")))
    else:
        for seed in parse_seeds(args.seeds):
            config = MarchingConfig(
                seed=seed,
                windows=args.windows,
                adam_steps_per_window=args.adam_steps,
                lbfgs_iterations_per_window=args.lbfgs_iterations,
                collocation=args.collocation,
                validation_collocation=args.validation_collocation,
                harmonics=args.harmonics,
                state_modes=args.state_modes,
                state_grid=args.state_grid,
                width=args.width,
                depth=args.depth,
                learning_rate=args.learning_rate,
                causal_bins=4,
                causal_epsilon=0.01,
                attention_exponent=0.5,
                gradient_clip=100.0,
            )
            results.append(run_seed(config, reference, args.output, device))
    if not results:
        raise RuntimeError("no marching results")
    summarize(results, args.output)
    make_figure(results, reference, args.output)


if __name__ == "__main__":
    main()
