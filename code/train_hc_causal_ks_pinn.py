#!/usr/bin/env python3
"""Train and audit scale-correct PINNs for the periodic KS equation.

The full method combines (i) an exact periodic Fourier embedding, (ii) a hard
initial-condition transformation, (iii) causal temporal weighting and a
time-horizon curriculum, and (iv) bounded residual-based attention.  The code
is an independent implementation of published ideas; it does not copy the
unlicensed 2026 HC-PINN repository.

Every saved model is evaluated against the converged ETDRK4 field, on unseen
collocation points in CPU float64, and for periodic jumps through the third
spatial derivative.  Four ablation levels are available:

  vanilla         soft IC and soft periodic derivative penalties
  periodic_soft   exact periodic embedding and soft IC
  hard_periodic   exact periodic embedding and hard IC
  hc_causal       hard_periodic + causal curriculum + residual attention
"""

from __future__ import annotations

# Import torch before packages that may load another OpenMP runtime on macOS.
import torch

import argparse
import copy
import csv
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "ks_reference_corrected.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "2026_recalculation" / "pinn"
METHODS = ("vanilla", "periodic_soft", "hard_periodic", "hc_causal")


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grad(output: torch.Tensor, variable: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        variable,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


class FourierMLP(nn.Module):
    def __init__(
        self,
        method: str,
        x_left: float,
        length: float,
        final_time: float,
        harmonics: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.method = method
        self.x_left = float(x_left)
        self.length = float(length)
        self.final_time = float(final_time)
        self.harmonics = int(harmonics)
        self.periodic = method != "vanilla"
        self.hard_initial = method in ("hard_periodic", "hc_causal")
        input_width = 2 if not self.periodic else 2 + 2 * harmonics
        layers: list[nn.Module] = [nn.Linear(input_width, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        if self.hard_initial:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = t / self.final_time
        if not self.periodic:
            xi = 2.0 * (x - self.x_left) / self.length - 1.0
            return torch.cat((xi, 2.0 * s - 1.0), dim=1)
        theta = 2.0 * math.pi * (x - self.x_left) / self.length
        modes = torch.arange(
            1, self.harmonics + 1, device=x.device, dtype=x.dtype
        ).reshape(1, -1)
        angles = theta * modes
        return torch.cat(
            (
                2.0 * s - 1.0,
                torch.ones_like(s),
                torch.cos(angles),
                torch.sin(angles),
            ),
            dim=1,
        )

    @staticmethod
    def initial_condition(x: torch.Tensor) -> torch.Tensor:
        return -torch.sin(x / 3.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.network(self.features(x, t))
        if not self.hard_initial:
            return raw
        s = t / self.final_time
        # Published hard-constraint family: exact at t=0, with a smooth bypass
        # that retains a direct path for the initial state.
        return torch.exp(-s) * self.initial_condition(x) + s * raw


def ks_residual(
    model: nn.Module, x: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    u = model(x, t)
    ut = grad(u, t)
    ux = grad(u, x)
    uxx = grad(ux, x)
    uxxx = grad(uxx, x)
    uxxxx = grad(uxxx, x)
    return ut + u * ux + uxx + uxxxx, (u, ux, uxx, uxxx)


@dataclass(frozen=True)
class TrainConfig:
    method: str
    seed: int
    steps: int
    collocation: int
    harmonics: int
    width: int
    depth: int
    learning_rate: float
    causal_bins: int
    causal_epsilon: float
    ic_weight: float
    bc_weight: float
    attention_exponent: float
    gradient_clip: float


def sample_collocation(
    config: TrainConfig,
    step: int,
    x_left: float,
    length: float,
    final_time: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    causal = config.method == "hc_causal"
    if causal:
        # Five nested horizons; the last 40% of optimization sees the full
        # domain. This retains causal ordering without training separate models.
        fraction = step / max(config.steps - 1, 1)
        horizon_fraction = (0.2, 0.4, 0.6, 0.8, 1.0)[
            min(4, int(fraction * 8.0))
        ]
    else:
        horizon_fraction = 1.0
    horizon = final_time * horizon_fraction
    if causal:
        bins = config.causal_bins
        per_bin = math.ceil(config.collocation / bins)
        bin_id = torch.arange(bins, device=device).repeat_interleave(per_bin)
        bin_id = bin_id[: config.collocation]
        within = torch.rand((config.collocation, 1), device=device)
        t = horizon * (bin_id.reshape(-1, 1) + within) / bins
    else:
        t = horizon * torch.rand((config.collocation, 1), device=device)
    x = x_left + length * torch.rand((config.collocation, 1), device=device)
    return x.requires_grad_(True), t.requires_grad_(True), horizon


def causal_attention_loss(
    residual: torch.Tensor, config: TrainConfig
) -> tuple[torch.Tensor, list[float]]:
    r2 = residual.square().reshape(-1)
    if config.method != "hc_causal":
        return r2.mean(), [1.0]
    # Bounded residual attention is detached: it reallocates emphasis without
    # allowing the model to reduce a weight instead of its residual.
    normalized = r2 / (r2.mean().detach() + 1e-12)
    attention = torch.clamp(
        normalized.pow(config.attention_exponent), min=0.2, max=5.0
    ).detach()
    bins = config.causal_bins
    usable = (r2.numel() // bins) * bins
    losses = (attention[:usable] * r2[:usable]).reshape(bins, -1).mean(dim=1)
    prefix = torch.cumsum(
        torch.cat((torch.zeros(1, device=r2.device), losses[:-1].detach())), dim=0
    )
    weights = torch.exp(-config.causal_epsilon * prefix)
    weights = torch.clamp(weights, min=1e-4)
    return torch.mean(weights.detach() * losses), [float(value) for value in weights.detach().cpu()]


def soft_constraint_loss(
    model: FourierMLP,
    method: str,
    x_left: float,
    length: float,
    final_time: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), device=device)
    if method in ("hard_periodic", "hc_causal"):
        ic_loss = zero
    else:
        x0 = x_left + length * torch.rand((256, 1), device=device)
        t0 = torch.zeros_like(x0)
        ic_loss = torch.mean((model(x0, t0) - model.initial_condition(x0)) ** 2)
    if method != "vanilla":
        return ic_loss, zero

    count = 64
    tb = final_time * torch.rand((count, 1), device=device)
    xl = torch.full((count, 1), x_left, device=device, requires_grad=True)
    xr = torch.full((count, 1), x_left + length, device=device, requires_grad=True)
    left = model(xl, tb)
    right = model(xr, tb)
    differences = [left - right]
    for _ in range(3):
        left = grad(left, xl)
        right = grad(right, xr)
        differences.append(left - right)
    bc_loss = sum(torch.mean(value.square()) for value in differences)
    return ic_loss, bc_loss


def train_one(
    config: TrainConfig,
    reference: dict[str, np.ndarray],
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(config.seed)
    x_grid = reference["x"]
    t_grid = reference["t"]
    x_left = -3.0 * math.pi
    length = 6.0 * math.pi
    final_time = 15.0
    model = FourierMLP(
        config.method,
        x_left,
        length,
        final_time,
        config.harmonics,
        config.width,
        config.depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps, eta_min=config.learning_rate * 0.05
    )
    history: list[dict[str, object]] = []
    synchronize(device)
    start = time.perf_counter()
    model.train()
    for step in range(config.steps):
        x, t, horizon = sample_collocation(
            config, step, x_left, length, final_time, device
        )
        optimizer.zero_grad(set_to_none=True)
        residual, _ = ks_residual(model, x, t)
        pde_loss, causal_weights = causal_attention_loss(residual, config)
        ic_loss, bc_loss = soft_constraint_loss(
            model, config.method, x_left, length, final_time, device
        )
        loss = pde_loss + config.ic_weight * ic_loss + config.bc_weight * bc_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        scheduler.step()
        if step % 250 == 0 or step + 1 == config.steps:
            record = {
                "step": step,
                "total": float(loss.detach().cpu()),
                "pde": float(pde_loss.detach().cpu()),
                "ic": float(ic_loss.detach().cpu()),
                "bc": float(bc_loss.detach().cpu()),
                "gradient_norm_before_clip": float(
                    torch.as_tensor(gradient_norm).detach().cpu()
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "horizon": horizon,
                "causal_weights": causal_weights,
            }
            history.append(record)
            print(
                f"{config.method} seed={config.seed} step={step:5d} "
                f"loss={record['total']:.3e} pde={record['pde']:.3e} "
                f"horizon={horizon:.1f}"
            )
    synchronize(device)
    training_seconds = time.perf_counter() - start

    model.eval()
    prediction = predict_grid(model, x_grid, t_grid, device)
    metrics = field_metrics(prediction, reference["u_archive"])
    cpu_metrics = residual_and_boundary_audit(
        model, x_left, length, final_time, seed=config.seed + 10000
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "parameter_count": parameter_count,
        "training_seconds": training_seconds,
        "history": history,
        **metrics,
        **cpu_metrics,
    }
    stem = f"{config.method}_seed{config.seed}"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": asdict(config),
            "geometry": {
                "x_left": x_left,
                "length": length,
                "final_time": final_time,
            },
        },
        output / f"{stem}.pt",
    )
    np.savez_compressed(output / f"{stem}_prediction.npz", x=x_grid, t=t_grid, u=prediction)
    (output / f"{stem}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


@torch.no_grad()
def predict_grid(
    model: FourierMLP,
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
    batch: int = 8192,
) -> np.ndarray:
    xx, tt = np.meshgrid(x, t)
    flat_x = xx.ravel()
    flat_t = tt.ravel()
    values = []
    for start in range(0, flat_x.size, batch):
        stop = min(start + batch, flat_x.size)
        xb = torch.as_tensor(flat_x[start:stop, None], dtype=torch.float32, device=device)
        tb = torch.as_tensor(flat_t[start:stop, None], dtype=torch.float32, device=device)
        values.append(model(xb, tb).detach().cpu().numpy().ravel())
    return np.concatenate(values).reshape(t.size, x.size).astype(np.float64)


def phase_invariant_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = []
    for p, q in zip(predicted, target):
        correlation = np.fft.ifft(np.fft.fft(p) * np.conj(np.fft.fft(q))).real
        shift = int(np.argmax(correlation))
        aligned = np.roll(p, -shift)
        values.append(np.linalg.norm(aligned - q) / np.linalg.norm(q))
    return np.asarray(values)


def field_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    difference = predicted - target
    per_time = np.linalg.norm(difference, axis=1) / np.linalg.norm(target, axis=1)
    phase = phase_invariant_errors(predicted, target)
    return {
        "global_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(target)),
        "median_time_relative_l2": float(np.median(per_time)),
        "maximum_time_relative_l2": float(np.max(per_time)),
        "final_time_relative_l2": float(per_time[-1]),
        "median_phase_invariant_relative_l2": float(np.median(phase)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "predicted_mean_drift": float(
            np.max(np.abs(np.mean(predicted, axis=1) - np.mean(predicted[0])))
        ),
        "time_relative_l2": per_time.tolist(),
        "time_phase_invariant_relative_l2": phase.tolist(),
    }


def residual_and_boundary_audit(
    trained: FourierMLP,
    x_left: float,
    length: float,
    final_time: float,
    seed: int,
) -> dict[str, object]:
    # MPS cannot cast in-place to float64; transfer first, then cast on CPU.
    model = copy.deepcopy(trained).to(device="cpu").to(dtype=torch.float64).eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    residual_values = []
    for _ in range(8):
        x = (
            x_left
            + length * torch.rand((256, 1), generator=generator, dtype=torch.float64)
        ).requires_grad_(True)
        t = (
            final_time * torch.rand((256, 1), generator=generator, dtype=torch.float64)
        ).requires_grad_(True)
        residual, _ = ks_residual(model, x, t)
        residual_values.append(residual.detach().numpy().ravel())
    residual_array = np.concatenate(residual_values)

    t_boundary = torch.linspace(0.0, final_time, 65, dtype=torch.float64).reshape(-1, 1)
    xl = torch.full_like(t_boundary, x_left, requires_grad=True)
    xr = torch.full_like(t_boundary, x_left + length, requires_grad=True)
    left = model(xl, t_boundary)
    right = model(xr, t_boundary)
    jumps = []
    for order in range(4):
        jump = (left - right).detach().numpy().ravel()
        jumps.append(
            {
                "derivative_order": order,
                "rms": float(np.sqrt(np.mean(jump**2))),
                "maximum_absolute": float(np.max(np.abs(jump))),
            }
        )
        if order < 3:
            left = grad(left, xl)
            right = grad(right, xr)

    x_ic = torch.linspace(x_left, x_left + length, 513, dtype=torch.float64)[:-1].reshape(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    predicted_ic = model(x_ic, t_ic).detach().numpy().ravel()
    exact_ic = -np.sin(x_ic.numpy().ravel() / 3.0)
    return {
        "cpu_float64_unseen_residual": {
            "points": int(residual_array.size),
            "rms": float(np.sqrt(np.mean(residual_array**2))),
            "mean_absolute": float(np.mean(np.abs(residual_array))),
            "q95_absolute": float(np.quantile(np.abs(residual_array), 0.95)),
            "maximum_absolute": float(np.max(np.abs(residual_array))),
        },
        "cpu_float64_periodic_jumps": jumps,
        "initial_condition_relative_l2": float(
            np.linalg.norm(predicted_ic - exact_ic) / np.linalg.norm(exact_ic)
        ),
    }


def summarize_results(results: list[dict[str, object]], output: Path) -> None:
    scalar_keys = (
        "global_relative_l2",
        "median_time_relative_l2",
        "maximum_time_relative_l2",
        "final_time_relative_l2",
        "median_phase_invariant_relative_l2",
        "maximum_absolute_error",
        "training_seconds",
        "initial_condition_relative_l2",
    )
    rows = []
    for result in results:
        row = {
            "method": result["config"]["method"],
            "seed": result["config"]["seed"],
            **{key: result[key] for key in scalar_keys},
            "residual_rms_cpu_float64": result["cpu_float64_unseen_residual"]["rms"],
            "periodic_u_jump_rms_cpu_float64": result["cpu_float64_periodic_jumps"][0]["rms"],
            "periodic_uxxx_jump_rms_cpu_float64": result["cpu_float64_periodic_jumps"][3]["rms"],
            "parameter_count": result["parameter_count"],
        }
        rows.append(row)
    with (output / "pinn_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates: dict[str, object] = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        method_summary: dict[str, object] = {"runs": len(subset)}
        method_summary.update({
            key: {
                "median": float(np.median([float(row[key]) for row in subset])),
                "q25": float(np.quantile([float(row[key]) for row in subset], 0.25)),
                "q75": float(np.quantile([float(row[key]) for row in subset], 0.75)),
            }
            for key in scalar_keys + ("residual_rms_cpu_float64",)
        })
        aggregates[method] = method_summary
    (output / "pinn_summary.json").write_text(
        json.dumps(aggregates, indent=2) + "\n", encoding="utf-8"
    )


def make_comparison_figure(
    results: list[dict[str, object]], reference: dict[str, np.ndarray], output: Path
) -> None:
    best = min(results, key=lambda item: float(item["global_relative_l2"]))
    method = best["config"]["method"]
    seed = best["config"]["seed"]
    prediction = np.load(output / f"{method}_seed{seed}_prediction.npz")["u"]
    target = reference["u_archive"]
    x = reference["x"]
    t = reference["t"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(grid[0, 0])
    image = ax0.pcolormesh(x, t, prediction, cmap="RdBu_r", shading="auto")
    ax0.set(xlabel=r"$X$", ylabel=r"$T$", title=f"(a) Best PINN: {method}, seed {seed}")
    fig.colorbar(image, ax=ax0, label=r"$u_\theta$")
    ax1 = fig.add_subplot(grid[0, 1])
    err = np.abs(prediction - target)
    image = ax1.pcolormesh(x, t, err, cmap="magma", shading="auto")
    ax1.set(xlabel=r"$X$", ylabel=r"$T$", title="(b) Absolute error")
    fig.colorbar(image, ax=ax1, label=r"$|u_\theta-u_{ETD}|$")
    ax2 = fig.add_subplot(grid[1, 0])
    for result in results:
        ax2.semilogy(
            t,
            result["time_relative_l2"],
            alpha=0.72,
            lw=1.0,
            label=f"{result['config']['method']} s{result['config']['seed']}",
        )
    ax2.set(xlabel=r"$T$", ylabel="relative $L_2$", title="(c) Error growth")
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(frameon=False, fontsize=6, ncol=2)
    ax3 = fig.add_subplot(grid[1, 1])
    for method_name in METHODS:
        subset = [r for r in results if r["config"]["method"] == method_name]
        if not subset:
            continue
        ax3.scatter(
            [r["training_seconds"] for r in subset],
            [r["global_relative_l2"] for r in subset],
            label=method_name,
            s=30,
        )
    ax3.axhline(8.7304e-11, color="black", ls="--", lw=1.0, label="ETDRK4 archive error")
    ax3.set(
        xscale="log",
        yscale="log",
        xlabel="training wall time (s)",
        ylabel="global relative $L_2$",
        title="(d) Accuracy-cost comparison",
    )
    ax3.grid(True, which="both", alpha=0.22)
    ax3.legend(frameon=False, fontsize=7)
    fig.savefig(output / "figure_pinn_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_seeds(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--collocation", type=int, default=1024)
    parser.add_argument("--harmonics", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-summary", action="store_true")
    args = parser.parse_args()
    requested_methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    invalid = sorted(set(requested_methods) - set(METHODS))
    if invalid:
        raise ValueError(f"unknown methods: {invalid}")
    seeds = parse_seeds(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    torch.set_float32_matmul_precision("highest")
    reference_npz = np.load(args.reference)
    reference = {key: np.asarray(reference_npz[key]) for key in reference_npz.files}
    print(f"device={device}; torch={torch.__version__}")

    results: list[dict[str, object]] = []
    if args.resume_summary:
        for path in sorted(args.output.glob("*_metrics.json")):
            results.append(json.loads(path.read_text(encoding="utf-8")))
    else:
        for method in requested_methods:
            for seed in seeds:
                config = TrainConfig(
                    method=method,
                    seed=seed,
                    steps=args.steps,
                    collocation=args.collocation,
                    harmonics=args.harmonics,
                    width=args.width,
                    depth=args.depth,
                    learning_rate=args.learning_rate,
                    causal_bins=8,
                    causal_epsilon=0.01,
                    ic_weight=100.0,
                    bc_weight=10.0,
                    attention_exponent=0.5,
                    gradient_clip=100.0,
                )
                results.append(train_one(config, reference, args.output, device))
    if not results:
        raise RuntimeError("no PINN results available")
    summarize_results(results, args.output)
    make_comparison_figure(results, reference, args.output)


if __name__ == "__main__":
    main()
