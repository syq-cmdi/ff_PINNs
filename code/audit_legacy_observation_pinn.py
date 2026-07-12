#!/usr/bin/env python3
"""Re-evaluate the observation-dominated legacy checkpoint without retraining.

The checkpoint was trained after mapping the archive to x in [-10,10] and t in
[0,50] while imposing a unit-coefficient KS residual.  This audit distinguishes
field-fit error from (a) that imposed but scale-inconsistent residual and (b)
the correctly transformed residual in the checkpoint coordinates.
"""

from __future__ import annotations

import torch

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "code" / "obs10k_w120_pde8k.pth"
DEFAULT_REFERENCE = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "ks_reference_corrected.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "2026_recalculation" / "legacy_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def derivative(output: torch.Tensor, variable: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        variable,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


class LegacyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("fourier_B", torch.empty(2, 32))
        self.register_buffer("x_lb", torch.tensor(-10.0))
        self.register_buffer("x_ub", torch.tensor(10.0))
        self.register_buffer("t_lb", torch.tensor(0.0))
        self.register_buffer("t_ub", torch.tensor(50.0))
        layers: list[nn.Module] = []
        sizes = [66] + [128] * 6 + [1]
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xn = 2.0 * (x - self.x_lb) / (self.x_ub - self.x_lb) - 1.0
        tn = 2.0 * (t - self.t_lb) / (self.t_ub - self.t_lb) - 1.0
        coordinates = torch.cat((xn, tn), dim=-1)
        projection = 2.0 * math.pi * coordinates @ self.fourier_B
        encoded = torch.cat((coordinates, torch.sin(projection), torch.cos(projection)), dim=-1)
        return self.net(encoded)


@torch.no_grad()
def predict(model: LegacyNet, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    xx, tt = np.meshgrid(x, t)
    x_wrong = xx.ravel() * 10.0 / (3.0 * math.pi)
    t_wrong = tt.ravel() * 50.0 / 15.0
    values = []
    for start in range(0, x_wrong.size, 8192):
        stop = min(start + 8192, x_wrong.size)
        xb = torch.as_tensor(x_wrong[start:stop, None], dtype=torch.float64)
        tb = torch.as_tensor(t_wrong[start:stop, None], dtype=torch.float64)
        values.append(model(xb, tb).numpy().ravel())
    return np.concatenate(values).reshape(t.size, x.size)


def residual_audit(model: LegacyNet, seed: int, points: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    wrong_unit = []
    transformed = []
    coefficients = (1.0 / math.pi, 10.0 / (3.0 * math.pi**2), 1000.0 / (27.0 * math.pi**4))
    for _ in range(math.ceil(points / 256)):
        count = min(256, points - len(wrong_unit) * 256)
        if count <= 0:
            break
        x = (-10.0 + 20.0 * torch.rand((count, 1), generator=generator, dtype=torch.float64)).requires_grad_(True)
        t = (50.0 * torch.rand((count, 1), generator=generator, dtype=torch.float64)).requires_grad_(True)
        u = model(x, t)
        ut = derivative(u, t)
        ux = derivative(u, x)
        uxx = derivative(ux, x)
        uxxx = derivative(uxx, x)
        uxxxx = derivative(uxxx, x)
        wrong_unit.append((ut + u * ux + uxx + uxxxx).detach().numpy().ravel())
        a, b, c = coefficients
        transformed.append((ut + a * u * ux + b * uxx + c * uxxxx).detach().numpy().ravel())

    def summarize(chunks: list[np.ndarray]) -> dict[str, float]:
        values = np.concatenate(chunks)
        return {
            "rms": float(np.sqrt(np.mean(values**2))),
            "mean_absolute": float(np.mean(np.abs(values))),
            "q95_absolute": float(np.quantile(np.abs(values), 0.95)),
            "maximum_absolute": float(np.max(np.abs(values))),
        }

    transformed_summary = summarize(transformed)
    corrected_unit = np.concatenate(transformed) / (15.0 / 50.0)
    return {
        "points": points,
        "imposed_unit_coefficient_residual_in_x10_t50_coordinates": summarize(wrong_unit),
        "scale_correct_transformed_residual_in_x10_t50_coordinates": {
            "coefficients": list(coefficients),
            **transformed_summary,
        },
        "equivalent_unit_KS_residual_in_corrected_X_T_coordinates": {
            "coordinate_relation": (
                "the transformed residual in x'=10*x_raw,t'=50*t_raw equals "
                "(15/50) times the unit-KS residual in X=3*pi*x_raw,T=15*t_raw"
            ),
            **{
                "rms": float(np.sqrt(np.mean(corrected_unit**2))),
                "mean_absolute": float(np.mean(np.abs(corrected_unit))),
                "q95_absolute": float(np.quantile(np.abs(corrected_unit), 0.95)),
                "maximum_absolute": float(np.max(np.abs(corrected_unit))),
            },
        },
    }


def periodic_audit(model: LegacyNet) -> dict[str, list[dict[str, float]]]:
    t = torch.linspace(0.0, 50.0, 129, dtype=torch.float64).reshape(-1, 1)
    xl = torch.full_like(t, -10.0, requires_grad=True)
    xr = torch.full_like(t, 10.0, requires_grad=True)
    left = model(xl, t)
    right = model(xr, t)
    rows_checkpoint = []
    rows_corrected = []
    for order in range(4):
        jump = (left - right).detach().numpy().ravel()
        summary = (
            {
                "derivative_order_in_checkpoint_x": order,
                "rms": float(np.sqrt(np.mean(jump**2))),
                "maximum_absolute": float(np.max(np.abs(jump))),
            }
        )
        rows_checkpoint.append(summary)
        coordinate_factor = (10.0 / (3.0 * math.pi)) ** order
        rows_corrected.append(
            {
                "derivative_order_in_corrected_X": order,
                "rms": summary["rms"] * coordinate_factor,
                "maximum_absolute": summary["maximum_absolute"] * coordinate_factor,
            }
        )
        if order < 3:
            left = derivative(left, xl)
            right = derivative(right, xr)
    return {
        "checkpoint_xprime_derivatives": rows_checkpoint,
        "corrected_X_derivatives": rows_corrected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points", type=int, default=4096)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = LegacyNet().to(torch.float64)
    model.load_state_dict(state, strict=True)
    model.eval()
    archive = np.load(args.reference)
    prediction = predict(model, archive["x"], archive["t"])
    target = archive["u_archive"]
    difference = prediction - target
    per_time = np.linalg.norm(difference, axis=1) / np.linalg.norm(target, axis=1)

    x_ic = torch.linspace(-10.0, 10.0, 513, dtype=torch.float64)[:-1].reshape(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    predicted_ic = model(x_ic, t_ic).detach().numpy().ravel()
    exact_ic = -np.sin(np.pi * x_ic.numpy().ravel() / 10.0)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "classification": (
            "observation-dominated field surrogate trained with a scale-inconsistent "
            "unit KS loss; not a forward PINN solution of the corrected KS problem"
        ),
        "field_fit_to_archive": {
            "global_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(target)),
            "median_time_relative_l2": float(np.median(per_time)),
            "maximum_time_relative_l2": float(np.max(per_time)),
            "final_time_relative_l2": float(per_time[-1]),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
        },
        "initial_condition_relative_l2": float(
            np.linalg.norm(predicted_ic - exact_ic) / np.linalg.norm(exact_ic)
        ),
        "residual_audit_cpu_float64_unseen": residual_audit(model, 20260711, args.points),
        "periodic_jumps_cpu_float64": periodic_audit(model),
        "coordinate_warning": (
            "If x'=10*x_raw and t'=50*t_raw, the archived PDE coefficients are "
            "1/pi, 10/(3*pi^2), and 1000/(27*pi^4), not one."
        ),
    }
    (args.output / "legacy_observation_pinn_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "legacy_observation_pinn_prediction.npz",
        x=archive["x"],
        t=archive["t"],
        u=prediction,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
