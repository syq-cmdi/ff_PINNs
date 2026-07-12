#!/usr/bin/env python3
"""Generate the forensic legacy-PINN audit figure from the locked checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from audit_legacy_observation_pinn import LegacyNet, derivative


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "code" / "obs10k_w120_pde8k.pth"
REFERENCE = ROOT / "results" / "2026_recalculation" / "reference" / "ks_reference_corrected.npz"
AUDIT = ROOT / "results" / "2026_recalculation" / "legacy_audit" / "legacy_observation_pinn_audit.json"
PREDICTION = ROOT / "results" / "2026_recalculation" / "legacy_audit" / "legacy_observation_pinn_prediction.npz"
OUTPUT = ROOT / "results" / "2026_recalculation" / "legacy_audit" / "figure_legacy_audit.png"


def residual_samples(model: LegacyNet, points: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator().manual_seed(20260711)
    unit, correct = [], []
    a, b, c = 1.0 / math.pi, 10.0 / (3.0 * math.pi**2), 1000.0 / (27.0 * math.pi**4)
    for start in range(0, points, 256):
        count = min(256, points - start)
        x = (-10.0 + 20.0 * torch.rand((count, 1), generator=generator, dtype=torch.float64)).requires_grad_(True)
        t = (50.0 * torch.rand((count, 1), generator=generator, dtype=torch.float64)).requires_grad_(True)
        u = model(x, t)
        ut = derivative(u, t)
        ux = derivative(u, x)
        uxx = derivative(ux, x)
        uxxx = derivative(uxx, x)
        uxxxx = derivative(uxxx, x)
        unit.append((ut + u * ux + uxx + uxxxx).detach().numpy().ravel())
        correct.append((ut + a * u * ux + b * uxx + c * uxxxx).detach().numpy().ravel())
    return np.concatenate(unit), np.concatenate(correct) / (15.0 / 50.0)


def schematic(axis) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    entries = [
        (0.69, "1  Coordinate/specification", "Is the stated PDE consistent\nwith the stored coordinates?", "#D6EAF8"),
        (0.39, "2  Numerical approximation", "Do solvers approximate the same\ninitial-boundary-value problem?", "#E8F6F3"),
        (0.09, "3  Model and observation", "Does that PDE and state map\nrepresent the experiment?", "#FCE8E6"),
    ]
    for y, title, body, color in entries:
        patch = FancyBboxPatch(
            (0.07, y), 0.84, 0.20,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color, edgecolor="#31465A", linewidth=1.1,
        )
        axis.add_patch(patch)
        axis.text(0.11, y + 0.135, title, fontsize=8.2, weight="bold", va="center")
        axis.text(0.11, y + 0.062, body, fontsize=7.1, va="center")
    for y0, y1 in ((0.69, 0.59), (0.39, 0.29)):
        axis.add_patch(FancyArrowPatch((0.49, y0), (0.49, y1), arrowstyle="-|>", mutation_scale=10, color="#607D8B"))
    axis.set_title("(a) Three non-interchangeable error levels", loc="left")


def main() -> None:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = LegacyNet().to(torch.float64).eval()
    model.load_state_dict(state)
    unit, correct = residual_samples(model)
    report = json.loads(AUDIT.read_text(encoding="utf-8"))
    reference = np.load(REFERENCE)
    prediction = np.load(PREDICTION)["u"]
    target = reference["u_archive"]
    per_time = np.linalg.norm(prediction - target, axis=1) / np.linalg.norm(target, axis=1)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    schematic(axes[0, 0])

    bins = np.logspace(-3, 3, 60)
    axes[0, 1].hist(np.abs(unit), bins=bins, density=True, histtype="step", lw=1.5, label="imposed unit coefficients")
    axes[0, 1].hist(np.abs(correct), bins=bins, density=True, histtype="step", lw=1.5, label="unit KS in corrected $X,T$")
    axes[0, 1].axvline(np.sqrt(np.mean(unit**2)), color="#20639B", ls="--", lw=1.0)
    axes[0, 1].axvline(np.sqrt(np.mean(correct**2)), color="#F28E2B", ls="--", lw=1.0)
    axes[0, 1].set(
        xscale="log", yscale="log", xlabel=r"unseen $|r_\theta|$", ylabel="density",
        title="(b) CPU-float64 residuals at 4096 points",
    )
    axes[0, 1].legend(frameon=False, fontsize=7)

    jumps = report["periodic_jumps_cpu_float64"]["corrected_X_derivatives"]
    orders = np.arange(4)
    values = [item["rms"] for item in jumps]
    axes[1, 0].bar(orders, values, color=["#20639B", "#4E79A7", "#F28E2B", "#E15759"])
    axes[1, 0].set(
        yscale="log",
        xticks=orders,
        xticklabels=(r"$u$", r"$u_x$", r"$u_{xx}$", r"$u_{xxx}$"),
        ylabel="endpoint-jump RMS",
        title="(c) Periodicity deteriorates with derivative order",
    )

    axes[1, 1].semilogy(reference["t"], per_time, color="#20639B")
    axes[1, 1].axhline(
        report["field_fit_to_archive"]["global_relative_l2"],
        color="#E15759", ls="--", lw=1.1, label="global field-fit error",
    )
    axes[1, 1].set(
        xlabel=r"corrected $T$", ylabel="relative $L_2$",
        title="(d) Sample fit does not certify the PDE",
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    for axis in axes.ravel()[1:]:
        axis.grid(True, which="both", alpha=0.20)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
