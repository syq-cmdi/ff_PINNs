#!/usr/bin/env python3
"""Add deterministic stitched-field diagnostics to completed marching runs."""

from __future__ import annotations

import torch  # load the workstation's OpenMP runtime before NumPy/SciPy

import argparse
import json
import math
from pathlib import Path

import numpy as np

from train_hc_causal_ks_pinn import field_metrics
from train_hc_marching_ks_pinn import energy_budget_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("hc_marching_seed*_metrics.json"))
    if not paths:
        raise RuntimeError(f"no completed metrics in {args.directory}")
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "2026_recalculation"
        / "reference"
        / "ks_reference_corrected.npz"
    )
    reference = np.load(reference_path)
    target = np.asarray(reference["u_archive"])
    for metrics_path in paths:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        seed = report["config"]["seed"]
        prediction_path = args.directory / f"hc_marching_seed{seed}_prediction.npz"
        archive = np.load(prediction_path)
        predicted = np.asarray(archive["u"])
        report.update(field_metrics(predicted, target))
        exact_initial = -np.sin(np.asarray(archive["x"]) / 3.0)
        report["initial_condition_relative_l2"] = float(
            np.linalg.norm(predicted[0] - exact_initial) / np.linalg.norm(exact_initial)
        )
        report["left_state_constraint"] = (
            "exact with respect to the supplied float32 Fourier state in every window; "
            "the reported initial error includes float32 Fourier coefficient quantization"
        )
        report.update(
            energy_budget_audit(
                predicted, np.asarray(archive["t"]), 6.0 * math.pi
            )
        )
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"seed {seed}: energy relative RMS "
            f"{report['energy_balance_relative_rms']:.6e}"
        )


if __name__ == "__main__":
    main()
