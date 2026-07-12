#!/usr/bin/env python3
"""Aggregate and plot the reproducible sparse Fourier/Galerkin sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "2026_recalculation" / "inverse"
OUTPUT = INPUT / "summary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(input_dir.glob("baseline_sweep_seed*_noise*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        estimator = report["estimators"]["fourier"]
        rows.append(
            {
                "seed": report["sample"]["seed"],
                "noise_relative_std": report["sample"]["noise_relative_to_global_field_std"],
                "sampling_fraction": report["sample"]["sampling_fraction"],
                "lambda_1": estimator["estimate"][0],
                "lambda_2": estimator["estimate"][1],
                "lambda_4": estimator["estimate"][2],
                "mean_relative_coefficient_error": estimator["mean_relative_coefficient_error"],
                "field_reconstruction_relative_l2": estimator[
                    "field_reconstruction_relative_l2_at_sampled_times"
                ],
                "wall_seconds": estimator["wall_seconds"],
                "source_metrics": str(path.relative_to(ROOT)),
            }
        )
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 sweep runs, found {len(rows)}")
    with (output_dir / "inverse_fourier_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    noise_levels = sorted({row["noise_relative_std"] for row in rows})
    aggregate = {}
    for noise in noise_levels:
        subset = [row for row in rows if row["noise_relative_std"] == noise]
        aggregate[f"{noise:.4f}"] = {
            "runs": len(subset),
            "mean_coefficients": [
                float(np.mean([row[key] for row in subset]))
                for key in ("lambda_1", "lambda_2", "lambda_4")
            ],
            "sample_std_coefficients": [
                float(np.std([row[key] for row in subset], ddof=1))
                for key in ("lambda_1", "lambda_2", "lambda_4")
            ],
            "mean_absolute_error_all_coefficients": float(
                np.mean(
                    [
                        abs(row[key] - 1.0)
                        for row in subset
                        for key in ("lambda_1", "lambda_2", "lambda_4")
                    ]
                )
            ),
            "mean_field_reconstruction_relative_l2": float(
                np.mean([row["field_reconstruction_relative_l2"] for row in subset])
            ),
        }
    (output_dir / "inverse_fourier_sweep_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
    colors = ("#20639B", "#F28E2B", "#3A923A")
    labels = (r"$\lambda_1$ ($u u_X$)", r"$\lambda_2$ ($u_{XX}$)", r"$\lambda_4$ ($u_{XXXX}$)")
    x = np.asarray(noise_levels) * 100.0
    for index, (key, label, color) in enumerate(
        zip(("lambda_1", "lambda_2", "lambda_4"), labels, colors)
    ):
        means = []
        standard = []
        for noise in noise_levels:
            values = [row[key] for row in rows if row["noise_relative_std"] == noise]
            means.append(np.mean(values))
            standard.append(np.std(values, ddof=1))
        axes[0].errorbar(x, means, yerr=standard, marker="o", capsize=3, color=color, label=label)
    axes[0].axhline(1.0, color="black", ls="--", lw=1.0, label="target")
    axes[0].set(
        xlabel=r"noise standard deviation / $\mathrm{std}(u)$ (%)",
        ylabel="identified coefficient",
        title="(a) Five-seed coefficient recovery",
    )
    axes[0].legend(frameon=False, fontsize=7)
    mean_errors = [
        100.0 * aggregate[f"{noise:.4f}"]["mean_absolute_error_all_coefficients"]
        for noise in noise_levels
    ]
    reconstruction = [
        100.0 * aggregate[f"{noise:.4f}"]["mean_field_reconstruction_relative_l2"]
        for noise in noise_levels
    ]
    axes[1].plot(x, mean_errors, "o-", label="coefficient mean absolute error")
    axes[1].plot(x, reconstruction, "s-", label="sampled-time field reconstruction")
    axes[1].set(
        xlabel=r"noise standard deviation / $\mathrm{std}(u)$ (%)",
        ylabel="error (%)",
        title="(b) Noise sensitivity at 12.56% sampling",
    )
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(True, alpha=0.22)
    fig.savefig(output_dir / "figure_inverse_noise_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sampling_patterns = {
        "51 x 64\n8/8 modes": "sampling_m8_51_64_seed*_noise0.0100/metrics.json",
        "101 x 64\n8/8 modes": "sampling_m8_101_64_seed*_noise0.0100/metrics.json",
        "51 x 128\n12/8 modes": "sampling_51_128_seed*_noise0.0100/metrics.json",
        "101 x 128\n12/8 modes": "baseline_sweep_seed*_noise0.0100/metrics.json",
    }
    sampling_rows = []
    for label, pattern in sampling_patterns.items():
        paths = sorted(input_dir.glob(pattern))
        if len(paths) != 5:
            raise RuntimeError(f"expected five runs for {label}, found {len(paths)}")
        for path in paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            estimator = report["estimators"]["fourier"]
            sampling_rows.append(
                {
                    "configuration": label.replace("\n", " "),
                    "seed": report["sample"]["seed"],
                    "n_times": report["sample"]["n_time_snapshots"],
                    "n_space": report["sample"]["n_spatial_observations_per_snapshot"],
                    "sampling_fraction": report["sample"]["sampling_fraction"],
                    "fourier_modes": estimator["spatial_modes_reconstructed"],
                    "galerkin_modes": estimator["galerkin_modes_regressed"],
                    "lambda_1": estimator["estimate"][0],
                    "lambda_2": estimator["estimate"][1],
                    "lambda_4": estimator["estimate"][2],
                    "mean_relative_coefficient_error": estimator[
                        "mean_relative_coefficient_error"
                    ],
                    "field_reconstruction_relative_l2": estimator[
                        "field_reconstruction_relative_l2_at_sampled_times"
                    ],
                }
            )
    with (output_dir / "inverse_fourier_sampling_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sampling_rows[0]))
        writer.writeheader()
        writer.writerows(sampling_rows)
    sampling_aggregate = {}
    for label in sampling_patterns:
        flat_label = label.replace("\n", " ")
        subset = [row for row in sampling_rows if row["configuration"] == flat_label]
        sampling_aggregate[flat_label] = {
            "runs": len(subset),
            "sampling_fraction": subset[0]["sampling_fraction"],
            "fourier_modes": subset[0]["fourier_modes"],
            "galerkin_modes": subset[0]["galerkin_modes"],
            "mean_coefficients": [
                float(np.mean([row[key] for row in subset]))
                for key in ("lambda_1", "lambda_2", "lambda_4")
            ],
            "sample_std_coefficients": [
                float(np.std([row[key] for row in subset], ddof=1))
                for key in ("lambda_1", "lambda_2", "lambda_4")
            ],
            "mean_relative_coefficient_error": float(
                np.mean([row["mean_relative_coefficient_error"] for row in subset])
            ),
            "std_relative_coefficient_error": float(
                np.std([row["mean_relative_coefficient_error"] for row in subset], ddof=1)
            ),
            "mean_field_reconstruction_relative_l2": float(
                np.mean([row["field_reconstruction_relative_l2"] for row in subset])
            ),
        }
    (output_dir / "inverse_fourier_sampling_summary.json").write_text(
        json.dumps(sampling_aggregate, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    for key, label, color in zip(("lambda_1", "lambda_2", "lambda_4"), labels, colors):
        means, standard = [], []
        for noise in noise_levels:
            values = [row[key] for row in rows if row["noise_relative_std"] == noise]
            means.append(np.mean(values))
            standard.append(np.std(values, ddof=1))
        axes[0, 0].errorbar(x, means, yerr=standard, marker="o", capsize=3, color=color, label=label)
    axes[0, 0].axhline(1.0, color="black", ls="--", lw=1.0, label="target")
    axes[0, 0].set(
        xlabel=r"noise / $\mathrm{std}(u)$ (%)",
        ylabel="identified coefficient",
        title="(a) Five-seed noise sweep",
    )
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].plot(x, mean_errors, "o-", label="coefficient error")
    axes[0, 1].plot(x, reconstruction, "s-", label="field reconstruction")
    axes[0, 1].set(
        xlabel=r"noise / $\mathrm{std}(u)$ (%)",
        ylabel="error (%)",
        title="(b) Error at 12.56% sampling",
    )
    axes[0, 1].legend(frameon=False, fontsize=7)

    sampling_labels = list(sampling_patterns)
    positions = np.arange(len(sampling_labels))
    width = 0.22
    for idx, (key, coeff_label, color) in enumerate(
        zip(("lambda_1", "lambda_2", "lambda_4"), labels, colors)
    ):
        means = []
        stds = []
        for label in sampling_labels:
            subset = [
                row[key]
                for row in sampling_rows
                if row["configuration"] == label.replace("\n", " ")
            ]
            means.append(np.mean(subset))
            stds.append(np.std(subset, ddof=1))
        axes[1, 0].bar(
            positions + (idx - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            color=color,
            label=coeff_label,
        )
    axes[1, 0].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[1, 0].set_xticks(positions, sampling_labels, fontsize=7)
    axes[1, 0].set(ylabel="identified coefficient", title="(c) Sampling allocation at 1% noise")
    axes[1, 0].legend(frameon=False, fontsize=7)

    fractions, coefficient_errors, coefficient_stds, field_errors = [], [], [], []
    for label in sampling_labels:
        item = sampling_aggregate[label.replace("\n", " ")]
        fractions.append(100.0 * item["sampling_fraction"])
        coefficient_errors.append(100.0 * item["mean_relative_coefficient_error"])
        coefficient_stds.append(100.0 * item["std_relative_coefficient_error"])
        field_errors.append(100.0 * item["mean_field_reconstruction_relative_l2"])
    axes[1, 1].errorbar(
        fractions,
        coefficient_errors,
        yerr=coefficient_stds,
        fmt="o",
        capsize=3,
        label="coefficient error",
    )
    for fraction, error, label in zip(fractions, coefficient_errors, sampling_labels):
        axes[1, 1].annotate(
            label.split("\n")[0],
            (fraction, error),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[1, 1].plot(fractions, field_errors, "s", label="field reconstruction")
    axes[1, 1].set(
        xlabel="observed grid fraction (%)",
        ylabel="error (%)",
        title="(d) Space-time sampling sensitivity",
        yscale="log",
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    for axis in axes.ravel():
        axis.grid(True, alpha=0.22)
    fig.savefig(output_dir / "figure_inverse_identification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
