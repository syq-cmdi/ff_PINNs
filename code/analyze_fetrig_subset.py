#!/usr/bin/env python3
"""Audit and summarize the public FETRIG Keyence subset.

Only values carrying the instrument status ``GO`` enter dimensional summary
statistics.  ``ALARM`` values are the sentinel -99.9999 mm.  For spectra only,
short alarm gaps are linearly interpolated; the interpolation fraction and
maximum gap are reported for every trace.  The three directory-labelled Y
stations are processed independently because the archive does not establish
that their acquisitions were synchronous.

Source: Wirth et al., DaRUS V1 (2025), doi:10.18419/DARUS-4998, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import gaussian_kde, kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "experimental" / "fetrig"
DEFAULT_REFERENCE = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "ks_reference_corrected.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "2026_recalculation" / "fetrig"


@dataclass
class KeyenceTrace:
    path: Path
    sampling_seconds: float
    trigger: np.ndarray
    height_mm: np.ndarray
    valid: np.ndarray


def parse_sampling_seconds(text: str) -> float:
    value = text.strip().lower()
    match = re.fullmatch(r"([0-9.]+)\s*(us|ms|s)", value)
    if not match:
        raise ValueError(f"unrecognized sampling cycle: {text!r}")
    magnitude = float(match.group(1))
    return magnitude * {"us": 1e-6, "ms": 1e-3, "s": 1.0}[match.group(2)]


def read_keyence(path: Path) -> KeyenceTrace:
    lines = path.read_text(encoding="utf-8-sig", errors="strict").splitlines()
    sampling_line = next(line for line in lines if line.startswith("Sampling cycle"))
    sampling_row = next(csv.reader([sampling_line]))
    sampling_seconds = parse_sampling_seconds(sampling_row[1])
    header = next(i for i, line in enumerate(lines) if line.startswith("Trigger count"))
    rows = list(csv.reader(lines[header + 1 :]))
    trigger = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    height = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
    valid = np.asarray([len(row) > 2 and row[2] == "GO" for row in rows], dtype=bool)
    if not np.all(np.diff(trigger) == 1):
        raise RuntimeError(f"non-consecutive trigger count in {path}")
    if np.any(height[~valid] != -99.9999):
        raise RuntimeError(f"unexpected ALARM sentinel in {path}")
    return KeyenceTrace(path, sampling_seconds, trigger, height, valid)


def run_lengths(mask: np.ndarray) -> np.ndarray:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)


def interpolate_alarm_gaps(trace: KeyenceTrace) -> np.ndarray:
    ids = np.arange(trace.height_mm.size)
    if trace.valid.sum() < 2:
        raise RuntimeError(f"not enough valid samples in {trace.path}")
    return np.interp(ids, ids[trace.valid], trace.height_mm[trace.valid])


def spectral_summary(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    centered = signal - np.mean(signal)
    fs = 1.0 / dt
    nperseg = min(16384, signal.size)
    frequency, density = welch(
        centered,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        scaling="density",
    )
    admissible = (frequency >= 0.5) & (frequency <= min(500.0, 0.45 * fs))
    peak_index = np.flatnonzero(admissible)[np.argmax(density[admissible])]
    positive = frequency > 0
    weights = density[positive]
    weights = weights / np.sum(weights)
    cumulative = np.cumsum(weights)
    f_positive = frequency[positive]
    return frequency, density, {
        "dominant_frequency_hz_0p5_to_500": float(frequency[peak_index]),
        "spectral_centroid_hz": float(np.sum(f_positive * weights)),
        "frequency_below_95pct_power_hz": float(
            f_positive[min(np.searchsorted(cumulative, 0.95), f_positive.size - 1)]
        ),
        "spectral_entropy": float(-np.sum(weights * np.log(weights + 1e-300)) / np.log(weights.size)),
    }


def block_interval(
    signal: np.ndarray,
    valid: np.ndarray,
    block_samples: int,
    statistic: str,
) -> tuple[float, float]:
    usable = (signal.size // block_samples) * block_samples
    blocks = signal[:usable].reshape(-1, block_samples)
    masks = valid[:usable].reshape(-1, block_samples)
    masked = np.where(masks, blocks, np.nan)
    if statistic == "mean":
        values = np.nanmean(masked, axis=1)
    elif statistic == "std":
        values = np.nanstd(masked, axis=1, ddof=1)
    else:
        raise ValueError(statistic)
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def summarize_trace(trace: KeyenceTrace) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    valid_height = trace.height_mm[trace.valid]
    interpolated = interpolate_alarm_gaps(trace)
    alarm_lengths = run_lengths(~trace.valid)
    frequency, density, spectral = spectral_summary(interpolated, trace.sampling_seconds)
    position = int(re.search(r"Y_(\d+)", str(trace.path)).group(1))
    liquid_re = int(re.search(r"ReL(\d+)", trace.path.name).group(1))
    mean_mm = float(np.mean(valid_height))
    std_mm = float(np.std(valid_height, ddof=1))
    mean_interval = block_interval(
        trace.height_mm,
        trace.valid,
        round(1.0 / trace.sampling_seconds),
        "mean",
    )
    std_interval = block_interval(
        trace.height_mm,
        trace.valid,
        round(1.0 / trace.sampling_seconds),
        "std",
    )
    summary: dict[str, object] = {
        "directory_position_label_mm": position,
        "file_liquid_reynolds_label": liquid_re,
        "gas_reynolds_label": 0,
        "samples": int(trace.height_mm.size),
        "sampling_frequency_hz": float(1.0 / trace.sampling_seconds),
        "duration_seconds": float(trace.height_mm.size * trace.sampling_seconds),
        "valid_samples": int(trace.valid.sum()),
        "valid_fraction": float(np.mean(trace.valid)),
        "alarm_runs": int(alarm_lengths.size),
        "maximum_alarm_gap_samples": int(alarm_lengths.max()) if alarm_lengths.size else 0,
        "maximum_alarm_gap_milliseconds": (
            float(alarm_lengths.max() * trace.sampling_seconds * 1000.0)
            if alarm_lengths.size
            else 0.0
        ),
        "mean_height_mm_GO_only": mean_mm,
        "std_height_mm_GO_only": std_mm,
        "coefficient_of_variation": std_mm / mean_mm,
        "skewness_GO_only": float(skew(valid_height, bias=False)),
        "excess_kurtosis_GO_only": float(kurtosis(valid_height, fisher=True, bias=False)),
        "minimum_height_mm_GO_only": float(np.min(valid_height)),
        "maximum_height_mm_GO_only": float(np.max(valid_height)),
        "one_second_block_mean_interval_2p5_97p5_mm": mean_interval,
        "one_second_block_std_interval_2p5_97p5_mm": std_interval,
        **spectral,
        "spectrum_preprocessing": (
            "linear interpolation across instrument ALARM gaps, mean removal, "
            "linear detrending within 16384-sample Hann windows, 50% overlap"
        ),
    }
    return summary, frequency, density


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    scalar_rows = []
    for row in rows:
        scalar = {}
        for key, value in row.items():
            scalar[key] = json.dumps(value) if isinstance(value, (tuple, list)) else value
        scalar_rows.append(scalar)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def standardized_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    standardized = (values - np.mean(values)) / np.std(values, ddof=1)
    # Deterministic thinning avoids fitting a KDE to hundreds of thousands of
    # nearly redundant points while retaining the full 20 s record span.
    if standardized.size > 10000:
        standardized = standardized[np.linspace(0, standardized.size - 1, 10000, dtype=int)]
    return gaussian_kde(standardized, bw_method="scott")(grid)


def make_statistics_figure(
    summaries: list[dict[str, object]],
    traces: dict[tuple[int, int], KeyenceTrace],
    ks_values: np.ndarray,
    output: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    colors = {100: "#20639B", 350: "#F28E2B", 600: "#3A923A"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    for position in (100, 350, 600):
        subset = sorted(
            [row for row in summaries if row["directory_position_label_mm"] == position],
            key=lambda row: row["file_liquid_reynolds_label"],
        )
        reynolds = [row["file_liquid_reynolds_label"] for row in subset]
        axes[0, 0].plot(
            reynolds,
            [row["mean_height_mm_GO_only"] for row in subset],
            "o-",
            color=colors[position],
            label=rf"reported $x={position}$ mm",
        )
        axes[0, 1].plot(
            reynolds,
            [row["coefficient_of_variation"] for row in subset],
            "o-",
            color=colors[position],
        )
        axes[1, 0].plot(
            reynolds,
            [row["skewness_GO_only"] for row in subset],
            "o-",
            color=colors[position],
        )
    axes[0, 0].set(
        xlabel=r"file label $Re_l$",
        ylabel="mean valid thickness (mm)",
        title="(a) Local dimensional mean",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set(
        xlabel=r"file label $Re_l$",
        ylabel=r"$\sigma_h/\bar h$",
        title="(b) Local fluctuation intensity",
    )
    ks_skew = float(skew(ks_values, bias=False))
    axes[1, 0].axhline(ks_skew, color="black", ls="--", lw=1.2, label="unit KS field")
    axes[1, 0].set(
        xlabel=r"file label $Re_l$",
        ylabel="standardized skewness",
        title="(c) Shape statistic (no amplitude mapping)",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    grid = np.linspace(-4.0, 6.0, 500)
    for position in (100, 350, 600):
        trace = traces[(position, 740)]
        valid = trace.height_mm[trace.valid]
        axes[1, 1].plot(
            grid,
            standardized_kde(valid, grid),
            color=colors[position],
            label=rf"reported $x={position}$ mm, $Re_l=740$",
        )
    axes[1, 1].plot(
        grid,
        standardized_kde(ks_values, grid),
        color="black",
        ls="--",
        lw=1.3,
        label="unit KS field",
    )
    axes[1, 1].set(
        xlabel="standardized fluctuation",
        ylabel="probability density",
        title="(d) Out-of-regime distribution stress test",
        xlim=(-4, 6),
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    for axis in axes.ravel():
        axis.grid(True, alpha=0.22)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_spectrum_figure(
    spectra: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], output: Path
) -> None:
    colors = {100: "#20639B", 350: "#F28E2B", 600: "#3A923A"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True, constrained_layout=True)
    for axis, reynolds in zip(axes, (500, 740, 1100)):
        for position in (100, 350, 600):
            f, p = spectra[(position, reynolds)]
            mask = (f >= 0.5) & (f <= 500)
            area = np.trapezoid(p[mask], f[mask])
            axis.loglog(
                f[mask], p[mask] / area, color=colors[position],
                label=rf"reported $x={position}$ mm"
            )
        axis.set(xlabel="frequency (Hz)", title=rf"file label $Re_l={reynolds}$")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("normalized PSD (Hz$^{-1}$)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input.glob("Y_*/*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"expected 18 Keyence traces, found {len(paths)}")
    summaries: list[dict[str, object]] = []
    traces: dict[tuple[int, int], KeyenceTrace] = {}
    spectra: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for path in paths:
        trace = read_keyence(path)
        summary, frequency, density = summarize_trace(trace)
        position = int(summary["directory_position_label_mm"])
        reynolds = int(summary["file_liquid_reynolds_label"])
        summaries.append(summary)
        traces[(position, reynolds)] = trace
        spectra[(position, reynolds)] = (frequency, density)
        print(
            f"Y_{position} ReL{reynolds}: mean={summary['mean_height_mm_GO_only']:.4f} mm, "
            f"CV={summary['coefficient_of_variation']:.3f}, "
            f"valid={100*summary['valid_fraction']:.2f}%"
        )

    reference = np.load(args.reference)
    ks_values = np.asarray(reference["u_archive"]).ravel()
    ks_summary = {
        "standardized_skewness": float(skew(ks_values, bias=False)),
        "standardized_excess_kurtosis": float(kurtosis(ks_values, fisher=True, bias=False)),
        "mean": float(np.mean(ks_values)),
        "standard_deviation": float(np.std(ks_values, ddof=1)),
        "warning": (
            "The standard KS state H has no calibrated millimetre mapping for FETRIG; "
            "only standardized distribution shape is shown, as an out-of-regime stress test."
        ),
    }
    audit = {
        "source": {
            "citation": (
                "M. Wirth, J. Hagedorn, B. Weigand, and S. Kabelac, "
                "Replication Data for FETRIG initial measurement campaign, "
                "DaRUS V1 (2025)."
            ),
            "doi": "10.18419/DARUS-4998",
            "license": "CC BY 4.0",
            "associated_apparatus_article": {
                "citation": (
                    "M. Wirth, J. Hagedorn, B. Weigand, and S. Kabelac, "
                    "Design of a test rig for the investigation of falling film "
                    "flows with counter-current gas flows, Review of Scientific "
                    "Instruments 96, 055112 (2025)."
                ),
                "doi": "10.1063/5.0263158",
            },
        },
        "reported_experimental_context_from_associated_article": {
            "liquid": "distilled water",
            "gas": "humid air",
            "temperature_K": 298.15,
            "relative_humidity_percent": 80,
            "inclination_degrees": 90,
            "liquid_reynolds_definition": "Re_liq = mass_flow_rate / (wetted_width * dynamic_viscosity)",
            "reported_CCI_positions_mm": [100, 350, 600],
            "sensor_head_as_printed": "Keyence CL-L030",
            "reported_sensor_resolution_micrometres": 0.25,
            "reported_minimum_detectable_water_film_micrometres": 200,
            "reported_selected_sampling_frequency_hz": 5000,
        },
        "selection": (
            "18 ReG0 Keyence files; directory labels Y_100, Y_350, Y_600; "
            "file labels ReL500, 620, 740, 860, 980, 1100"
        ),
        "interpretation_limits": [
            "Reynolds-number settings are read from filenames; their definition is cited from the associated apparatus article.",
            "The archive uses Y_100/Y_350/Y_600 directory labels, whereas the associated article reports streamwise x=100/350/600 mm; this correspondence is stated, not silently renamed.",
            "The three stations are analyzed independently; synchronization is not assumed.",
            "The high-Re measurements are not a validation of weakly nonlinear periodic KS.",
            "No conversion from the standard KS state to millimetres or h(x,t) is asserted.",
        ],
        "ks_standardized_reference": ks_summary,
        "traces": summaries,
    }
    write_csv(summaries, args.output / "fetrig_statistics.csv")
    (args.output / "fetrig_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    make_statistics_figure(
        summaries, traces, ks_values, args.output / "figure_fetrig_stress_test.png"
    )
    make_spectrum_figure(spectra, args.output / "figure_fetrig_spectra.png")


if __name__ == "__main__":
    main()
