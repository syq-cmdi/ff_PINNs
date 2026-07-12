#!/usr/bin/env python3
"""Build an audited five-seed publication metrics record for HC-marching.

This script is intentionally strict.  It requires completed seeds 0--4 with
identical configurations (apart from ``seed``), matching prediction archives,
and one checkpoint per marching window.  It recomputes field and energy
diagnostics from the saved predictions before aggregating them.  Missing or
inconsistent inputs cause a non-zero exit *before* publication_metrics.json is
created or replaced.

The output uses medians and interquartile ranges for seed-level scalar metrics.
Interface transfer error, mean drift, initial-condition error, and periodic
boundary jumps are reported as worst cases across all five seeds.

For the runtime ratio, a validated ``etdrk4_runtime_audit.json`` takes
precedence and contributes its repeated-timing median (with q25/q75 recorded).
Only when that file is absent does the script use the single timing in
``etdrk4_convergence.csv`` and label it explicitly as a fallback.

Run after all five training runs finish:

    python3 code/build_publication_metrics.py

Validation without writing:

    python3 code/build_publication_metrics.py --check-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "results" / "2026_recalculation" / "marching_final"
DEFAULT_REFERENCE = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "ks_reference_corrected.npz"
)
DEFAULT_SCALE_AUDIT = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "scale_reference_audit.json"
)
DEFAULT_CONVERGENCE = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "etdrk4_convergence.csv"
)
DEFAULT_RUNTIME_AUDIT = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "etdrk4_runtime_audit.json"
)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_PERIODIC_ORDERS = (0, 1, 2, 3)
EXPECTED_LOCKED_CONFIG = {
    "method": "hc_marching",
    "windows": 40,
    "adam_steps_per_window": 1000,
    "lbfgs_iterations_per_window": 200,
    "collocation": 1024,
    "validation_collocation": 4096,
    "harmonics": 8,
    "state_modes": 24,
    "state_grid": 512,
    "width": 64,
    "depth": 5,
    "learning_rate": 0.001,
    "causal_bins": 4,
    "causal_epsilon": 0.01,
    "attention_exponent": 0.5,
    "gradient_clip": 100.0,
}
METRIC_PREFIX = "hc_marching_seed"
OUTPUT_NAME = "publication_metrics.json"


class PublicationMetricsError(RuntimeError):
    """Input audit failed; no publication aggregate may be written."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--scale-audit", type=Path, default=DEFAULT_SCALE_AUDIT)
    parser.add_argument("--convergence", type=Path, default=DEFAULT_CONVERGENCE)
    parser.add_argument("--runtime-audit", type=Path, default=DEFAULT_RUNTIME_AUDIT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Perform every audit and aggregation calculation without writing JSON",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise PublicationMetricsError(message)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"missing {label}: {path}")


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be a JSON object")
    return value


def require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        fail(f"{label} must be a JSON array")
    return value


def finite_float(value: Any, label: str, *, nonnegative: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicationMetricsError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        fail(f"{label} is not finite: {number!r}")
    if nonnegative and number < 0.0:
        fail(f"{label} must be non-negative, got {number}")
    return number


def integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        fail(f"{label} must be an integer, got boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PublicationMetricsError(f"{label} is not an integer: {value!r}") from exc
    if number != value:
        fail(f"{label} is not an exact integer: {value!r}")
    if minimum is not None and number < minimum:
        fail(f"{label} must be at least {minimum}, got {number}")
    return number


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    require_file(path, "provenance file")
    content = path.read_bytes()
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
        displayed = str(relative)
    except ValueError:
        displayed = str(path.resolve())
    return {
        "path": displayed,
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def load_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require_file(path, label)
    content = path.read_bytes()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PublicationMetricsError(f"invalid JSON in {path}: {exc}") from exc
    report = dict(require_mapping(parsed, label))
    provenance = file_record(path)
    # The record must describe the same byte snapshot that was parsed.  A
    # concurrent training write therefore cannot silently enter the aggregate.
    if provenance["sha256"] != sha256_bytes(content):
        fail(f"{path} changed while it was being audited")
    return report, provenance


def metric_path(directory: Path, seed: int) -> Path:
    return directory / f"{METRIC_PREFIX}{seed}_metrics.json"


def prediction_path(directory: Path, seed: int) -> Path:
    return directory / f"{METRIC_PREFIX}{seed}_prediction.npz"


def checkpoint_paths(directory: Path, seed: int) -> list[Path]:
    return sorted(directory.glob(f"{METRIC_PREFIX}{seed}_window*.pt"))


def preflight_seed_files(directory: Path) -> list[Path]:
    discovered: dict[int, Path] = {}
    name_pattern = re.compile(r"hc_marching_seed([0-9]+)_metrics\.json")
    for path in sorted(directory.glob("hc_marching_seed*_metrics.json")):
        match = name_pattern.fullmatch(path.name)
        if match is None:
            fail(f"unexpected marching metric filename: {path.name}")
        seed = int(match.group(1))
        if seed in discovered:
            fail(f"duplicate metric files for seed {seed}")
        discovered[seed] = path
    present_seeds = sorted(discovered)
    extra_seeds = sorted(set(present_seeds) - set(EXPECTED_SEEDS))
    if extra_seeds:
        fail(
            f"unexpected extra completed seed metrics {extra_seeds}; the locked "
            "publication seed set is exactly 0--4"
        )
    expected = [metric_path(directory, seed) for seed in EXPECTED_SEEDS]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        present = [seed for seed in EXPECTED_SEEDS if seed in discovered]
        absent = [
            seed
            for seed, path in zip(EXPECTED_SEEDS, expected)
            if not path.is_file()
        ]
        fail(
            "five completed seed metrics are required before publication aggregation; "
            f"present seeds={present}, missing seeds={absent}; no output was written"
        )
    return expected


def assert_close(
    actual: float,
    expected: float,
    label: str,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        fail(f"{label} mismatch: stored={actual:.17g}, recomputed={expected:.17g}")


def phase_invariant_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.empty(predicted.shape[0], dtype=np.float64)
    for index, (prediction, reference) in enumerate(zip(predicted, target)):
        correlation = np.fft.ifft(
            np.fft.fft(prediction) * np.conj(np.fft.fft(reference))
        ).real
        shift = int(np.argmax(correlation))
        aligned = np.roll(prediction, -shift)
        values[index] = np.linalg.norm(aligned - reference) / np.linalg.norm(reference)
    return values


def recompute_field_metrics(
    predicted: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    difference = predicted - target
    time_relative = np.linalg.norm(difference, axis=1) / np.linalg.norm(target, axis=1)
    phase_relative = phase_invariant_errors(predicted, target)
    return {
        "global_relative_l2": float(
            np.linalg.norm(difference) / np.linalg.norm(target)
        ),
        "median_time_relative_l2": float(np.median(time_relative)),
        "maximum_time_relative_l2": float(np.max(time_relative)),
        "final_time_relative_l2": float(time_relative[-1]),
        "median_phase_invariant_relative_l2": float(np.median(phase_relative)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "predicted_mean_drift": float(
            np.max(np.abs(np.mean(predicted, axis=1) - np.mean(predicted[0])))
        ),
        "time_relative_l2": time_relative,
        "time_phase_invariant_relative_l2": phase_relative,
    }


def recompute_energy_metrics(
    field: np.ndarray, times: np.ndarray, length: float
) -> dict[str, float]:
    if times.size < 7:
        fail("at least seven output times are required for the energy audit")
    dt = np.diff(times)
    if not np.allclose(dt, dt[0], rtol=1.0e-12, atol=1.0e-14):
        fail("energy audit requires a uniform time grid")
    spatial_points = field.shape[1]
    wavenumber = 2.0 * np.pi * np.fft.fftfreq(
        spatial_points, d=length / spatial_points
    )
    spectrum = np.fft.fft(field, axis=1)
    ux = np.fft.ifft(1j * wavenumber[None, :] * spectrum, axis=1).real
    uxx = np.fft.ifft(-(wavenumber[None, :] ** 2) * spectrum, axis=1).real
    energy = 0.5 * np.mean(field**2, axis=1)
    energy_derivative = savgol_filter(
        energy,
        window_length=7,
        polyorder=5,
        deriv=1,
        delta=float(dt[0]),
        mode="interp",
    )
    right_hand_side = np.mean(ux**2, axis=1) - np.mean(uxx**2, axis=1)
    balance_residual = energy_derivative - right_hand_side
    core = slice(3, -3)
    return {
        "energy_balance_relative_rms": float(
            np.linalg.norm(balance_residual[core])
            / np.linalg.norm(right_hand_side[core])
        ),
        "energy_balance_absolute_rms": float(
            np.sqrt(np.mean(balance_residual[core] ** 2))
        ),
    }


def validate_configurations(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    locked: list[dict[str, Any]] = []
    seen_seeds: list[int] = []
    for position, report in enumerate(reports):
        config = dict(require_mapping(report.get("config"), f"seed {position} config"))
        seed = integer(config.get("seed"), f"seed {position} config.seed", minimum=0)
        seen_seeds.append(seed)
        if seed != EXPECTED_SEEDS[position]:
            fail(
                f"metric file for expected seed {EXPECTED_SEEDS[position]} declares seed {seed}"
            )
        if config.get("method") != "hc_marching":
            fail(f"seed {seed} method must be 'hc_marching', got {config.get('method')!r}")
        config.pop("seed")
        locked.append(config)
    if tuple(seen_seeds) != EXPECTED_SEEDS:
        fail(f"seed set/order mismatch: expected {EXPECTED_SEEDS}, got {tuple(seen_seeds)}")
    signatures = [canonical_json(config) for config in locked]
    if len(set(signatures)) != 1:
        differences = {
            seed: config for seed, config in zip(EXPECTED_SEEDS, locked)
        }
        fail(
            "seed configurations differ after removing only the seed field: "
            + canonical_json(differences)
        )
    if canonical_json(locked[0]) != canonical_json(EXPECTED_LOCKED_CONFIG):
        fail(
            "five seeds are internally consistent but do not match the manuscript's "
            "locked production configuration: expected="
            + canonical_json(EXPECTED_LOCKED_CONFIG)
            + ", actual="
            + canonical_json(locked[0])
        )
    return locked[0]


def validate_shared_runtime_metadata(
    reports: Sequence[dict[str, Any]], locked_config: Mapping[str, Any]
) -> dict[str, Any]:
    devices = [str(report.get("device")) for report in reports]
    torch_versions = [str(report.get("torch_version")) for report in reports]
    python_versions = [str(report.get("python_version")) for report in reports]
    if len(set(devices)) != 1:
        fail(f"training devices differ across seeds: {devices}")
    if len(set(torch_versions)) != 1:
        fail(f"PyTorch versions differ across seeds: {torch_versions}")
    if len(set(python_versions)) != 1:
        fail(f"Python versions differ across seeds: {python_versions}")
    if devices[0] != "mps":
        fail(f"locked publication training device must be mps, got {devices[0]!r}")
    if not torch_versions[0].startswith("2.10"):
        fail(
            "manuscript specifies PyTorch 2.10, but per-seed metrics record "
            f"{torch_versions[0]!r}"
        )
    per_window_counts = [
        integer(
            report.get("parameter_count_per_window"),
            f"seed {seed} parameter_count_per_window",
            minimum=1,
        )
        for seed, report in zip(EXPECTED_SEEDS, reports)
    ]
    all_window_counts = [
        integer(
            report.get("parameter_count_all_windows"),
            f"seed {seed} parameter_count_all_windows",
            minimum=1,
        )
        for seed, report in zip(EXPECTED_SEEDS, reports)
    ]
    if len(set(per_window_counts)) != 1 or len(set(all_window_counts)) != 1:
        fail("parameter counts differ across seeds")
    windows = integer(locked_config.get("windows"), "locked config windows", minimum=1)
    if all_window_counts[0] != per_window_counts[0] * windows:
        fail(
            "parameter_count_all_windows is not parameter_count_per_window * windows"
        )
    return {
        "training_device": devices[0],
        "torch_version": torch_versions[0],
        "python_version": python_versions[0],
        "training_model_and_collocation_precision": (
            "float32 according to the hashed training source"
        ),
        "precision_evidence": (
            "Per-seed metrics do not carry an explicit dtype field. The hashed current "
            "training source constructs model/collocation tensors in float32 and sets "
            "float32 matmul precision to highest; CPU audit precision is explicit in "
            "the metric key names."
        ),
        "independent_residual_and_periodic_audit": "CPU float64",
        "parameter_count_per_window": per_window_counts[0],
        "parameter_count_all_windows": all_window_counts[0],
    }


def validate_window_records(
    report: Mapping[str, Any], seed: int, locked_config: Mapping[str, Any]
) -> dict[str, Any]:
    windows = integer(locked_config.get("windows"), "locked config windows", minimum=1)
    diagnostics = require_sequence(report.get("diagnostics"), f"seed {seed} diagnostics")
    transfers = require_sequence(
        report.get("interface_transfers"), f"seed {seed} interface_transfers"
    )
    if len(diagnostics) != windows:
        fail(f"seed {seed} has {len(diagnostics)} diagnostics, expected {windows}")
    if len(transfers) != windows:
        fail(f"seed {seed} has {len(transfers)} transfers, expected {windows}")
    expected_edges = np.linspace(0.0, 15.0, windows + 1)
    window_seconds: list[float] = []
    for window, item_raw in enumerate(diagnostics):
        item = require_mapping(item_raw, f"seed {seed} diagnostic {window}")
        if integer(item.get("window"), f"seed {seed} diagnostic window") != window:
            fail(f"seed {seed} diagnostic windows are not contiguous")
        assert_close(
            finite_float(item.get("t_left"), f"seed {seed} window {window} t_left"),
            float(expected_edges[window]),
            f"seed {seed} window {window} left edge",
        )
        assert_close(
            finite_float(item.get("t_right"), f"seed {seed} window {window} t_right"),
            float(expected_edges[window + 1]),
            f"seed {seed} window {window} right edge",
        )
        window_seconds.append(
            finite_float(item.get("seconds"), f"seed {seed} window {window} seconds")
        )
        finite_float(
            item.get("validation_residual_rms"),
            f"seed {seed} window {window} validation residual RMS",
        )
        finite_float(
            item.get("reference_global_relative_l2_evaluation_only"),
            f"seed {seed} window {window} local reference error",
        )
        if item.get("lbfgs_status") != "completed":
            fail(f"seed {seed} window {window} LBFGS status is not completed")
        integer(
            item.get("lbfgs_closures"),
            f"seed {seed} window {window} LBFGS closures",
            minimum=1,
        )
    relative_values: list[float] = []
    absolute_values: list[float] = []
    for window, item_raw in enumerate(transfers):
        item = require_mapping(item_raw, f"seed {seed} transfer {window}")
        if integer(item.get("window"), f"seed {seed} transfer window") != window:
            fail(f"seed {seed} transfer windows are not contiguous")
        relative_values.append(
            finite_float(
                item.get("relative_fourier_transfer_error"),
                f"seed {seed} window {window} relative transfer error",
            )
        )
        absolute_values.append(
            finite_float(
                item.get("maximum_absolute_fourier_transfer_error"),
                f"seed {seed} window {window} absolute transfer error",
            )
        )
    return {
        "maximum_relative_transfer_error": max(relative_values[:-1]),
        "maximum_absolute_transfer_error": max(absolute_values[:-1]),
        "final_diagnostic_relative_projection_error": relative_values[-1],
        "final_diagnostic_maximum_absolute_projection_error": absolute_values[-1],
        "sum_window_seconds": float(sum(window_seconds)),
    }


def validate_residual_and_periodicity(
    report: Mapping[str, Any], seed: int, windows: int
) -> dict[str, Any]:
    residual = require_mapping(
        report.get("cpu_float64_unseen_residual"),
        f"seed {seed} CPU float64 unseen residual",
    )
    points = integer(residual.get("points"), f"seed {seed} residual points", minimum=1)
    expected_points = windows * 128
    if points != expected_points:
        fail(
            f"seed {seed} residual audit has {points} points; expected "
            f"128 per window x {windows} = {expected_points}"
        )
    residual_values = {
        "points": points,
        "rms": finite_float(residual.get("rms"), f"seed {seed} residual RMS"),
        "mean_absolute": finite_float(
            residual.get("mean_absolute"), f"seed {seed} residual mean absolute"
        ),
        "q95_absolute": finite_float(
            residual.get("q95_absolute"), f"seed {seed} residual q95 absolute"
        ),
        "maximum_absolute": finite_float(
            residual.get("maximum_absolute"), f"seed {seed} residual max absolute"
        ),
    }
    if not (
        residual_values["mean_absolute"]
        <= residual_values["q95_absolute"]
        <= residual_values["maximum_absolute"]
    ):
        fail(f"seed {seed} residual absolute summaries are not ordered")
    jumps = require_sequence(
        report.get("cpu_float64_periodic_jumps"),
        f"seed {seed} CPU float64 periodic jumps",
    )
    if len(jumps) != len(EXPECTED_PERIODIC_ORDERS):
        fail(f"seed {seed} must contain periodic jumps for derivative orders 0--3")
    jump_values: dict[int, dict[str, float]] = {}
    for expected_order, item_raw in zip(EXPECTED_PERIODIC_ORDERS, jumps):
        item = require_mapping(item_raw, f"seed {seed} periodic order {expected_order}")
        order = integer(item.get("derivative_order"), f"seed {seed} derivative order")
        if order != expected_order:
            fail(
                f"seed {seed} periodic derivative order mismatch: "
                f"expected {expected_order}, got {order}"
            )
        jump_values[order] = {
            "rms": finite_float(item.get("rms"), f"seed {seed} order {order} jump RMS"),
            "maximum_absolute": finite_float(
                item.get("maximum_absolute"),
                f"seed {seed} order {order} maximum jump",
            ),
        }
    return {"residual": residual_values, "periodic_jumps": jump_values}


def validate_prediction_and_recompute(
    directory: Path,
    seed: int,
    report: Mapping[str, Any],
    reference_x: np.ndarray,
    reference_t: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = prediction_path(directory, seed)
    require_file(path, f"seed {seed} prediction archive")
    try:
        with np.load(path) as archive:
            if set(archive.files) != {"x", "t", "u"}:
                fail(
                    f"seed {seed} prediction archive keys must be x,t,u; "
                    f"got {archive.files}"
                )
            x = np.asarray(archive["x"], dtype=np.float64)
            t = np.asarray(archive["t"], dtype=np.float64)
            predicted = np.asarray(archive["u"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise PublicationMetricsError(
            f"cannot read seed {seed} prediction archive {path}: {exc}"
        ) from exc
    if x.shape != reference_x.shape or not np.array_equal(x, reference_x):
        fail(f"seed {seed} prediction x grid differs from the locked reference")
    if t.shape != reference_t.shape or not np.array_equal(t, reference_t):
        fail(f"seed {seed} prediction time grid differs from the locked reference")
    if predicted.shape != target.shape:
        fail(
            f"seed {seed} prediction shape {predicted.shape} != reference {target.shape}"
        )
    if not np.all(np.isfinite(predicted)):
        fail(f"seed {seed} prediction contains NaN or infinity")

    field = recompute_field_metrics(predicted, target)
    energy = recompute_energy_metrics(predicted, reference_t, 6.0 * np.pi)
    stored_scalar_keys = (
        "global_relative_l2",
        "median_time_relative_l2",
        "maximum_time_relative_l2",
        "final_time_relative_l2",
        "median_phase_invariant_relative_l2",
        "maximum_absolute_error",
        "predicted_mean_drift",
    )
    for key in stored_scalar_keys:
        assert_close(
            finite_float(report.get(key), f"seed {seed} stored {key}"),
            float(field[key]),
            f"seed {seed} {key}",
        )
    for key in ("energy_balance_relative_rms", "energy_balance_absolute_rms"):
        assert_close(
            finite_float(report.get(key), f"seed {seed} stored {key}"),
            energy[key],
            f"seed {seed} {key}",
            relative_tolerance=1.0e-9,
        )
    stored_time = np.asarray(
        require_sequence(report.get("time_relative_l2"), f"seed {seed} time errors"),
        dtype=np.float64,
    )
    if stored_time.shape != field["time_relative_l2"].shape or not np.allclose(
        stored_time, field["time_relative_l2"], rtol=1.0e-10, atol=1.0e-12
    ):
        fail(f"seed {seed} stored per-time relative errors fail recomputation")
    stored_phase_raw = report.get("time_phase_invariant_relative_l2")
    stored_phase_present = stored_phase_raw is not None
    if stored_phase_present:
        stored_phase = np.asarray(
            require_sequence(
                stored_phase_raw,
                f"seed {seed} phase-invariant time errors",
            ),
            dtype=np.float64,
        )
        if stored_phase.shape != field[
            "time_phase_invariant_relative_l2"
        ].shape or not np.allclose(
            stored_phase,
            field["time_phase_invariant_relative_l2"],
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            fail(f"seed {seed} stored phase-invariant errors fail recomputation")
    exact_initial = -np.sin(reference_x / 3.0)
    initial_error = float(
        np.linalg.norm(predicted[0] - exact_initial) / np.linalg.norm(exact_initial)
    )
    stored_initial_error = finite_float(
        report.get("initial_condition_relative_l2"),
        f"seed {seed} stored initial-condition error",
    )
    return (
        {
            **{key: float(field[key]) for key in stored_scalar_keys},
            **energy,
            "initial_condition_relative_l2": initial_error,
            "stored_initial_condition_relative_l2": stored_initial_error,
            "stored_initial_matches_stitched_prediction": math.isclose(
                stored_initial_error, initial_error, rel_tol=1.0e-10, abs_tol=1.0e-12
            ),
            "stored_phase_time_series_present": stored_phase_present,
        },
        file_record(path),
    )


def checkpoint_manifest(
    directory: Path, seed: int, expected_windows: int
) -> dict[str, Any]:
    paths = checkpoint_paths(directory, seed)
    expected_names = [
        f"{METRIC_PREFIX}{seed}_window{window:02d}.pt"
        for window in range(expected_windows)
    ]
    actual_names = [path.name for path in paths]
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        unexpected = sorted(set(actual_names) - set(expected_names))
        fail(
            f"seed {seed} checkpoint set is incomplete/inconsistent; "
            f"missing={missing}, unexpected={unexpected}"
        )
    records = [file_record(path) for path in paths]
    manifest_text = "".join(
        f"{record['path']}\t{record['bytes']}\t{record['sha256']}\n"
        for record in records
    ).encode("utf-8")
    return {
        "count": len(records),
        "manifest_sha256": sha256_bytes(manifest_text),
        "files": records,
    }


def seed_summary(
    directory: Path,
    seed: int,
    report: dict[str, Any],
    locked_config: Mapping[str, Any],
    reference_x: np.ndarray,
    reference_t: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    window = validate_window_records(report, seed, locked_config)
    residual_periodic = validate_residual_and_periodicity(
        report,
        seed,
        integer(locked_config["windows"], "locked windows", minimum=1),
    )
    field_energy, prediction_provenance = validate_prediction_and_recompute(
        directory,
        seed,
        report,
        reference_x,
        reference_t,
        target,
    )
    training_seconds = finite_float(
        report.get("training_seconds"), f"seed {seed} training seconds"
    )
    overhead_seconds = training_seconds - float(window["sum_window_seconds"])
    if overhead_seconds < -1.0e-6 or overhead_seconds > max(60.0, 0.05 * training_seconds):
        fail(
            f"seed {seed} training_seconds is inconsistent with the sum of the 40 "
            f"window timers: total={training_seconds}, windows={window['sum_window_seconds']}"
        )
    return (
        {
            "seed": seed,
            **field_energy,
            "training_seconds": training_seconds,
            "sum_window_seconds": window["sum_window_seconds"],
            "training_timer_overhead_seconds": overhead_seconds,
            "residual_rms_cpu_float64": residual_periodic["residual"]["rms"],
            "residual_q95_absolute_cpu_float64": residual_periodic["residual"][
                "q95_absolute"
            ],
            "residual_maximum_absolute_cpu_float64": residual_periodic["residual"][
                "maximum_absolute"
            ],
            "maximum_relative_transfer_error": window[
                "maximum_relative_transfer_error"
            ],
            "maximum_absolute_transfer_error": window[
                "maximum_absolute_transfer_error"
            ],
            "final_diagnostic_relative_projection_error": window[
                "final_diagnostic_relative_projection_error"
            ],
            "final_diagnostic_maximum_absolute_projection_error": window[
                "final_diagnostic_maximum_absolute_projection_error"
            ],
            "periodic_jumps": residual_periodic["periodic_jumps"],
        },
        {"prediction": prediction_provenance},
    )


def distribution(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    if tuple(sorted(values_by_seed)) != EXPECTED_SEEDS:
        fail(f"distribution seed set is incomplete: {sorted(values_by_seed)}")
    values = np.asarray(
        [finite_float(values_by_seed[seed], f"seed {seed} aggregate value") for seed in EXPECTED_SEEDS],
        dtype=np.float64,
    )
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {
        "n_seeds": len(EXPECTED_SEEDS),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "values_by_seed": {str(seed): float(values_by_seed[seed]) for seed in EXPECTED_SEEDS},
    }


def aggregate_key(per_seed: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return distribution({integer(row["seed"], "per-seed seed"): float(row[key]) for row in per_seed})


def load_reference(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require_file(path, "corrected KS reference archive")
    try:
        with np.load(path) as archive:
            required = {"x", "t", "u_archive", "u_etdrk4"}
            if not required.issubset(archive.files):
                fail(f"reference archive is missing {sorted(required - set(archive.files))}")
            x = np.asarray(archive["x"], dtype=np.float64)
            t = np.asarray(archive["t"], dtype=np.float64)
            target = np.asarray(archive["u_archive"], dtype=np.float64)
            etdrk4 = np.asarray(archive["u_etdrk4"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise PublicationMetricsError(f"cannot read reference archive {path}: {exc}") from exc
    if target.shape != (t.size, x.size):
        fail(f"reference field shape {target.shape} != {(t.size, x.size)}")
    if etdrk4.shape != target.shape:
        fail("u_etdrk4 and u_archive shapes differ")
    if not all(np.all(np.isfinite(array)) for array in (x, t, target, etdrk4)):
        fail("reference archive contains NaN or infinity")
    if x.size != 512 or t.size != 201:
        fail(f"locked publication reference must be 201 x 512, got {target.shape}")
    dx = np.diff(x)
    if not np.allclose(dx, dx[0], rtol=1.0e-12, atol=1.0e-14):
        fail("reference x grid is not uniform")
    length = float(dx[0] * x.size)
    assert_close(length, 6.0 * np.pi, "reference periodic length")
    assert_close(float(x[0]), -3.0 * np.pi, "reference left boundary")
    assert_close(float(t[0]), 0.0, "reference first time")
    assert_close(float(t[-1]), 15.0, "reference final time")
    archive_difference = float(
        np.linalg.norm(etdrk4 - target) / np.linalg.norm(target)
    )
    return x, t, target, {
        "field_used_for_PINN_error": "u_archive",
        "archive_shape_time_by_space": list(target.shape),
        "u_etdrk4_vs_u_archive_relative_l2": archive_difference,
        "file": file_record(path),
    }


def canonical_etdrk4_runtime(
    runtime_audit_path: Path,
    convergence_path: Path,
    scale_audit_path: Path,
    reference_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any] | None,
]:
    scale_audit, scale_provenance = load_json_snapshot(
        scale_audit_path, "reference scale audit"
    )
    reference = require_mapping(scale_audit.get("reference"), "scale audit reference")
    if reference.get("method") != "Fourier pseudospectral ETDRK4, 2/3 dealiased":
        fail("scale audit reference method is not the locked dealiased ETDRK4 method")
    if integer(reference.get("N"), "scale audit reference N") != 512:
        fail("scale audit reference N is not 512")
    assert_close(
        finite_float(reference.get("dt"), "scale audit reference dt"),
        0.0046875,
        "scale audit reference dt",
    )
    accuracy_keys = (
        "global_relative_l2",
        "median_time_relative_l2",
        "maximum_time_relative_l2",
        "final_time_relative_l2",
        "maximum_absolute_error",
    )
    scale_accuracy = {
        key: finite_float(reference.get(key), f"scale audit reference {key}")
        for key in accuracy_keys
    }
    runtime = require_mapping(scale_audit.get("runtime"), "scale audit runtime")

    if runtime_audit_path.is_file():
        repeated, repeated_provenance = load_json_snapshot(
            runtime_audit_path, "repeated ETDRK4 runtime audit"
        )
        if repeated.get("method") != "Fourier pseudospectral ETDRK4, 2/3 dealiased":
            fail("repeated runtime audit method is not the locked dealiased ETDRK4")
        configuration = require_mapping(
            repeated.get("configuration"), "repeated runtime audit configuration"
        )
        if integer(configuration.get("N"), "runtime audit N") != 512:
            fail("repeated runtime audit N is not 512")
        assert_close(
            finite_float(configuration.get("dt"), "runtime audit dt"),
            0.0046875,
            "repeated runtime audit dt",
        )
        assert_close(
            finite_float(configuration.get("final_time"), "runtime audit final_time"),
            15.0,
            "repeated runtime audit final_time",
        )
        assert_close(
            finite_float(configuration.get("output_dt"), "runtime audit output_dt"),
            0.075,
            "repeated runtime audit output_dt",
        )
        warmup_repeats = integer(
            configuration.get("warmup_repeats"),
            "runtime audit warmup_repeats",
            minimum=0,
        )
        timed_repeats = integer(
            configuration.get("timed_repeats"),
            "runtime audit timed_repeats",
            minimum=5,
        )
        seconds = np.asarray(
            require_sequence(
                repeated.get("wall_seconds_all"), "runtime audit wall_seconds_all"
            ),
            dtype=np.float64,
        )
        if seconds.shape != (timed_repeats,) or not np.all(np.isfinite(seconds)):
            fail(
                "runtime audit wall_seconds_all must contain one finite value per "
                "timed repeat"
            )
        if np.any(seconds <= 0.0):
            fail("runtime audit wall times must all be positive")
        q25, median, q75 = np.quantile(
            seconds, [0.25, 0.5, 0.75], method="linear"
        )
        for key, recomputed in (
            ("wall_seconds_median", median),
            ("wall_seconds_q25", q25),
            ("wall_seconds_q75", q75),
            ("wall_seconds_minimum", np.min(seconds)),
            ("wall_seconds_maximum", np.max(seconds)),
        ):
            assert_close(
                finite_float(repeated.get(key), f"runtime audit {key}"),
                float(recomputed),
                f"repeated runtime audit {key}",
            )
        scope = repeated.get("scope")
        if not isinstance(scope, str) or not scope.strip():
            fail("repeated runtime audit must provide a non-empty timing scope")
        errors = np.asarray(
            require_sequence(
                repeated.get("global_relative_l2_all"),
                "runtime audit global_relative_l2_all",
            ),
            dtype=np.float64,
        )
        if errors.shape != (timed_repeats,) or not np.all(np.isfinite(errors)):
            fail(
                "runtime audit global_relative_l2_all must contain one finite value "
                "per timed repeat"
            )
        for index, value in enumerate(errors):
            if not math.isclose(
                float(value),
                scale_accuracy["global_relative_l2"],
                rel_tol=5.0e-6,
                abs_tol=1.0e-15,
            ):
                fail(
                    f"runtime repeat {index} accuracy {value:.17g} differs from the "
                    "locked reference audit"
                )
        expected_reference_hash = file_record(reference_path)["sha256"]
        if repeated.get("reference_sha256") != expected_reference_hash:
            fail("runtime audit reference SHA256 differs from the locked reference")
        repeated_runtime = require_mapping(
            repeated.get("runtime"), "repeated runtime audit environment"
        )
        convergence_provenance = (
            file_record(convergence_path) if convergence_path.is_file() else None
        )
        return (
            {
                "source": "repeated_runtime_audit",
                "method": repeated["method"],
                "N": 512,
                "dt": 0.0046875,
                "wall_seconds": float(median),
                "denominator_statistic": "median of repeated wall times",
                "wall_seconds_median": float(median),
                "wall_seconds_q25": float(q25),
                "wall_seconds_q75": float(q75),
                "wall_seconds_iqr": float(q75 - q25),
                "wall_seconds_all": seconds.tolist(),
                "timing_repeats": timed_repeats,
                "warmup_repeats": warmup_repeats,
                "timing_scope": scope.strip(),
                "execution_backend": "CPU NumPy/FFT",
                "recorded_platform": repeated_runtime.get("platform"),
                "recorded_processor": repeated_runtime.get("processor"),
                "recorded_python": repeated_runtime.get("python"),
                "recorded_numpy": repeated_runtime.get("numpy"),
                "accuracy_against_archive": {
                    "global_relative_l2_all": errors.tolist(),
                    "locked_scale_audit": scale_accuracy,
                },
                "source_file": repeated_provenance,
                "fallback_used": False,
            },
            convergence_provenance,
            scale_provenance,
            repeated_provenance,
        )

    require_file(convergence_path, "ETDRK4 convergence table fallback")
    with convergence_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row
        for row in rows
        if row.get("method") == "dealiased_ETDRK4"
        and row.get("N") == "512"
        and math.isclose(float(row.get("dt", "nan")), 0.0046875, abs_tol=1.0e-15)
    ]
    if len(matches) != 1:
        fail(
            "expected exactly one dealiased_ETDRK4 convergence row with "
            f"N=512, dt=0.0046875; found {len(matches)}"
        )
    row = matches[0]
    denominator_seconds = finite_float(row.get("wall_seconds"), "ETDRK4 wall seconds")
    accuracy: dict[str, float] = {}
    for key in accuracy_keys:
        csv_value = finite_float(row.get(key), f"ETDRK4 convergence {key}")
        assert_close(csv_value, scale_accuracy[key], f"canonical ETDRK4 {key}")
        accuracy[key] = csv_value
    convergence_provenance = file_record(convergence_path)
    return (
        {
            "source": "convergence_csv_single_run_fallback",
            "method": row["method"],
            "N": 512,
            "dt": 0.0046875,
            "wall_seconds": denominator_seconds,
            "denominator_statistic": "single recorded wall time",
            "wall_seconds_median": denominator_seconds,
            "wall_seconds_q25": None,
            "wall_seconds_q75": None,
            "wall_seconds_iqr": None,
            "execution_backend": "CPU NumPy/FFT",
            "recorded_platform": runtime.get("platform"),
            "recorded_python": runtime.get("python"),
            "recorded_numpy": runtime.get("numpy"),
            "timing_repeats": 1,
            "warmup_repeats": None,
            "timing_scope": (
                "single wall_seconds value stored in etdrk4_convergence.csv; the CSV "
                "does not itself encode timing scope"
            ),
            "accuracy_against_archive": accuracy,
            "source_file": convergence_provenance,
            "fallback_used": True,
            "fallback_reason": (
                f"repeated runtime audit absent at {runtime_audit_path.resolve()}"
            ),
        },
        convergence_provenance,
        scale_provenance,
        None,
    )


def training_command(
    locked_config: Mapping[str, Any],
    directory: Path,
    reference: Path,
    device: str,
) -> str:
    option_map = (
        ("windows", "--windows"),
        ("adam_steps_per_window", "--adam-steps"),
        ("lbfgs_iterations_per_window", "--lbfgs-iterations"),
        ("collocation", "--collocation"),
        ("validation_collocation", "--validation-collocation"),
        ("harmonics", "--harmonics"),
        ("state_modes", "--state-modes"),
        ("state_grid", "--state-grid"),
        ("width", "--width"),
        ("depth", "--depth"),
        ("learning_rate", "--learning-rate"),
    )
    command = [
        "python3",
        "code/train_hc_marching_ks_pinn.py",
        "--reference",
        str(reference.resolve()),
        "--output",
        str(directory.resolve()),
        "--seeds",
        ",".join(str(seed) for seed in EXPECTED_SEEDS),
    ]
    for key, option in option_map:
        if key not in locked_config:
            fail(f"locked config is missing command option {key}")
        command.extend((option, str(locked_config[key])))
    command.extend(("--device", device))
    return shlex.join(command)


def validate_training_precision_source(path: Path) -> dict[str, Any]:
    require_file(path, "training source")
    text = path.read_text(encoding="utf-8")
    required_snippets = (
        "dtype=torch.float32",
        'torch.set_float32_matmul_precision("highest")',
        "torch.clamp(normalized.pow(exponent), 0.2, 5.0).detach()",
        "min=1e-3",
        "eta_min=config.learning_rate * 0.1",
        'line_search_fn="strong_wolfe"',
    )
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if ".to(torch.float64)" not in text and ".to(dtype=torch.float64)" not in text:
        missing.append("explicit model cast to torch.float64")
    if missing:
        fail(
            "training source no longer contains the precision statements assumed by "
            f"this publication audit: {missing}"
        )
    return {
        "float32_training_tensor_construction_found": True,
        "float32_matmul_highest_setting_found": True,
        "CPU_float64_audit_cast_found": True,
        "source_identity_limit": (
            "The per-seed metrics do not embed a training-source hash. The hash below "
            "identifies the source present at aggregation time, which is assumed but "
            "cannot be cryptographically proven to be the exact bytes used to launch "
            "every already-completed seed."
        ),
    }


def aggregation_command(args: argparse.Namespace, output: Path) -> str:
    command = [
        "python3",
        "code/build_publication_metrics.py",
        "--directory",
        str(args.directory.resolve()),
        "--reference",
        str(args.reference.resolve()),
        "--scale-audit",
        str(args.scale_audit.resolve()),
        "--convergence",
        str(args.convergence.resolve()),
        "--runtime-audit",
        str(args.runtime_audit.resolve()),
        "--output",
        str(output.resolve()),
    ]
    return shlex.join(command)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    directory = args.directory.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else directory / OUTPUT_NAME
    )
    metric_paths = preflight_seed_files(directory)
    reference_x, reference_t, target, reference_record = load_reference(
        args.reference.resolve()
    )
    reports: list[dict[str, Any]] = []
    metric_provenance: dict[str, Any] = {}
    for seed, path in zip(EXPECTED_SEEDS, metric_paths):
        report, provenance = load_json_snapshot(path, f"seed {seed} metric")
        reports.append(report)
        metric_provenance[str(seed)] = provenance
    locked_config = validate_configurations(reports)
    runtime_metadata = validate_shared_runtime_metadata(reports, locked_config)
    windows = integer(locked_config.get("windows"), "locked windows", minimum=1)

    per_seed: list[dict[str, Any]] = []
    prediction_provenance: dict[str, Any] = {}
    checkpoint_provenance: dict[str, Any] = {}
    for seed, report in zip(EXPECTED_SEEDS, reports):
        summary, provenance = seed_summary(
            directory,
            seed,
            report,
            locked_config,
            reference_x,
            reference_t,
            target,
        )
        per_seed.append(summary)
        prediction_provenance[str(seed)] = provenance["prediction"]
        checkpoint_provenance[str(seed)] = checkpoint_manifest(
            directory, seed, windows
        )

    (
        etdrk4_runtime,
        convergence_provenance,
        scale_provenance,
        runtime_audit_provenance,
    ) = (
        canonical_etdrk4_runtime(
            args.runtime_audit.resolve(),
            args.convergence.resolve(),
            args.scale_audit.resolve(),
            args.reference.resolve(),
        )
    )
    denominator = float(etdrk4_runtime["wall_seconds"])
    for row in per_seed:
        row["training_to_etdrk4_runtime_ratio"] = (
            float(row["training_seconds"]) / denominator
        )

    distribution_keys = (
        "global_relative_l2",
        "median_time_relative_l2",
        "maximum_time_relative_l2",
        "final_time_relative_l2",
        "median_phase_invariant_relative_l2",
        "maximum_absolute_error",
        "energy_balance_relative_rms",
        "energy_balance_absolute_rms",
        "residual_rms_cpu_float64",
        "residual_q95_absolute_cpu_float64",
        "residual_maximum_absolute_cpu_float64",
        "training_seconds",
        "training_to_etdrk4_runtime_ratio",
    )
    aggregate = {key: aggregate_key(per_seed, key) for key in distribution_keys}
    periodic_worst: dict[str, Any] = {}
    for order in EXPECTED_PERIODIC_ORDERS:
        seed_rms = {
            int(row["seed"]): float(row["periodic_jumps"][order]["rms"])
            for row in per_seed
        }
        seed_maximum = {
            int(row["seed"]): float(
                row["periodic_jumps"][order]["maximum_absolute"]
            )
            for row in per_seed
        }
        winning_rms_seed = max(seed_rms, key=seed_rms.get)
        winning_max_seed = max(seed_maximum, key=seed_maximum.get)
        periodic_worst[str(order)] = {
            "all_seed_maximum_of_seed_rms": seed_rms[winning_rms_seed],
            "seed_for_maximum_rms": winning_rms_seed,
            "all_seed_maximum_absolute": seed_maximum[winning_max_seed],
            "seed_for_maximum_absolute": winning_max_seed,
        }

    def worst_case(key: str) -> dict[str, Any]:
        values = {int(row["seed"]): float(row[key]) for row in per_seed}
        winning_seed = max(values, key=values.get)
        return {
            "all_seed_maximum": values[winning_seed],
            "seed": winning_seed,
            "values_by_seed": {str(seed): values[seed] for seed in EXPECTED_SEEDS},
        }

    training_source = ROOT / "code" / "train_hc_marching_ks_pinn.py"
    refresh_source = ROOT / "code" / "refresh_marching_metrics.py"
    aggregation_source = Path(__file__).resolve()
    for path, label in (
        (training_source, "training source"),
        (refresh_source, "metric refresh source"),
        (aggregation_source, "publication aggregation source"),
    ):
        require_file(path, label)
    precision_source_audit = validate_training_precision_source(training_source)
    if etdrk4_runtime["source"] == "repeated_runtime_audit":
        runtime_warning = (
            "The numerator uses PyTorch MPS accelerator execution whereas the "
            "denominator uses CPU NumPy/FFT, so the backends are heterogeneous. "
            "The denominator is the median of a repeated integration-loop timing "
            "audit, with its q25/q75 recorded, but the PINN numerator has a broader "
            "scope that includes local evaluation, transfers, CPU copies, and "
            "checkpoint writes. Exact machine load is not controlled. Treat the "
            "ratio as wall-clock context, not a controlled hardware speedup benchmark."
        )
    else:
        runtime_warning = (
            "The numerator uses PyTorch MPS accelerator execution whereas the "
            "denominator is one CPU NumPy/FFT wall time from the convergence CSV. "
            "The backends and timing scopes are heterogeneous, the fallback denominator "
            "has no repeat distribution, and exact machine load is not controlled. "
            "Treat the ratio as wall-clock context, not a controlled hardware speedup "
            "benchmark."
        )

    payload: dict[str, Any] = {
        "schema": "ks_hc_marching_publication_metrics_v1",
        "status": "validated_five_seed_aggregate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_policy": {
            "required_seeds": list(EXPECTED_SEEDS),
            "number_of_seeds": len(EXPECTED_SEEDS),
            "no_missing_or_extra_seed_metrics": True,
        },
        "locked_configuration": locked_config,
        "execution_and_precision": {
            **runtime_metadata,
            "training_source_precision_audit": precision_source_audit,
            "aggregation_environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "hardware_identity_limit": (
                "Per-seed metrics record only the MPS device string, Python, and "
                "PyTorch versions; they do not record the exact Mac model, SoC, "
                "power state, or concurrent system load."
            ),
        },
        "aggregation_policy": {
            "seed_level_scalars": (
                "median, q25, q75, and IQR=q75-q25 over exactly five seeds; "
                "NumPy linear quantiles"
            ),
            "worst_case_metrics": (
                "interface transfer errors, predicted mean drift, initial-condition "
                "relative L2, and periodic jumps are maxima across every seed"
            ),
            "reference_field": "u_archive on the locked 201 x 512 grid",
        },
        "metric_definitions": {
            "global_relative_l2": (
                "Frobenius norm ||U_PINN-U_archive||_F / ||U_archive||_F over "
                "all 201 times and 512 periodic spatial points."
            ),
            "median_time_relative_l2": (
                "For each stored time, compute ||u_PINN-u_archive||_2 / "
                "||u_archive||_2 over 512 x points, then take the median over 201 times."
            ),
            "maximum_time_relative_l2": (
                "Maximum of the 201 snapshot-wise relative L2 errors."
            ),
            "final_time_relative_l2": (
                "Snapshot-wise relative L2 error at T=15."
            ),
            "median_phase_invariant_relative_l2": (
                "At each time, circularly shift the predicted 512-point periodic "
                "profile by the integer-grid shift maximizing FFT cross-correlation "
                "with the archive, compute relative L2 after alignment, then take "
                "the median over 201 times. Sub-grid shifts are not optimized."
            ),
            "predicted_mean_drift": (
                "Within each seed, max_T |mean_X u_PINN(T)-mean_X u_PINN(0)|; "
                "the publication value is the maximum of this quantity over seeds."
            ),
            "energy_balance": (
                "For E(T)=0.5*mean_X(u^2), test dE/dT = mean_X(u_X^2) - "
                "mean_X(u_XX^2). Spatial derivatives are FFT derivatives. dE/dT "
                "uses a 7-point, degree-5 Savitzky-Golay derivative; three time "
                "samples at each end are excluded. Absolute RMS is sqrt(mean(r_E^2)); "
                "relative RMS is ||r_E||_2/||right-hand side||_2 on the retained times."
            ),
            "cpu_float64_unseen_residual_rms": (
                "sqrt(mean(r_i^2)) for r=u_T+u*u_X+u_XX+u_XXXX at 128 newly "
                "sampled CPU-float64 points per window (5120 points for 40 windows); "
                "sampling seed is training seed+20000. These points were not used "
                "for Adam or LBFGS training."
            ),
            "cpu_float64_unseen_residual_q95_absolute": (
                "Empirical linear 0.95 quantile of |r_i| over the same unseen "
                "CPU-float64 residual points."
            ),
            "periodic_jumps": (
                "For derivative orders 0--3, compare left and right endpoints at "
                "nine times per window in the CPU-float64 audit. RMS is over all "
                "window/time endpoint differences; maximum_absolute is the largest "
                "absolute endpoint difference. Publication values are maxima over seeds."
            ),
            "interface_transfer_error": (
                "Error introduced when each predicted right-interface field is "
                "projected to the truncated Fourier state. For all but the final "
                "window that state is supplied exactly to the next window; the final "
                "projection is diagnostic only and is retained per seed but excluded "
                "from the interface aggregate. Publication values are maxima over "
                "the 39 transferred interfaces and five seeds."
            ),
            "initial_condition_relative_l2": (
                "Relative L2 error of the saved stitched prediction at T=0 against "
                "-sin(X/3). This is recomputed from each prediction archive and is "
                "the conservative publication value. The older stored scalar may be "
                "zero because it described the hard model ansatz before float32 "
                "Fourier-state/prediction serialization; both values are retained per seed."
            ),
        },
        "aggregate_median_iqr": aggregate,
        "all_seed_worst_case": {
            "interface_transfer_relative": worst_case(
                "maximum_relative_transfer_error"
            ),
            "interface_transfer_maximum_absolute": worst_case(
                "maximum_absolute_transfer_error"
            ),
            "predicted_mean_drift": worst_case("predicted_mean_drift"),
            "initial_condition_relative_l2": worst_case(
                "initial_condition_relative_l2"
            ),
            "periodic_jump_by_derivative_order": periodic_worst,
        },
        "runtime_comparison": {
            "numerator": {
                "quantity": "HC-marching PINN training_seconds for each seed",
                "definition": (
                    "Wall time from immediately before window construction/training "
                    "through all window training, local evaluation, state transfer, "
                    "CPU model copies, and per-window checkpoint writes; excludes the "
                    "subsequent stitched-field metrics, CPU-float64 audit, energy audit, "
                    "and final prediction/metrics serialization."
                ),
                "backend": runtime_metadata["training_device"],
                "distribution": aggregate["training_seconds"],
            },
            "denominator": etdrk4_runtime,
            "ratio": {
                "formula": (
                    "training_seconds_per_seed / "
                    f"{denominator:.17g} s"
                ),
                "distribution": aggregate["training_to_etdrk4_runtime_ratio"],
                "denominator_uncertainty_policy": (
                    "Ratios use the selected denominator point estimate only. The "
                    "repeated-audit q25/q75, when available, are reported but are not "
                    "propagated into the five-seed ratio IQR."
                ),
            },
            "heterogeneous_hardware_warning": runtime_warning,
        },
        "per_seed": per_seed,
        "audit_points": {
            "metric_files_complete": True,
            "declared_seed_set_exactly_0_to_4": True,
            "configuration_identical_except_seed": True,
            "device_and_versions_identical_across_seeds": True,
            "prediction_grids_and_shapes_match_reference": True,
            "stored_field_metrics_recomputed_from_predictions": True,
            "phase_time_series_policy": (
                "Recomputed from every prediction. A stored per-time phase vector is "
                "verified when present but is not required because early training "
                "metrics contain only the scalar phase median."
            ),
            "stored_energy_metrics_recomputed_from_predictions": True,
            "initial_condition_policy": (
                "Use the recomputed stitched-prediction error for the all-seed maximum. "
                "The original stored value is retained separately and is not required "
                "to equal the serialized prediction error."
            ),
            "all_values_finite": True,
            "all_windows_present_and_contiguous_from_T0_to_T15": True,
            "training_seconds_consistent_with_sum_of_window_timers": True,
            "all_lbfgs_status_values_completed": True,
            "one_checkpoint_per_window_per_seed": True,
            "cpu_float64_residual_point_count_is_128_per_window": True,
            "periodic_derivative_orders_exactly_0_to_3": True,
            "runtime_denominator_source": etdrk4_runtime["source"],
            "runtime_denominator_repeated_timing_used": (
                etdrk4_runtime["source"] == "repeated_runtime_audit"
            ),
            "residual_and_periodic_values_not_recomputed_by_aggregator": (
                "Validated structurally from per-seed metrics and tied to checkpoint "
                "hashes; independent recomputation would require reloading all models."
            ),
        },
        "reference": reference_record,
        "reproducibility": {
            "commands": {
                "five_seed_training": training_command(
                    locked_config,
                    directory,
                    args.reference.resolve(),
                    runtime_metadata["training_device"],
                ),
                "publication_aggregation": aggregation_command(args, output),
            },
            "non_cli_locked_training_defaults": {
                key: locked_config[key]
                for key in (
                    "causal_bins",
                    "causal_epsilon",
                    "attention_exponent",
                    "gradient_clip",
                )
                if key in locked_config
            },
            "sha256": {
                "aggregation_source": file_record(aggregation_source),
                "training_source": file_record(training_source),
                "metric_refresh_source": file_record(refresh_source),
                "reference_archive": reference_record["file"],
                "reference_scale_audit": scale_provenance,
                "etdrk4_convergence_table": convergence_provenance,
                "etdrk4_runtime_audit": runtime_audit_provenance,
                "seed_metrics": metric_provenance,
                "seed_predictions": prediction_provenance,
                "seed_checkpoint_manifests": checkpoint_provenance,
            },
        },
        "publication_caveats": [
            "The five seeds quantify optimizer/sampling variability for one locked "
            "architecture and training budget; they are not an architecture sweep.",
            "Phase-invariant error removes only integer-grid translations and should "
            "be reported alongside, not instead of, unaligned global/final errors.",
            "The CPU-float64 residual audit samples points deterministically from each "
            "seed but does not constitute an exhaustive residual bound.",
            "Runtime ratios mix MPS and CPU backends and must not be described as a "
            "controlled speedup measurement.",
        ],
    }
    return payload, output


def main() -> int:
    args = parse_args()
    try:
        payload, output = build(args)
        if args.check_only:
            print(
                "Validation passed for seeds 0--4; --check-only selected, so no output "
                "was written."
            )
            return 0
        atomic_write_json(output, payload)
        print(f"Wrote validated publication aggregate: {output}")
        return 0
    except PublicationMetricsError as exc:
        print(
            f"ERROR: publication metrics validation failed: {exc}\n"
            "No publication_metrics.json was created or replaced.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
