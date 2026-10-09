#!/usr/bin/env python3
"""Promote a complete locked HC-marching result set without overwriting.

The source is ``results/2026_recalculation/marching_probe`` and the target is
its sibling ``marching_final``.  Promotion is deliberately conservative:

* exactly seeds 0--4 must each have one metrics JSON, one prediction NPZ, and
  checkpoints for windows 00--39;
* every seed configuration and the five-run summary must equal the manuscript
  locked configuration below;
* the run CSV must contain exactly those five seeds and agree with the metrics;
* the final Figure 3 PNG must exist and be structurally valid;
* only explicitly allow-listed files are copied;
* copied bytes are verified and recorded in ``SHA256_MANIFEST.json``;
* the sibling temporary directory is renamed with an OS no-replace primitive.

The target is never removed or overwritten.  The default action is validation
only.  Actual promotion requires the explicit ``--promote`` flag:

    python3 code/promote_marching_results.py --promote
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "2026_recalculation" / "marching_probe"
DEFAULT_TARGET = ROOT / "results" / "2026_recalculation" / "marching_final"
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_WINDOWS = 40
MANIFEST_NAME = "SHA256_MANIFEST.json"

# This dictionary is the paper's locked production design.  Equality is exact
# after removing only the per-run seed field.
LOCKED_CONFIG: dict[str, Any] = {
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

SUMMARY_NAME = "hc_marching_summary.json"
RUNS_NAME = "hc_marching_runs.csv"
FIGURE_NAME = "figure_hc_marching.png"
OPTIONAL_PUBLICATION_METRICS = "publication_metrics.json"

SUMMARY_DISTRIBUTIONS = (
    "global_relative_l2",
    "median_time_relative_l2",
    "maximum_time_relative_l2",
    "final_time_relative_l2",
    "median_phase_invariant_relative_l2",
    "training_seconds",
    "residual_rms_cpu_float64",
    "residual_q95_cpu_float64",
    "maximum_interface_transfer_relative_error",
    "predicted_mean_drift",
    "energy_balance_relative_rms",
    "initial_condition_relative_l2",
    "training_to_etdrk4_runtime_ratio",
)

CSV_COLUMNS = (
    "method",
    "seed",
    "global_relative_l2",
    "median_time_relative_l2",
    "maximum_time_relative_l2",
    "final_time_relative_l2",
    "median_phase_invariant_relative_l2",
    "training_seconds",
    "residual_rms_cpu_float64",
    "residual_q95_cpu_float64",
    "periodic_u_jump_rms_cpu_float64",
    "periodic_uxxx_jump_rms_cpu_float64",
    "maximum_interface_transfer_relative_error",
    "predicted_mean_drift",
    "energy_balance_relative_rms",
    "initial_condition_relative_l2",
    "training_to_etdrk4_runtime_ratio",
)


class PromotionError(RuntimeError):
    """A release gate failed; no final directory may be created."""


def fail(message: str) -> None:
    raise PromotionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy, hash, and atomically publish after every validation passes",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        fail(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        fail(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        fail(f"empty {label}: {path}")


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be a JSON object")
    return value


def require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        fail(f"{label} must be a JSON array")
    return value


def exact_integer(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        fail(f"{label} must be an integer, not a boolean")
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[0-9]+", stripped) is None:
            fail(f"{label} is not a canonical non-negative integer: {value!r}")
        number = int(stripped)
        if minimum is not None and number < minimum:
            fail(f"{label} must be >= {minimum}, got {number}")
        return number
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PromotionError(f"{label} is not an integer: {value!r}") from exc
    if number != value:
        fail(f"{label} is not an exact integer: {value!r}")
    if minimum is not None and number < minimum:
        fail(f"{label} must be >= {minimum}, got {number}")
    return number


def finite_float(value: Any, label: str, nonnegative: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PromotionError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        fail(f"{label} is not finite: {number!r}")
    if nonnegative and number < 0.0:
        fail(f"{label} must be non-negative, got {number}")
    return number


def assert_close(
    actual: float,
    expected: float,
    label: str,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        fail(f"{label} mismatch: actual={actual:.17g}, expected={expected:.17g}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_regular_file(path, label)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot parse {label} {path}: {exc}") from exc
    return dict(require_mapping(parsed, label))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, relative_name: str) -> dict[str, Any]:
    require_regular_file(path, relative_name)
    return {
        "path": relative_name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def expected_seed_names() -> set[str]:
    names: set[str] = set()
    for seed in EXPECTED_SEEDS:
        names.add(f"hc_marching_seed{seed}_metrics.json")
        names.add(f"hc_marching_seed{seed}_prediction.npz")
        for window in range(EXPECTED_WINDOWS):
            names.add(f"hc_marching_seed{seed}_window{window:02d}.pt")
    return names


def discover_and_validate_seed_names(source: Path) -> set[str]:
    expected = expected_seed_names()
    discovered = {
        path.name
        for path in source.iterdir()
        if path.name.startswith("hc_marching_seed")
    }
    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)
    if missing or unexpected:
        fail(
            "per-seed artifact set is not exactly seeds 0--4 x metrics/prediction/"
            f"40 checkpoints; missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(expected):
        require_regular_file(source / name, f"per-seed artifact {name}")
    return expected


def validate_config(config_raw: Any, seed: int, label: str) -> None:
    config = dict(require_mapping(config_raw, label))
    declared_seed = exact_integer(config.pop("seed", None), f"{label}.seed", minimum=0)
    if declared_seed != seed:
        fail(f"{label} declares seed {declared_seed}, expected {seed}")
    if canonical_json(config) != canonical_json(LOCKED_CONFIG):
        fail(
            f"{label} differs from paper locked configuration; "
            f"actual={canonical_json(config)}, locked={canonical_json(LOCKED_CONFIG)}"
        )


def validate_metric(path: Path, seed: int) -> dict[str, Any]:
    report = load_json(path, f"seed {seed} metrics")
    validate_config(report.get("config"), seed, f"seed {seed} config")
    finite_float(report.get("training_seconds"), f"seed {seed} training_seconds")
    for key in (
        "global_relative_l2",
        "median_time_relative_l2",
        "maximum_time_relative_l2",
        "final_time_relative_l2",
        "median_phase_invariant_relative_l2",
        "maximum_absolute_error",
        "predicted_mean_drift",
        "initial_condition_relative_l2",
        "energy_balance_relative_rms",
        "energy_balance_absolute_rms",
    ):
        finite_float(report.get(key), f"seed {seed} {key}")

    diagnostics = require_sequence(report.get("diagnostics"), f"seed {seed} diagnostics")
    transfers = require_sequence(
        report.get("interface_transfers"), f"seed {seed} interface_transfers"
    )
    if len(diagnostics) != EXPECTED_WINDOWS or len(transfers) != EXPECTED_WINDOWS:
        fail(f"seed {seed} must have exactly 40 diagnostics and 40 transfers")
    edges = np.linspace(0.0, 15.0, EXPECTED_WINDOWS + 1)
    for window, item_raw in enumerate(diagnostics):
        item = require_mapping(item_raw, f"seed {seed} diagnostic {window}")
        if exact_integer(item.get("window"), f"seed {seed} diagnostic window") != window:
            fail(f"seed {seed} diagnostic windows are not contiguous")
        assert_close(
            finite_float(item.get("t_left"), f"seed {seed} window {window} t_left"),
            float(edges[window]),
            f"seed {seed} window {window} t_left",
        )
        assert_close(
            finite_float(item.get("t_right"), f"seed {seed} window {window} t_right"),
            float(edges[window + 1]),
            f"seed {seed} window {window} t_right",
        )
        if item.get("lbfgs_status") != "completed":
            fail(f"seed {seed} window {window} LBFGS is not completed")
    for window, item_raw in enumerate(transfers):
        item = require_mapping(item_raw, f"seed {seed} transfer {window}")
        if exact_integer(item.get("window"), f"seed {seed} transfer window") != window:
            fail(f"seed {seed} transfer windows are not contiguous")
        finite_float(
            item.get("relative_fourier_transfer_error"),
            f"seed {seed} window {window} transfer error",
        )

    residual = require_mapping(
        report.get("cpu_float64_unseen_residual"), f"seed {seed} residual audit"
    )
    if exact_integer(residual.get("points"), f"seed {seed} residual points") != 5120:
        fail(f"seed {seed} must have 5120 unseen residual audit points")
    for key in ("rms", "mean_absolute", "q95_absolute", "maximum_absolute"):
        finite_float(residual.get(key), f"seed {seed} residual {key}")
    jumps = require_sequence(
        report.get("cpu_float64_periodic_jumps"), f"seed {seed} periodic jumps"
    )
    if len(jumps) != 4:
        fail(f"seed {seed} must report periodic derivative orders 0--3")
    for order, item_raw in enumerate(jumps):
        item = require_mapping(item_raw, f"seed {seed} periodic jump {order}")
        if exact_integer(item.get("derivative_order"), "derivative order") != order:
            fail(f"seed {seed} periodic derivative orders are not 0--3")
        finite_float(item.get("rms"), f"seed {seed} order {order} periodic RMS")
        finite_float(
            item.get("maximum_absolute"),
            f"seed {seed} order {order} periodic maximum",
        )
    return report


def validate_prediction(
    path: Path,
    seed: int,
    canonical_x: np.ndarray | None,
    canonical_t: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    require_regular_file(path, f"seed {seed} prediction")
    try:
        with np.load(path) as archive:
            if set(archive.files) != {"x", "t", "u"}:
                fail(f"seed {seed} prediction keys must be exactly x,t,u")
            x = np.asarray(archive["x"], dtype=np.float64)
            t = np.asarray(archive["t"], dtype=np.float64)
            u = np.asarray(archive["u"], dtype=np.float64)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise PromotionError(f"cannot read seed {seed} prediction: {exc}") from exc
    if x.shape != (512,) or t.shape != (201,) or u.shape != (201, 512):
        fail(f"seed {seed} prediction has wrong shapes x={x.shape}, t={t.shape}, u={u.shape}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(u)):
        fail(f"seed {seed} prediction contains NaN or infinity")
    assert_close(float(x[0]), -3.0 * np.pi, f"seed {seed} x left")
    assert_close(float((x[1] - x[0]) * x.size), 6.0 * np.pi, f"seed {seed} period")
    assert_close(float(t[0]), 0.0, f"seed {seed} first time")
    assert_close(float(t[-1]), 15.0, f"seed {seed} final time")
    if canonical_x is not None and not np.array_equal(x, canonical_x):
        fail(f"seed {seed} x grid differs from seed 0")
    if canonical_t is not None and not np.array_equal(t, canonical_t):
        fail(f"seed {seed} t grid differs from seed 0")
    return x, t


def validate_checkpoint(path: Path, seed: int, window: int) -> None:
    require_regular_file(path, f"seed {seed} window {window} checkpoint")
    if not zipfile.is_zipfile(path):
        fail(f"seed {seed} window {window} checkpoint is not a valid ZIP-based torch save")
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise PromotionError(
            f"cannot validate seed {seed} window {window} checkpoint: {exc}"
        ) from exc
    if bad_member is not None:
        fail(f"seed {seed} window {window} checkpoint has corrupt member {bad_member}")
    if not any(name.endswith("data.pkl") for name in names):
        fail(f"seed {seed} window {window} checkpoint lacks torch data.pkl")


def per_seed_summary_values(report: Mapping[str, Any]) -> dict[str, float]:
    residual = require_mapping(report["cpu_float64_unseen_residual"], "residual")
    transfers = require_sequence(report["interface_transfers"], "transfers")
    return {
        "global_relative_l2": float(report["global_relative_l2"]),
        "median_time_relative_l2": float(report["median_time_relative_l2"]),
        "maximum_time_relative_l2": float(report["maximum_time_relative_l2"]),
        "final_time_relative_l2": float(report["final_time_relative_l2"]),
        "median_phase_invariant_relative_l2": float(
            report["median_phase_invariant_relative_l2"]
        ),
        "training_seconds": float(report["training_seconds"]),
        "residual_rms_cpu_float64": float(residual["rms"]),
        "residual_q95_cpu_float64": float(residual["q95_absolute"]),
        "maximum_interface_transfer_relative_error": max(
            float(require_mapping(item, "transfer")["relative_fourier_transfer_error"])
            for item in transfers[:-1]
        ),
        "predicted_mean_drift": float(report["predicted_mean_drift"]),
        "energy_balance_relative_rms": float(report["energy_balance_relative_rms"]),
        "initial_condition_relative_l2": float(report["initial_condition_relative_l2"]),
    }


def validate_summary(path: Path, reports: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    summary = load_json(path, "five-seed summary")
    if exact_integer(summary.get("runs"), "summary runs") != 5:
        fail(f"summary runs must equal 5, got {summary.get('runs')!r}")
    locked = require_mapping(summary.get("locked_config"), "summary locked_config")
    if canonical_json(locked) != canonical_json(LOCKED_CONFIG):
        fail("summary locked_config differs from the paper locked configuration")
    denominator = finite_float(
        summary.get("canonical_etdrk4_seconds"), "summary canonical ETDRK4 seconds"
    )
    seed_values = {seed: per_seed_summary_values(report) for seed, report in reports.items()}
    for seed in EXPECTED_SEEDS:
        seed_values[seed]["training_to_etdrk4_runtime_ratio"] = (
            seed_values[seed]["training_seconds"] / denominator
        )
    for key in SUMMARY_DISTRIBUTIONS:
        distribution = require_mapping(summary.get(key), f"summary {key}")
        values = np.asarray([seed_values[seed][key] for seed in EXPECTED_SEEDS])
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        for field, expected in (("median", median), ("q25", q25), ("q75", q75)):
            assert_close(
                finite_float(distribution.get(field), f"summary {key} {field}"),
                float(expected),
                f"summary {key} {field}",
            )
    return summary


def validate_runs_csv(
    path: Path,
    reports: Mapping[int, Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require_regular_file(path, "five-seed run CSV")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = tuple(reader.fieldnames or ())
    if header != CSV_COLUMNS:
        fail(f"run CSV header differs from locked schema: {header}")
    if len(rows) != 5:
        fail(f"run CSV must contain exactly five rows, got {len(rows)}")
    denominator = finite_float(
        summary.get("canonical_etdrk4_seconds"), "summary ETDRK4 seconds"
    )
    seen: set[int] = set()
    for row in rows:
        seed = exact_integer(row.get("seed"), "run CSV seed", minimum=0)
        if seed in seen or seed not in EXPECTED_SEEDS:
            fail(f"run CSV contains duplicate or unexpected seed {seed}")
        seen.add(seed)
        if row.get("method") != "hc_marching":
            fail(f"run CSV seed {seed} method is not hc_marching")
        report = reports[seed]
        expected = per_seed_summary_values(report)
        jumps = require_sequence(report["cpu_float64_periodic_jumps"], "periodic jumps")
        expected.update(
            {
                "periodic_u_jump_rms_cpu_float64": float(
                    require_mapping(jumps[0], "u jump")["rms"]
                ),
                "periodic_uxxx_jump_rms_cpu_float64": float(
                    require_mapping(jumps[3], "uxxx jump")["rms"]
                ),
                "training_to_etdrk4_runtime_ratio": float(report["training_seconds"])
                / denominator,
            }
        )
        for key in CSV_COLUMNS[2:]:
            assert_close(
                finite_float(row.get(key), f"run CSV seed {seed} {key}"),
                expected[key],
                f"run CSV seed {seed} {key}",
            )
    if tuple(sorted(seen)) != EXPECTED_SEEDS:
        fail(f"run CSV seed set is {sorted(seen)}, expected {EXPECTED_SEEDS}")


def validate_png(path: Path) -> dict[str, int]:
    require_regular_file(path, "final Figure 3")
    content = path.read_bytes()
    header = content[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        fail("Figure 3 is not a structurally recognizable PNG")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1000 or height < 600:
        fail(f"Figure 3 resolution is unexpectedly small: {width} x {height}")
    offset = 8
    chunk_types: list[bytes] = []
    while offset < len(content):
        if offset + 12 > len(content):
            fail("Figure 3 PNG ends inside a chunk header")
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        chunk_type = content[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(content):
            fail(f"Figure 3 PNG has truncated {chunk_type!r} chunk")
        payload = content[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", content[offset + 8 + length : chunk_end])[0]
        computed_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            fail(f"Figure 3 PNG has a bad CRC in {chunk_type!r}")
        chunk_types.append(chunk_type)
        offset = chunk_end
        if chunk_type == b"IEND":
            break
    if not chunk_types or chunk_types[0] != b"IHDR" or b"IDAT" not in chunk_types:
        fail("Figure 3 PNG lacks required IHDR/IDAT structure")
    if chunk_types[-1] != b"IEND" or offset != len(content):
        fail("Figure 3 PNG lacks a terminal IEND or has trailing bytes")
    return {"pixel_width": width, "pixel_height": height}


def validate_optional_publication_metrics(
    path: Path, metric_records: Mapping[int, Mapping[str, Any]]
) -> bool:
    if not path.exists():
        return False
    report = load_json(path, "publication metrics")
    if report.get("schema") != "ks_hc_marching_publication_metrics_v1":
        fail("publication_metrics.json has the wrong schema")
    if report.get("status") != "validated_five_seed_aggregate":
        fail("publication_metrics.json is not a validated five-seed aggregate")
    policy = require_mapping(report.get("seed_policy"), "publication seed policy")
    if list(policy.get("required_seeds", [])) != list(EXPECTED_SEEDS):
        fail("publication_metrics.json does not declare seeds 0--4")
    locked = require_mapping(
        report.get("locked_configuration"), "publication locked configuration"
    )
    if canonical_json(locked) != canonical_json(LOCKED_CONFIG):
        fail("publication_metrics.json locked configuration differs")
    sha_root = require_mapping(
        require_mapping(report.get("reproducibility"), "publication reproducibility").get(
            "sha256"
        ),
        "publication SHA256",
    )
    recorded_metrics = require_mapping(sha_root.get("seed_metrics"), "seed metric hashes")
    for seed in EXPECTED_SEEDS:
        record = require_mapping(recorded_metrics.get(str(seed)), f"seed {seed} hash record")
        if record.get("sha256") != metric_records[seed]["sha256"]:
            fail(f"publication metric hash for seed {seed} is stale")
    return True


def validate_source(source: Path, target: Path) -> tuple[list[str], dict[str, Any]]:
    if source.is_symlink() or not source.is_dir():
        fail(f"source must be a real directory: {source}")
    if source == target:
        fail("source and target must differ")
    if source.parent != target.parent:
        fail("temporary/final promotion requires source and target to be siblings")
    if os.path.lexists(target):
        fail(f"target already exists and will not be overwritten or removed: {target}")

    allowlist = discover_and_validate_seed_names(source)
    reports: dict[int, dict[str, Any]] = {}
    canonical_x: np.ndarray | None = None
    canonical_t: np.ndarray | None = None
    for seed in EXPECTED_SEEDS:
        metric_name = f"hc_marching_seed{seed}_metrics.json"
        prediction_name = f"hc_marching_seed{seed}_prediction.npz"
        reports[seed] = validate_metric(source / metric_name, seed)
        canonical_x, canonical_t = validate_prediction(
            source / prediction_name,
            seed,
            canonical_x,
            canonical_t,
        )
        for window in range(EXPECTED_WINDOWS):
            validate_checkpoint(
                source / f"hc_marching_seed{seed}_window{window:02d}.pt",
                seed,
                window,
            )

    summary = validate_summary(source / SUMMARY_NAME, reports)
    validate_runs_csv(source / RUNS_NAME, reports, summary)
    figure_geometry = validate_png(source / FIGURE_NAME)
    allowlist.update({SUMMARY_NAME, RUNS_NAME, FIGURE_NAME})

    metric_hashes = {
        seed: file_record(
            source / f"hc_marching_seed{seed}_metrics.json",
            f"hc_marching_seed{seed}_metrics.json",
        )
        for seed in EXPECTED_SEEDS
    }
    publication_present = validate_optional_publication_metrics(
        source / OPTIONAL_PUBLICATION_METRICS, metric_hashes
    )
    if publication_present:
        allowlist.add(OPTIONAL_PUBLICATION_METRICS)

    ordered = sorted(allowlist)
    return ordered, {
        "figure_3": figure_geometry,
        "publication_metrics_included": publication_present,
        "summary_runs": 5,
        "locked_config": LOCKED_CONFIG,
    }


def no_replace_directory_rename(source: Path, target: Path) -> None:
    """Atomically rename a directory while refusing an existing target."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:
            fail("renamex_np is unavailable; refusing a non-exclusive rename")
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        rename_exclusive = 0x00000004
        result = renamex(source_bytes, target_bytes, rename_exclusive)
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            fail("renameat2 is unavailable; refusing a non-exclusive rename")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            source_bytes,
            at_fdcwd,
            target_bytes,
            rename_noreplace,
        )
    else:
        fail(
            f"no audited atomic no-replace directory rename for platform {sys.platform}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            fail(f"target appeared during promotion and was not overwritten: {target}")
        raise PromotionError(
            f"atomic no-replace rename failed: [errno {error_number}] "
            f"{os.strerror(error_number)}"
        )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_and_promote(
    source: Path,
    target: Path,
    allowlist: Sequence[str],
    validation: Mapping[str, Any],
) -> None:
    # Recheck before creating any sibling temporary state.
    if os.path.lexists(target):
        fail(f"target already exists and will not be overwritten or removed: {target}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    renamed = False
    try:
        source_records = {
            name: file_record(source / name, name) for name in allowlist
        }
        copied_records: list[dict[str, Any]] = []
        for name in allowlist:
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
            copied = file_record(destination, name)
            if copied != source_records[name]:
                fail(f"copied bytes differ from source for {name}")
            copied_records.append(copied)

        # Reject source mutation during the copy window.
        for name in allowlist:
            if file_record(source / name, name) != source_records[name]:
                fail(f"source changed during promotion: {name}")

        manifest_lines = "".join(
            f"{record['path']}\t{record['bytes']}\t{record['sha256']}\n"
            for record in copied_records
        ).encode("utf-8")
        manifest = {
            "schema": "ks_hc_marching_promotion_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(source),
            "target_directory": str(target),
            "promotion_policy": (
                "explicit allow-list; SHA256 verified copy; atomic OS no-replace "
                "directory rename; existing target is never overwritten or deleted"
            ),
            "validation": dict(validation),
            "file_count_excluding_manifest": len(copied_records),
            "files": copied_records,
            "allowlist_manifest_sha256": hashlib.sha256(manifest_lines).hexdigest(),
            "promotion_script": file_record(Path(__file__).resolve(), "code/promote_marching_results.py"),
        }
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        # Durability before the single namespace publication operation.
        for path in temporary.iterdir():
            if path.is_file():
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        fsync_directory(temporary)
        if os.path.lexists(target):
            fail(f"target appeared during promotion and was not overwritten: {target}")
        no_replace_directory_rename(temporary, target)
        renamed = True
        fsync_directory(target.parent)
    finally:
        # Delete only the unique temporary directory created by this invocation.
        # Never touch the target, even when a post-rename fsync reports an error.
        if not renamed and temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    args = parse_args()
    # abspath normalizes the CLI path without following a final symlink; this
    # preserves the target-exists safety check even for a broken symlink.
    source = Path(os.path.abspath(args.source.expanduser()))
    target = Path(os.path.abspath(args.target.expanduser()))
    try:
        allowlist, validation = validate_source(source, target)
        if not args.promote:
            print(
                f"Validation passed for {len(allowlist)} allow-listed files. "
                "No files copied; pass --promote to publish marching_final."
            )
            return 0
        copy_and_promote(source, target, allowlist, validation)
        print(f"Promoted immutable result set to {target}")
        return 0
    except PromotionError as exc:
        print(f"ERROR: promotion refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
