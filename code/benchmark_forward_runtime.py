#!/usr/bin/env python3
"""Repeat the canonical ETDRK4 integration for a load-sensitive timing audit.

Accuracy is checked against the locked corrected archive on every repetition.
The benchmark deliberately times only the integration loop, matching the
timer inside :class:`FourierKS`; file I/O and metric evaluation are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from ks_scale_reference_audit import FourierKS, error_metrics, initial_condition


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT / "results" / "2026_recalculation" / "reference" / "ks_reference_corrected.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "2026_recalculation" / "reference" / "etdrk4_runtime_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 5:
        raise ValueError("use at least five timed repeats and a nonnegative warmup")

    archive = np.load(args.reference)
    target = np.asarray(archive["u_archive"], dtype=np.float64)
    x = np.asarray(archive["x"], dtype=np.float64)
    solver = FourierKS(length=6.0 * np.pi, n=512, dealias=True)
    u0 = initial_condition(x)

    for _ in range(args.warmup):
        solver.solve_etdrk4(u0, final_time=15.0, dt=0.0046875, output_dt=0.075)

    seconds: list[float] = []
    errors: list[float] = []
    for index in range(args.repeats):
        _, field, elapsed = solver.solve_etdrk4(
            u0, final_time=15.0, dt=0.0046875, output_dt=0.075
        )
        metric = error_metrics(field, target)["global_relative_l2"]
        if not np.isclose(metric, 8.266172974248464e-11, rtol=5e-6, atol=1e-15):
            raise RuntimeError(f"accuracy changed on repeat {index}: {metric:.16e}")
        seconds.append(float(elapsed))
        errors.append(float(metric))

    values = np.asarray(seconds)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    result = {
        "scope": "canonical integration loop only; excludes file I/O and metric evaluation",
        "method": "Fourier pseudospectral ETDRK4, 2/3 dealiased",
        "configuration": {
            "N": 512,
            "dt": 0.0046875,
            "final_time": 15.0,
            "output_dt": 0.075,
            "warmup_repeats": args.warmup,
            "timed_repeats": args.repeats,
        },
        "wall_seconds_all": seconds,
        "wall_seconds_median": float(np.median(values)),
        "wall_seconds_q25": float(q25),
        "wall_seconds_q75": float(q75),
        "wall_seconds_minimum": float(np.min(values)),
        "wall_seconds_maximum": float(np.max(values)),
        "global_relative_l2_all": errors,
        "reference_path": str(args.reference),
        "reference_sha256": sha256(args.reference),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "command": (
            "python3 code/benchmark_forward_runtime.py --warmup "
            f"{args.warmup} --repeats {args.repeats}"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
