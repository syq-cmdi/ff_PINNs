#!/usr/bin/env python3
"""Reconstruct the archived KS scaling and build a converged reference.

This is intentionally independent of ``ks_pinn_benchmark.py``.  It treats the
MAT file as an archived field, estimates its PDE coefficients in the stored
coordinates, reconstructs the unique positive space/time scaling that yields
the unit-coefficient Kuramoto--Sivashinsky equation, and verifies the field
with a Fourier ETDRK4 integration.  A second-order semi-implicit BDF solver is
included as an algorithmically independent cross-check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.io import loadmat
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAT = ROOT / "code" / "KS_raissi.mat"
DEFAULT_OUT = ROOT / "results" / "2026_recalculation" / "reference"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_terms(u: np.ndarray, length: float) -> tuple[np.ndarray, ...]:
    """Return ux, uxx and uxxxx for ``u[x,t]`` on a periodic grid."""
    n = u.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    uhat = np.fft.fft(u, axis=0)
    ux = np.fft.ifft(1j * k[:, None] * uhat, axis=0).real
    uxx = np.fft.ifft(-(k[:, None] ** 2) * uhat, axis=0).real
    uxxxx = np.fft.ifft((k[:, None] ** 4) * uhat, axis=0).real
    return ux, uxx, uxxxx


def identify_raw_coefficients(
    x: np.ndarray, t: np.ndarray, u: np.ndarray
) -> dict[str, object]:
    """Fit ut + a*u*ux + b*uxx + c*uxxxx = 0 in stored coordinates."""
    ux, uxx, uxxxx = spectral_terms(u, float(x[-1] - x[0]))
    # A fifth-degree local polynomial uses only seven time samples.  The
    # reported fit excludes the three edge samples affected by extrapolation.
    ut = savgol_filter(
        u,
        window_length=7,
        polyorder=5,
        deriv=1,
        delta=float(t[1] - t[0]),
        axis=1,
        mode="interp",
    )
    core = (slice(None), slice(3, -3))
    matrix = np.column_stack(
        [
            (u * ux)[core].ravel(),
            uxx[core].ravel(),
            uxxxx[core].ravel(),
        ]
    )
    target = -ut[core].ravel()
    coeff, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = matrix @ coeff - target

    # Deterministic resampling of complete time slices quantifies sensitivity
    # to the retained output times without pretending that the archived field
    # is an ensemble of independent experimental observations.
    rng = np.random.default_rng(20260711)
    n_time = u[:, 3:-3].shape[1]
    bootstrap = []
    for _ in range(200):
        ids = rng.integers(0, n_time, size=n_time)
        blocks = []
        rhs = []
        for idx in ids:
            j = idx + 3
            blocks.append(
                np.column_stack(
                    [(u[:, j] * ux[:, j]), uxx[:, j], uxxxx[:, j]]
                )
            )
            rhs.append(-ut[:, j])
        b_matrix = np.vstack(blocks)
        b_target = np.concatenate(rhs)
        bootstrap.append(np.linalg.lstsq(b_matrix, b_target, rcond=None)[0])
    bootstrap_array = np.asarray(bootstrap)
    ci = np.quantile(bootstrap_array, [0.025, 0.975], axis=0)

    a, b, c = coeff
    space_scale = a / b
    time_scale = a * space_scale
    unit_coeff = np.array(
        [
            a * space_scale / time_scale,
            b * space_scale**2 / time_scale,
            c * space_scale**4 / time_scale,
        ]
    )
    wrong_space, wrong_time = 10.0, 50.0
    wrong_coeff = np.array(
        [
            a * wrong_space / wrong_time,
            b * wrong_space**2 / wrong_time,
            c * wrong_space**4 / wrong_time,
        ]
    )
    return {
        "raw_coefficients": coeff.tolist(),
        "raw_coefficients_interval95_time_slice_resampling": ci.T.tolist(),
        "relative_regression_residual": float(
            np.linalg.norm(residual) / np.linalg.norm(target)
        ),
        "reconstructed_space_scale": float(space_scale),
        "reconstructed_time_scale": float(time_scale),
        "unit_coordinate_coefficients": unit_coeff.tolist(),
        "wrong_x10_t50_coordinate_coefficients": wrong_coeff.tolist(),
        "analytical_raw_coefficients": [
            5.0 / np.pi,
            5.0 / (3.0 * np.pi**2),
            5.0 / (27.0 * np.pi**4),
        ],
        "analytical_space_scale": 3.0 * np.pi,
        "analytical_time_scale": 15.0,
    }


@dataclass
class FourierKS:
    length: float
    n: int
    dealias: bool = True

    def __post_init__(self) -> None:
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.length / self.n)
        self.linear = self.k**2 - self.k**4
        mode = np.fft.fftfreq(self.n) * self.n
        self.mask = np.abs(mode) <= self.n / 3

    def nonlinear(self, v: np.ndarray) -> np.ndarray:
        u = np.fft.ifft(v).real
        square_hat = np.fft.fft(u * u)
        if self.dealias:
            square_hat = square_hat * self.mask
        return -0.5j * self.k * square_hat

    def solve_etdrk4(
        self, u0: np.ndarray, final_time: float, dt: float, output_dt: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        steps_per_output = int(round(output_dt / dt))
        if not np.isclose(steps_per_output * dt, output_dt, rtol=0, atol=1e-13):
            raise ValueError("dt must divide output_dt")
        steps = int(round(final_time / dt))
        if steps % steps_per_output:
            raise ValueError("output interval must divide final_time")

        h = final_time / steps
        linear = self.linear
        e = np.exp(h * linear)
        e2 = np.exp(h * linear / 2.0)
        roots = np.exp(1j * np.pi * (np.arange(1, 65) - 0.5) / 64)
        lr = h * linear[:, None] + roots[None, :]
        q = h * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1))
        f1 = h * np.real(
            np.mean(
                (-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2))
                / lr**3,
                axis=1,
            )
        )
        f2 = h * np.real(
            np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1)
        )
        f3 = h * np.real(
            np.mean(
                (-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr))
                / lr**3,
                axis=1,
            )
        )

        v = np.fft.fft(u0)
        outputs = [u0.copy()]
        start = time.perf_counter()
        for step in range(1, steps + 1):
            nv = self.nonlinear(v)
            a = e2 * v + q * nv
            na = self.nonlinear(a)
            b = e2 * v + q * na
            nb = self.nonlinear(b)
            c = e2 * a + q * (2.0 * nb - nv)
            nc = self.nonlinear(c)
            v = e * v + f1 * nv + 2.0 * f2 * (na + nb) + f3 * nc
            if step % steps_per_output == 0:
                outputs.append(np.fft.ifft(v).real.copy())
        elapsed = time.perf_counter() - start
        times = np.linspace(0.0, final_time, len(outputs))
        return times, np.asarray(outputs), elapsed

    def solve_sbdf2(
        self, u0: np.ndarray, final_time: float, dt: float, output_dt: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Independent second-order semi-implicit BDF/Adams--Bashforth solve."""
        steps_per_output = int(round(output_dt / dt))
        if not np.isclose(steps_per_output * dt, output_dt, rtol=0, atol=1e-13):
            raise ValueError("dt must divide output_dt")
        steps = int(round(final_time / dt))
        v_prev = np.fft.fft(u0)
        n_prev = self.nonlinear(v_prev)
        outputs = [u0.copy()]
        start = time.perf_counter()
        v = (v_prev + dt * n_prev) / (1.0 - dt * self.linear)
        n_now = self.nonlinear(v)
        if 1 % steps_per_output == 0:
            outputs.append(np.fft.ifft(v).real.copy())
        for step in range(2, steps + 1):
            v_next = (
                4.0 * v
                - v_prev
                + 2.0 * dt * (2.0 * n_now - n_prev)
            ) / (3.0 - 2.0 * dt * self.linear)
            v_prev, v = v, v_next
            n_prev, n_now = n_now, self.nonlinear(v)
            if step % steps_per_output == 0:
                outputs.append(np.fft.ifft(v).real.copy())
        elapsed = time.perf_counter() - start
        times = np.linspace(0.0, final_time, len(outputs))
        return times, np.asarray(outputs), elapsed


def initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(x / 3.0)


def reference_on_archive_grid(solution: np.ndarray, n: int) -> np.ndarray:
    """Map a solver field ``[t,x]`` to the archive's 512-point unique grid."""
    if n == 512:
        return solution
    if n < 512 and 512 % n == 0:
        # Periodic Fourier interpolation, preserving the solver's normalization.
        v = np.fft.fft(solution, axis=1)
        padded = np.zeros((solution.shape[0], 512), dtype=complex)
        half = n // 2
        padded[:, :half] = v[:, :half]
        padded[:, -half + 1 :] = v[:, -half + 1 :]
        padded[:, half] = v[:, half] / 2.0
        padded[:, -half] = v[:, half] / 2.0
        return np.fft.ifft(padded, axis=1).real * (512 / n)
    if n > 512 and n % 512 == 0:
        return solution[:, :: n // 512]
    raise ValueError(f"unsupported grid conversion {n} -> 512")


def error_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = predicted - target
    per_time = np.linalg.norm(difference, axis=1) / np.linalg.norm(target, axis=1)
    return {
        "global_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(target)),
        "median_time_relative_l2": float(np.median(per_time)),
        "maximum_time_relative_l2": float(np.max(per_time)),
        "final_time_relative_l2": float(per_time[-1]),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def energy_budget(field: np.ndarray, times: np.ndarray, length: float) -> dict[str, float]:
    n = field.shape[1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    v = np.fft.fft(field, axis=1)
    ux = np.fft.ifft(1j * k[None, :] * v, axis=1).real
    uxx = np.fft.ifft(-(k[None, :] ** 2) * v, axis=1).real
    energy = 0.5 * np.mean(field**2, axis=1)
    denergy = savgol_filter(
        energy,
        window_length=7,
        polyorder=5,
        deriv=1,
        delta=float(times[1] - times[0]),
        mode="interp",
    )
    production_minus_dissipation = np.mean(ux**2, axis=1) - np.mean(uxx**2, axis=1)
    residual = denergy - production_minus_dissipation
    core = slice(3, -3)
    return {
        "initial_mean": float(np.mean(field[0])),
        "maximum_absolute_mean_drift": float(
            np.max(np.abs(np.mean(field, axis=1) - np.mean(field[0])))
        ),
        "energy_balance_relative_rms": float(
            np.linalg.norm(residual[core])
            / np.linalg.norm(production_minus_dissipation[core])
        ),
        "energy_balance_absolute_rms": float(np.sqrt(np.mean(residual[core] ** 2))),
    }


def save_convergence(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    x: np.ndarray,
    times: np.ndarray,
    archived: np.ndarray,
    reference: np.ndarray,
    convergence: list[dict[str, object]],
    output: Path,
) -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax0 = fig.add_subplot(grid[0, 0])
    image = ax0.pcolormesh(x, times, archived, shading="auto", cmap="RdBu_r")
    ax0.set(xlabel=r"$X$", ylabel=r"$T$", title="(a) Archived field, corrected coordinates")
    fig.colorbar(image, ax=ax0, label=r"$u$")

    ax1 = fig.add_subplot(grid[0, 1])
    error = np.abs(reference - archived)
    image_error = ax1.pcolormesh(
        x, times, np.maximum(error, 1e-16), shading="auto", cmap="magma",
        norm=plt.matplotlib.colors.LogNorm(vmin=1e-13, vmax=max(1e-12, error.max())),
    )
    ax1.set(xlabel=r"$X$", ylabel=r"$T$", title="(b) ETDRK4 absolute error")
    fig.colorbar(image_error, ax=ax1, label=r"$|u_{ETD}-u_{archive}|$")

    ax2 = fig.add_subplot(grid[1, 0])
    for n in sorted({int(row["N"]) for row in convergence}):
        subset = [row for row in convergence if int(row["N"]) == n]
        ax2.loglog(
            [float(row["dt"]) for row in subset],
            [float(row["global_relative_l2"]) for row in subset],
            "o-",
            label=f"N={n}",
        )
    ax2.invert_xaxis()
    dt_ticks = [0.0375, 0.01875, 0.009375, 0.0046875]
    ax2.set_xticks(dt_ticks)
    ax2.set_xticklabels(["0.0375", "0.01875", "0.009375", "0.0046875"], rotation=25)
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set(xlabel=r"internal $\Delta T$", ylabel="global relative $L_2$", title="(c) Space-time convergence")
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(grid[1, 1])
    per_time = np.linalg.norm(reference - archived, axis=1) / np.linalg.norm(archived, axis=1)
    ax3.semilogy(times, per_time, color="#1f5a94")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.set(xlabel=r"$T$", ylabel="relative $L_2$", title="(d) Error at each archived time")

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-sbdf2", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = loadmat(args.mat)
    x_raw = np.asarray(raw["x"]).ravel()
    t_raw = np.asarray(raw["tt"]).ravel()
    u_with_endpoint = np.asarray(raw["uu"], dtype=np.float64)
    u = u_with_endpoint[:-1, :]
    scale = identify_raw_coefficients(x_raw, t_raw, u)
    # The regression reconstructs these scales to approximately 1e-9.  Use
    # their exact analytical values for the canonical problem and time grid.
    space_scale = 3.0 * np.pi
    time_scale = 15.0
    x = x_raw[:-1] * space_scale
    times = t_raw * time_scale
    archived = u.T
    length = float(2.0 * space_scale)

    convergence: list[dict[str, object]] = []
    best_solution: np.ndarray | None = None
    best_times: np.ndarray | None = None
    for n in (128, 256, 512, 1024):
        solver = FourierKS(length, n, dealias=True)
        solver_x = np.linspace(-space_scale, space_scale, n, endpoint=False)
        u0 = initial_condition(solver_x)
        for dt in (0.0375, 0.01875, 0.009375, 0.0046875):
            t_etd, solution, elapsed = solver.solve_etdrk4(
                u0, final_time=time_scale, dt=dt, output_dt=0.075
            )
            mapped = reference_on_archive_grid(solution, n)
            metrics = error_metrics(mapped, archived)
            convergence.append(
                {
                    "method": "dealiased_ETDRK4",
                    "N": n,
                    "dt": dt,
                    "wall_seconds": elapsed,
                    **metrics,
                }
            )
            print(
                f"ETDRK4 N={n:4d} dt={dt:.7f}: "
                f"L2={metrics['global_relative_l2']:.3e}, {elapsed:.2f} s"
            )
            if n == 512 and np.isclose(dt, 0.0046875):
                best_solution = mapped
                best_times = t_etd

    if best_solution is None or best_times is None:
        raise RuntimeError("best reference was not generated")

    crosscheck: dict[str, object] | None = None
    if not args.skip_sbdf2:
        solver = FourierKS(length, 256, dealias=True)
        solver_x = np.linspace(-space_scale, space_scale, 256, endpoint=False)
        t_bdf, solution_bdf, elapsed = solver.solve_sbdf2(
            initial_condition(solver_x),
            final_time=time_scale,
            dt=0.0001875,
            output_dt=0.075,
        )
        mapped_bdf = reference_on_archive_grid(solution_bdf, 256)
        crosscheck = {
            "method": "dealiased_SBDF2_AB2",
            "N": 256,
            "dt": 0.0001875,
            "wall_seconds": elapsed,
            **error_metrics(mapped_bdf, archived),
        }
        print(
            "SBDF2 cross-check: "
            f"L2={crosscheck['global_relative_l2']:.3e}, {elapsed:.2f} s"
        )

    best_metrics = error_metrics(best_solution, archived)
    audit = {
        "created_utc": "2026-07-11",
        "mat_file": str(args.mat),
        "mat_sha256": file_sha256(args.mat),
        "stored_grid": {
            "x_range_including_periodic_endpoint": [float(x_raw[0]), float(x_raw[-1])],
            "t_range": [float(t_raw[0]), float(t_raw[-1])],
            "u_shape": list(u_with_endpoint.shape),
            "periodic_endpoint_max_jump": float(
                np.max(np.abs(u_with_endpoint[0] - u_with_endpoint[-1]))
            ),
            "initial_condition_relative_l2": float(
                np.linalg.norm(u[:, 0] - initial_condition(x))
                / np.linalg.norm(u[:, 0])
            ),
        },
        "scale_reconstruction": scale,
        "corrected_problem": {
            "equation": "u_T + u u_X + u_XX + u_XXXX = 0",
            "space_range": [-space_scale, space_scale],
            "time_range": [0.0, time_scale],
            "periodic_length": length,
            "initial_condition": "u(X,0)=-sin(X/3)",
        },
        "reference": {
            "method": "Fourier pseudospectral ETDRK4, 2/3 dealiased",
            "N": 512,
            "dt": 0.0046875,
            **best_metrics,
            **energy_budget(best_solution, best_times, length),
        },
        "independent_crosscheck": crosscheck,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "interpretation": (
            "The archive is scale-consistent with the standard unit-coefficient KS "
            "equation on X in [-3pi,3pi] and T in [0,15]. The x10/t50 mapping "
            "does not preserve unit coefficients."
        ),
    }

    save_convergence(convergence, args.output / "etdrk4_convergence.csv")
    (args.output / "scale_reference_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "ks_reference_corrected.npz",
        x=x,
        t=best_times,
        u_archive=archived,
        u_etdrk4=best_solution,
    )
    make_figure(
        x,
        best_times,
        archived,
        best_solution,
        convergence,
        args.output / "figure_scale_reference.png",
    )
    print(json.dumps(audit["reference"], indent=2))


if __name__ == "__main__":
    main()
