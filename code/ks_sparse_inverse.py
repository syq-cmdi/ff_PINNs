#!/usr/bin/env python3
"""Sparse/noisy coefficient identification for the corrected KS benchmark.

The target equation is

    u_t + lambda_1 u u_x + lambda_2 u_xx + lambda_4 u_xxxx = 0,

on x in [-3*pi, 3*pi), t in [0, 15].  The reference archive is produced by
the corrected ETDRK4 recalculation and has lambda_1=lambda_2=lambda_4=1.

Two deliberately different estimators are provided:

1. ``fourier``: a transparent, non-neural baseline.  It reconstructs a
   truncated periodic Fourier series at each observed time, smooths each
   modal coefficient in time, and regresses the *low-mode Galerkin form* of
   the KS equation.  The fourth derivative is therefore multiplication by
   k**4 rather than a noisy finite difference.
2. ``pinn``: an inverse PINN with exact periodicity from either Fourier input
   features (``embedded_mlp``) or a band-limited Fourier output layer
   (``spectral_output``, the default), and the hard initial-condition ansatz
   u(x,t) = u0(x) + (t/T) N_theta(x,t).  Positive coefficients are learned
   jointly with the field from sparse data and PDE collocation points.

The script records all sampling, noise, seed, optimization, and error
settings in JSON.  It never overwrites the reference archive.

Examples
--------
Fast deterministic baseline:

    python3 code/ks_sparse_inverse.py --method fourier \
      --noise-rel 0.01 --n-times 101 --n-space 128 --seed 20260712

Small PINN smoke test (checks the full differentiation/training path only):

    python3 code/ks_sparse_inverse.py --method both \
      --noise-rel 0.01 --n-times 41 --n-space 64 --seed 20260712 \
      --pinn-pretrain-steps 5 --pinn-steps 20 \
      --pinn-collocation 128 --pinn-batch 256 \
      --pinn-width 24 --pinn-depth 3 --device cpu --tag smoke

Publication-scale starting point (report convergence and multiple seeds):

    python3 code/ks_sparse_inverse.py --method both \
      --noise-rel 0.01 --n-times 101 --n-space 128 --seed 20260712 \
      --pinn-pretrain-steps 3000 --pinn-steps 20000 \
      --pinn-collocation 2048 --pinn-batch 2048 --data-weight 10 \
      --pinn-width 64 --pinn-depth 5 --pinn-modes 12 \
      --pinn-architecture spectral_output --device mps \
      --tag full_seed20260712

Run the final command with at least three explicitly reported seeds before
using PINN uncertainty claims in a manuscript.
"""

# Import torch before NumPy/SciPy on this workstation.  Reversing the order
# can load a second OpenMP runtime and abort the process.
import torch

import argparse
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT
    / "results"
    / "2026_recalculation"
    / "reference"
    / "ks_reference_corrected.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "2026_recalculation" / "inverse"
TARGET = np.ones(3, dtype=np.float64)


@dataclass(frozen=True)
class SparseSample:
    """One reproducible sparse/noisy sample of the reference field."""

    t_indices: np.ndarray
    x_indices_by_time: np.ndarray
    t: np.ndarray
    x: np.ndarray
    u_clean: np.ndarray
    u_noisy: np.ndarray
    noise_abs: float
    noise_rel: float
    seed: int
    full_grid_size: int

    @property
    def n_observations(self) -> int:
        return int(self.u_noisy.size)

    @property
    def sampling_fraction(self) -> float:
        return float(self.n_observations / self.full_grid_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=DEFAULT_REFERENCE, help="Corrected KS NPZ"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Result directory"
    )
    parser.add_argument("--tag", default="run", help="Filename-safe run label")
    parser.add_argument(
        "--method", choices=("fourier", "pinn", "both"), default="fourier"
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--noise-rel",
        type=float,
        default=0.01,
        help="Gaussian noise standard deviation divided by global std(u)",
    )
    parser.add_argument("--n-times", type=int, default=101)
    parser.add_argument("--n-space", type=int, default=128)

    # Fourier/Galerkin baseline.
    parser.add_argument("--fourier-modes", type=int, default=12)
    parser.add_argument("--galerkin-modes", type=int, default=8)
    parser.add_argument(
        "--smoothing-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for the noise-derived temporal spline target",
    )
    parser.add_argument("--time-trim", type=int, default=3)
    parser.add_argument(
        "--regression-ridge",
        type=float,
        default=0.0,
        help="Dimensionless ridge after RMS column scaling",
    )

    # Inverse PINN.
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--pinn-steps", type=int, default=5000)
    parser.add_argument(
        "--pinn-pretrain-steps",
        type=int,
        default=1000,
        help="Data-only field pretraining before joint inverse optimization",
    )
    parser.add_argument("--pinn-batch", type=int, default=1024)
    parser.add_argument("--pinn-collocation", type=int, default=1024)
    parser.add_argument("--pinn-width", type=int, default=48)
    parser.add_argument("--pinn-depth", type=int, default=4)
    parser.add_argument("--pinn-modes", type=int, default=10)
    parser.add_argument(
        "--pinn-architecture",
        choices=("embedded_mlp", "spectral_output"),
        default="spectral_output",
        help=(
            "embedded_mlp uses Fourier inputs; spectral_output predicts a "
            "band-limited Fourier coefficient vector from time"
        ),
    )
    parser.add_argument(
        "--pinn-time-max",
        type=float,
        default=None,
        help="Optional identification horizon; defaults to the full t interval",
    )
    parser.add_argument("--pinn-lr", type=float, default=1.0e-3)
    parser.add_argument("--coeff-lr", type=float, default=2.0e-3)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument(
        "--coeff-init",
        type=float,
        nargs=3,
        default=(0.5, 0.5, 0.5),
        metavar=("L1", "L2", "L4"),
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=4,
        help="Evaluate PINN on every Nth reference point in each dimension",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.noise_rel < 0:
        raise ValueError("--noise-rel must be non-negative")
    if args.n_times < 5 or args.n_space < 5:
        raise ValueError("At least five times and five spatial observations are required")
    if args.fourier_modes < 1 or args.galerkin_modes < 1:
        raise ValueError("Fourier mode counts must be positive")
    if args.galerkin_modes > args.fourier_modes:
        raise ValueError("--galerkin-modes cannot exceed --fourier-modes")
    if 1 + 2 * args.fourier_modes >= args.n_space:
        raise ValueError("Need n_space > 1 + 2*fourier_modes for spatial regression")
    if args.time_trim < 0 or 2 * args.time_trim >= args.n_times - 3:
        raise ValueError("--time-trim removes too many observed times")
    if args.pinn_steps < 0 or args.pinn_pretrain_steps < 0:
        raise ValueError("PINN step counts cannot be negative")
    if args.pinn_batch < 1 or args.pinn_collocation < 1:
        raise ValueError("PINN batch and collocation counts must be positive")
    if args.pinn_width < 1 or args.pinn_depth < 1 or args.pinn_modes < 1:
        raise ValueError("PINN architecture values must be positive")
    if args.log_every < 1 or args.eval_stride < 1:
        raise ValueError("--log-every and --eval-stride must be positive")
    if args.pinn_time_max is not None and args.pinn_time_max <= 0:
        raise ValueError("--pinn-time-max must be positive")
    if args.data_weight < 0 or args.pde_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if any(v <= 0 for v in args.coeff_init):
        raise ValueError("All --coeff-init values must be positive")


def load_reference(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(f"Reference archive not found: {path}")
    with np.load(path) as data:
        missing = {"x", "t"} - set(data.files)
        if missing:
            raise KeyError(f"Reference is missing arrays: {sorted(missing)}")
        field_key = "u_etdrk4" if "u_etdrk4" in data.files else "u_archive"
        if field_key not in data.files:
            raise KeyError("Reference needs u_etdrk4 or u_archive")
        x = np.asarray(data["x"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        u = np.asarray(data[field_key], dtype=np.float64)
    if u.shape != (t.size, x.size):
        raise ValueError(f"Expected u shape {(t.size, x.size)}, got {u.shape}")
    dx = np.diff(x)
    dt = np.diff(t)
    if not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-12):
        raise ValueError("The Fourier baseline requires a uniform periodic x grid")
    if not np.all(dt > 0):
        raise ValueError("Reference times must be strictly increasing")
    expected_left = -3.0 * np.pi
    expected_length = 6.0 * np.pi
    period = dx[0] * x.size
    if not np.isclose(x[0], expected_left, atol=1e-10) or not np.isclose(
        period, expected_length, atol=1e-10
    ):
        raise ValueError(
            "This experiment is defined for corrected coordinates "
            "x in [-3*pi,3*pi); refusing an ambiguously scaled archive"
        )
    return x, t, u, field_key


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_sparse(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    n_times: int,
    n_space: int,
    noise_rel: float,
    seed: int,
) -> SparseSample:
    if n_times > t.size or n_space > x.size:
        raise ValueError("Requested sparse sample is larger than the reference grid")
    rng = np.random.default_rng(seed)
    # Even temporal coverage makes the inverse problem interpretable; spatial
    # sensor locations are redrawn at each snapshot to avoid a privileged grid.
    t_indices = np.unique(
        np.rint(np.linspace(0, t.size - 1, n_times)).astype(np.int64)
    )
    if t_indices.size != n_times:
        raise ValueError("n_times is too large for unique rounded time indices")
    x_indices = np.empty((n_times, n_space), dtype=np.int64)
    clean = np.empty((n_times, n_space), dtype=np.float64)
    for row, time_index in enumerate(t_indices):
        x_indices[row] = np.sort(rng.choice(x.size, n_space, replace=False))
        clean[row] = u[time_index, x_indices[row]]
    noise_abs = float(noise_rel * np.std(u))
    noisy = clean + rng.normal(0.0, noise_abs, size=clean.shape)
    return SparseSample(
        t_indices=t_indices,
        x_indices_by_time=x_indices,
        t=t[t_indices],
        x=x[x_indices],
        u_clean=clean,
        u_noisy=noisy,
        noise_abs=noise_abs,
        noise_rel=float(noise_rel),
        seed=int(seed),
        full_grid_size=int(u.size),
    )


def real_fourier_design(
    x: np.ndarray, x_left: float, length: float, modes: int
) -> np.ndarray:
    phase = 2.0 * np.pi * (x - x_left) / length
    design = np.ones((x.size, 1 + 2 * modes), dtype=np.float64)
    for mode in range(1, modes + 1):
        design[:, 2 * mode - 1] = np.cos(mode * phase)
        design[:, 2 * mode] = np.sin(mode * phase)
    return design


def fourier_derivative_designs(
    x: np.ndarray, x_left: float, length: float, modes: int
) -> Tuple[np.ndarray, np.ndarray]:
    phase = 2.0 * np.pi * (x - x_left) / length
    value = real_fourier_design(x, x_left, length, modes)
    first = np.zeros_like(value)
    for mode in range(1, modes + 1):
        wavenumber = 2.0 * np.pi * mode / length
        cosine = np.cos(mode * phase)
        sine = np.sin(mode * phase)
        first[:, 2 * mode - 1] = -wavenumber * sine
        first[:, 2 * mode] = wavenumber * cosine
    return value, first


def estimate_fourier_galerkin(
    x: np.ndarray,
    t: np.ndarray,
    u_reference: np.ndarray,
    sample: SparseSample,
    modes: int,
    galerkin_modes: int,
    smoothing_multiplier: float,
    time_trim: int,
    regression_ridge: float,
) -> Dict[str, Any]:
    start = time.perf_counter()
    length = float((x[1] - x[0]) * x.size)
    coefficients = np.empty((sample.t.size, 1 + 2 * modes), dtype=np.float64)

    for row in range(sample.t.size):
        observed_x = sample.x[row]
        design = real_fourier_design(observed_x, x[0], length, modes)
        coefficients[row] = np.linalg.lstsq(
            design, sample.u_noisy[row], rcond=None
        )[0]

    # For independently sampled sensor noise, the variance of a sine/cosine
    # coefficient is approximately 2*sigma^2/n_x.  This supplies an explicit,
    # noise-derived spline target instead of visually tuning each curve.
    spline_target = float(
        smoothing_multiplier
        * sample.t.size
        * 2.0
        * sample.noise_abs**2
        / sample.x.shape[1]
    )
    smoothed = np.empty_like(coefficients)
    time_derivative = np.empty_like(coefficients)
    for column in range(coefficients.shape[1]):
        spline = UnivariateSpline(
            sample.t,
            coefficients[:, column],
            k=3,
            s=spline_target,
        )
        smoothed[:, column] = spline(sample.t)
        time_derivative[:, column] = spline.derivative(1)(sample.t)

    value_design, first_design = fourier_derivative_designs(
        x, x[0], length, modes
    )
    reconstructed = smoothed @ value_design.T
    reconstructed_x = smoothed @ first_design.T
    nonlinear_spectrum = np.fft.fft(
        reconstructed * reconstructed_x, axis=1
    ) / x.size

    rows: List[np.ndarray] = []
    targets: List[float] = []
    selected_times = range(time_trim, sample.t.size - time_trim)
    for time_row in selected_times:
        for mode in range(1, galerkin_modes + 1):
            k = 2.0 * np.pi * mode / length
            # a*cos + b*sin = Re[(a-i*b) exp(i*k*x_phase)].
            u_hat = (
                smoothed[time_row, 2 * mode - 1]
                - 1j * smoothed[time_row, 2 * mode]
            ) / 2.0
            ut_hat = (
                time_derivative[time_row, 2 * mode - 1]
                - 1j * time_derivative[time_row, 2 * mode]
            ) / 2.0
            feature = np.asarray(
                [
                    nonlinear_spectrum[time_row, mode],
                    -(k**2) * u_hat,
                    (k**4) * u_hat,
                ],
                dtype=np.complex128,
            )
            rows.extend((feature.real, feature.imag))
            targets.extend((-ut_hat.real, -ut_hat.imag))

    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    column_rms = np.sqrt(np.mean(matrix**2, axis=0))
    if np.any(column_rms <= np.finfo(float).eps):
        raise RuntimeError("Degenerate regression column; change retained modes")
    standardized = matrix / column_rms
    gram = standardized.T @ standardized
    rhs = standardized.T @ target
    scaled_solution = np.linalg.solve(
        gram + regression_ridge * np.eye(3, dtype=np.float64), rhs
    )
    estimate = scaled_solution / column_rms
    predicted = matrix @ estimate
    residual = predicted - target

    reconstructed_reference = u_reference[sample.t_indices]
    reconstruction_rel_l2 = float(
        np.linalg.norm(reconstructed - reconstructed_reference)
        / np.linalg.norm(reconstructed_reference)
    )
    errors = np.abs(estimate - TARGET)
    return {
        "method": "periodic_fourier_low_mode_galerkin",
        "estimate": estimate.tolist(),
        "target": TARGET.tolist(),
        "absolute_error": errors.tolist(),
        "relative_error": errors.tolist(),
        "mean_relative_coefficient_error": float(np.mean(errors)),
        "max_relative_coefficient_error": float(np.max(errors)),
        "modal_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "modal_target_rms": float(np.sqrt(np.mean(target**2))),
        "standardized_design_condition": float(np.linalg.cond(standardized)),
        "field_reconstruction_relative_l2_at_sampled_times": reconstruction_rel_l2,
        "spatial_modes_reconstructed": int(modes),
        "galerkin_modes_regressed": int(galerkin_modes),
        "temporal_spline_target_per_coefficient": spline_target,
        "smoothing_multiplier": float(smoothing_multiplier),
        "time_snapshots_used_in_regression": int(sample.t.size - 2 * time_trim),
        "complex_modal_equations": int(matrix.shape[0] // 2),
        "real_regression_rows": int(matrix.shape[0]),
        "regression_ridge_after_column_scaling": float(regression_ridge),
        "wall_seconds": float(time.perf_counter() - start),
    }


def inverse_softplus(value: float) -> float:
    # log(expm1(value)) is accurate for the coefficient initializations used here.
    return math.log(math.expm1(value))


class HardPeriodicICPINN(torch.nn.Module):
    """Fourier-periodic MLP with an exactly imposed analytical initial field."""

    def __init__(
        self,
        x_left: float,
        length: float,
        t_final: float,
        modes: int,
        width: int,
        depth: int,
        coefficient_initialization: Iterable[float],
    ) -> None:
        super().__init__()
        self.x_left = float(x_left)
        self.length = float(length)
        self.t_final = float(t_final)
        self.modes = int(modes)
        layers: List[torch.nn.Module] = []
        in_features = 2 + 2 * modes  # normalized t, constant, sine/cosine pairs
        for layer_index in range(depth):
            layers.append(
                torch.nn.Linear(in_features if layer_index == 0 else width, width)
            )
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(width, 1))
        self.network = torch.nn.Sequential(*layers)
        raw = [inverse_softplus(float(value)) for value in coefficient_initialization]
        self.raw_coefficients = torch.nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_modules = [
            module for module in self.network if isinstance(module, torch.nn.Linear)
        ]
        for module in linear_modules:
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)
        # Starting from N_theta=0 gives u(x,t)=u0(x), avoiding a random
        # high-frequency fourth derivative that can overwhelm the first updates.
        torch.nn.init.zeros_(linear_modules[-1].weight)
        torch.nn.init.zeros_(linear_modules[-1].bias)

    @property
    def coefficients(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_coefficients)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau = t / self.t_final
        phase = 2.0 * math.pi * (x - self.x_left) / self.length
        features = [tau, torch.ones_like(tau)]
        for mode in range(1, self.modes + 1):
            features.extend((torch.cos(mode * phase), torch.sin(mode * phase)))
        network_value = self.network(torch.cat(features, dim=1))
        initial_value = -torch.sin(x / 3.0)
        return initial_value + tau * network_value


class HardPeriodicICSpectralPINN(torch.nn.Module):
    """Time network with an explicitly band-limited periodic spatial output."""

    def __init__(
        self,
        x_left: float,
        length: float,
        t_final: float,
        modes: int,
        width: int,
        depth: int,
        coefficient_initialization: Iterable[float],
    ) -> None:
        super().__init__()
        self.x_left = float(x_left)
        self.length = float(length)
        self.t_final = float(t_final)
        self.modes = int(modes)
        layers: List[torch.nn.Module] = []
        for layer_index in range(depth):
            layers.append(torch.nn.Linear(1 if layer_index == 0 else width, width))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(width, 1 + 2 * modes))
        self.network = torch.nn.Sequential(*layers)
        raw = [inverse_softplus(float(value)) for value in coefficient_initialization]
        self.raw_coefficients = torch.nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_modules = [
            module for module in self.network if isinstance(module, torch.nn.Linear)
        ]
        for module in linear_modules:
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)
        torch.nn.init.zeros_(linear_modules[-1].weight)
        torch.nn.init.zeros_(linear_modules[-1].bias)

    @property
    def coefficients(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_coefficients)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau = t / self.t_final
        modal_coefficients = self.network(tau)
        phase = 2.0 * math.pi * (x - self.x_left) / self.length
        update = modal_coefficients[:, 0:1]
        for mode in range(1, self.modes + 1):
            update = (
                update
                + modal_coefficients[:, 2 * mode - 1 : 2 * mode]
                * torch.cos(mode * phase)
                + modal_coefficients[:, 2 * mode : 2 * mode + 1]
                * torch.sin(mode * phase)
            )
        initial_value = -torch.sin(x / 3.0)
        return initial_value + tau * update


def derivative(
    value: torch.Tensor, coordinate: torch.Tensor, order: int = 1
) -> torch.Tensor:
    result = value
    for _ in range(order):
        result = torch.autograd.grad(
            result,
            coordinate,
            grad_outputs=torch.ones_like(result),
            create_graph=True,
            retain_graph=True,
        )[0]
    return result


def pinn_residual(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    prediction = model(x, t)
    ut = derivative(prediction, t)
    ux = derivative(prediction, x)
    uxx = derivative(ux, x)
    uxxx = derivative(uxx, x)
    uxxxx = derivative(uxxx, x)
    lambda_1, lambda_2, lambda_4 = model.coefficients
    return ut + lambda_1 * prediction * ux + lambda_2 * uxx + lambda_4 * uxxxx


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1)


def hard_constraint_diagnostics(
    model: torch.nn.Module,
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        initial_x = tensor(x, device)
        initial_t = torch.zeros_like(initial_x)
        initial_prediction = model(initial_x, initial_t)
        initial_exact = -torch.sin(initial_x / 3.0)
        initial_error = float(
            torch.max(torch.abs(initial_prediction - initial_exact)).cpu()
        )

    check_times = torch.linspace(
        float(t[0]), float(t[-1]), 17, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    x_left = torch.full_like(check_times, float(x[0]), requires_grad=True)
    x_right = torch.full_like(
        check_times,
        float(x[0] + (x[1] - x[0]) * x.size),
        requires_grad=True,
    )
    left_value = model(x_left, check_times)
    right_value = model(x_right, check_times)
    jumps: Dict[str, float] = {}
    for order in range(4):
        jumps[f"order_{order}"] = float(
            torch.max(torch.abs(left_value - right_value)).detach().cpu()
        )
        if order < 3:
            left_value = derivative(left_value, x_left)
            right_value = derivative(right_value, x_right)
    return {
        "initial_condition_max_abs_error": initial_error,
        "periodic_jump_max_abs_over_17_times": jumps,
        "periodic_derivative_orders_checked": [0, 1, 2, 3],
    }


def estimate_inverse_pinn(
    x: np.ndarray,
    t: np.ndarray,
    u_reference: np.ndarray,
    sample: SparseSample,
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, Any]:
    start = time.perf_counter()
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    length = float((x[1] - x[0]) * x.size)
    identification_time_max = float(
        t[-1] if args.pinn_time_max is None else min(args.pinn_time_max, t[-1])
    )
    observed_time_rows = sample.t <= identification_time_max + 1.0e-12
    if int(np.sum(observed_time_rows)) < 5:
        raise ValueError("The PINN identification horizon contains fewer than five snapshots")
    model_class = (
        HardPeriodicICPINN
        if args.pinn_architecture == "embedded_mlp"
        else HardPeriodicICSpectralPINN
    )
    model = model_class(
        x_left=float(x[0]),
        length=length,
        t_final=identification_time_max,
        modes=args.pinn_modes,
        width=args.pinn_width,
        depth=args.pinn_depth,
        coefficient_initialization=args.coeff_init,
    ).to(device)

    network_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "raw_coefficients"
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": network_parameters, "lr": args.pinn_lr},
            {"params": [model.raw_coefficients], "lr": args.coeff_lr},
        ]
    )

    observed_x = tensor(sample.x[observed_time_rows].ravel(), device)
    observed_t = tensor(
        np.repeat(
            sample.t[observed_time_rows, None], sample.x.shape[1], axis=1
        ).ravel(),
        device,
    )
    observed_u = tensor(sample.u_noisy[observed_time_rows].ravel(), device)
    u_scale = float(np.std(u_reference))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 104729)
    history: List[Dict[str, Any]] = []
    n_observations = observed_u.shape[0]

    for pretrain_step in range(1, args.pinn_pretrain_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_size = min(args.pinn_batch, n_observations)
        indices_cpu = torch.randint(
            0, n_observations, (batch_size,), generator=generator
        )
        indices = indices_cpu.to(device)
        data_prediction = model(observed_x[indices], observed_t[indices])
        data_loss = torch.mean(((data_prediction - observed_u[indices]) / u_scale) ** 2)
        if not torch.isfinite(data_loss):
            raise FloatingPointError(
                f"Non-finite data loss at pretraining step {pretrain_step}"
            )
        data_loss.backward()
        optimizer.step()
        should_log = (
            pretrain_step == 1
            or pretrain_step == args.pinn_pretrain_steps
            or pretrain_step % args.log_every == 0
        )
        if should_log:
            record = {
                "phase": "data_pretraining",
                "step": int(pretrain_step),
                "loss": float(data_loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "pde_loss": None,
                "estimate": model.coefficients.detach().cpu().numpy().tolist(),
                "elapsed_seconds": float(time.perf_counter() - start),
            }
            history.append(record)
            print(
                "PINN pretrain "
                f"{pretrain_step:6d}: data={record['data_loss']:.4e}",
                flush=True,
            )

    for step in range(1, args.pinn_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_size = min(args.pinn_batch, n_observations)
        indices_cpu = torch.randint(
            0, n_observations, (batch_size,), generator=generator
        )
        indices = indices_cpu.to(device)
        data_prediction = model(observed_x[indices], observed_t[indices])
        data_loss = torch.mean(((data_prediction - observed_u[indices]) / u_scale) ** 2)

        # Collocation points are redrawn every step from a deterministic CPU RNG,
        # then moved to the requested accelerator.
        collocation_x = (
            float(x[0])
            + length
            * torch.rand(
                (args.pinn_collocation, 1), generator=generator, dtype=torch.float32
            )
        ).to(device)
        collocation_t = (
            identification_time_max
            * torch.rand(
                (args.pinn_collocation, 1), generator=generator, dtype=torch.float32
            )
        ).to(device)
        collocation_x.requires_grad_(True)
        collocation_t.requires_grad_(True)
        residual = pinn_residual(model, collocation_x, collocation_t)
        pde_loss = torch.mean(residual**2)
        loss = args.data_weight * data_loss + args.pde_weight * pde_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at PINN step {step}")
        loss.backward()
        optimizer.step()

        should_log = (
            step == 1 or step == args.pinn_steps or step % args.log_every == 0
        )
        if should_log:
            record = {
                "phase": "joint_inverse",
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "estimate": model.coefficients.detach().cpu().numpy().tolist(),
                "elapsed_seconds": float(time.perf_counter() - start),
            }
            history.append(record)
            print(
                "PINN step "
                f"{step:6d}: loss={record['loss']:.4e}, "
                f"data={record['data_loss']:.4e}, pde={record['pde_loss']:.4e}, "
                f"lambda={np.asarray(record['estimate']).round(6).tolist()}",
                flush=True,
            )

    estimate = model.coefficients.detach().cpu().numpy().astype(np.float64)
    errors = np.abs(estimate - TARGET)

    # A strided reference-grid diagnostic is enough to detect a PINN that fits
    # coefficients while failing to represent the state.  Chunking bounds memory.
    x_eval = x[:: args.eval_stride]
    t_within_horizon = t[t <= identification_time_max + 1.0e-12]
    t_eval = t_within_horizon[:: args.eval_stride]
    predictions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for time_value in t_eval:
            x_tensor = tensor(x_eval, device)
            t_tensor = torch.full_like(x_tensor, float(time_value))
            predictions.append(model(x_tensor, t_tensor).cpu().numpy().ravel())
    prediction = np.asarray(predictions)
    reference_within_horizon = u_reference[
        t <= identification_time_max + 1.0e-12
    ]
    truth = reference_within_horizon[:: args.eval_stride, :: args.eval_stride]
    field_rel_l2 = float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))
    constraint_diagnostics = hard_constraint_diagnostics(
        model, x, np.asarray([t[0], identification_time_max]), device
    )

    checkpoint_path = run_dir / "inverse_pinn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {
                "x_left": float(x[0]),
                "length": length,
                "t_final": identification_time_max,
                "modes": args.pinn_modes,
                "width": args.pinn_width,
                "depth": args.pinn_depth,
                "architecture": args.pinn_architecture,
            },
            "seed": args.seed,
            "target_coefficients": TARGET.tolist(),
        },
        checkpoint_path,
    )
    return {
        "method": "hard_periodic_hard_initial_condition_inverse_pinn",
        "estimate": estimate.tolist(),
        "target": TARGET.tolist(),
        "absolute_error": errors.tolist(),
        "relative_error": errors.tolist(),
        "mean_relative_coefficient_error": float(np.mean(errors)),
        "max_relative_coefficient_error": float(np.max(errors)),
        "field_relative_l2_strided_grid": field_rel_l2,
        "evaluation_stride": int(args.eval_stride),
        "steps": int(args.pinn_steps),
        "data_pretraining_steps": int(args.pinn_pretrain_steps),
        "observation_batch": int(min(args.pinn_batch, n_observations)),
        "observations_within_identification_horizon": int(n_observations),
        "time_snapshots_within_identification_horizon": int(
            np.sum(observed_time_rows)
        ),
        "identification_time_max": identification_time_max,
        "collocation_points_per_step": int(args.pinn_collocation),
        "pde_weight": float(args.pde_weight),
        "data_weight": float(args.data_weight),
        "network_width": int(args.pinn_width),
        "network_depth": int(args.pinn_depth),
        "periodic_fourier_modes": int(args.pinn_modes),
        "architecture": args.pinn_architecture,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "network_learning_rate": float(args.pinn_lr),
        "coefficient_learning_rate": float(args.coeff_lr),
        "coefficient_initialization": [float(v) for v in args.coeff_init],
        "coefficient_constraint": "positive via softplus transform; no value prior",
        "device": str(device),
        "dtype": "float32",
        "hard_initial_condition": (
            "u(x,t)=-sin(x/3)+(t/T_id)N_theta(x,t), "
            f"T_id={identification_time_max:g}"
        ),
        "hard_constraint_diagnostics": constraint_diagnostics,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "wall_seconds": float(time.perf_counter() - start),
    }


def environment_record() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready_sample(sample: SparseSample) -> Dict[str, Any]:
    realized_noise = sample.u_noisy - sample.u_clean
    signal_rms = float(np.sqrt(np.mean(sample.u_clean**2)))
    noise_rms = float(np.sqrt(np.mean(realized_noise**2)))
    return {
        "seed": sample.seed,
        "n_time_snapshots": int(sample.t.size),
        "n_spatial_observations_per_snapshot": int(sample.x.shape[1]),
        "n_observations": sample.n_observations,
        "full_grid_size": sample.full_grid_size,
        "sampling_fraction": sample.sampling_fraction,
        "noise_distribution": "independent Gaussian, zero mean",
        "noise_relative_to_global_field_std": sample.noise_rel,
        "noise_absolute_std": sample.noise_abs,
        "realized_noise_mean": float(np.mean(realized_noise)),
        "realized_noise_std": float(np.std(realized_noise)),
        "realized_noise_rms": noise_rms,
        "realized_snr_db": float(20.0 * np.log10(signal_rms / noise_rms))
        if noise_rms > 0
        else None,
        "time_index_selection": "rounded evenly spaced indices including endpoints",
        "space_index_selection": "uniform without replacement, redrawn per time",
        "time_indices": sample.t_indices.tolist(),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    x, t, u, field_key = load_reference(args.reference.resolve())
    sample = sample_sparse(
        x=x,
        t=t,
        u=u,
        n_times=args.n_times,
        n_space=args.n_space,
        noise_rel=args.noise_rel,
        seed=args.seed,
    )

    safe_tag = "".join(character if character.isalnum() or character in "-_" else "_" for character in args.tag)
    run_name = f"{safe_tag}_seed{args.seed}_noise{args.noise_rel:.4f}"
    run_dir = args.output_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "status": "completed",
        "equation": "u_t + lambda_1*u*u_x + lambda_2*u_xx + lambda_4*u_xxxx = 0",
        "coefficient_order": ["lambda_1", "lambda_2", "lambda_4"],
        "target_coefficients": TARGET.tolist(),
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference.resolve()),
            "field": field_key,
            "shape": list(u.shape),
            "x_interval_periodic": [float(x[0]), float(x[0] + (x[1] - x[0]) * x.size)],
            "t_interval": [float(t[0]), float(t[-1])],
            "global_field_std": float(np.std(u)),
        },
        "sample": json_ready_sample(sample),
        "requested_method": args.method,
        "environment": environment_record(),
        "command_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "estimators": {},
    }

    if args.method in ("fourier", "both"):
        result["estimators"]["fourier"] = estimate_fourier_galerkin(
            x=x,
            t=t,
            u_reference=u,
            sample=sample,
            modes=args.fourier_modes,
            galerkin_modes=args.galerkin_modes,
            smoothing_multiplier=args.smoothing_multiplier,
            time_trim=args.time_trim,
            regression_ridge=args.regression_ridge,
        )
        estimate = result["estimators"]["fourier"]["estimate"]
        print(f"Fourier/Galerkin estimate: {np.asarray(estimate).round(8).tolist()}")

    if args.method in ("pinn", "both"):
        result["estimators"]["pinn"] = estimate_inverse_pinn(
            x=x,
            t=t,
            u_reference=u,
            sample=sample,
            args=args,
            run_dir=run_dir,
        )

    result_path = run_dir / "metrics.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
