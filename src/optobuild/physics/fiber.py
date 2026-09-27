"""Linear single-mode fiber (physics_models.md 3.5).

Model (our FFT convention, retarded frame moving at the group velocity):

    A~(L, w) = A~(0, w) exp(-alpha L / 2) exp(-i (beta2/2 w^2 + beta3/6 w^3) L)
    t0 -> t0 + beta1 L,        beta1 = n_g / c

Symbols: L [m] length; alpha [1/m] power attenuation coefficient
(P(L) = P(0) exp(-alpha L)); beta1 [s/m]; beta2 [s^2/m]; beta3 [s^3/m];
w [rad/s] baseband angular frequency; n_g group index.
beta2 and beta3 follow from D [s/m^2] and slope S [s/m^3] at the carrier
wavelength (core.units.dispersion_to_beta2 / dispersion_slope_to_beta3).

Assumptions: linear propagation (no Kerr effect, no Raman), scalar field
(no PMD / birefringence), frequency-independent loss, dispersion expanded to
third order about the carrier, single mode.
Validity: low launch powers such that the nonlinear length exceeds L by a
large factor; |w| within the simulated band. For nonlinear propagation use
SSFM (Phase 4, not implemented).

Numerics: exact in the frequency domain for the sampled (periodic) signal;
circular in time, so a dispersive spread comparable to the window wraps
around (see :func:`dispersive_spread`).

Analytic reference (unchirped Gaussian, |A(0,T)|^2 = P0 exp(-T^2/T0^2)):
    A(L, T) = sqrt(P0) T0 / sqrt(T0^2 + i beta2 L) exp(-T^2 / (2 (T0^2 + i beta2 L)))
    T1/T0 = sqrt(1 + (L/L_D)^2),  L_D = T0^2 / |beta2|
(Agrawal, Nonlinear Fiber Optics, 6th ed., sec. 3.2, with the sign of i
flipped for our convention.)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid


def fiber_transfer_function(
    angular_frequency: ArrayLike, length: float, alpha: float, beta2: float, beta3: float = 0.0
) -> NDArray[np.complex128]:
    """Field transfer function H(w) of a linear fiber (FFT-order input)."""
    w = np.asarray(angular_frequency, dtype=float)
    phase = (0.5 * beta2 * w**2 + beta3 * w**3 / 6.0) * length
    return math.exp(-0.5 * alpha * length) * np.exp(-1j * phase)


def group_delay(length: float, group_index: float) -> float:
    """Group delay beta1 L = n_g L / c [s]."""
    return group_index * length / SPEED_OF_LIGHT


def propagate_linear(
    field: ArrayLike,
    grid: TimeGrid,
    length: float,
    alpha: float,
    beta2: float,
    beta3: float = 0.0,
    group_index: float = 1.0,
) -> tuple[NDArray[np.complex128], TimeGrid]:
    """Propagate a (n_pol, N) envelope; returns (field, grid with t0 += beta1 L)."""
    h = fiber_transfer_function(grid.angular_frequency(), length, alpha, beta2, beta3)
    out = apply_transfer_function(field, h, grid)
    return out, grid.with_t0(grid.t0 + group_delay(length, group_index))


def dispersion_length(t0: float, beta2: float) -> float:
    """L_D = T0^2 / |beta2| [m] (infinite for beta2 = 0)."""
    return math.inf if beta2 == 0 else t0**2 / abs(beta2)


def dispersive_spread(bandwidth: float, length: float, beta2: float, beta3: float = 0.0) -> float:
    """Differential group delay [s] across a band of full width ``bandwidth`` [Hz].

    Estimate |beta2| L dw + |beta3| L dw^2 / 8 with dw = 2 pi bandwidth; used for the
    window wrap-around diagnostic.
    """
    dw = 2.0 * math.pi * bandwidth
    return abs(beta2) * length * dw + abs(beta3) * length * dw**2 / 8.0


__all__ = [
    "dispersion_length",
    "dispersive_spread",
    "fiber_transfer_function",
    "group_delay",
    "propagate_linear",
]
