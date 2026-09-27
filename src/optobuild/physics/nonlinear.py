"""Kerr nonlinearity of optical fibers (physics_models.md sec. 3.10).

Nonlinear Schroedinger equation in OUR convention (E = Re{A exp(+i w0 t)},
forward FT kernel exp(-i w t), retarded time T = t - beta1 z):

    dA/dz = -alpha/2 A  - i (beta2/2 w^2 + beta3/6 w^3) A   [frequency domain]
                        - i gamma |A|^2 A                    [time domain]

(Agrawal, Nonlinear Fiber Optics, eq. 2.3.46, has +i gamma |A|^2 A: the sign of
i is flipped by the carrier convention, exactly as for dispersion.)
Physical check: the SPM phase phi(T) = -gamma |A(T)|^2 L_eff gives an
instantaneous frequency shift d(phi)/dT < 0 where the power rises, i.e. the
leading edge is red-shifted, as observed experimentally.

Symbols: gamma = 2 pi n2 / (lambda A_eff) [1/(W m)]; n2 [m^2/W] nonlinear
index; A_eff [m^2] effective mode area; alpha [1/m]; L_eff(h) =
(1 - exp(-alpha h))/alpha [m].

The nonlinear-plus-loss sub-step dA/dz = -alpha/2 A - i gamma |A|^2 A is solved
exactly (|A|^2 decays as exp(-alpha z), so the accumulated phase is
-gamma |A0|^2 L_eff(h)); the solver combines it with the exact linear
dispersion operator by symmetric splitting.

Assumptions: scalar field (single polarization), instantaneous Kerr response
(no Raman), no self-steepening, gamma frequency independent. Valid for pulses
longer than ~1 ps and powers well below the stimulated Raman/Brillouin
thresholds; Raman and self-steepening are planned for Phase 9.
References: G. P. Agrawal, Nonlinear Fiber Optics, 6th ed., ch. 2, 4, 5.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gamma_from_n2(n2: float, wavelength: float, effective_area: float) -> float:
    """Nonlinear coefficient gamma = 2 pi n2 / (lambda A_eff) [1/(W m)]."""
    return 2.0 * math.pi * n2 / (wavelength * effective_area)


def effective_length(length: float, alpha: float) -> float:
    """L_eff = (1 - exp(-alpha L)) / alpha [m] (= L for alpha = 0)."""
    if alpha == 0.0:
        return length
    return -math.expm1(-alpha * length) / alpha


def kerr_loss_step(
    field: ArrayLike, gamma: float, alpha: float, step: float
) -> NDArray[np.complex128]:
    """Exact solution over ``step`` [m] of dA/dz = -alpha/2 A - i gamma |A|^2 A.

    ``field`` has shape (n_pol, N); the power driving the phase is summed over
    polarizations (scalar model for n_pol = 1).
    """
    a = np.asarray(field, dtype=complex)
    power = np.sum(np.abs(a) ** 2, axis=0, keepdims=True)
    phase = -gamma * power * effective_length(step, alpha)
    return a * math.exp(-0.5 * alpha * step) * np.exp(1j * phase)


def nonlinear_length(gamma: float, peak_power: float) -> float:
    """L_NL = 1 / (gamma P0) [m] (infinite if gamma P0 = 0)."""
    return math.inf if gamma * peak_power == 0 else 1.0 / (gamma * peak_power)


def soliton_order(gamma: float, peak_power: float, t0: float, beta2: float) -> float:
    """N = sqrt(gamma P0 T0^2 / |beta2|) (sech pulse, anomalous dispersion)."""
    return math.sqrt(gamma * peak_power * t0**2 / abs(beta2)) if beta2 else math.inf


__all__ = [
    "effective_length",
    "gamma_from_n2",
    "kerr_loss_step",
    "nonlinear_length",
    "soliton_order",
]
