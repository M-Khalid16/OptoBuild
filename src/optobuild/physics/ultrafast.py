"""Generalized NLSE for ultrafast pulses (physics_models.md 3.27).

In our convention (E = Re{A e^{+i w0 t}}, forward FT with e^{-i w t}, time in
the retarded frame, w baseband angular frequency, positive = higher optical
frequency) the generalized NLSE in the frequency domain is

    dA~/dz = D(w) A~ + N(A),
    D(w)   = -alpha/2 - i sum_{k>=2} beta_k w^k / k!
    N(A)   = -i gamma (1 + w / w0) F{ A [ (1 - f_R) |A|^2 + f_R (h_R * |A|^2) ] }

(the complex conjugate of the usual form, e.g. Dudley, Genty & Coen, Rev.
Mod. Phys. 78, 1135 (2006), eq. 5). The factor (1 + w/w0) is self-steepening
(the time derivative (1 - (i/w0) d/dT) of our convention); * is the causal
convolution; setting f_R = 0 and dropping (1 + w/w0) recovers the scalar NLSE
of physics.nonlinear / solvers.ssfm.

Raman response (Blow & Wood, IEEE J. Quantum Electron. 25, 2665 (1989)):
h_R(t) = (tau1^2 + tau2^2)/(tau1 tau2^2) exp(-t/tau2) sin(t/tau1), t >= 0, with
the exact spectrum H_R(w) = C / ((1/tau2 + i w)^2 + 1/tau1^2),
C = (tau1^2 + tau2^2)/(tau1^2 tau2^2) (H_R(0) = 1); silica: f_R = 0.18,
tau1 = 12.2 fs, tau2 = 32 fs. First moment T_R = f_R int t h_R dt
= f_R 2 tau1^2 tau2 / (tau1^2 + tau2^2).

Conserved quantities (lossless): with self-steepening and Raman the photon
number N_ph = sum |A~(w)|^2 / (w0 + w) is conserved (energy is not: the
Raman shift converts photon energy to phonons); without them the energy
sum |A~|^2 is conserved too.

Closed forms used for validation:
* dispersionless, f_R = 0: the intensity obeys dI/dz + (3 gamma / w0) I dI/dT = 0,
  so I(z, T) = I0(T - 3 gamma I z / w0) (implicit, exact before the shock);
* soliton self-frequency shift of a fundamental soliton (linear Raman
  approximation, Gordon, Opt. Lett. 11, 662 (1986)):
  d<w>/dz = -8 |beta2| T_R / (15 T0^4) (red shift, T0 the sech width).

Assumptions: scalar field (single polarization), frequency-independent gamma
and effective area apart from the (1 + w/w0) factor, Taylor dispersion about
w0, silica-like Raman response, no noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.numerics.fft import inverse_spectrum, spectrum
from optobuild.numerics.grid import TimeGrid

SILICA_RAMAN_FRACTION = 0.18
SILICA_TAU1 = 12.2e-15
SILICA_TAU2 = 32e-15


def dispersion_operator(
    omega: ArrayLike, betas: Sequence[float], alpha: float = 0.0
) -> NDArray[np.complex128]:
    """D(w) = -alpha/2 - i sum_k beta_k w^k / k! with betas = (beta2, beta3, ...) [s^k/m]."""
    w = np.asarray(omega, dtype=float)
    d = np.full(w.shape, -0.5 * alpha, dtype=complex)
    for k, b in enumerate(betas, start=2):
        if b != 0.0:
            d -= 1j * b * w**k / math.factorial(k)
    return d


def raman_response_spectrum(
    omega: ArrayLike, tau1: float = SILICA_TAU1, tau2: float = SILICA_TAU2
) -> NDArray[np.complex128]:
    """Exact Fourier transform H_R(w) of the Blow-Wood response (H_R(0) = 1)."""
    w = np.asarray(omega, dtype=float)
    c = (tau1**2 + tau2**2) / (tau1**2 * tau2**2)
    return c / ((1 / tau2 + 1j * w) ** 2 + 1 / tau1**2)


def raman_time(
    fraction: float = SILICA_RAMAN_FRACTION, tau1: float = SILICA_TAU1, tau2: float = SILICA_TAU2
) -> float:
    """T_R = f_R int t h_R(t) dt = f_R 2 tau1^2 tau2 / (tau1^2 + tau2^2) [s]."""
    return fraction * 2 * tau1**2 * tau2 / (tau1**2 + tau2**2)


class NonlinearOperator:
    """N(A~) of the GNLSE on a fixed grid (precomputes the frequency-domain factors)."""

    def __init__(
        self,
        grid: TimeGrid,
        gamma: float,
        center_frequency: float,
        *,
        self_steepening: bool = True,
        raman_fraction: float = SILICA_RAMAN_FRACTION,
        tau1: float = SILICA_TAU1,
        tau2: float = SILICA_TAU2,
    ) -> None:
        self.grid = grid
        w = grid.angular_frequency()
        w0 = 2 * math.pi * center_frequency
        self.prefactor = -1j * gamma * ((1 + w / w0) if self_steepening else np.ones_like(w))
        self.fraction = raman_fraction
        self.h_r = raman_response_spectrum(w, tau1, tau2) if raman_fraction else None

    def time_domain_term(self, field: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """A [(1 - f_R)|A|^2 + f_R (h_R * |A|^2)] in the time domain."""
        intensity = np.abs(field) ** 2
        if self.h_r is None:
            return field * intensity
        conv = np.real(inverse_spectrum(self.h_r * spectrum(intensity, self.grid), self.grid))
        return field * ((1 - self.fraction) * intensity + self.fraction * conv)

    def __call__(self, spec: NDArray[np.complex128]) -> NDArray[np.complex128]:
        field = inverse_spectrum(spec, self.grid)
        return self.prefactor * spectrum(self.time_domain_term(field), self.grid)


def photon_number(spec: ArrayLike, grid: TimeGrid, center_frequency: float) -> float:
    """sum |A~(w)|^2 df / (h_bar (w0 + w)) up to the constant h_bar [photon-number units]."""
    s = np.asarray(spec)
    w = 2 * math.pi * (center_frequency + grid.frequency())
    return float(np.sum(np.abs(s) ** 2 / w) * grid.df)


def soliton_self_frequency_shift_rate(beta2: float, t0: float, raman_time_s: float) -> float:
    """Gordon: d<w>/dz = -8 |beta2| T_R / (15 T0^4) [rad/s per m]."""
    return -8 * abs(beta2) * raman_time_s / (15 * t0**4)


def shock_intensity_delay(gamma: float, intensity: float, center_frequency: float) -> float:
    """dT/dz = 3 gamma I / w0 [s/m] of an intensity level I (dispersionless, f_R = 0)."""
    return 3 * gamma * intensity / (2 * math.pi * center_frequency)


__all__ = [
    "SILICA_RAMAN_FRACTION",
    "SILICA_TAU1",
    "SILICA_TAU2",
    "NonlinearOperator",
    "dispersion_operator",
    "photon_number",
    "raman_response_spectrum",
    "raman_time",
    "shock_intensity_delay",
    "soliton_self_frequency_shift_rate",
]
