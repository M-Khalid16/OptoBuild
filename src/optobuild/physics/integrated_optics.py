"""Integrated-optics building blocks: waveguides, couplers, rings, MZIs, Bragg gratings
(physics_models.md 3.22-3.24).

Convention (consistent with the complex envelope E = Re{A exp(+i w t)}): a
forward wave accumulates exp(-i beta z), so a lossy waveguide of length L has
the field transmission

    h(nu) = exp(-alpha L / 2) exp(-i beta(nu) L),
    beta(nu) = 2 pi / c [n_eff nu_0 + n_g (nu - nu_0)]                 (1)

(first-order dispersion about the design frequency nu_0 = c / lambda_0:
phase index n_eff and group index n_g at lambda_0; alpha [1/m] power loss).
Group delay tau = -d arg(h) / d omega = n_g L / c.

Symmetric directional coupler with power coupling ratio kappa^2 (amplitude
kappa, through t = sqrt(1 - kappa^2)), ports (1, 2) in -> (3, 4) out:

    [b3, b4]^T = [[t, -i kappa], [-i kappa, t]] [a1, a2]^T              (2)

Rings (Bogaerts et al., Laser Photonics Rev. 6, 47 (2012), eqs. 1-3, in our
sign convention), round-trip field factor A = a exp(-i phi), a = exp(-alpha L/2),
phi = beta L:

    all-pass through:  H_t = (t - A) / (1 - t A)                           (3)
    add-drop through:  H_t = (t1 - t2 A) / (1 - t1 t2 A)
    add-drop drop:     H_d = -kappa1 kappa2 sqrt(A) / (1 - t1 t2 A)        (4)

with sqrt(A) = sqrt(a) exp(-i phi / 2) (half round trip). Resonances at
phi = 2 pi m; FSR = lambda^2 / (n_g L) (in wavelength) = c / (n_g L) (in
frequency); all-pass loaded Q = pi n_g L sqrt(t a) / (lambda (1 - t a)),
FWHM = lambda^2 (1 - t a) / (pi n_g L sqrt(t a)).

Mach-Zehnder interferometer: coupler (2) - arms h_1, h_2 - coupler (2);
with 3-dB couplers, bar = (h1 - h2)/2, cross = -i (h1 + h2)/2
(|cross|^2 = cos^2(dphi/2) for equal arm losses); FSR = lambda^2 / (n_g dL).

Uniform Bragg grating (coupled-mode theory; Erdogan, J. Lightwave Technol.
15, 1277 (1997); Kogelnik 1972), coupling coefficient kappa_g [1/m],
detuning delta = beta - pi / Lambda, s = sqrt(kappa_g^2 - delta^2):

    r = -i kappa_g sinh(sL) / (s cosh(sL) + i delta sinh(sL))
    t = s / (s cosh(sL) + i delta sinh(sL))                             (5)

(t excludes the carrier phase exp(-i pi L / Lambda) of the reference wave,
which is included by ``bragg_grating``). Peak reflectance tanh^2(kappa_g L);
bandwidth between first zeros dlambda = lambda^2 / (pi n_g) sqrt(kappa_g^2 +
(pi / L)^2). For a first-order square-wave index modulation of depth dn
(50 % duty), kappa_g = 2 dn / lambda.

Assumptions: single mode per waveguide, linear dispersion about lambda_0 (no
n_g dispersion), wavelength-independent coupling ratios, lossless couplers,
no back-reflections except in the grating, unidirectional excitation.
Validation: tests/validation/test_integrated_optics.py.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.constants import SPEED_OF_LIGHT


def propagation_constant(
    frequency: ArrayLike, n_eff: float, n_group: float, design_wavelength: float
) -> NDArray[np.float64]:
    """beta(nu) [rad/m] from eq. (1) at absolute optical frequencies ``frequency`` [Hz]."""
    nu = np.asarray(frequency, dtype=float)
    nu0 = SPEED_OF_LIGHT / design_wavelength
    return 2 * math.pi / SPEED_OF_LIGHT * (n_eff * nu0 + n_group * (nu - nu0))


def waveguide_transmission(
    frequency: ArrayLike,
    length: float,
    n_eff: float,
    n_group: float,
    design_wavelength: float,
    loss: float = 0.0,
) -> NDArray[np.complex128]:
    """Field transmission exp(-alpha L/2 - i beta L) of a waveguide; ``loss`` alpha [1/m]."""
    beta = propagation_constant(frequency, n_eff, n_group, design_wavelength)
    return np.exp(-0.5 * loss * length - 1j * beta * length)


def coupler_matrix(power_coupling: float) -> NDArray[np.complex128]:
    """Eq. (2): [[t, -i kappa], [-i kappa, t]] for power coupling ratio kappa^2."""
    if not 0.0 <= power_coupling <= 1.0:
        raise ValueError(f"power coupling ratio must be in [0, 1], got {power_coupling}")
    k = math.sqrt(power_coupling)
    t = math.sqrt(1.0 - power_coupling)
    return np.array([[t, -1j * k], [-1j * k, t]])


def all_pass_ring(round_trip: ArrayLike, power_coupling: float) -> NDArray[np.complex128]:
    """Eq. (3): through transmission for round-trip factor A = a exp(-i phi)."""
    a = np.asarray(round_trip, dtype=complex)
    t = math.sqrt(1.0 - power_coupling)
    return (t - a) / (1.0 - t * a)


def add_drop_ring(
    half_round_trip: ArrayLike, power_coupling_in: float, power_coupling_drop: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Eq. (4): (through, drop) from the half-round-trip factor h = sqrt(a) exp(-i phi/2).

    A = h^2 is the round-trip factor (passing h avoids the square-root branch ambiguity).
    """
    h = np.asarray(half_round_trip, dtype=complex)
    t1, t2 = math.sqrt(1 - power_coupling_in), math.sqrt(1 - power_coupling_drop)
    k1, k2 = math.sqrt(power_coupling_in), math.sqrt(power_coupling_drop)
    a = h * h
    den = 1.0 - t1 * t2 * a
    return (t1 - t2 * a) / den, -k1 * k2 * h / den


def mzi(
    arm1: ArrayLike, arm2: ArrayLike, coupling_in: float = 0.5, coupling_out: float = 0.5
) -> NDArray[np.complex128]:
    """2x2 transfer matrix C_out diag(h1, h2) C_in, shape (2, 2, N) (rows: outputs)."""
    h1 = np.asarray(arm1, dtype=complex)
    h2 = np.asarray(arm2, dtype=complex)
    ci, co = coupler_matrix(coupling_in), coupler_matrix(coupling_out)
    arms = np.zeros((2, 2, h1.size), dtype=complex)
    arms[0, 0], arms[1, 1] = h1, h2
    return np.einsum("ij,jkn,kl->iln", co, arms, ci)


def all_pass_fwhm_phase(power_coupling: float, round_trip_amplitude: float) -> float:
    """Exact full width [rad of round-trip phase] of an all-pass dip at half its depth.

    |H_t|^2 = 1 - D / (B + C sin^2(phi/2)), B = (1 - ta)^2, C = 4 ta; the half-depth
    level (T_max + T_min)/2 is reached at sin^2(phi/2) = B / (2B + C), so
    FWHM_phi = 4 arcsin((1 - ta) / sqrt(2 (1 + t^2 a^2))). For ta -> 1 this tends to
    2 (1 - ta) / sqrt(ta), the Lorentzian result behind ``all_pass_loaded_q``.
    """
    ta = math.sqrt(1 - power_coupling) * round_trip_amplitude
    return 4 * math.asin((1 - ta) / math.sqrt(2 * (1 + ta * ta)))


def ring_fsr_wavelength(wavelength: float, n_group: float, length: float) -> float:
    """FSR = lambda^2 / (n_g L) [m] (length = ring circumference)."""
    return wavelength**2 / (n_group * length)


def all_pass_loaded_q(
    wavelength: float, n_group: float, length: float, power_coupling: float, loss: float
) -> float:
    """Loaded Q = pi n_g L sqrt(t a) / (lambda (1 - t a)), a = exp(-alpha L / 2).

    High-Q (Lorentzian) limit of the exact half-depth width (``all_pass_fwhm_phase``).
    """
    ta = math.sqrt(1 - power_coupling) * math.exp(-0.5 * loss * length)
    return math.pi * n_group * length * math.sqrt(ta) / (wavelength * (1 - ta))


def bragg_coupling_square(index_step: float, wavelength: float) -> float:
    """kappa_g = 2 dn / lambda for a first-order 50 %-duty square index modulation."""
    return 2.0 * index_step / wavelength


def bragg_grating(
    frequency: ArrayLike,
    length: float,
    period: float,
    coupling: float,
    n_eff: float,
    n_group: float,
    design_wavelength: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Eq. (5): (reflection r, transmission t incl. the propagation phase) of a uniform grating.

    ``coupling`` kappa_g [1/m]; ``period`` Lambda [m]; Bragg wavelength 2 n_eff Lambda
    (at the design point).
    """
    beta = propagation_constant(frequency, n_eff, n_group, design_wavelength)
    delta = beta - math.pi / period
    s = np.sqrt(complex(coupling) ** 2 - delta.astype(complex) ** 2)
    sl = s * length
    # s -> 0 limit handled by sinh(sL)/s -> L
    small = np.abs(s) < 1e-12 / max(length, 1e-30)
    s_safe = np.where(small, 1.0, s)
    sinh_over_s = np.where(small, length, np.sinh(sl) / s_safe)
    den = np.cosh(sl) + 1j * delta * sinh_over_s
    r = -1j * coupling * sinh_over_s / den
    t = np.exp(-1j * math.pi / period * length) / den
    return r, t


def bragg_peak_reflectance(coupling: float, length: float) -> float:
    """tanh^2(kappa_g L)."""
    return math.tanh(coupling * length) ** 2


def bragg_bandwidth(wavelength: float, n_group: float, coupling: float, length: float) -> float:
    """Full width between the first reflection zeros: lambda^2/(pi n_g) sqrt(kappa^2+(pi/L)^2)."""
    return wavelength**2 / (math.pi * n_group) * math.hypot(coupling, math.pi / length)


__all__ = [
    "add_drop_ring",
    "all_pass_fwhm_phase",
    "all_pass_loaded_q",
    "all_pass_ring",
    "bragg_bandwidth",
    "bragg_coupling_square",
    "bragg_grating",
    "bragg_peak_reflectance",
    "coupler_matrix",
    "mzi",
    "propagation_constant",
    "ring_fsr_wavelength",
    "waveguide_transmission",
]
