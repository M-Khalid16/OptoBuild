"""Electrical drive waveforms and the Mach-Zehnder modulator (physics_models.md 3.3-3.4).

NRZ waveform
------------
    v(t) = V0 + (V1 - V0) * sum_k b_k p(t - k T_b)

``p`` is the unit rectangle of width T_b = 1/R_b, optionally convolved with a
Gaussian impulse response ``h(t) ∝ exp(-t^2 / (2 sigma^2))``, which gives a
10-90 % rise time ``t_r = 2 * z_0.9 * sigma`` with ``z_0.9 = Phi^-1(0.9)
= 1.2815515655446004`` (erf-shaped edges). The filter is applied in the
frequency domain, ``H(f) = exp(-2 pi^2 sigma^2 f^2)``, i.e. circularly over the
window (the bit pattern is treated as periodic).
Units: V (volts), s, bit/s.

Mach-Zehnder modulator (push-pull, dual-arm, finite extinction ratio)
---------------------------------------------------------------------
    A_out = A_in * sqrt(IL) * [a1 exp(+i dphi/2) + a2 exp(-i dphi/2)]
          = A_in * sqrt(IL) * [cos(dphi/2) + i eps sin(dphi/2)]
    dphi(t) = pi (V(t) + V_bias) / V_pi + phi0
    a1 = (1 + eps)/2, a2 = (1 - eps)/2, eps = 1/sqrt(ER)

Power transfer ``T = IL [cos^2(dphi/2) + eps^2 sin^2(dphi/2)]``: maximum IL at
dphi = 0, minimum IL/ER at dphi = pi, so the static extinction ratio is ER.
Residual chirp: output phase ``arctan(eps tan(dphi/2))`` (zero for eps = 0).
Symbols: V_pi [V], V_bias [V], phi0 [rad], IL linear power transmission in
(0, 1], ER linear >= 1.
Assumptions: instantaneous (quasi-static) response, no electro-optic bandwidth
limit, polarization-independent, lossless arms apart from IL.
With the drive NRZ between -V_pi/2 and +V_pi/2 and V_bias = -V_pi/2 the
modulator is biased at quadrature and swings between minimum and maximum.
References: Agrawal, Fiber-Optic Communication Systems, 5th ed., sec. 3.4;
Saleh & Teich, Fundamentals of Photonics, 3rd ed., ch. 21.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid

Z_090 = 1.2815515655446004
"""Standard-normal quantile Phi^-1(0.9)."""


def gaussian_sigma_from_rise_time(rise_time: float) -> float:
    """Gaussian impulse-response width sigma [s] giving a 10-90 % rise time [s]."""
    return rise_time / (2.0 * Z_090)


def nrz_grid(n_bits: int, bit_rate: float, samples_per_bit: int) -> TimeGrid:
    """Grid with ``samples_per_bit`` samples per bit: dt = 1 / (R_b * sps)."""
    return TimeGrid(n_bits * samples_per_bit, 1.0 / (bit_rate * samples_per_bit))


def nrz_waveform(
    bits: ArrayLike,
    samples_per_bit: int,
    grid: TimeGrid,
    low: float,
    high: float,
    rise_time: float = 0.0,
) -> NDArray[np.float64]:
    """NRZ voltage waveform [V] on ``grid`` (bit k occupies samples [k sps, (k+1) sps))."""
    b = np.asarray(bits, dtype=float)
    rect = low + (high - low) * np.repeat(b, samples_per_bit)
    if rise_time <= 0.0:
        return rect
    sigma = gaussian_sigma_from_rise_time(rise_time)
    h = np.exp(-2.0 * np.pi**2 * sigma**2 * grid.frequency() ** 2)
    return np.real(apply_transfer_function(rect, h, grid))


def mzm_field_transfer(
    voltage: ArrayLike,
    v_pi: float,
    v_bias: float = 0.0,
    insertion_loss: float = 1.0,
    extinction_ratio: float = math.inf,
    phi0: float = 0.0,
) -> NDArray[np.complex128]:
    """Complex field transmission A_out/A_in of the MZM for drive voltage(s) [V]."""
    eps = 0.0 if math.isinf(extinction_ratio) else 1.0 / math.sqrt(extinction_ratio)
    half = 0.5 * (np.pi * (np.asarray(voltage, dtype=float) + v_bias) / v_pi + phi0)
    return math.sqrt(insertion_loss) * (np.cos(half) + 1j * eps * np.sin(half))


def mzm_power_transfer(
    voltage: ArrayLike,
    v_pi: float,
    v_bias: float = 0.0,
    insertion_loss: float = 1.0,
    extinction_ratio: float = math.inf,
    phi0: float = 0.0,
) -> NDArray[np.float64]:
    """Closed-form power transmission T = IL [cos^2(dphi/2) + eps^2 sin^2(dphi/2)]."""
    eps2 = 0.0 if math.isinf(extinction_ratio) else 1.0 / extinction_ratio
    half = 0.5 * (np.pi * (np.asarray(voltage, dtype=float) + v_bias) / v_pi + phi0)
    return insertion_loss * (np.cos(half) ** 2 + eps2 * np.sin(half) ** 2)


__all__ = [
    "Z_090",
    "gaussian_sigma_from_rise_time",
    "mzm_field_transfer",
    "mzm_power_transfer",
    "nrz_grid",
    "nrz_waveform",
]
