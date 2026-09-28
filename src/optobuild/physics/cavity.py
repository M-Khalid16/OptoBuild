"""Lumped laser-cavity elements for round-trip models (physics_models.md 3.28).

All act on a scalar envelope A(t) [sqrt(W)] on a periodic time window (one
round trip of a single circulating pulse):

* energy-saturated gain (gain recovery slow compared with the round trip,
  fast compared with many round trips: the gain sees the pulse energy E):
  power gain G = exp(g0 / (1 + E / E_sat)), g0 = small-signal ln-gain;
* fast saturable absorber: power transmission T(t) = 1 - q0 / (1 + P(t)/P_sat) - l_ns;
* Gaussian bandpass filter: field transfer exp(-w^2 / W^2) (W the 1/e
  half-width of the field transfer in angular frequency);
* Gaussian time gate (active amplitude modulator in the parabolic
  approximation): field transmission exp(-t^2 / T_m^2);
* output coupler: fraction T of the power leaves, sqrt(1 - T) of the field stays.

Closed form (validation of the cavity iteration): a loop gain -> gate ->
filter -> coupler maps a chirp-free Gaussian exp(-a t^2) onto itself with
a -> F(a + c), c = 1/T_m^2, F(x) = x / (1 + 4 x / W^2); fixed point
a* = (-c + sqrt(c^2 + c W^2)) / 2. The gate and filter then transmit the
energy fraction a* / (a* + c), so with a passive transmission eta the
steady energy at the gain input is E* = E_sat (g0 / L - 1),
L = -ln(eta (1 - T) a* / (a* + c)) (lasing requires g0 > L).
References: Haus, IEEE J. Sel. Top. Quantum Electron. 6, 1173 (2000)
(master-equation view); Siegman, *Lasers* (1986) ch. 27-28.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid


def saturated_gain(field: ArrayLike, grid: TimeGrid, g0: float, e_sat: float) -> NDArray:
    """Multiply by sqrt(G), G = exp(g0 / (1 + E / E_sat)), E = sum |A|^2 dt."""
    a = np.asarray(field, dtype=complex)
    energy = float(np.sum(np.abs(a) ** 2) * grid.dt)
    return a * math.exp(0.5 * g0 / (1 + energy / e_sat))


def saturable_absorber(
    field: ArrayLike, modulation_depth: float, saturation_power: float, nonsaturable: float = 0.0
) -> NDArray:
    """Instantaneous absorber: field x sqrt(1 - q0/(1 + P/P_sat) - l_ns)."""
    a = np.asarray(field, dtype=complex)
    t = 1 - modulation_depth / (1 + np.abs(a) ** 2 / saturation_power) - nonsaturable
    return a * np.sqrt(np.clip(t, 0.0, None))


def gaussian_filter(field: ArrayLike, grid: TimeGrid, half_width: float) -> NDArray:
    """Field transfer exp(-w^2 / W^2), W [rad/s]."""
    h = np.exp(-((grid.angular_frequency() / half_width) ** 2))
    return apply_transfer_function(np.asarray(field, dtype=complex), h, grid)


def gaussian_gate(field: ArrayLike, grid: TimeGrid, gate_time: float) -> NDArray:
    """Field transmission exp(-t^2 / T_m^2) about the window centre (sample N // 2)."""
    t = np.arange(grid.n_samples) * grid.dt - (grid.n_samples // 2) * grid.dt
    return np.asarray(field, dtype=complex) * np.exp(-(t**2) / gate_time**2)


def output_coupler(field: ArrayLike, transmission: float) -> tuple[NDArray, NDArray]:
    """(field kept in the cavity, output field) for power output fraction T."""
    a = np.asarray(field, dtype=complex)
    return a * math.sqrt(1 - transmission), a * math.sqrt(transmission)


def gate_filter_fixed_point(gate_time: float, filter_half_width: float) -> float:
    """a* of the Gaussian gate + filter round-trip map (see module docstring) [1/s^2]."""
    c = 1 / gate_time**2
    return 0.5 * (-c + math.sqrt(c * c + c * filter_half_width**2))


def gate_filter_steady_energy(
    gate_time: float,
    filter_half_width: float,
    g0: float,
    e_sat: float,
    output_transmission: float,
    passive_transmission: float = 1.0,
) -> float:
    """E* at the gain input (0 below threshold)."""
    a = gate_filter_fixed_point(gate_time, filter_half_width)
    c = 1 / gate_time**2
    loss = -math.log(passive_transmission * (1 - output_transmission) * a / (a + c))
    return max(0.0, e_sat * (g0 / loss - 1))


__all__ = [
    "gate_filter_fixed_point",
    "gate_filter_steady_energy",
    "gaussian_filter",
    "gaussian_gate",
    "output_coupler",
    "saturable_absorber",
    "saturated_gain",
]
