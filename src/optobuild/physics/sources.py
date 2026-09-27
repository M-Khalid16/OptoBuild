"""Optical source models.

CW laser (ideal)
----------------
    A_p(t) = sqrt(P0) * exp(i (phi0 + 2 pi df t)) * j_p

Symbols: P0 [W] average (= instantaneous) power; phi0 [rad] phase;
df [Hz] offset of the laser frequency from the signal reference frequency;
j = unit Jones vector (single polarization: j = [1]).

Assumptions: zero linewidth, no relative-intensity noise, no chirp.
Validity: any P0 >= 0; |df| < fs/2 to be representable. When df is not an
integer multiple of the grid resolution 1/T, the periodic extension has a
phase jump at the window edge (spectral leakage) - flagged by
:func:`frequency_offset_is_periodic`.
Validation: tests/validation/test_sources.py (mean |A|^2 = P0 exactly).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from optobuild.numerics.grid import TimeGrid


def cw_field(
    grid: TimeGrid, power: float, phase: float = 0.0, frequency_offset: float = 0.0
) -> NDArray[np.complex128]:
    """Single-polarization CW envelope, shape (1, N), in sqrt(W)."""
    if power < 0:
        raise ValueError(f"power must be >= 0 W, got {power}.")
    t = grid.time()
    return (math.sqrt(power) * np.exp(1j * (phase + 2.0 * np.pi * frequency_offset * t)))[None, :]


def frequency_offset_is_periodic(
    grid: TimeGrid, frequency_offset: float, rtol: float = 1e-9
) -> bool:
    """True if ``frequency_offset`` is an integer multiple of the grid resolution df."""
    k = frequency_offset / grid.df
    return abs(k - round(k)) <= rtol * max(1.0, abs(k))


PULSE_SHAPES = ("gaussian", "sech")


def pulse_field(
    grid: TimeGrid, shape: str, peak_power: float, width: float, center: float | None = None
) -> NDArray[np.complex128]:
    """Single unchirped pulse, shape (1, N), in sqrt(W).

    gaussian: A = sqrt(P0) exp(-(t - tc)^2 / (2 T0^2))   (T0: 1/e-intensity half width)
    sech:     A = sqrt(P0) sech((t - tc) / T0)
    FWHM of |A|^2: 2 sqrt(ln 2) T0 (gaussian), 2 ln(1 + sqrt 2) T0 (sech).
    ``center`` defaults to the middle of the window. The pulse is not
    periodised; its tails must be negligible at the window edges (checked by
    the component).
    """
    if peak_power < 0 or width <= 0:
        raise ValueError("peak_power must be >= 0 W and width > 0 s.")
    tc = grid.t0 + 0.5 * grid.duration if center is None else center
    x = (grid.time() - tc) / width
    if shape == "gaussian":
        env = np.exp(-0.5 * x**2)
    elif shape == "sech":
        env = 1.0 / np.cosh(np.clip(x, -700, 700))
    else:
        raise ValueError(f"Unknown pulse shape {shape!r}; use one of {PULSE_SHAPES}.")
    return (math.sqrt(peak_power) * env).astype(complex)[None, :]


__all__ = ["PULSE_SHAPES", "cw_field", "frequency_offset_is_periodic", "pulse_field"]
