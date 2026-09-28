"""Spectral analysis of device responses: resonances, FSR, FWHM, Q, extinction, group delay.

Inputs are sampled responses on an ascending axis x (wavelength [m] or
frequency [Hz]) with power transmission T = |H|^2 (linear, 0..1) or complex
field transfer H.

* Resonances: local minima (``kind="dip"``) or maxima (``"peak"``) of T with a
  prominence of at least ``min_depth`` (linear), via ``scipy.signal.find_peaks``.
* FWHM: full width at the half-depth level of the resonance,
  T_half = (T_edge + T_extremum)/2 with T_edge the largest (dip) or smallest
  (peak) value between neighbouring resonances; crossings located by linear
  interpolation (error O(dx^2 T'') - resolve the resonance with >= 10 samples
  across the FWHM; a ``Diagnostic``-worthy condition reported via
  ``samples_per_fwhm``).
* Q = x_res / FWHM (dimensionless, for x in wavelength or frequency).
* FSR = mean spacing of adjacent resonances.
* Extinction ratio = T_edge / T_extremum (dip) or its inverse (peak), in dB.
* Insertion loss at resonance (peak) = -10 log10 T_peak.
* Group delay tau = -d arg(H) / d omega [s] (our e^{+i w t} convention: a pure
  delay tau is H = exp(-i omega tau)), by central differences of the
  unwrapped phase on the angular-frequency axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Resonance:
    """One resonance of a sampled power response."""

    index: int
    position: float
    extremum: float
    edge: float
    fwhm: float
    samples_per_fwhm: float

    @property
    def q_factor(self) -> float:
        return self.position / self.fwhm if self.fwhm > 0 else math.inf

    @property
    def extinction_ratio_db(self) -> float:
        lo, hi = sorted((self.extremum, self.edge))
        return 10 * math.log10(hi / lo) if lo > 0 else math.inf


def _crossing(x: NDArray, y: NDArray, i0: int, step: int, level: float) -> float:
    """Walk from i0 in direction ``step`` to the first crossing of ``level``; interpolate."""
    i = i0
    while 0 <= i + step < y.size:
        j = i + step
        if (y[i] - level) * (y[j] - level) <= 0 and y[j] != y[i]:
            return float(x[i] + (level - y[i]) * (x[j] - x[i]) / (y[j] - y[i]))
        i = j
    return math.nan


def find_resonances(
    x: ArrayLike, transmission: ArrayLike, kind: str = "dip", min_depth: float = 0.01
) -> list[Resonance]:
    """Resonances of a sampled power response ``transmission`` on ascending axis ``x``."""
    xs = np.asarray(x, dtype=float)
    t = np.asarray(transmission, dtype=float)
    if xs.shape != t.shape or xs.ndim != 1 or np.any(np.diff(xs) <= 0):
        raise ValueError("x must be ascending and match the transmission shape")
    if kind not in ("dip", "peak"):
        raise ValueError(f"kind must be 'dip' or 'peak', got {kind!r}")
    sign = -1.0 if kind == "dip" else 1.0
    idx, _ = find_peaks(sign * t, prominence=min_depth)
    out = []
    bounds = np.concatenate([[0], (idx[:-1] + idx[1:]) // 2, [t.size - 1]])
    for k, i in enumerate(idx):
        seg = t[bounds[k] : bounds[k + 1] + 1]
        edge = float(seg.max() if kind == "dip" else seg.min())
        level = 0.5 * (edge + t[i])
        left = _crossing(xs, t, int(i), -1, level)
        right = _crossing(xs, t, int(i), 1, level)
        width = right - left
        dx = float(xs[min(i + 1, xs.size - 1)] - xs[max(i - 1, 0)]) / 2
        out.append(
            Resonance(int(i), float(xs[i]), float(t[i]), edge, width, width / dx if dx else 0.0)
        )
    return out


def free_spectral_range(resonances: list[Resonance]) -> float:
    """Mean spacing of adjacent resonance positions (needs >= 2 resonances)."""
    if len(resonances) < 2:
        raise ValueError("FSR needs at least two resonances in the sweep")
    return float(np.mean(np.diff([r.position for r in resonances])))


def group_delay(frequency: ArrayLike, transfer: ArrayLike) -> NDArray[np.float64]:
    """tau(f) = -d arg H / d omega [s] on an ascending frequency axis [Hz]."""
    f = np.asarray(frequency, dtype=float)
    phase = np.unwrap(np.angle(np.asarray(transfer, dtype=complex)))
    return -np.gradient(phase, 2 * math.pi * f)


__all__ = ["Resonance", "find_resonances", "free_spectral_range", "group_delay"]
