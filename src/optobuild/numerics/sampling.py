"""Sampling diagnostics (numerical_conventions.md sec. 6).

These functions never modify data; they return ``Diagnostic`` records that
components attach to their results via ``context.warn``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.numerics.fft import power_spectral_density
from optobuild.numerics.grid import TimeGrid

ALIASING_BAND_FRACTION = 0.1
"""Outer fraction of the band (on each side) inspected for aliasing risk."""
ALIASING_THRESHOLD = 1e-4
"""Energy fraction in the outer band above which a warning is issued."""


def band_edge_energy_fraction(
    x: ArrayLike, grid: TimeGrid, band_fraction: float = ALIASING_BAND_FRACTION
) -> float:
    """Fraction of spectral energy with |f| > (1 - band_fraction) fs/2."""
    psd = power_spectral_density(x, grid)
    if psd.ndim > 1:
        psd = psd.sum(axis=tuple(range(psd.ndim - 1)))
    total = float(psd.sum())
    if total == 0.0:
        return 0.0
    edge = np.abs(grid.frequency()) > (1.0 - band_fraction) * grid.nyquist_frequency
    return float(psd[edge].sum() / total)


def aliasing_diagnostic(
    x: ArrayLike, grid: TimeGrid, source: str = "", threshold: float = ALIASING_THRESHOLD
) -> Diagnostic | None:
    """Warn if significant energy lies near +-fs/2, where aliasing corrupts the result."""
    frac = band_edge_energy_fraction(x, grid)
    if frac <= threshold:
        return None
    return Diagnostic(
        Severity.WARNING,
        "sampling.aliasing_risk",
        f"{frac:.2e} of the signal energy lies in the outer {ALIASING_BAND_FRACTION:.0%} of the "
        f"simulated band (|f| > {(1 - ALIASING_BAND_FRACTION) * grid.nyquist_frequency:.4g} Hz); "
        "content beyond fs/2 would have folded back (aliasing).",
        hint="Increase samples per bit, or band-limit the signal (e.g. finite rise time).",
        source=source,
    )


def occupied_bandwidth(x: ArrayLike, grid: TimeGrid, fraction: float = 0.99) -> float:
    """Smallest symmetric full width [Hz] about f = 0 containing ``fraction`` of the power."""
    psd = power_spectral_density(x, grid)
    if psd.ndim > 1:
        psd = psd.sum(axis=tuple(range(psd.ndim - 1)))
    total = psd.sum()
    if total == 0:
        return 0.0
    f = np.abs(grid.frequency())
    order = np.argsort(f, kind="stable")
    cum = np.cumsum(psd[order]) / total
    idx = int(np.searchsorted(cum, fraction))
    return 2.0 * float(f[order][min(idx, len(f) - 1)])


def time_edge_energy_fraction(x: ArrayLike, edge_fraction: float = 0.05) -> float:
    """Fraction of energy in the first and last ``edge_fraction`` of the window."""
    arr = np.asarray(x)
    p = (
        np.sum(np.abs(arr) ** 2, axis=tuple(range(arr.ndim - 1)))
        if arr.ndim > 1
        else np.abs(arr) ** 2
    )
    total = float(p.sum())
    if total == 0.0:
        return 0.0
    k = max(1, int(edge_fraction * p.size))
    return float((p[:k].sum() + p[-k:].sum()) / total)


def samples_per_symbol_diagnostic(samples_per_symbol: int, source: str = "") -> Diagnostic | None:
    """Warn for fewer than 4 samples per symbol (filters and eye diagrams become coarse)."""
    if samples_per_symbol >= 4:
        return None
    return Diagnostic(
        Severity.WARNING,
        "sampling.samples_per_symbol",
        f"Only {samples_per_symbol} samples per symbol.",
        hint="Use at least 8 samples per symbol for waveform-level accuracy.",
        source=source,
    )


__all__ = [
    "ALIASING_BAND_FRACTION",
    "ALIASING_THRESHOLD",
    "aliasing_diagnostic",
    "band_edge_energy_fraction",
    "occupied_bandwidth",
    "time_edge_energy_fraction",
    "samples_per_symbol_diagnostic",
]
