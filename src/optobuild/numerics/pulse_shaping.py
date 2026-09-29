"""Raised-cosine (RC) and root-raised-cosine (RRC) pulse shaping on a grid.

Normalized raised-cosine spectrum (H(0) = 1), symbol period T, roll-off b:

    H_rc(f) = 1                                              |f| <= (1-b)/(2T)
            = 1/2 [1 + cos(pi T / b (|f| - (1-b)/(2T)))]    (1-b)/(2T) < |f| <= (1+b)/(2T)
            = 0                                              otherwise
    H_rrc(f) = sqrt(H_rc(f))

For b = 0 the value at exactly |f| = 1/(2T) is 1/2 (the b -> 0 limit), which
keeps the folded spectrum flat on a discrete grid.

Transmit shaping (``shape``): an impulse train x[k sps] = s_k filtered by
sps * H(f); receive matched filtering (``matched_filter``) applies H(f) with
unit DC gain. With RRC at both ends the cascade is sps * H_rc, whose impulse
response is 1 at t = 0 and 0 at all other symbol instants (Nyquist): the
samples z[k sps] equal s_k exactly when (1+b)/(2T) <= fs/2. For i.i.d.
symbols the RRC-shaped waveform has mean power E[|s|^2] (Parseval).
All filtering is circular (periodic symbol pattern). Reference: Proakis &
Salehi, Digital Communications, 5th ed., sec. 9.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import InvalidParameterError, SamplingError
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid


def raised_cosine(frequency: ArrayLike, symbol_rate: float, rolloff: float) -> NDArray[np.float64]:
    """H_rc(f) with H_rc(0) = 1."""
    if not 0.0 <= rolloff <= 1.0:
        raise InvalidParameterError(f"roll-off must be in [0, 1], got {rolloff}.")
    f = np.abs(np.asarray(frequency, dtype=float))
    t = 1.0 / symbol_rate
    f1, f2 = (1 - rolloff) / (2 * t), (1 + rolloff) / (2 * t)
    h = np.zeros_like(f)
    h[f < f1] = 1.0
    if rolloff == 0.0:
        h[np.isclose(f, f1, rtol=1e-12, atol=0.0)] = 0.5
    else:
        band = (f >= f1) & (f <= f2)
        h[band] = 0.5 * (1 + np.cos(np.pi * t / rolloff * (f[band] - f1)))
    return h


def root_raised_cosine(
    frequency: ArrayLike, symbol_rate: float, rolloff: float
) -> NDArray[np.float64]:
    """H_rrc(f) = sqrt(H_rc(f))."""
    return np.sqrt(raised_cosine(frequency, symbol_rate, rolloff))


def _response(grid: TimeGrid, symbol_rate: float, rolloff: float, kind: str) -> NDArray[np.float64]:
    if (1 + rolloff) * symbol_rate / 2 > grid.nyquist_frequency * (1 + 1e-12):
        raise SamplingError(
            f"Pulse bandwidth (1+b)R_s/2 = {(1 + rolloff) * symbol_rate / 2:g} Hz exceeds fs/2.",
            hint="Increase samples per symbol.",
        )
    if kind == "rrc":
        return root_raised_cosine(grid.frequency(), symbol_rate, rolloff)
    if kind == "rc":
        return raised_cosine(grid.frequency(), symbol_rate, rolloff)
    raise InvalidParameterError(f"Unknown pulse kind {kind!r}; use 'rrc' or 'rc'.")


def shape(
    symbols: ArrayLike, samples_per_symbol: int, grid: TimeGrid, rolloff: float, kind: str = "rrc"
) -> NDArray[np.complex128]:
    """Pulse-shaped complex baseband waveform (symbol k centred at sample k*sps)."""
    s = np.asarray(symbols, dtype=complex)
    if s.shape[-1] * samples_per_symbol != grid.n_samples:
        raise SamplingError(
            f"{s.shape[-1]} symbols x {samples_per_symbol} samples != {grid.n_samples} samples."
        )
    x = np.zeros(s.shape[:-1] + (grid.n_samples,), dtype=complex)
    x[..., ::samples_per_symbol] = s
    symbol_rate = grid.sample_rate / samples_per_symbol
    h = samples_per_symbol * _response(grid, symbol_rate, rolloff, kind)
    return apply_transfer_function(x, h, grid)


def matched_filter(
    waveform: ArrayLike, samples_per_symbol: int, grid: TimeGrid, rolloff: float, kind: str = "rrc"
) -> NDArray[np.complex128]:
    """Receive filter H(f) (unit DC gain)."""
    symbol_rate = grid.sample_rate / samples_per_symbol
    return apply_transfer_function(waveform, _response(grid, symbol_rate, rolloff, kind), grid)


__all__ = ["matched_filter", "raised_cosine", "root_raised_cosine", "shape"]
