"""Eye-diagram data generation.

The waveform is cut into overlapping traces of ``symbols_per_trace`` bit
periods, each starting at a bit boundary shifted by ``offset`` samples. The
window is treated as periodic (circular indexing), consistent with the
simulation's periodic signals. No interpolation is performed: trace samples are
the simulated samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.analysis.decision import bit_matrix


@dataclass(frozen=True)
class EyeDiagram:
    """Eye traces and their time axis."""

    traces: NDArray[np.float64]
    """Array (n_traces, symbols_per_trace * sps + 1) of waveform values."""
    time_s: NDArray[np.float64]
    """Time axis of one trace [s], starting at 0."""
    samples_per_symbol: int


def eye_diagram(
    samples: ArrayLike,
    samples_per_symbol: int,
    symbol_period: float,
    symbols_per_trace: int = 2,
    offset: int = 0,
) -> EyeDiagram:
    """Build eye traces from a waveform with ``samples_per_symbol`` samples per symbol."""
    x = np.asarray(samples, dtype=float)
    n_sym = bit_matrix(x, samples_per_symbol).shape[0]
    length = symbols_per_trace * samples_per_symbol + 1
    starts = (np.arange(n_sym) * samples_per_symbol + offset)[:, None]
    idx = (starts + np.arange(length)[None, :]) % x.size
    dt = symbol_period / samples_per_symbol
    return EyeDiagram(x[idx], dt * np.arange(length), samples_per_symbol)


__all__ = ["EyeDiagram", "eye_diagram"]
