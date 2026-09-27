"""Clock sampling and threshold decision for binary signals.

Bit k of a waveform with ``sps`` samples per bit occupies samples
[k sps, (k+1) sps). The decision samples are ``x[k sps + offset]`` with an
integer ``offset`` in [0, sps). Decided bit = 1 if sample > threshold.

Timing recovery ``max_variance``: choose the offset maximizing the variance of
the sampled values across bits (maximum eye spread for a balanced pattern);
a standard data-independent timing criterion. It does not use the reference
bits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import SamplingError


def bit_matrix(samples: ArrayLike, samples_per_bit: int) -> NDArray[np.float64]:
    """Reshape a waveform into (n_bits, sps); N must be a multiple of sps."""
    x = np.asarray(samples, dtype=float)
    if samples_per_bit < 1 or x.size % samples_per_bit:
        raise SamplingError(
            f"Waveform length {x.size} is not a multiple of samples_per_bit={samples_per_bit}.",
            hint="Generate the waveform with an integer number of samples per bit.",
        )
    return x.reshape(-1, samples_per_bit)


def max_variance_offset(samples: ArrayLike, samples_per_bit: int) -> int:
    """Sampling offset (in samples) with the largest variance across bits."""
    return int(np.argmax(bit_matrix(samples, samples_per_bit).var(axis=0)))


def sample_bits(samples: ArrayLike, samples_per_bit: int, offset: int) -> NDArray[np.float64]:
    """Decision samples x[k sps + offset], one per bit."""
    if not 0 <= offset < samples_per_bit:
        raise SamplingError(f"offset must be in [0, {samples_per_bit}), got {offset}.")
    return bit_matrix(samples, samples_per_bit)[:, offset].copy()


def decide(values: ArrayLike, threshold: float) -> NDArray[np.uint8]:
    """Binary decision: 1 where value > threshold."""
    return (np.asarray(values, dtype=float) > threshold).astype(np.uint8)


__all__ = ["bit_matrix", "decide", "max_variance_offset", "sample_bits"]
