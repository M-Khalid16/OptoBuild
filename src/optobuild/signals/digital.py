"""Unsampled bit sequences (docs/signal_model.md §6)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from optobuild.core.errors import SamplingError
from optobuild.signals.base import Signal, frozen_metadata
from optobuild.signals.kinds import SignalKind


@dataclass(frozen=True, eq=False)
class DigitalSequence(Signal):
    """A sequence of bits (0/1) with a bit rate.

    Attributes
    ----------
    bits:
        uint8 array of shape (n_bits,), values in {0, 1}. Read-only.
    bit_rate:
        Bit rate R_b [bit/s] (> 0).
    metadata:
        Read-only annotations (e.g. generator type, PRBS order).
    """

    kind: ClassVar[SignalKind] = SignalKind.DIGITAL

    bits: NDArray[np.uint8]
    bit_rate: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = np.asarray(self.bits)
        if raw.ndim != 1 or raw.size == 0:
            raise SamplingError(f"bits must be a non-empty 1-D array, got shape {raw.shape}.")
        if raw.dtype.kind not in "biu" or np.any((raw != 0) & (raw != 1)):
            raise SamplingError(
                "bits must contain only the integers 0 and 1.",
                hint="Map logical values to 0/1 before creating a DigitalSequence.",
            )
        arr = raw.astype(np.uint8, copy=True)
        arr.setflags(write=False)
        object.__setattr__(self, "bits", arr)
        r = float(self.bit_rate)
        if not (math.isfinite(r) and r > 0):
            raise SamplingError(f"bit_rate must be finite and > 0 bit/s, got {self.bit_rate!r}.")
        object.__setattr__(self, "bit_rate", r)
        object.__setattr__(self, "metadata", frozen_metadata(self.metadata))

    @property
    def n_bits(self) -> int:
        """Number of bits."""
        return int(self.bits.size)

    @property
    def bit_period(self) -> float:
        """Bit period T_b = 1 / R_b [s]."""
        return 1.0 / self.bit_rate

    @property
    def duration(self) -> float:
        """Sequence duration n_bits * T_b [s]."""
        return self.n_bits / self.bit_rate

    def __repr__(self) -> str:
        return f"DigitalSequence(n_bits={self.n_bits}, bit_rate={self.bit_rate:g} bit/s)"


__all__ = ["DigitalSequence"]
