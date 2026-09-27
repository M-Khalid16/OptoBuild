"""Documented metadata keys shared between components.

Components may rely only on the keys listed here (docs/signal_model.md sec. 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optobuild.core.errors import SignalTypeError

BIT_RATE = "bit_rate"
"""Bit rate R_b [bit/s] of the data carried by a sampled waveform."""
SAMPLES_PER_BIT = "samples_per_bit"
"""Integer samples per bit of a sampled waveform."""
PATTERN = "pattern"
"""Human-readable description of the bit pattern (e.g. 'PRBS7')."""


def require_timing(metadata: Mapping[str, Any], who: str) -> tuple[float, int]:
    """Return (bit_rate, samples_per_bit) from metadata or raise with a hint."""
    try:
        return float(metadata[BIT_RATE]), int(metadata[SAMPLES_PER_BIT])
    except KeyError:
        raise SignalTypeError(
            f"'{who}' needs the signal's bit timing ('{BIT_RATE}', '{SAMPLES_PER_BIT}' metadata), "
            "which this signal does not carry.",
            hint="Drive the link from an NRZ generator, which annotates its waveform.",
        ) from None


__all__ = ["BIT_RATE", "PATTERN", "SAMPLES_PER_BIT", "require_timing"]
