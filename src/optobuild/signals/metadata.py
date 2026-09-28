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
SYMBOL_RATE = "symbol_rate"
"""Symbol rate R_s [Bd] of a sampled waveform carrying symbols."""
SAMPLES_PER_SYMBOL = "samples_per_symbol"
"""Integer samples per symbol of a sampled waveform."""
MODULATION = "modulation"
"""Constellation name, e.g. 'qpsk', '16qam' (analysis.constellations)."""
PULSE_ROLLOFF = "pulse_rolloff"
"""Roll-off factor of the (root-)raised-cosine transmit pulse."""
CARRIER_WAVELENGTH = "carrier_wavelength"
"""Vacuum wavelength [m] of the optical carrier a detected signal came from."""


def require_symbol_timing(metadata: Mapping[str, Any], who: str) -> tuple[float, int]:
    """Return (symbol_rate, samples_per_symbol) from metadata or raise with a hint."""
    try:
        return float(metadata[SYMBOL_RATE]), int(metadata[SAMPLES_PER_SYMBOL])
    except KeyError:
        raise SignalTypeError(
            f"'{who}' needs the signal's symbol timing ('{SYMBOL_RATE}', "
            f"'{SAMPLES_PER_SYMBOL}' metadata), which this signal does not carry.",
            hint="Drive the link from a pulse shaper, which annotates its waveforms.",
        ) from None


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


__all__ = [
    "BIT_RATE",
    "CARRIER_WAVELENGTH",
    "MODULATION",
    "PATTERN",
    "PULSE_ROLLOFF",
    "SAMPLES_PER_BIT",
    "SAMPLES_PER_SYMBOL",
    "SYMBOL_RATE",
    "require_symbol_timing",
    "require_timing",
]
