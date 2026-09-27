"""Signal kinds used to type component ports (ADR-0001, ADR-0004)."""

from __future__ import annotations

import enum


class SignalKind(enum.Enum):
    """The physical/logical domain a port carries.

    Two ports may be connected only if their kinds are equal (ADR-0004).
    Conversion between kinds always requires an explicit component
    (e.g. a photodiode converts OPTICAL to ELECTRICAL).
    """

    OPTICAL = "optical"
    """Sampled complex envelope of an optical field, sqrt(W), per polarization."""

    ELECTRICAL = "electrical"
    """Sampled real-valued voltage [V] or current [A] waveform."""

    DIGITAL = "digital"
    """Unsampled sequence of bits with a bit rate."""

    SYMBOLS = "symbols"
    """Unsampled sequence of complex/integer symbols with a symbol rate."""


__all__ = ["SignalKind"]
