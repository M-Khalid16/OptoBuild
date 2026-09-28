"""Unsampled complex symbol sequences (docs/signal_model.md sec. 6)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from optobuild.core.errors import SamplingError
from optobuild.signals.base import Signal, frozen_array, frozen_metadata
from optobuild.signals.kinds import SignalKind


@dataclass(frozen=True, eq=False)
class SymbolSequence(Signal):
    """Complex symbols with a symbol rate.

    Attributes
    ----------
    symbols:
        complex128 array of shape (n_symbols,) (single polarization) or
        (n_pol, n_symbols). Read-only.
    symbol_rate:
        Symbol rate R_s [Bd] (> 0).
    metadata:
        Read-only annotations; ``modulation`` names the constellation
        (see ``optobuild.analysis.constellations``).
    """

    kind: ClassVar[SignalKind] = SignalKind.SYMBOLS

    symbols: NDArray[np.complex128]
    symbol_rate: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = frozen_array(self.symbols, np.complex128, what="Symbols")
        if (
            arr.ndim not in (1, 2)
            or arr.shape[-1] == 0
            or (arr.ndim == 2 and arr.shape[0] not in (1, 2))
        ):
            raise SamplingError(
                f"symbols must have shape (n,) or (n_pol, n) with n_pol in {{1, 2}}, "
                f"got {arr.shape}."
            )
        object.__setattr__(self, "symbols", arr)
        r = float(self.symbol_rate)
        if not (math.isfinite(r) and r > 0):
            raise SamplingError(f"symbol_rate must be finite and > 0 Bd, got {self.symbol_rate!r}.")
        object.__setattr__(self, "symbol_rate", r)
        object.__setattr__(self, "metadata", frozen_metadata(self.metadata))

    @property
    def n_symbols(self) -> int:
        """Number of symbols (per polarization)."""
        return int(self.symbols.shape[-1])

    @property
    def symbol_period(self) -> float:
        """T_s = 1 / R_s [s]."""
        return 1.0 / self.symbol_rate

    def average_energy(self) -> float:
        """mean |s|^2 (per polarization, averaged)."""
        return float(np.mean(np.abs(self.symbols) ** 2))

    def __repr__(self) -> str:
        return (
            f"SymbolSequence(n_symbols={self.n_symbols}, symbol_rate={self.symbol_rate:g} Bd, "
            f"modulation={self.metadata.get('modulation', '?')})"
        )


__all__ = ["SymbolSequence"]
