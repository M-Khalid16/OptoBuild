"""Real-valued electrical signals (docs/signal_model.md §4)."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid
from optobuild.signals.base import SampledSignal, frozen_array, frozen_metadata
from optobuild.signals.kinds import SignalKind


class ElectricalQuantity(enum.Enum):
    """Physical quantity carried by an electrical signal."""

    VOLTAGE = "V"
    CURRENT = "A"


@dataclass(frozen=True, eq=False)
class ElectricalSignal(SampledSignal):
    """Sampled real waveform x(t_n) in volts or amperes.

    Attributes
    ----------
    grid:
        Sampling grid.
    samples:
        Real float64 array of shape (N,). Read-only.
    quantity:
        VOLTAGE (V) or CURRENT (A).
    metadata:
        Read-only annotation mapping.
    """

    kind: ClassVar[SignalKind] = SignalKind.ELECTRICAL

    grid: TimeGrid
    samples: NDArray[np.float64]
    quantity: ElectricalQuantity = ElectricalQuantity.VOLTAGE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = np.asarray(self.samples)
        if np.iscomplexobj(raw):
            raise SamplingError(
                "Electrical samples must be real-valued.",
                hint="Take the real part explicitly, or use an optical/complex signal.",
            )
        arr = frozen_array(raw, np.float64, what="Electrical samples")
        if arr.ndim != 1:
            raise SamplingError(f"Electrical samples must be 1-D, got shape {arr.shape}.")
        object.__setattr__(self, "samples", arr)
        self._check_length(arr, "Electrical samples")
        if not isinstance(self.quantity, ElectricalQuantity):
            raise SamplingError(f"quantity must be an ElectricalQuantity, got {self.quantity!r}.")
        object.__setattr__(self, "metadata", frozen_metadata(self.metadata))

    @property
    def unit(self) -> str:
        """SI unit symbol of the samples ("V" or "A")."""
        return self.quantity.value

    def mean(self) -> float:
        """Window mean of the samples [V or A]."""
        return float(np.mean(self.samples))

    def replace(
        self,
        *,
        samples: ArrayLike | None = None,
        grid: TimeGrid | None = None,
        quantity: ElectricalQuantity | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ElectricalSignal:
        """New signal with some attributes replaced; metadata is *merged*."""
        merged = dict(self.metadata)
        merged.update(metadata or {})
        return ElectricalSignal(
            grid=self.grid if grid is None else grid,
            samples=self.samples if samples is None else samples,
            quantity=self.quantity if quantity is None else quantity,
            metadata=merged,
        )

    def __repr__(self) -> str:
        return (
            f"ElectricalSignal(N={self.grid.n_samples}, dt={self.grid.dt:g} s, "
            f"quantity={self.quantity.name}, mean={self.mean():g} {self.unit})"
        )


__all__ = ["ElectricalQuantity", "ElectricalSignal"]
