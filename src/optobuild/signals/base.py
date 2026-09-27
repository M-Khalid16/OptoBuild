"""Common machinery for immutable signals (ADR-0001)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid
from optobuild.signals.kinds import SignalKind


def frozen_array(values: ArrayLike, dtype: DTypeLike, *, what: str) -> np.ndarray:
    """Copy ``values`` to a read-only array of ``dtype``; reject NaN/Inf."""
    arr = np.array(values, dtype=dtype, copy=True)
    if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
        raise SamplingError(
            f"{what} contains NaN or Inf values.",
            hint="Check the upstream component parameters for overflow or invalid inputs.",
        )
    arr.setflags(write=False)
    return arr


def frozen_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Read-only copy of a metadata mapping."""
    return MappingProxyType(dict(metadata or {}))


class Signal:
    """Marker base class for all signals. Subclasses are frozen dataclasses."""

    kind: ClassVar[SignalKind]
    metadata: Mapping[str, Any]


class SampledSignal(Signal):
    """A signal sampled on a :class:`TimeGrid` (time is the last array axis)."""

    grid: TimeGrid

    def _check_length(self, arr: np.ndarray, what: str) -> None:
        if arr.shape[-1] != self.grid.n_samples:
            raise SamplingError(
                f"{what} has {arr.shape[-1]} samples but the grid has {self.grid.n_samples}.",
                hint="Create the TimeGrid with the same number of samples as the data.",
            )


def require_same_grid(a: SampledSignal, b: SampledSignal, *, what: str = "signals") -> None:
    """Raise :class:`SamplingError` unless both signals have identical N and dt."""
    if not a.grid.is_compatible(b.grid):
        raise SamplingError(
            f"Incompatible sampling of {what}: N={a.grid.n_samples}, dt={a.grid.dt:g} s vs "
            f"N={b.grid.n_samples}, dt={b.grid.dt:g} s.",
            hint="Use the same sample rate and number of samples for both signal paths.",
        )


__all__ = ["SampledSignal", "Signal", "frozen_array", "frozen_metadata", "require_same_grid"]
