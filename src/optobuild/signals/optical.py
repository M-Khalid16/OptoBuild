"""Optical signals as sampled complex envelopes (docs/signal_model.md §3)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid
from optobuild.signals.base import SampledSignal, frozen_array, frozen_metadata
from optobuild.signals.kinds import SignalKind


@dataclass(frozen=True, eq=False)
class OpticalSignal(SampledSignal):
    """Complex envelope ``A_p(t)`` [sqrt(W)] of an optical field.

    ``E_p(t) = Re{A_p(t) exp(+i 2 pi f_ref t)}`` and ``P(t) = sum_p |A_p(t)|^2`` [W].

    Attributes
    ----------
    grid:
        Sampling grid.
    field:
        Complex array of shape ``(n_pol, N)`` with ``n_pol`` in {1, 2}. A 1-D
        input of length N is stored as ``(1, N)``. Read-only.
    center_frequency:
        Reference (carrier) frequency ``f_ref`` [Hz] (> 0).
    metadata:
        Read-only provenance/annotation mapping.
    """

    kind: ClassVar[SignalKind] = SignalKind.OPTICAL

    grid: TimeGrid
    field: NDArray[np.complex128]
    center_frequency: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = frozen_array(self.field, np.complex128, what="Optical field")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] not in (1, 2):
            raise SamplingError(
                f"Optical field must have shape (N,), (1, N) or (2, N); got {arr.shape}.",
                hint="Use axis 0 for polarization (1 or 2) and axis 1 for time.",
            )
        object.__setattr__(self, "field", arr)
        self._check_length(arr, "Optical field")
        f = float(self.center_frequency)
        if not (math.isfinite(f) and f > 0):
            raise SamplingError(
                f"center_frequency must be finite and > 0 Hz, got {self.center_frequency!r}.",
                hint="Specify the carrier as a positive optical frequency, e.g. 193.4e12 Hz.",
            )
        object.__setattr__(self, "center_frequency", f)
        object.__setattr__(self, "metadata", frozen_metadata(self.metadata))

    # --- derived quantities --------------------------------------------------------------
    @property
    def n_pol(self) -> int:
        """Number of polarization components (1 or 2)."""
        return int(self.field.shape[0])

    @property
    def wavelength(self) -> float:
        """Vacuum wavelength of the reference carrier, c / f_ref [m]."""
        return SPEED_OF_LIGHT / self.center_frequency

    def power(self) -> NDArray[np.float64]:
        """Instantaneous power P(t_n) = sum_p |A_p|^2 [W], shape (N,)."""
        return np.sum(np.abs(self.field) ** 2, axis=0)

    def average_power(self) -> float:
        """Window-averaged power [W]."""
        return float(np.mean(self.power()))

    def phase(self) -> NDArray[np.float64]:
        """Phase angle(A_p) [rad] relative to the reference carrier, shape (n_pol, N)."""
        return np.angle(self.field)

    def absolute_frequency(self) -> NDArray[np.float64]:
        """Absolute optical frequency f_ref + f_k [Hz] of each FFT bin (FFT order)."""
        return self.center_frequency + self.grid.frequency()

    # --- derivation ----------------------------------------------------------------------
    def replace(
        self,
        *,
        field: ArrayLike | None = None,
        grid: TimeGrid | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OpticalSignal:
        """New signal with some attributes replaced; metadata is *merged*."""
        merged = dict(self.metadata)
        merged.update(metadata or {})
        return OpticalSignal(
            grid=self.grid if grid is None else grid,
            field=self.field if field is None else field,
            center_frequency=self.center_frequency,
            metadata=merged,
        )

    def __repr__(self) -> str:
        return (
            f"OpticalSignal(N={self.grid.n_samples}, dt={self.grid.dt:g} s, n_pol={self.n_pol}, "
            f"f_ref={self.center_frequency:g} Hz, P_avg={self.average_power():g} W)"
        )


__all__ = ["OpticalSignal"]
