"""Engine-side implementation of the component ``RunContext`` protocol."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any

import numpy as np

from optobuild.core.diagnostics import Diagnostic
from optobuild.core.errors import SimulationCancelledError
from optobuild.core.log import component_logger
from optobuild.core.rng import component_generator
from optobuild.numerics.layout import SimulationLayout


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation; running components stop at their next check."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`SimulationCancelledError` if cancellation was requested."""
        if self._event.is_set():
            raise SimulationCancelledError(
                "The simulation was cancelled.", hint="Run it again to obtain results."
            )


def _freeze_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.array(value, copy=True)
        arr.setflags(write=False)
        return arr
    if isinstance(value, np.generic):
        return value.item()  # canonical Python scalar
    if isinstance(value, (int, float, complex, str, bool, type(None))):
        return value
    if isinstance(value, tuple) or is_dataclass(value):
        return value
    raise TypeError(
        f"Recorded results must be numbers, strings, arrays, tuples or dataclasses; "
        f"got {type(value).__name__}."
    )


class ExecutionContext:
    """Per-component context: seeded RNG, logger, cancellation, progress, results."""

    def __init__(
        self,
        component_name: str,
        root_seed: int,
        *,
        stochastic: bool = True,
        cancel: CancellationToken | None = None,
        progress: Callable[[float, str], None] | None = None,
        layout: SimulationLayout | None = None,
        trial: int | None = None,
    ) -> None:
        self._name = component_name
        self._seed = root_seed
        self._layout = layout
        self._trial = trial
        self._stochastic = stochastic
        self._rng: np.random.Generator | None = None
        self._cancel = cancel
        self._progress = progress
        self.results: dict[str, Any] = {}
        self.diagnostics: list[Diagnostic] = []

    @property
    def rng(self) -> np.random.Generator:
        """Generator from (root seed, component name[, trial]); created on first use."""
        if self._rng is None:
            if not self._stochastic:
                raise RuntimeError(
                    f"Component '{self._name}' uses context.rng but does not declare "
                    "stochastic = True; its cached results would ignore the seed (ADR-0012)."
                )
            extra = () if self._trial is None else (self._trial,)
            self._rng = component_generator(self._seed, self._name, *extra)
        return self._rng

    @property
    def layout(self) -> SimulationLayout | None:
        """Global simulation layout of this run, if any."""
        return self._layout

    @property
    def logger(self) -> logging.Logger:
        """Logger ``optobuild.run.<component name>``."""
        return component_logger(self._name)

    def check_cancelled(self) -> None:
        """Raise ``SimulationCancelledError`` if cancelled."""
        if self._cancel is not None:
            self._cancel.raise_if_cancelled()

    def report_progress(self, fraction: float, message: str = "") -> None:
        """Forward intra-component progress in [0, 1]."""
        if self._progress is not None:
            self._progress(min(max(float(fraction), 0.0), 1.0), message)

    def record(self, key: str, value: Any) -> None:
        """Store a result; arrays are copied read-only, NumPy scalars become Python scalars."""
        if key in self.results:
            raise KeyError(f"Result '{key}' was already recorded by '{self._name}'.")
        self.results[key] = _freeze_result(value)

    def warn(self, diagnostic: Diagnostic) -> None:
        """Attach a diagnostic produced during the run."""
        self.diagnostics.append(diagnostic)


__all__ = ["CancellationToken", "ExecutionContext"]
