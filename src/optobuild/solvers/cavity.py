"""Round-trip iteration of a laser cavity to a steady (mode-locked) state (ADR-0020).

A cavity is an ordered list of elements, each a callable ``field -> field``
or ``field -> (field, output)`` (an output coupler); one application of the
list is one round trip. Starting from an initial field (e.g. seeded noise),
the round trip is repeated until the pulse reproduces itself:

    delta_n = || |A_n|^2 - |A_{n-1}|^2 || / || |A_n|^2 ||  <  tolerance

for ``patience`` consecutive round trips (intensity, so a constant carrier
phase slip per round trip is allowed), or ``max_round_trips`` is reached
(then ``converged`` is False and the caller reports it). ``recenter`` rolls
the circulating field by whole samples so that its intensity peak sits at the
window centre (compensates a group-delay drift of the pulse relative to the
window; a numerical device that does not change the pulse). The energy
history and the last output are returned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optobuild.numerics.grid import TimeGrid

Element = Callable[[NDArray[np.complex128]], Any]


@dataclass(frozen=True)
class CavityResult:
    field: NDArray[np.complex128]
    """Circulating field at the end of the last round trip."""
    output: NDArray[np.complex128] | None
    """Field extracted by the last output-coupling element in the last round trip."""
    round_trips: int
    converged: bool
    residual: float
    energies: NDArray[np.float64]
    """Circulating energy [J] after each round trip."""


def run_cavity(
    elements: Sequence[Element],
    initial: NDArray[np.complex128],
    grid: TimeGrid,
    *,
    max_round_trips: int = 1000,
    tolerance: float = 1e-8,
    patience: int = 3,
    recenter: bool = False,
    check_cancelled: Callable[[], None] | None = None,
    progress: Callable[[float], None] | None = None,
) -> CavityResult:
    a = np.asarray(initial, dtype=complex).reshape(-1)
    prev = np.abs(a) ** 2
    energies = []
    output = None
    calm = 0
    residual = np.inf
    for n in range(1, max_round_trips + 1):
        for el in elements:
            res = el(a)
            if isinstance(res, tuple):
                a, output = res
            else:
                a = res
        if recenter:
            shift = grid.n_samples // 2 - int(np.argmax(np.abs(a)))
            if shift:
                a = np.roll(a, shift)
                if output is not None:
                    output = np.roll(output, shift)
        inten = np.abs(a) ** 2
        energies.append(float(np.sum(inten) * grid.dt))
        norm = float(np.linalg.norm(inten))
        residual = float(np.linalg.norm(inten - prev) / norm) if norm > 0 else np.inf
        prev = inten
        calm = calm + 1 if residual < tolerance else 0
        if check_cancelled is not None:
            check_cancelled()
        if progress is not None:
            progress(n / max_round_trips)
        if calm >= patience:
            return CavityResult(a, output, n, True, residual, np.asarray(energies))
    return CavityResult(a, output, max_round_trips, False, residual, np.asarray(energies))


__all__ = ["CavityResult", "run_cavity"]
