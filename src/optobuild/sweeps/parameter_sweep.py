"""Parameter sweeps and generic Monte Carlo over projects (ADR-0021).

A sweep runs a project at every point of a Cartesian grid of component
parameters (``Axis``) and for every trial index, and collects scalar results
(``Probe``). The project given by the caller is never modified: each point
works on a copy made through the project's plain-data form.

Reproducibility: stochastic components draw from generators seeded by
(root seed, component name, trial) (ADR-0008/0012), independent of the grid
point and of the execution order. The same trial indices at every point give
common random numbers (differences between points are not blurred by
independent noise), and serial and parallel execution give identical values.

Parallel execution (``workers > 1``) uses separate processes. The project
crosses the process boundary as its JSON text (never pickled) and only the
probed scalars come back; each worker keeps its own result cache.
Serial execution reuses one cache for all points, so components upstream of
the swept parameters run once.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optobuild.components.registry import ComponentRegistry, builtin_registry
from optobuild.engine.cache import ResultCache
from optobuild.engine.context import CancellationToken
from optobuild.engine.executor import FeedForwardExecutor, SimulationResult
from optobuild.persistence.project import (
    Project,
    dumps_project,
    loads_project,
    project_from_dict,
    project_to_dict,
    run_project,
)


@dataclass(frozen=True)
class Axis:
    """Values of one component parameter (SI) to sweep."""

    node: str
    parameter: str
    values: tuple[Any, ...]

    @property
    def label(self) -> str:
        return f"{self.node}.{self.parameter}"


@dataclass(frozen=True)
class Probe:
    """A scalar result recorded by a node."""

    node: str
    key: str

    @property
    def label(self) -> str:
        return f"{self.node}.{self.key}"

    def read(self, result: SimulationResult) -> float:
        value = result.result(self.node, self.key)
        if isinstance(value, bool):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TypeError(f"Result {self.label} is not a scalar number.") from None


@dataclass(frozen=True)
class SweepResult:
    """values[label] has shape (len(axis_1), ..., len(axis_k), n_trials)."""

    axes: tuple[Axis, ...]
    probes: tuple[Probe, ...]
    trials: tuple[int, ...]
    values: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def mean(self, probe: str) -> NDArray[np.float64]:
        """Mean over trials."""
        return np.mean(self.values[probe], axis=-1)

    def standard_error(self, probe: str) -> NDArray[np.float64]:
        """Standard error of the trial mean (NaN for a single trial)."""
        v = self.values[probe]
        n = v.shape[-1]
        if n < 2:
            return np.full(v.shape[:-1], math.nan)
        return np.std(v, axis=-1, ddof=1) / math.sqrt(n)

    def to_csv(self) -> str:
        """One row per grid point and trial: axis values, trial, probe values."""
        head = [a.label for a in self.axes] + ["trial"] + [p.label for p in self.probes]
        lines = [",".join(head)]
        for idx in itertools.product(*(range(len(a.values)) for a in self.axes)):
            for t_i, trial in enumerate(self.trials):
                row = [repr(a.values[i]) for a, i in zip(self.axes, idx, strict=True)]
                row.append(str(trial))
                row += [repr(float(self.values[p.label][(*idx, t_i)])) for p in self.probes]
                lines.append(",".join(row))
        return "\n".join(lines) + "\n"


def clone_project(project: Project, registry: ComponentRegistry | None = None) -> Project:
    """Independent copy of a project (through its validated plain-data form)."""
    return project_from_dict(project_to_dict(project), registry or builtin_registry())


def _points(axes: Sequence[Axis]) -> list[tuple[int, ...]]:
    return list(itertools.product(*(range(len(a.values)) for a in axes)))


def _run_point(
    project: Project,
    axes: Sequence[Axis],
    idx: tuple[int, ...],
    probes: Sequence[Probe],
    trials: Sequence[int],
    seed: int | None,
    executor: FeedForwardExecutor,
    cancel: CancellationToken | None,
) -> list[list[float]]:
    for axis, i in zip(axes, idx, strict=True):
        project.graph.set_parameters(axis.node, **{axis.parameter: axis.values[i]})
    out = []
    for trial in trials:
        res = run_project(project, seed=seed, executor=executor, trial=trial, cancel=cancel)
        out.append([p.read(res) for p in probes])
    return out


def _worker(
    project_json: str,
    axes: tuple[Axis, ...],
    idx: tuple[int, ...],
    probes: tuple[Probe, ...],
    trials: tuple[int, ...],
    seed: int | None,
) -> list[list[float]]:
    """Process-pool entry: rebuild the project from JSON (built-in components only)."""
    project = loads_project(project_json)
    return _run_point(
        project, axes, idx, probes, trials, seed, FeedForwardExecutor(ResultCache()), None
    )


def sweep(
    project: Project,
    axes: Sequence[Axis],
    probes: Sequence[Probe],
    *,
    trials: int | Sequence[int] = 1,
    seed: int | None = None,
    workers: int = 1,
    registry: ComponentRegistry | None = None,
    cancel: CancellationToken | None = None,
    on_point: Callable[[int, int], None] | None = None,
) -> SweepResult:
    """Run the grid of ``axes`` (empty for a plain Monte Carlo) x ``trials``.

    ``trials`` is a count (indices 0..n-1) or explicit trial indices.
    ``on_point(done, total)`` reports progress. ``workers > 1`` needs a project
    made of built-in components (the JSON is re-loaded in the workers).
    """
    trial_ids = tuple(range(trials)) if isinstance(trials, int) else tuple(trials)
    if not trial_ids:
        raise ValueError("need at least one trial")
    axes, probes = tuple(axes), tuple(probes)
    for a in axes:
        if a.node not in project.graph:
            raise KeyError(f"The project has no node '{a.node}' (axis {a.label}).")
        if not a.values:
            raise ValueError(f"Axis {a.label} has no values.")
    for p in probes:
        if p.node not in project.graph:
            raise KeyError(f"The project has no node '{p.node}' (probe {p.label}).")
    points = _points(axes)
    shape = tuple(len(a.values) for a in axes) + (len(trial_ids),)
    values = {p.label: np.empty(shape) for p in probes}

    def store(idx: tuple[int, ...], rows: list[list[float]]) -> None:
        for t_i, row in enumerate(rows):
            for p, v in zip(probes, row, strict=True):
                values[p.label][(*idx, t_i)] = v

    if workers > 1 and len(points) * len(trial_ids) > 1:
        text = dumps_project(project)
        # "spawn": no fork of a possibly multi-threaded parent (GUI), same on every OS
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(_worker, text, axes, idx, probes, trial_ids, seed) for idx in points
            ]
            for done, (idx, fut) in enumerate(zip(points, futures, strict=True), start=1):
                store(idx, fut.result())
                if on_point is not None:
                    on_point(done, len(points))
    else:
        work = clone_project(project, registry)
        executor = FeedForwardExecutor(ResultCache())
        for done, idx in enumerate(points, start=1):
            store(idx, _run_point(work, axes, idx, probes, trial_ids, seed, executor, cancel))
            if on_point is not None:
                on_point(done, len(points))
    return SweepResult(axes, probes, trial_ids, values)


@dataclass(frozen=True)
class MonteCarloEstimate:
    """Sample statistics of one probe over independent trials."""

    probe: str
    n_trials: int
    mean: float
    std: float
    standard_error: float

    def interval(self, z: float = 1.959964) -> tuple[float, float]:
        """Normal-approximation confidence interval mean +- z SE (95 % by default)."""
        return self.mean - z * self.standard_error, self.mean + z * self.standard_error


def monte_carlo(
    project: Project,
    probes: Sequence[Probe],
    n_trials: int,
    *,
    first_trial: int = 0,
    seed: int | None = None,
    workers: int = 1,
    registry: ComponentRegistry | None = None,
) -> dict[str, MonteCarloEstimate]:
    """Mean, standard deviation and standard error of scalar results over trials."""
    if n_trials < 2:
        raise ValueError("Monte Carlo statistics need at least 2 trials")
    res = sweep(
        project,
        (),
        probes,
        trials=range(first_trial, first_trial + n_trials),
        seed=seed,
        workers=workers,
        registry=registry,
    )
    out = {}
    for p in probes:
        v = res.values[p.label]
        std = float(np.std(v, ddof=1))
        out[p.label] = MonteCarloEstimate(
            p.label, n_trials, float(np.mean(v)), std, std / math.sqrt(n_trials)
        )
    return out


__all__ = [
    "Axis",
    "MonteCarloEstimate",
    "Probe",
    "SweepResult",
    "clone_project",
    "monte_carlo",
    "sweep",
]
