"""Optimization of component parameters against a simulated figure of merit (ADR-0021).

Variables are component parameters (SI) with bounds; the objective is a scalar
result recorded by a node, maximized or minimized, averaged over a fixed set
of trial indices (common random numbers: every evaluation sees the same noise
realizations, so the objective is a deterministic function of the variables
and derivative-free optimizers behave). Methods (scipy.optimize):

* one variable: bounded Brent search (``minimize_scalar(method="bounded")``),
  converging to ``xatol`` in the variable;
* several variables: Nelder-Mead with bounds (scipy >= 1.7), or Powell.

The project is copied; the returned ``OptimizationResult.project`` carries the
optimum. Every evaluation is recorded (history) for reports. The engine cache
is shared across evaluations, so components upstream of the variables run once.
Limitations: local optimizers (the result is a local optimum from the initial
point); noisy objectives need enough trials (the averaging is exact only for
the chosen trials, i.e. the optimum is that of the sample average).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from optobuild.components.registry import ComponentRegistry
from optobuild.engine.cache import ResultCache
from optobuild.engine.context import CancellationToken
from optobuild.engine.executor import FeedForwardExecutor
from optobuild.persistence.project import Project, run_project
from optobuild.sweeps.parameter_sweep import clone_project


@dataclass(frozen=True)
class Variable:
    node: str
    parameter: str
    lower: float
    upper: float
    initial: float | None = None

    @property
    def label(self) -> str:
        return f"{self.node}.{self.parameter}"

    def start(self) -> float:
        return 0.5 * (self.lower + self.upper) if self.initial is None else self.initial


@dataclass(frozen=True)
class Objective:
    node: str
    key: str
    sense: str = "max"
    trials: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.sense not in ("max", "min"):
            raise ValueError(f"sense must be 'max' or 'min', got {self.sense!r}")
        if not self.trials:
            raise ValueError("the objective needs at least one trial index")

    @property
    def label(self) -> str:
        return f"{self.node}.{self.key}"


@dataclass(frozen=True)
class OptimizationResult:
    variables: tuple[Variable, ...]
    objective: Objective
    x: dict[str, float]
    value: float
    n_evaluations: int
    success: bool
    message: str
    project: Project
    history: tuple[tuple[tuple[float, ...], float], ...] = field(default=())


def optimize(
    project: Project,
    variables: Sequence[Variable],
    objective: Objective,
    *,
    method: str = "auto",
    xatol: float | None = None,
    max_evaluations: int = 200,
    seed: int | None = None,
    registry: ComponentRegistry | None = None,
    cancel: CancellationToken | None = None,
    on_evaluation: Callable[[int, tuple[float, ...], float], None] | None = None,
) -> OptimizationResult:
    """Maximize or minimize ``objective`` over ``variables`` within their bounds."""
    variables = tuple(variables)
    if not variables:
        raise ValueError("need at least one variable")
    for v in variables:
        if v.node not in project.graph:
            raise KeyError(f"The project has no node '{v.node}' (variable {v.label}).")
        if not v.lower < v.upper:
            raise ValueError(f"{v.label}: lower bound must be below the upper bound.")
        if not v.lower <= v.start() <= v.upper:
            raise ValueError(f"{v.label}: the initial value lies outside the bounds.")
    if objective.node not in project.graph:
        raise KeyError(f"The project has no node '{objective.node}'.")
    work = clone_project(project, registry)
    executor = FeedForwardExecutor(ResultCache())
    sign = -1.0 if objective.sense == "max" else 1.0
    history: list[tuple[tuple[float, ...], float]] = []

    def evaluate(x: Sequence[float]) -> float:
        xs = tuple(float(np.clip(xi, v.lower, v.upper)) for xi, v in zip(x, variables,
                                                                          strict=True))  # fmt: skip
        for xi, v in zip(xs, variables, strict=True):
            work.graph.set_parameters(v.node, **{v.parameter: xi})
        total = 0.0
        for trial in objective.trials:
            res = run_project(work, seed=seed, executor=executor, trial=trial, cancel=cancel)
            total += float(res.result(objective.node, objective.key))
        value = total / len(objective.trials)
        if not math.isfinite(value):
            value = math.inf * -sign  # worst possible
        history.append((xs, value))
        if on_evaluation is not None:
            on_evaluation(len(history), xs, value)
        return sign * value

    chosen = method
    if chosen == "auto":
        chosen = "bounded" if len(variables) == 1 else "nelder-mead"
    if chosen == "bounded":
        if len(variables) != 1:
            raise ValueError("method 'bounded' optimizes a single variable")
        v = variables[0]
        tol = xatol if xatol is not None else 1e-6 * (v.upper - v.lower)
        res: Any = minimize_scalar(
            lambda z: evaluate((z,)),
            bounds=(v.lower, v.upper),
            method="bounded",
            options={"xatol": tol, "maxiter": max_evaluations},
        )
        x_best, success, message = (float(res.x),), bool(res.success), str(res.message)
    elif chosen in ("nelder-mead", "powell"):
        spans = [v.upper - v.lower for v in variables]
        opts: dict[str, Any] = {"maxfev": max_evaluations}
        if chosen == "nelder-mead":
            opts["xatol"] = xatol if xatol is not None else 1e-6 * min(spans)
            opts["fatol"] = 0.0
        else:
            opts["xtol"] = xatol if xatol is not None else 1e-6
        res = minimize(
            evaluate,
            np.array([v.start() for v in variables]),
            method="Nelder-Mead" if chosen == "nelder-mead" else "Powell",
            bounds=[(v.lower, v.upper) for v in variables],
            options=opts,
        )
        x_best = tuple(
            float(np.clip(xi, v.lower, v.upper)) for xi, v in zip(res.x, variables, strict=True)
        )
        success, message = bool(res.success), str(res.message)
    else:
        raise ValueError(f"unknown method {method!r}; use auto, bounded, nelder-mead or powell")
    # re-evaluate at the reported optimum so that ``project`` and ``value`` agree
    best_value = sign * evaluate(x_best)
    return OptimizationResult(
        variables,
        objective,
        {v.label: xi for v, xi in zip(variables, x_best, strict=True)},
        best_value,
        len(history),
        success,
        message,
        work,
        tuple(history),
    )


__all__ = ["Objective", "OptimizationResult", "Variable", "optimize"]
