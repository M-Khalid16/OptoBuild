"""Feed-forward (DAG) executor (ADR-0005)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import (
    ComponentExecutionError,
    InvalidGraphError,
    OptoBuildError,
    SignalTypeError,
    SimulationCancelledError,
)
from optobuild.core.log import get_logger
from optobuild.core.rng import validate_seed
from optobuild.engine.cache import ResultCache, node_key
from optobuild.engine.context import CancellationToken, ExecutionContext
from optobuild.graph.model import SimulationGraph
from optobuild.numerics.layout import SimulationLayout

_LOG = get_logger("engine")


@dataclass(frozen=True)
class ProgressEvent:
    """Progress notification: overall ``fraction`` in [0, 1] while running ``node``."""

    node: str
    index: int
    total: int
    fraction: float
    message: str = ""


@dataclass(frozen=True)
class NodeResult:
    """Everything one node produced."""

    outputs: Mapping[str, Any]
    results: Mapping[str, Any]
    diagnostics: tuple[Diagnostic, ...]
    key: str
    cache_hit: bool = False
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    """Outputs of a feed-forward run, addressable by node and port/result key."""

    order: tuple[str, ...]
    nodes: Mapping[str, NodeResult]
    seed: int
    diagnostics: tuple[Diagnostic, ...] = field(default=())
    layout: SimulationLayout | None = None
    trial: int | None = None

    def signal(self, node: str, port: str) -> Any:
        """Signal produced on output ``port`` of ``node`` (intermediate inspection)."""
        try:
            return self.nodes[node].outputs[port]
        except KeyError:
            raise KeyError(f"No output '{port}' recorded for node '{node}'.") from None

    def result(self, node: str, key: str) -> Any:
        """Analysis result ``key`` recorded by ``node``."""
        try:
            return self.nodes[node].results[key]
        except KeyError:
            raise KeyError(f"No result '{key}' recorded by node '{node}'.") from None

    def all_diagnostics(self) -> list[Diagnostic]:
        """Graph-level plus per-node diagnostics."""
        out = list(self.diagnostics)
        for n in self.order:
            out.extend(self.nodes[n].diagnostics)
        return out


ProgressCallback = Callable[[ProgressEvent], None]


class FeedForwardExecutor:
    """Runs an acyclic :class:`SimulationGraph` once, in deterministic topological order."""

    def __init__(self, cache: ResultCache | None = None) -> None:
        self.cache = cache

    def run(
        self,
        graph: SimulationGraph,
        *,
        seed: int = 0,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
        layout: SimulationLayout | None = None,
        trial: int | None = None,
    ) -> SimulationResult:
        """Validate and execute ``graph``.

        ``layout`` is the global simulation layout passed to every component
        (ADR-0011). ``trial`` selects an independent Monte Carlo realization:
        every component's generator becomes (seed, name, trial) (ADR-0008);
        ``None`` is the default realization.

        Raises
        ------
        InvalidGraphError
            If validation finds errors (all errors are listed in the message).
        SimulationCycleError
            If the graph is cyclic.
        ComponentExecutionError / SignalTypeError
            If a component fails or returns outputs that do not match its ports.
        SimulationCancelledError
            If ``cancel`` is triggered.
        """
        root_seed = validate_seed(seed)
        if trial is not None and (
            isinstance(trial, bool) or not isinstance(trial, int) or trial < 0
        ):
            raise ValueError(f"trial must be a non-negative integer or None, got {trial!r}.")
        diagnostics = graph.validate()
        errors = [d for d in diagnostics if d.severity is Severity.ERROR]
        if errors:
            raise InvalidGraphError(
                "The graph is not valid:\n" + "\n".join(f"  - {d}" for d in errors),
                hint="Fix the listed problems before running.",
            )
        order = graph.topological_order()
        total = len(order)
        keys: dict[str, str] = {}
        nodes: dict[str, NodeResult] = {}
        for i, name in enumerate(order):
            if cancel is not None:
                cancel.raise_if_cancelled()
            comp = graph.node(name)
            incoming = graph.incoming(name)
            input_keys = {
                port: (keys[c.source.node], c.source.port) for port, c in incoming.items()
            }
            key = node_key(comp, root_seed, input_keys, layout=layout, trial=trial)
            keys[name] = key

            if progress is not None:
                progress(ProgressEvent(name, i, total, i / total, "start"))
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                nodes[name] = NodeResult(
                    cached.outputs, cached.results, cached.diagnostics, key, cache_hit=True
                )
                _LOG.debug("node %s: cache hit", name)
            else:
                inputs = {
                    port: nodes[c.source.node].outputs[c.source.port]
                    for port, c in incoming.items()
                }
                nodes[name] = self._run_node(
                    comp, inputs, key, root_seed, i, total, progress, cancel, layout, trial
                )
                if self.cache is not None:
                    self.cache.put(key, nodes[name])
            if progress is not None:
                progress(ProgressEvent(name, i, total, (i + 1) / total, "done"))
        return SimulationResult(
            tuple(order),
            MappingProxyType(nodes),
            root_seed,
            tuple(d for d in diagnostics if d.severity is not Severity.ERROR),
            layout=layout,
            trial=trial,
        )

    @staticmethod
    def _run_node(
        comp: Any,
        inputs: dict[str, Any],
        key: str,
        root_seed: int,
        index: int,
        total: int,
        progress: ProgressCallback | None,
        cancel: CancellationToken | None,
        layout: SimulationLayout | None,
        trial: int | None,
    ) -> NodeResult:
        def sub_progress(fraction: float, message: str) -> None:
            if progress is not None:
                progress(
                    ProgressEvent(comp.name, index, total, (index + fraction) / total, message)
                )

        ctx = ExecutionContext(
            comp.name, root_seed, cancel=cancel, progress=sub_progress, layout=layout, trial=trial
        )
        start = time.perf_counter()
        try:
            raw = comp.run(inputs, ctx)
        except SimulationCancelledError:
            raise
        except Exception as exc:
            detail = exc.message if isinstance(exc, OptoBuildError) else f"{exc}"
            raise ComponentExecutionError(
                f"Component '{comp.name}' ({comp.type_id}) failed: {type(exc).__name__}: {detail}",
                hint=getattr(exc, "hint", None) or "Check the component's inputs and parameters.",
            ) from exc
        elapsed = time.perf_counter() - start
        outputs = dict(raw or {})
        declared = {p.name: p for p in comp.outputs()}
        if set(outputs) != set(declared):
            raise ComponentExecutionError(
                f"Component '{comp.name}' returned outputs {sorted(outputs)} but declares "
                f"{sorted(declared)}.",
                hint="This is a bug in the component implementation.",
            )
        for port, sig in outputs.items():
            if getattr(sig, "kind", None) is not declared[port].kind:
                raise SignalTypeError(
                    f"Component '{comp.name}' produced {type(sig).__name__} on port '{port}', "
                    f"which is declared {declared[port].kind.value}.",
                    hint="This is a bug in the component implementation.",
                )
        if _LOG.isEnabledFor(logging.INFO):
            _LOG.info("ran %s (%s) in %.3g s", comp.name, comp.type_id, elapsed)
        return NodeResult(
            MappingProxyType(outputs),
            MappingProxyType(dict(ctx.results)),
            tuple(ctx.diagnostics),
            key,
            elapsed_s=elapsed,
        )


__all__ = [
    "FeedForwardExecutor",
    "NodeResult",
    "ProgressEvent",
    "SimulationResult",
]
