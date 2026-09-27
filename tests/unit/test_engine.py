from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from optobuild.components.base import Component, RunContext
from optobuild.components.reference import Gain, GaussianNoise, RampSource, Recorder
from optobuild.components.spec import ComponentCategory, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import (
    ComponentExecutionError,
    InvalidGraphError,
    SignalTypeError,
    SimulationCancelledError,
    SimulationCycleError,
)
from optobuild.engine import CancellationToken, FeedForwardExecutor, ProgressEvent, ResultCache
from optobuild.engine.context import ExecutionContext
from optobuild.graph import SimulationGraph
from optobuild.signals import DigitalSequence, SignalKind

E = SignalKind.ELECTRICAL


def _chain(n: int = 3) -> SimulationGraph:
    g = SimulationGraph()
    g.add(RampSource("src", {"n_samples": 8}))
    prev = "src"
    for i in range(n):
        g.add(Gain(f"g{i}", {"gain": 2.0}))
        g.connect(prev, "out", f"g{i}", "in")
        prev = f"g{i}"
    g.add(Recorder("rec"))
    g.connect(prev, "out", "rec", "in")
    return g


def test_signal_propagates_through_several_components() -> None:
    res = FeedForwardExecutor().run(_chain(3), seed=0)
    assert res.order == ("src", "g0", "g1", "g2", "rec")
    np.testing.assert_array_equal(res.result("rec", "samples"), 8.0 * np.arange(8))
    # intermediate inspection
    np.testing.assert_array_equal(res.signal("g0", "out").samples, 2.0 * np.arange(8))
    assert res.result("rec", "mean") == pytest.approx(28.0)
    with pytest.raises(KeyError):
        res.signal("g0", "nope")
    with pytest.raises(KeyError):
        res.result("rec", "nope")


def test_recorded_arrays_are_read_only() -> None:
    res = FeedForwardExecutor().run(_chain(1))
    with pytest.raises(ValueError):
        res.result("rec", "samples")[0] = 1.0


def test_invalid_graph_is_refused_with_all_errors() -> None:
    g = SimulationGraph()
    g.add(Gain("a"))
    g.add(Gain("b"))
    with pytest.raises(InvalidGraphError) as info:
        FeedForwardExecutor().run(g)
    assert str(info.value).count("graph.unconnected_input") == 2


def test_cycle_refused() -> None:
    g = SimulationGraph()
    g.add(Gain("a"))
    g.add(Gain("b"))
    g.connect("a", "out", "b", "in")
    g.connect("b", "out", "a", "in")
    with pytest.raises((SimulationCycleError, InvalidGraphError)):
        FeedForwardExecutor().run(g)


class _Broken(Component):
    type_id = "tests.broken"
    version = "1"
    display_name = "broken"
    category = ComponentCategory.ELECTRICAL
    input_ports = (PortSpec("in", E),)
    output_ports = (PortSpec("out", E),)
    mode = "raise"

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        if self.mode == "raise":
            raise ZeroDivisionError("boom")
        if self.mode == "missing":
            return {}
        return {"out": DigitalSequence(np.array([1]), 1.0)}


@pytest.mark.parametrize(
    ("mode", "exc", "fragment"),
    [
        ("raise", ComponentExecutionError, "Component 'bad' (tests.broken) failed"),
        ("missing", ComponentExecutionError, "returned outputs []"),
        ("wrong_kind", SignalTypeError, "declared electrical"),
    ],
)
def test_component_failures_are_explained(mode: str, exc: type, fragment: str) -> None:
    g = _chain(0)
    g.remove("rec")
    bad = _Broken("bad")
    bad.mode = mode
    g.add(bad)
    g.connect("src", "out", "bad", "in")
    with pytest.raises(exc) as info:
        FeedForwardExecutor().run(g)
    assert fragment in str(info.value)
    if mode == "raise":
        assert isinstance(info.value.__cause__, ZeroDivisionError)


def test_progress_events_are_monotonic_and_complete() -> None:
    events: list[ProgressEvent] = []
    FeedForwardExecutor().run(_chain(2), progress=events.append)
    fractions = [e.fraction for e in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert {e.node for e in events} == {"src", "g0", "g1", "rec"}


def test_cancellation_before_and_during_run() -> None:
    token = CancellationToken()
    token.cancel()
    with pytest.raises(SimulationCancelledError):
        FeedForwardExecutor().run(_chain(1), cancel=token)

    token2 = CancellationToken()

    def cancel_after_first(e: ProgressEvent) -> None:
        if e.node == "g0" and e.message == "done":
            token2.cancel()

    with pytest.raises(SimulationCancelledError):
        FeedForwardExecutor().run(_chain(3), cancel=token2, progress=cancel_after_first)


def test_cache_hits_and_downstream_invalidation() -> None:
    cache = ResultCache()
    ex = FeedForwardExecutor(cache)
    g = _chain(3)
    first = ex.run(g, seed=1)
    assert not any(n.cache_hit for n in first.nodes.values())
    second = ex.run(g, seed=1)
    assert all(n.cache_hit for n in second.nodes.values())
    g.set_parameters("g1", gain=3.0)
    third = ex.run(g, seed=1)
    hits = {name: n.cache_hit for name, n in third.nodes.items()}
    assert hits == {"src": True, "g0": True, "g1": False, "g2": False, "rec": False}
    np.testing.assert_array_equal(third.result("rec", "samples"), 12.0 * np.arange(8))
    # a different seed invalidates everything (seed is part of every key)
    fourth = ex.run(g, seed=2)
    assert not any(n.cache_hit for n in fourth.nodes.values())


def test_cache_is_bounded() -> None:
    cache = ResultCache(max_entries=2)
    for i in range(5):
        cache.put(str(i), i)
    assert len(cache) == 2 and cache.get("0") is None and cache.get("4") == 4
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(ValueError):
        ResultCache(0)


def test_execution_context_services() -> None:
    progress: list[tuple[float, str]] = []
    ctx = ExecutionContext("n", 3, progress=lambda f, m: progress.append((f, m)))
    ctx.report_progress(1.5, "x")
    assert progress == [(1.0, "x")]
    ctx.record("a", np.arange(3))
    with pytest.raises(KeyError):
        ctx.record("a", 1)
    with pytest.raises(TypeError):
        ctx.record("b", {"not": "allowed"})
    ctx.warn(Diagnostic(Severity.WARNING, "c", "m"))
    assert len(ctx.diagnostics) == 1
    assert ctx.logger.name == "optobuild.run.n"
    ctx.check_cancelled()  # no token -> no-op


def test_noise_component_uses_context_rng() -> None:
    g = _chain(0)
    g.remove("rec")
    g.add(GaussianNoise("noise", {"sigma": 1.0}))
    g.add(Recorder("rec"))
    g.connect("src", "out", "noise", "in")
    g.connect("noise", "out", "rec", "in")
    a = FeedForwardExecutor().run(g, seed=5).result("rec", "samples")
    b = FeedForwardExecutor().run(g, seed=5).result("rec", "samples")
    c = FeedForwardExecutor().run(g, seed=6).result("rec", "samples")
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
