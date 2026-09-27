from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from optobuild.components.base import Component, RunContext
from optobuild.components.reference import Adder, Gain, RampSource, Recorder
from optobuild.components.spec import ComponentCategory, PortSpec
from optobuild.core.diagnostics import Severity
from optobuild.core.errors import (
    InvalidGraphError,
    InvalidParameterError,
    PortTypeMismatchError,
    SimulationCycleError,
)
from optobuild.graph import Connection, Endpoint, SimulationGraph
from optobuild.signals.kinds import SignalKind

OPT = SignalKind.OPTICAL


class _OptSource(Component):
    type_id = "tests.opt_source"
    version = "1"
    display_name = "opt src"
    category = ComponentCategory.SOURCE
    output_ports = (PortSpec("out", OPT),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        return {}


class _OptThrough(Component):
    type_id = "tests.opt_through"
    version = "1"
    display_name = "opt through"
    category = ComponentCategory.PASSIVE
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("out", OPT),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        return {}


class _OptMeter(Component):
    type_id = "tests.opt_meter"
    version = "1"
    display_name = "opt meter"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("in", OPT, tap=True),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        return {}


def _chain() -> SimulationGraph:
    g = SimulationGraph()
    g.add(RampSource("src"))
    g.add(Gain("g1"))
    g.add(Recorder("rec"))
    g.connect("src", "out", "g1", "in")
    g.connect("g1", "out", "rec", "in")
    return g


def test_build_and_query() -> None:
    g = _chain()
    assert g.node_names == ("src", "g1", "rec") and len(g) == 3 and "g1" in g
    assert g.incoming("rec")["in"] == Connection(Endpoint("g1", "out"), Endpoint("rec", "in"))
    assert g.outgoing("src")[0].target == Endpoint("g1", "in")
    assert g.downstream("src") == {"g1", "rec"}
    assert str(g.connections[0]) == "src.out -> g1.in"
    assert g.validate() == []


def test_duplicate_names_and_missing_nodes() -> None:
    g = _chain()
    with pytest.raises(InvalidGraphError, match="already exists"):
        g.add(Gain("g1"))
    with pytest.raises(InvalidGraphError, match="No component named"):
        g.connect("nope", "out", "rec", "in")
    with pytest.raises(InvalidGraphError):
        g.add("not a component")  # type: ignore[arg-type]


def test_invalid_ports_rejected() -> None:
    g = _chain()
    g.add(Gain("g2"))
    with pytest.raises(InvalidGraphError, match="no output port 'x'"):
        g.connect("src", "x", "g2", "in")
    with pytest.raises(InvalidGraphError, match="no input port 'out'"):
        g.connect("src", "out", "g2", "out")
    with pytest.raises(InvalidGraphError, match="already connected"):
        g.connect("src", "out", "rec", "in")


def test_kind_mismatch_rejected() -> None:
    g = SimulationGraph()
    g.add(_OptSource("laser"))
    g.add(Gain("amp"))
    with pytest.raises(PortTypeMismatchError, match=r"optical\) to amp.in \(electrical"):
        g.connect("laser", "out", "amp", "in")
    assert g.connections == ()


def test_optical_fan_out_rule() -> None:
    g = SimulationGraph()
    g.add(_OptSource("laser"))
    g.add(_OptThrough("a"))
    g.add(_OptThrough("b"))
    g.add(_OptMeter("m1"))
    g.add(_OptMeter("m2"))
    g.connect("laser", "out", "m1", "in")  # taps are always fine
    g.connect("laser", "out", "a", "in")
    g.connect("laser", "out", "m2", "in")
    with pytest.raises(InvalidGraphError, match="duplicate optical power"):
        g.connect("laser", "out", "b", "in")


def test_electrical_fan_out_allowed() -> None:
    g = _chain()
    g.add(Gain("g2"))
    g.connect("src", "out", "g2", "in")
    assert len(g.outgoing("src")) == 2


def test_topological_order_is_deterministic_insertion_tiebreak() -> None:
    g = SimulationGraph()
    g.add(Adder("sum"))  # added first but depends on others
    g.add(RampSource("b"))
    g.add(RampSource("a"))
    g.add(Recorder("rec"))
    g.connect("a", "out", "sum", "a")
    g.connect("b", "out", "sum", "b")
    g.connect("sum", "out", "rec", "in")
    assert g.topological_order() == ["b", "a", "sum", "rec"]


def test_cycle_detection() -> None:
    g = SimulationGraph()
    g.add(RampSource("src"))
    g.add(Adder("add"))
    g.add(Gain("fb"))
    g.connect("src", "out", "add", "a")
    g.connect("add", "out", "fb", "in")
    g.connect("fb", "out", "add", "b")
    assert g.find_cycle() in (["add", "fb", "add"], ["fb", "add", "fb"])
    with pytest.raises(SimulationCycleError, match="cycle") as info:
        g.topological_order()
    assert "iterative executor" in str(info.value)
    assert any(d.code == "graph.cycle" for d in g.validate())


def test_self_loop_is_a_cycle() -> None:
    g = SimulationGraph()
    g.add(Adder("add"))
    g.add(RampSource("src"))
    g.connect("src", "out", "add", "a")
    g.connect("add", "out", "add", "b")
    assert g.find_cycle() == ["add", "add"]


def test_validation_reports_unconnected_inputs_and_empty_graph() -> None:
    g = SimulationGraph()
    assert [d.code for d in g.validate()] == ["graph.empty"]
    g.add(Adder("add"))
    diags = g.validate()
    assert {d.code for d in diags} == {"graph.unconnected_input"}
    assert all(d.severity is Severity.ERROR and d.source == "add" for d in diags)
    assert len(diags) == 2


def test_remove_and_disconnect() -> None:
    g = _chain()
    g.disconnect(g.connections[1])
    assert len(g.connections) == 1
    with pytest.raises(InvalidGraphError):
        g.disconnect(Connection(Endpoint("x", "y"), Endpoint("z", "w")))
    g.remove("g1")
    assert g.connections == () and "g1" not in g


def test_set_parameters_replaces_instance_and_validates() -> None:
    g = _chain()
    old = g.node("g1")
    new = g.set_parameters("g1", gain=5.0)
    assert g.node("g1") is new and new is not old and new.parameters["gain"] == 5.0
    with pytest.raises(InvalidParameterError):
        g.set_parameters("src", n_samples=1)
    assert g.node("src").parameters["n_samples"] == 16
