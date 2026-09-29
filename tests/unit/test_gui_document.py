from __future__ import annotations

from pathlib import Path

import pytest

from optobuild.cli.demos import optical_link_project
from optobuild.core.errors import InvalidGraphError, InvalidParameterError, PortTypeMismatchError
from optobuild.gui.document import ProjectDocument, auto_positions
from optobuild.numerics.layout import SimulationLayout
from optobuild.persistence import project_to_dict, run_project


def _doc() -> ProjectDocument:
    return ProjectDocument(optical_link_project(prbs_order=7))


def test_positions_are_created_for_every_node() -> None:
    doc = _doc()
    pos = doc.project.schematic["positions"]
    assert set(pos) == set(doc.project.graph.node_names)
    auto = auto_positions(doc.project.graph)
    assert auto["prbs"][0] < auto["nrz"][0] < auto["mzm"][0] < auto["fiber"][0]


def test_add_connect_undo_redo() -> None:
    doc = ProjectDocument()
    events: list[str] = []
    doc.subscribe(events.append)
    a = doc.add_component("optobuild.reference.ramp_source", position=(10, 20))
    b = doc.add_component("optobuild.reference.recorder")
    assert (a, b) == ("ramp_source_1", "recorder_1")
    doc.connect(a, "out", b, "in")
    assert len(doc.project.graph.connections) == 1 and doc.dirty
    assert doc.undo_text == "Connect ramp_source_1.out -> recorder_1.in"
    doc.undo()
    assert doc.project.graph.connections == ()
    doc.undo()
    assert b not in doc.project.graph
    doc.redo()
    doc.redo()
    assert len(doc.project.graph.connections) == 1
    assert doc.position(a) == (10.0, 20.0)
    assert events.count("changed") == 7 and not doc.can_redo


def test_rejected_edits_leave_document_unchanged() -> None:
    doc = _doc()
    before = project_to_dict(doc.project)
    depth = len(doc._undo)
    with pytest.raises(PortTypeMismatchError):
        doc.connect("pin", "out", "rx_power", "in")
    with pytest.raises(InvalidParameterError):
        doc.set_parameters("mzm", v_pi=-1.0)
    with pytest.raises(InvalidGraphError):
        doc.add_component("optobuild.detector.pin", name="pin")
    with pytest.raises(InvalidGraphError, match="Invalid component name"):
        doc.add_component("optobuild.detector.pin", name="1bad/name")
    assert project_to_dict(doc.project) == before and len(doc._undo) == depth
    assert not doc.dirty


def test_rename_keeps_connections_parameters_and_position() -> None:
    doc = _doc()
    doc.set_parameters("fiber", length=30e3)
    pos = doc.position("fiber")
    doc.rename_component("fiber", "span 1")
    g = doc.project.graph
    assert "fiber" not in g and g.node("span 1").parameters["length"] == 30e3
    assert {str(c) for c in g.connections} >= {
        "mzm.optical_out -> span 1.in",
        "span 1.out -> pin.in",
    }
    assert doc.position("span 1") == pos
    with pytest.raises(InvalidGraphError, match="already exists"):
        doc.rename_component("span 1", "pin")
    doc.undo()
    assert "fiber" in doc.project.graph


def test_remove_move_disconnect_and_simulation_settings() -> None:
    doc = _doc()
    doc.move_component("pin", (5.0, 6.0))
    assert doc.position("pin") == (5.0, 6.0)
    doc.disconnect("fiber", "out", "rx_power", "in")
    assert not any(str(c) == "fiber.out -> rx_power.in" for c in doc.project.graph.connections)
    with pytest.raises(InvalidGraphError):
        doc.disconnect("fiber", "out", "rx_power", "in")
    doc.remove_components(["eye", "tx_spectrum"])
    assert "eye" not in doc.project.schematic["positions"]
    doc.set_simulation(seed=99, layout=SimulationLayout(10e9, 127, 8))
    assert doc.project.seed == 99 and doc.project.layout.samples_per_bit == 8
    doc.set_simulation(layout=None)
    assert doc.project.layout is None
    doc.undo()
    assert doc.project.layout is not None


def test_save_open_and_stale_results(tmp_path: Path) -> None:
    doc = _doc()
    with pytest.raises(InvalidGraphError, match="no file name"):
        doc.save()
    path = doc.save(tmp_path / "link.json")
    assert not doc.dirty
    reopened = ProjectDocument.open(path)
    assert reopened.position("mzm") == doc.position("mzm")
    doc.set_result(run_project(doc.snapshot_project()))
    assert not doc.result_stale
    doc.set_parameters("fiber", length=1e3)
    assert doc.result_stale
    doc.replace_project(optical_link_project())
    assert doc.last_result is None and not doc.can_undo


def test_snapshot_is_independent() -> None:
    doc = _doc()
    snap = doc.snapshot_project()
    doc.set_parameters("fiber", length=1.0)
    assert snap.graph.node("fiber").parameters["length"] == 50e3
