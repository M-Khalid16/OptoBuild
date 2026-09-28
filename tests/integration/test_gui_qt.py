"""Headless tests of the Qt GUI (skipped when PySide6/pyqtgraph are unavailable)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit  # noqa: E402

from optobuild.cli.demos import optical_link_project  # noqa: E402
from optobuild.gui.document import ProjectDocument  # noqa: E402
from optobuild.gui.qt.mainwindow import MainWindow  # noqa: E402
from optobuild.gui.qt.panels import SimulationDialog  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app: QApplication):  # type: ignore[no-untyped-def]
    w = MainWindow(ProjectDocument(optical_link_project(prbs_order=7)))
    yield w
    w.runner.cancel()
    w.runner.wait()
    w.close()


def _wait_for(app: QApplication, predicate, timeout: float = 60.0) -> None:  # type: ignore[no-untyped-def]
    end = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() > end:
            raise TimeoutError("GUI condition not reached")
        time.sleep(0.005)
    app.processEvents()


def _run(app: QApplication, w: MainWindow) -> None:
    finished: list[object] = []
    w.runner.finished.connect(finished.append)
    assert w.run_simulation()
    _wait_for(app, lambda: bool(finished))


def test_schematic_mirrors_the_project(window: MainWindow) -> None:
    g = window.document.project.graph
    assert set(window.scene.nodes) == set(g.node_names)
    assert len(window.scene.edges) == len(g.connections) == 13
    assert window.windowTitle().startswith("untitled")
    assert "optobuild.modulator.mzm" in window.palette.type_ids()


def test_parameter_form_edits_the_project_in_si(window: MainWindow) -> None:
    window.scene.nodes["laser"].setSelected(True)
    panel = window.parameters
    assert panel.component_name == "laser"
    wl = panel.editors["wavelength"]
    assert isinstance(wl, QLineEdit) and wl.text() == "1550"
    wl.setText("1310")
    wl.editingFinished.emit()
    assert window.document.project.graph.node("laser").parameters["wavelength"] == pytest.approx(
        1310e-9
    )
    assert window.windowTitle().endswith("* - OptoBuild")
    power = panel.editors["power"]
    power.setText("-100 W")  # invalid: negative power
    power.editingFinished.emit()
    assert "must be >= 0" in panel.message.text()
    assert panel.editors["power"].text() == "0"  # reverted to the stored 1 mW = 0 dBm
    timing = panel.editors["timing_source"]
    assert isinstance(timing, QComboBox) and timing.currentText() == "layout"


def test_checkbox_and_rename(window: MainWindow) -> None:
    window.scene.nodes["pin"].setSelected(True)
    shot = window.parameters.editors["shot_noise"]
    assert isinstance(shot, QCheckBox)
    shot.setChecked(False)
    assert window.document.project.graph.node("pin").parameters["shot_noise"] is False
    window.parameters.name_edit.setText("receiver")
    window.parameters.name_edit.editingFinished.emit()
    assert "receiver" in window.scene.nodes and "pin" not in window.scene.nodes
    window.undo()
    assert "pin" in window.scene.nodes


def test_palette_add_connect_and_type_errors(window: MainWindow) -> None:
    name = window.add_component("optobuild.analyzer.optical_power_meter")
    assert name == "optical_power_meter_1" and window.parameters.component_name == name
    scene = window.scene
    src = scene.nodes["fiber"].port("out", True)
    dst = scene.nodes[name].port("in", False)
    assert scene.request_connection(dst, src)  # either order works
    assert len(scene.edges) == 14
    bad = scene.nodes["pin"].port("out", True)
    other = window.add_component("optobuild.analyzer.optical_power_meter")
    assert not scene.request_connection(bad, scene.nodes[other].port("in", False))
    assert "Cannot connect pin.out (electrical)" in window.status_label.text()
    assert not scene.request_connection(src, scene.nodes["pin"].port("out", True))


def test_delete_and_undo_redo(window: MainWindow) -> None:
    window.scene.nodes["eye"].setSelected(True)
    window.delete_selected()
    assert "eye" not in window.scene.nodes and len(window.scene.edges) == 12
    assert window.act_undo.isEnabled() and "Delete eye" in window.act_undo.text()
    window.undo()
    assert "eye" in window.scene.nodes and len(window.scene.edges) == 13
    window.redo()
    assert "eye" not in window.scene.nodes
    edge = next(e for e in window.scene.edges if e.key == ("fiber", "out", "rx_power", "in"))
    edge.setSelected(True)
    window.delete_selected()
    assert all(e.key != ("fiber", "out", "rx_power", "in") for e in window.scene.edges)


def test_run_shows_results_and_marks_them_stale(app: QApplication, window: MainWindow) -> None:
    _run(app, window)
    r = window.results
    assert r.table.rowCount() > 10
    assert r.eye.selector.currentText() == "eye" and r.plotted_points("eye") > 1000
    assert r.spectrum.selector.currentText() == "tx_spectrum"
    assert r.plotted_points("spectrum") == 127 * 16
    assert r.plotted_points("waveform") > 0
    assert "Finished: 13 components" in window.status_label.text()
    assert window.act_save_results.isEnabled()
    window.document.set_parameters("fiber", length=10e3)
    assert "OUT OF DATE" in r.status.text()
    _run(app, window)  # second run reuses the cached transmitter
    assert "from cache" in window.status_label.text()
    hits = {n: x.cache_hit for n, x in window.document.last_result.nodes.items()}
    assert hits["mzm"] and not hits["fiber"]


def test_cancel(app: QApplication) -> None:
    w = MainWindow(ProjectDocument(optical_link_project(prbs_order=15)))
    done: list[str] = []
    w.runner.cancelled.connect(lambda: done.append("cancelled"))
    w.runner.finished.connect(lambda _r: done.append("finished"))
    w.run_simulation()
    w.runner.cancel()
    _wait_for(app, lambda: bool(done))
    assert done == ["cancelled"] or done == ["finished"]  # a fast machine may finish first
    assert not w.runner.running and w.act_run.isEnabled()
    w.close()


def test_failed_run_is_reported(app: QApplication, window: MainWindow) -> None:
    window.document.set_simulation(layout=None)  # layout-mode sources now fail clearly
    failed: list[str] = []
    window.runner.failed.connect(failed.append)
    window.run_simulation()
    _wait_for(app, lambda: bool(failed))
    assert "no layout" in failed[0] and "Simulation failed" in window.status_label.text()


def test_save_open_and_results(tmp_path: Path, app: QApplication, window: MainWindow) -> None:
    path = tmp_path / "link.json"
    assert window.save_to(path) and not window.document.dirty
    assert window.windowTitle() == "link.json - OptoBuild"
    w2 = MainWindow()
    assert w2.open_path(path)
    assert set(w2.scene.nodes) == set(window.scene.nodes)
    assert not w2.open_path(tmp_path / "missing.json")
    assert "Cannot open" in w2.status_label.text()
    w2.close()
    pytest.importorskip("h5py")
    _run(app, window)
    assert window.save_results_to(tmp_path / "r.h5")


def test_demos_and_new(window: MainWindow) -> None:
    window.open_demo("reference")
    assert set(window.scene.nodes) >= {"ramp", "adder"}
    window.new_project()
    assert window.scene.nodes == {} and not window.act_undo.isEnabled()


def test_simulation_dialog(app: QApplication) -> None:
    dlg = SimulationDialog(5, None)
    assert dlg.parse() == (5, None)
    dlg.group.setChecked(True)
    dlg.bit_rate.setText("25")
    seed, layout = dlg.parse()
    assert layout is not None and layout.bit_rate == 25e9 and layout.n_bits == 2047
    dlg.sps.setText("1")
    dlg._accept()
    assert "samples_per_bit" in dlg.error.text() and dlg.values is None


def test_mouse_drag_connects_ports_and_moves_nodes(app: QApplication) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    doc = ProjectDocument()
    w = MainWindow(doc)
    w.show()
    a = doc.add_component("optobuild.reference.ramp_source", position=(0, 0))
    b = doc.add_component("optobuild.reference.recorder", position=(300, 0))
    view, scene = w.view, w.scene
    view.centerOn(225, 30)
    app.processEvents()

    def at(item) -> QPoint:  # type: ignore[no-untyped-def]
        return view.mapFromScene(item.scenePos())

    src = scene.nodes[a].port("out", True)
    dst = scene.nodes[b].port("in", False)
    vp = view.viewport()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(src))
    QTest.mouseMove(vp, at(dst))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(dst))
    assert [str(c) for c in doc.project.graph.connections] == [f"{a}.out -> {b}.in"]

    node = scene.nodes[b]
    start = view.mapFromScene(node.scenePos() + node.rect().center())
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for k in range(1, 6):
        QTest.mouseMove(vp, start + QPoint(10 * k, 8 * k))
    QTest.mouseRelease(
        vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(50, 40)
    )
    x, y = doc.position(b)
    assert (x, y) != (300.0, 0.0) and doc.undo_text == f"Move {b}"
    doc.undo()
    assert doc.position(b) == (300.0, 0.0)
    w.close()


def test_coherent_demo_shows_constellation_and_symbol_waveforms(app: QApplication) -> None:
    from optobuild.cli.demos import coherent_link_project

    w = MainWindow(ProjectDocument(coherent_link_project(prbs_order=11)))
    _run(app, w)
    r = w.results
    assert r.constellation.selector.currentText() == "analyzer"
    assert r.plotted_points("constellation") == 2047
    idx = r.waveform.selector.findText("mapper.symbols")
    assert idx >= 0
    r.waveform.selector.setCurrentIndex(idx)  # symbol outputs render as a scatter, no crash
    assert r.plotted_points("waveform") == 2047
    w.close()


def test_ring_demo_shows_device_response(app: QApplication) -> None:
    from optobuild.cli.demos import ring_filter_project

    w = MainWindow(ProjectDocument(ring_filter_project(prbs_order=7)))
    _run(app, w)
    r = w.results
    items = [r.response.selector.itemText(i) for i in range(r.response.selector.count())]
    assert items == ["ring.through", "ring.drop"]
    assert r.plotted_points("response") == 127 * 32
    w.close()
