"""Main window: palette | schematic | parameters, results below, background runs."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
)

from optobuild.cli.demos import DEMOS
from optobuild.core.errors import OptoBuildError
from optobuild.gui.document import ProjectDocument
from optobuild.gui.qt.canvas import SchematicScene, SchematicView
from optobuild.gui.qt.panels import PalettePanel, ParameterPanel, ResultsPanel, SimulationDialog
from optobuild.gui.qt.worker import SimulationRunner
from optobuild.persistence.project import Project

PROJECT_FILTER = "OptoBuild projects (*.json *.yaml *.yml)"


class MainWindow(QMainWindow):
    """The OptoBuild editor."""

    def __init__(self, document: ProjectDocument | None = None) -> None:
        super().__init__()
        self.document = document or ProjectDocument()
        self.runner = SimulationRunner(self)
        self.scene = SchematicScene(self.document)
        self.view = SchematicView(self.scene)
        self.setCentralWidget(self.view)

        self.palette = PalettePanel(self.document.registry)
        self.parameters = ParameterPanel(self.document)
        self.results = ResultsPanel()
        self._dock(self.palette, "Components", Qt.DockWidgetArea.LeftDockWidgetArea)
        self._dock(self.parameters, "Parameters", Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock(self.results, "Results", Qt.DockWidgetArea.BottomDockWidgetArea)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.status_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

        self._build_actions()
        self.palette.add_requested.connect(self.add_component)
        self.scene.node_selected.connect(self.parameters.show_component)
        self.scene.error.connect(self.show_error)
        self.parameters.error.connect(self.show_error)
        self.runner.started.connect(self._run_started)
        self.runner.progress.connect(self._run_progress)
        self.runner.finished.connect(self._run_finished)
        self.runner.failed.connect(self._run_failed)
        self.runner.cancelled.connect(self._run_cancelled)
        self.document.subscribe(self._document_changed)
        self.resize(1400, 900)
        self._refresh_all()

    # --- construction ----------------------------------------------------------------------
    def _dock(self, widget: Any, title: str, area: Qt.DockWidgetArea) -> None:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)

    def _action(self, text: str, slot: Any, shortcut: Any = None) -> QAction:
        act = QAction(text, self)
        if shortcut is not None:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        return act

    def _build_actions(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction(self._action("&New", self.new_project, QKeySequence.StandardKey.New))
        f.addAction(self._action("&Open...", self._open_dialog, QKeySequence.StandardKey.Open))
        demos = f.addMenu("Open &demo")
        for name in sorted(DEMOS):
            demos.addAction(self._action(name, lambda _=False, n=name: self.open_demo(n)))
        f.addAction(self._action("&Save", self._save, QKeySequence.StandardKey.Save))
        f.addAction(
            self._action("Save &as...", self._save_as_dialog, QKeySequence.StandardKey.SaveAs)
        )
        f.addSeparator()
        self.act_save_results = self._action("Save &results (HDF5)...", self._save_results)
        f.addAction(self.act_save_results)
        f.addSeparator()
        f.addAction(self._action("&Quit", self.close, QKeySequence.StandardKey.Quit))
        e = mb.addMenu("&Edit")
        self.act_undo = self._action("&Undo", self.undo, QKeySequence.StandardKey.Undo)
        self.act_redo = self._action("&Redo", self.redo, QKeySequence.StandardKey.Redo)
        self.act_delete = self._action(
            "&Delete", self.delete_selected, QKeySequence.StandardKey.Delete
        )
        for a in (self.act_undo, self.act_redo, self.act_delete):
            e.addAction(a)
        s = mb.addMenu("&Simulation")
        self.act_run = self._action("&Run", self.run_simulation, "F5")
        self.act_cancel = self._action("&Cancel", self.runner.cancel, "Esc")
        s.addAction(self.act_run)
        s.addAction(self.act_cancel)
        s.addAction(self._action("&Settings...", self._settings_dialog))
        tb = self.addToolBar("Main")
        tb.setObjectName("Main")
        for a in (self.act_undo, self.act_redo, self.act_run, self.act_cancel):
            tb.addAction(a)

    # --- document events -------------------------------------------------------------------
    def _document_changed(self, event: str) -> None:
        if event in ("changed", "loaded"):
            self.scene.rebuild()
            self.parameters.show_component(self.parameters.component_name)
        if event == "loaded":
            self.results.show_result(self.document.last_result)
        self.results.set_stale(self.document.result_stale)
        self._refresh_actions()

    def _refresh_all(self) -> None:
        self.scene.rebuild()
        self.results.show_result(self.document.last_result)
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        d = self.document
        self.act_undo.setEnabled(d.can_undo)
        self.act_redo.setEnabled(d.can_redo)
        self.act_undo.setText(f"&Undo {d.undo_text}".rstrip())
        self.act_redo.setText(f"&Redo {d.redo_text}".rstrip())
        running = self.runner.running
        self.act_run.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        self.act_save_results.setEnabled(d.last_result is not None)
        name = d.path.name if d.path else "untitled"
        self.setWindowTitle(f"{name}{' *' if d.dirty else ''} - OptoBuild")

    def show_error(self, message: str) -> None:
        """Show a rejected action's explanation in the status bar."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #c92a2a")

    def show_info(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet("")

    # --- editing ---------------------------------------------------------------------------
    def add_component(self, type_id: str) -> str | None:
        """Add a component at the centre of the view and select it."""
        try:
            name = self.document.add_component(type_id, position=self.view.free_position())
        except OptoBuildError as exc:
            self.show_error(str(exc))
            return None
        self.scene.clearSelection()
        self.scene.nodes[name].setSelected(True)
        return name

    def delete_selected(self) -> None:
        edges = [e.key for e in self.scene.selected_edges()]
        nodes = self.scene.selected_node_names()
        try:
            for key in edges:
                if not set(key[::2]) & set(nodes):
                    self.document.disconnect(*key)
            if nodes:
                self.document.remove_components(nodes)
        except OptoBuildError as exc:
            self.show_error(str(exc))

    def undo(self) -> None:
        self.document.undo()

    def redo(self) -> None:
        self.document.redo()

    # --- files -----------------------------------------------------------------------------
    def _confirm_discard(self) -> bool:
        if not self.document.dirty or QApplication.platformName() == "offscreen":
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", "Discard unsaved changes to the current project?"
        )
        return answer == QMessageBox.StandardButton.Yes

    def new_project(self) -> None:
        if self._confirm_discard():
            from optobuild.graph.model import SimulationGraph

            self.document.replace_project(Project(graph=SimulationGraph()))

    def open_path(self, path: str | Path) -> bool:
        """Open a project file; errors are shown, not raised."""
        try:
            doc = ProjectDocument.open(path, self.document.registry)
        except (OptoBuildError, OSError) as exc:
            self.show_error(f"Cannot open {path}: {exc}")
            return False
        self.document.replace_project(doc.project, path)
        self.show_info(f"Opened {path}")
        return True

    def open_demo(self, name: str) -> None:
        if self._confirm_discard():
            self.document.replace_project(DEMOS[name]())
            self.show_info(f"Loaded demo '{name}'")

    def save_to(self, path: str | Path) -> bool:
        try:
            self.document.save(path)
        except (OptoBuildError, OSError) as exc:
            self.show_error(f"Cannot save: {exc}")
            return False
        self.show_info(f"Saved {path}")
        self._refresh_actions()
        return True

    def _open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if path:
            self.open_path(path)

    def _save(self) -> None:
        if self.document.path is None:
            self._save_as_dialog()
        else:
            self.save_to(self.document.path)

    def _save_as_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "project.json", PROJECT_FILTER)
        if path:
            self.save_to(path)

    def _save_results(self) -> None:
        if self.document.last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results", "results.h5", "HDF5 (*.h5 *.hdf5)"
        )
        if path:
            self.save_results_to(path)

    def save_results_to(self, path: str | Path) -> bool:
        """Write the last results (with the project snapshot) to HDF5."""
        from optobuild.persistence.results import save_results

        try:
            save_results(self.document.last_result, path, project=self.document.project)
        except (OptoBuildError, OSError) as exc:
            self.show_error(f"Cannot save results: {exc}")
            return False
        self.show_info(f"Saved results to {path}")
        return True

    def _settings_dialog(self) -> None:
        dlg = SimulationDialog(self.document.project.seed, self.document.project.layout, self)
        if dlg.exec() and dlg.values is not None:
            seed, layout = dlg.values
            self.document.set_simulation(seed=seed, layout=layout)

    # --- running ---------------------------------------------------------------------------
    def run_simulation(self) -> bool:
        """Start a background run of a snapshot of the project."""
        if self.runner.running:
            return False
        try:
            snapshot = self.document.snapshot_project()
        except OptoBuildError as exc:
            self.show_error(str(exc))
            return False
        self.runner.start(snapshot)
        return True

    def _run_started(self) -> None:
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.show_info("Running...")
        self._refresh_actions()

    def _run_progress(self, fraction: float, node: str) -> None:
        self.progress.setValue(int(1000 * fraction))
        self.status_label.setText(f"Running: {node}")

    def _run_finished(self, result: Any) -> None:
        self.progress.setVisible(False)
        self.document.set_result(result)
        self.results.show_result(result)
        hits = sum(n.cache_hit for n in result.nodes.values())
        self.show_info(
            f"Finished: {len(result.order)} components ({hits} from cache), "
            f"{len(result.all_diagnostics())} diagnostics."
        )
        self._refresh_actions()

    def _run_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.show_error(f"Simulation failed: {message}")
        self._refresh_actions()

    def _run_cancelled(self) -> None:
        self.progress.setVisible(False)
        self.show_info("Simulation cancelled.")
        self._refresh_actions()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self.runner.cancel()
        self.runner.wait()
        event.accept()


def main(argv: Sequence[str] | None = None) -> int:
    """Start the GUI: ``optobuild-gui [project.json | demo:<name>]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication([sys.argv[0], *args])
    window = MainWindow()
    if args:
        target = args[0]
        if target.startswith("demo:") and target[5:] in DEMOS:
            window.open_demo(target[5:])
        else:
            window.open_path(target)
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
