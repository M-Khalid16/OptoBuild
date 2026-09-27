"""Palette, parameter editor, results viewer and simulation-settings dialog."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from optobuild.components.registry import ComponentRegistry
from optobuild.core.errors import OptoBuildError
from optobuild.core.units import from_si, to_si
from optobuild.engine.executor import SimulationResult
from optobuild.gui.document import ProjectDocument
from optobuild.gui.forms import FieldModel, WidgetKind, fields_for
from optobuild.gui.plotdata import Curve, eye_curve, scalar_rows, spectrum_curve, waveform_curve
from optobuild.numerics.layout import SimulationLayout

pg.setConfigOptions(background="w", foreground="k", antialias=False)


class PalettePanel(QTreeWidget):
    """Components grouped by category; double-click (or Enter) requests adding one."""

    add_requested = Signal(str)

    def __init__(self, registry: ComponentRegistry) -> None:
        super().__init__()
        self.setHeaderLabels(["Component"])
        groups: dict[str, QTreeWidgetItem] = {}
        for type_id in registry:
            cls = registry.get(type_id)
            cat = cls.category.value.replace("_", " ")
            if cat not in groups:
                groups[cat] = QTreeWidgetItem(self, [cat.capitalize()])
            item = QTreeWidgetItem(groups[cat], [cls.display_name])
            item.setData(0, Qt.ItemDataRole.UserRole, type_id)
            item.setToolTip(0, f"{type_id} v{cls.version}\n\n{cls.documentation()}")
        self.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.expandAll()
        self.itemActivated.connect(self._activated)
        self.itemDoubleClicked.connect(self._activated)

    def _activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        type_id = item.data(0, Qt.ItemDataRole.UserRole)
        if type_id:
            self.add_requested.emit(type_id)

    def type_ids(self) -> list[str]:
        out = []
        for i in range(self.topLevelItemCount()):
            g = self.topLevelItem(i)
            out += [g.child(j).data(0, Qt.ItemDataRole.UserRole) for j in range(g.childCount())]
        return out


class ParameterPanel(QWidget):
    """Auto-generated editor for the selected component (from ParameterSpec)."""

    error = Signal(str)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.component_name: str | None = None
        self.editors: dict[str, QWidget] = {}
        self._layout = QVBoxLayout(self)
        self._title = QLabel("No component selected")
        self._title.setWordWrap(True)
        self._layout.addWidget(self._title)
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._rename)
        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #c92a2a")
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #e67700")
        for w in (self.name_edit, self._form_host, self.message, self.notes):
            self._layout.addWidget(w)
        self._layout.addStretch(1)
        self.show_component(None)

    def show_component(self, name: str | None) -> None:
        """Rebuild the form for component ``name`` (or clear it)."""
        self.component_name = name
        self.editors = {}
        while self._form.rowCount():
            self._form.removeRow(0)
        self.message.setText("")
        graph = self.document.project.graph
        if name is None or name not in graph:
            self.component_name = None
            self._title.setText("No component selected")
            self.name_edit.setVisible(False)
            self.notes.setText("")
            return
        comp = graph.node(name)
        self._title.setText(
            f"<b>{comp.display_name}</b><br><small>{comp.type_id} "
            f"v{comp.version}</small><br><small>{comp.documentation()}</small>"
        )
        self.name_edit.setVisible(True)
        self.name_edit.setText(name)
        for f in fields_for(comp.parameter_specs):
            editor = self._make_editor(f, comp.parameters[f.name])
            editor.setToolTip(f.tooltip)
            self._form.addRow(f.label, editor)
            self.editors[f.name] = editor
        self.notes.setText("\n".join(str(d) for d in comp.diagnostics))

    def _make_editor(self, field: FieldModel, value: Any) -> QWidget:
        kind = field.widget
        if kind is WidgetKind.CHECKBOX:
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(lambda v, f=field: self._commit(f, v))
            return box
        if kind is WidgetKind.CHOICE:
            combo = QComboBox()
            for c in field.choices:
                combo.addItem(str(c), c)
            combo.setCurrentIndex(list(field.choices).index(value))
            combo.currentIndexChanged.connect(
                lambda i, f=field, c=combo: self._commit(f, c.itemData(i))
            )
            return combo
        edit = QLineEdit(str(field.to_display(value)))
        edit.setProperty("committed", edit.text())
        edit.editingFinished.connect(lambda f=field, e=edit: self._commit_text(f, e))
        return edit

    def _commit_text(self, field: FieldModel, edit: QLineEdit) -> None:
        if edit.text() == edit.property("committed"):
            return
        self._commit(field, edit.text())

    def _commit(self, field: FieldModel, raw: Any) -> None:
        name = self.component_name
        if name is None:
            return
        try:
            value = field.from_display(raw)
            self.document.set_parameters(name, **{field.name: value})
        except OptoBuildError as exc:
            self.message.setText(str(exc))
            self.error.emit(str(exc))
            self._revert(field)
            return
        self.message.setText("")

    def _revert(self, field: FieldModel) -> None:
        if self.component_name is None:
            return
        comp = self.document.project.graph.node(self.component_name)
        editor = self.editors[field.name]
        value = comp.parameters[field.name]
        editor.blockSignals(True)
        if isinstance(editor, QLineEdit):
            editor.setText(str(field.to_display(value)))
        elif isinstance(editor, QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, QComboBox):
            editor.setCurrentIndex(list(field.choices).index(value))
        editor.blockSignals(False)

    def _rename(self) -> None:
        old, new = self.component_name, self.name_edit.text().strip()
        if old is None or new == old:
            return
        try:
            self.document.rename_component(old, new)
        except OptoBuildError as exc:
            self.message.setText(str(exc))
            self.name_edit.setText(old)
            return
        self.component_name = new


def _plot(widget: pg.PlotWidget, curve: Curve) -> None:
    widget.clear()
    widget.plot(curve.x, curve.y, pen=pg.mkPen("#1c7ed6", width=1), connect="finite")
    widget.setLabel("bottom", curve.x_label)
    widget.setLabel("left", curve.y_label)
    widget.setTitle(curve.title)


class _PlotTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.selector = QComboBox()
        top.addWidget(QLabel("Source:"))
        top.addWidget(self.selector, 1)
        self.extra = QHBoxLayout()
        top.addLayout(self.extra)
        lay.addLayout(top)
        self.plot = pg.PlotWidget()
        lay.addWidget(self.plot, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.3)


class ResultsPanel(QTabWidget):
    """Summary table, eye diagram, optical spectrum, waveforms and diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.result: SimulationResult | None = None
        summary = QWidget()
        sl = QVBoxLayout(summary)
        self.status = QLabel("No results yet. Run the simulation (F5).")
        sl.addWidget(self.status)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Component", "Result", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        sl.addWidget(self.table)
        self.addTab(summary, "Summary")
        self.eye = _PlotTab()
        self.eye.selector.currentTextChanged.connect(self._draw_eye)
        self.addTab(self.eye, "Eye diagram")
        self.spectrum = _PlotTab()
        self.freq_axis = QCheckBox("frequency axis")
        self.spectrum.extra.addWidget(self.freq_axis)
        self.spectrum.selector.currentTextChanged.connect(self._draw_spectrum)
        self.freq_axis.toggled.connect(lambda _: self._draw_spectrum())
        self.addTab(self.spectrum, "Spectrum")
        self.waveform = _PlotTab()
        self.waveform.selector.currentTextChanged.connect(self._draw_waveform)
        self.addTab(self.waveform, "Waveform")
        self.diagnostics = QListWidget()
        self.addTab(self.diagnostics, "Diagnostics")

    def set_stale(self, stale: bool) -> None:
        if self.result is not None:
            self.status.setText(
                "Results are OUT OF DATE: the project changed since this run."
                if stale
                else f"Results of run with seed {self.result.seed}."
            )
            self.status.setStyleSheet("color: #c92a2a" if stale else "")

    def show_result(self, result: SimulationResult | None) -> None:
        self.result = result
        self.table.setRowCount(0)
        self.diagnostics.clear()
        for tab in (self.eye, self.spectrum, self.waveform):
            tab.selector.blockSignals(True)
            tab.selector.clear()
            tab.plot.clear()
        if result is None:
            self.status.setText("No results yet. Run the simulation (F5).")
            for tab in (self.eye, self.spectrum, self.waveform):
                tab.selector.blockSignals(False)
            return
        self.set_stale(False)
        rows = scalar_rows(result)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(text))
        for d in result.all_diagnostics():
            self.diagnostics.addItem(str(d))
        for n in result.order:
            res = result.nodes[n].results
            if "traces" in res:
                self.eye.selector.addItem(n)
            if "power_per_rbw_w" in res:
                self.spectrum.selector.addItem(n)
            for port in result.nodes[n].outputs:
                self.waveform.selector.addItem(f"{n}.{port}")
        for tab in (self.eye, self.spectrum, self.waveform):
            tab.selector.blockSignals(False)
        self._draw_eye()
        self._draw_spectrum()
        self._draw_waveform()

    def _draw_eye(self, *_: Any) -> None:
        name = self.eye.selector.currentText()
        if self.result is None or not name:
            return
        res = dict(self.result.nodes[name].results)
        _plot(self.eye.plot, eye_curve(res, str(res.get("unit", "A"))))

    def _draw_spectrum(self, *_: Any) -> None:
        name = self.spectrum.selector.currentText()
        if self.result is None or not name:
            return
        axis = "frequency" if self.freq_axis.isChecked() else "wavelength"
        _plot(self.spectrum.plot, spectrum_curve(dict(self.result.nodes[name].results), axis))

    def _draw_waveform(self, *_: Any) -> None:
        text = self.waveform.selector.currentText()
        if self.result is None or not text:
            return
        node, port = text.rsplit(".", 1)
        _plot(self.waveform.plot, waveform_curve(self.result.signal(node, port)))

    def plotted_points(self, tab: str) -> int:
        """Number of finite points currently drawn in a plot tab (for tests)."""
        widget = {"eye": self.eye, "spectrum": self.spectrum, "waveform": self.waveform}[tab]
        items = widget.plot.getPlotItem().listDataItems()
        return int(sum(np.isfinite(i.yData).sum() for i in items if i.yData is not None))


class SimulationDialog(QDialog):
    """Seed and global layout (bit rate, number of bits, samples per bit)."""

    def __init__(
        self, seed: int, layout: SimulationLayout | None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Simulation settings")
        form = QFormLayout(self)
        self.seed = QLineEdit(str(seed))
        form.addRow("Random seed", self.seed)
        self.group = QGroupBox("Global layout (sources with timing_source = layout)")
        self.group.setCheckable(True)
        self.group.setChecked(layout is not None)
        g = QFormLayout(self.group)
        lay = layout or SimulationLayout(10e9, 2047, 16)
        self.bit_rate = QLineEdit(f"{float(from_si(lay.bit_rate, 'Gb/s')):.12g}")
        self.n_bits = QLineEdit(str(lay.n_bits))
        self.sps = QLineEdit(str(lay.samples_per_bit))
        g.addRow("Bit rate [Gb/s]", self.bit_rate)
        g.addRow("Number of bits", self.n_bits)
        g.addRow("Samples per bit", self.sps)
        form.addRow(self.group)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #c92a2a")
        self.error.setWordWrap(True)
        form.addRow(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.values: tuple[int, SimulationLayout | None] | None = None

    def parse(self) -> tuple[int, SimulationLayout | None]:
        """Validated (seed, layout); raises ValueError/OptoBuildError with a message."""
        seed = int(self.seed.text().strip())
        if not self.group.isChecked():
            return seed, None
        layout = SimulationLayout(
            float(to_si(float(self.bit_rate.text()), "Gb/s")),
            int(self.n_bits.text().strip()),
            int(self.sps.text().strip()),
        )
        return seed, layout

    def _accept(self) -> None:
        try:
            self.values = self.parse()
        except (ValueError, OptoBuildError) as exc:
            self.error.setText(str(exc))
            return
        self.accept()
