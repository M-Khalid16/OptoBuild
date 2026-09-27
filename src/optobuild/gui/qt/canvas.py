"""Schematic editor: components as boxes with typed ports, connections as curves.

The canvas only *displays* the document and forwards user intent (move,
connect, select) to it; all rules (port kinds, fan-out) are enforced by the
graph through ``ProjectDocument``. Rejected connections are reported through
the ``error`` signal with the graph's explanation.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from optobuild.components.spec import PortSpec
from optobuild.core.errors import OptoBuildError
from optobuild.gui.document import ProjectDocument
from optobuild.signals.kinds import SignalKind

KIND_COLORS = {
    SignalKind.OPTICAL: QColor("#d9480f"),
    SignalKind.ELECTRICAL: QColor("#1c7ed6"),
    SignalKind.DIGITAL: QColor("#2f9e44"),
    SignalKind.SYMBOLS: QColor("#7048e8"),
}
NODE_W, NODE_H, PORT_R, PORT_DY = 150.0, 56.0, 6.0, 18.0


class PortItem(QGraphicsEllipseItem):
    """A typed port on a node."""

    def __init__(self, node: NodeItem, spec: PortSpec, is_output: bool, index: int) -> None:
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, node)
        self.node, self.spec, self.is_output = node, spec, is_output
        color = KIND_COLORS[spec.kind]
        self.setBrush(QBrush(color if is_output or not spec.tap else QColor("white")))
        self.setPen(QPen(color, 2))
        self.setPos(NODE_W if is_output else 0.0, PORT_DY + index * PORT_DY)
        tap = " (tap)" if spec.tap else ""
        self.setToolTip(
            f"{spec.name}: {spec.kind.value}{tap}"
            + (f"\n{spec.description}" if spec.description else "")
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def scene_center(self) -> QPointF:
        return self.scenePos()


class NodeItem(QGraphicsRectItem):
    """A component instance."""

    def __init__(
        self,
        name: str,
        type_id: str,
        display_name: str,
        inputs: tuple[PortSpec, ...],
        outputs: tuple[PortSpec, ...],
        has_warnings: bool,
    ) -> None:
        height = max(NODE_H, PORT_DY * (max(len(inputs), len(outputs)) + 1))
        super().__init__(0, 0, NODE_W, height)
        self.name = name
        self.setBrush(QBrush(QColor("#fff9db" if has_warnings else "#f8f9fa")))
        self.setPen(QPen(QColor("#495057"), 1.5))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        title = QGraphicsSimpleTextItem(name, self)
        title.setPos(8, height - 36)
        sub = QGraphicsSimpleTextItem(display_name, self)
        sub.setBrush(QBrush(QColor("#868e96")))
        sub.setPos(8, height - 20)
        self.setToolTip(f"{name}\n{type_id}")
        self.ports = [PortItem(self, p, False, i) for i, p in enumerate(inputs)]
        self.ports += [PortItem(self, p, True, i) for i, p in enumerate(outputs)]
        self.edges: list[EdgeItem] = []

    def port(self, name: str, is_output: bool) -> PortItem:
        return next(p for p in self.ports if p.spec.name == name and p.is_output == is_output)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for e in self.edges:
                e.update_path()
        return super().itemChange(change, value)


class EdgeItem(QGraphicsPathItem):
    """A connection between an output and an input port."""

    def __init__(self, src: PortItem, dst: PortItem) -> None:
        super().__init__()
        self.src, self.dst = src, dst
        self.setPen(QPen(KIND_COLORS[src.spec.kind], 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)
        src.node.edges.append(self)
        dst.node.edges.append(self)
        self.update_path()

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.src.node.name, self.src.spec.name, self.dst.node.name, self.dst.spec.name)

    def update_path(self) -> None:
        a, b = self.src.scene_center(), self.dst.scene_center()
        dx = max(40.0, abs(b.x() - a.x()) / 2)
        path = QPainterPath(a)
        path.cubicTo(a + QPointF(dx, 0), b - QPointF(dx, 0), b)
        self.setPath(path)


class SchematicScene(QGraphicsScene):
    """Scene mirroring a ProjectDocument."""

    node_selected = Signal(object)  # component name or None
    error = Signal(str)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.nodes: dict[str, NodeItem] = {}
        self.edges: list[EdgeItem] = []
        self._drag_port: PortItem | None = None
        self._drag_line: QGraphicsPathItem | None = None
        self._moving: dict[str, QPointF] = {}
        self.selectionChanged.connect(self._on_selection)
        self.rebuild()

    # --- building --------------------------------------------------------------------------
    def rebuild(self) -> None:
        """Re-create all items from the document (keeps the selection by name)."""
        selected = {i.name for i in self.selectedItems() if isinstance(i, NodeItem)}
        self.blockSignals(True)
        self.clear()
        self.nodes, self.edges = {}, []
        graph = self.document.project.graph
        for comp in graph:
            warn = bool(comp.diagnostics)
            item = NodeItem(
                comp.name, comp.type_id, comp.display_name, comp.inputs(), comp.outputs(), warn
            )
            item.setPos(*self.document.position(comp.name))
            self.addItem(item)
            self.nodes[comp.name] = item
            item.setSelected(comp.name in selected)
        for c in graph.connections:
            src = self.nodes[c.source.node].port(c.source.port, True)
            dst = self.nodes[c.target.node].port(c.target.port, False)
            edge = EdgeItem(src, dst)
            self.addItem(edge)
            self.edges.append(edge)
        self.blockSignals(False)
        self._on_selection()

    def selected_node_names(self) -> list[str]:
        return [i.name for i in self.selectedItems() if isinstance(i, NodeItem)]

    def selected_edges(self) -> list[EdgeItem]:
        return [i for i in self.selectedItems() if isinstance(i, EdgeItem)]

    def _on_selection(self) -> None:
        names = self.selected_node_names()
        self.node_selected.emit(names[0] if len(names) == 1 else None)

    # --- connecting ------------------------------------------------------------------------
    def request_connection(self, a: PortItem, b: PortItem) -> bool:
        """Connect two ports (either order); errors are emitted, not raised."""
        if a.is_output == b.is_output:
            self.error.emit("Connect an output port to an input port.")
            return False
        src, dst = (a, b) if a.is_output else (b, a)
        try:
            self.document.connect(src.node.name, src.spec.name, dst.node.name, dst.spec.name)
        except OptoBuildError as exc:
            self.error.emit(str(exc))
            return False
        return True

    def _port_at(self, pos: QPointF) -> PortItem | None:
        for item in self.items(pos):
            if isinstance(item, PortItem):
                return item
        return None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        port = self._port_at(event.scenePos())
        if port is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_port = port
            self._drag_line = QGraphicsPathItem()
            self._drag_line.setPen(QPen(KIND_COLORS[port.spec.kind], 1.5, Qt.PenStyle.DashLine))
            self.addItem(self._drag_line)
            event.accept()
            return
        self._moving = {n: item.pos() for n, item in self.nodes.items()}
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_port is not None and self._drag_line is not None:
            path = QPainterPath(self._drag_port.scene_center())
            path.lineTo(event.scenePos())
            self._drag_line.setPath(path)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_port is not None:
            start = self._drag_port
            if self._drag_line is not None:
                self.removeItem(self._drag_line)
            self._drag_port, self._drag_line = None, None
            target = self._port_at(event.scenePos())
            if target is not None and target is not start:
                self.request_connection(start, target)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        moved = [
            n
            for n, item in self.nodes.items()
            if n in self._moving and item.pos() != self._moving[n]
        ]
        self._moving = {}
        for n in moved:  # one undoable step per moved node
            p = self.nodes[n].pos()
            self.document.move_component(n, (p.x(), p.y()))


class SchematicView(QGraphicsView):
    """Zoomable view of the schematic."""

    def __init__(self, scene: SchematicScene) -> None:
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setSceneRect(QRectF(-2000, -2000, 6000, 4000))

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def free_position(self) -> tuple[float, float]:
        """Scene position at the centre of the visible area."""
        c = self.mapToScene(self.viewport().rect().center())
        return c.x() - NODE_W / 2, c.y() - NODE_H / 2
