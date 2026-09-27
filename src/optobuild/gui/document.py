"""Editable project document for the GUI (Qt-free).

Every edit goes through :class:`ProjectDocument`, which

* validates it with the same graph/component rules as scripts (a rejected
  edit leaves the document unchanged and raises the domain error);
* records an undo snapshot (the complete project in its JSON-compatible
  form, so undo/redo restore exactly what save/load would);
* keeps schematic positions in ``project.schematic["positions"]``;
* marks the last simulation result as stale;
* notifies observers (the Qt views) with a short event name.

No physics happens here; running is delegated to the engine.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from optobuild.components.registry import ComponentRegistry, builtin_registry
from optobuild.core.errors import InvalidGraphError, OptoBuildError
from optobuild.engine.executor import SimulationResult
from optobuild.graph.model import SimulationGraph
from optobuild.numerics.layout import SimulationLayout
from optobuild.persistence.project import (
    Project,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)

Position = tuple[float, float]
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\- ]*$")
MAX_UNDO = 200


def auto_positions(
    graph: SimulationGraph, dx: float = 220.0, dy: float = 130.0
) -> dict[str, Position]:
    """Place nodes by longest-path depth (columns) in insertion order (rows).

    Cycles (not runnable, but editable) are tolerated: remaining nodes are
    appended in an extra column.
    """
    depth: dict[str, int] = {}
    try:
        order = graph.topological_order()
    except OptoBuildError:
        order = list(graph.node_names)
    for name in order:
        preds = [c.source.node for c in graph.incoming(name).values()]
        depth[name] = 1 + max((depth.get(p, 0) for p in preds), default=-1)
    rows: dict[int, int] = {}
    out: dict[str, Position] = {}
    for name in graph.node_names:
        col = depth.get(name, 0)
        row = rows.get(col, 0)
        rows[col] = row + 1
        out[name] = (col * dx, row * dy)
    return out


class ProjectDocument:
    """A project being edited, with undo/redo and change notification."""

    def __init__(
        self,
        project: Project | None = None,
        path: str | Path | None = None,
        registry: ComponentRegistry | None = None,
    ) -> None:
        self.registry = builtin_registry() if registry is None else registry
        self.project = project if project is not None else Project(graph=SimulationGraph())
        self.path: Path | None = None if path is None else Path(path)
        self.dirty = False
        self.last_result: SimulationResult | None = None
        self.result_stale = False
        self._undo: list[tuple[str, dict[str, Any]]] = []
        self._redo: list[tuple[str, dict[str, Any]]] = []
        self._listeners: list[Callable[[str], None]] = []
        self._ensure_positions()

    # --- observers -------------------------------------------------------------------------
    def subscribe(self, listener: Callable[[str], None]) -> None:
        """Call ``listener(event)`` after every change (events: changed, loaded, saved,
        result)."""
        self._listeners.append(listener)

    def _notify(self, event: str) -> None:
        for fn in list(self._listeners):
            fn(event)

    # --- files -----------------------------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path, registry: ComponentRegistry | None = None) -> ProjectDocument:
        """Load a project file."""
        reg = builtin_registry() if registry is None else registry
        return cls(load_project(path, reg), path, reg)

    def save(self, path: str | Path | None = None) -> Path:
        """Save to ``path`` (or the current path)."""
        target = Path(path) if path is not None else self.path
        if target is None:
            raise InvalidGraphError(
                "The document has no file name yet.", hint="Use 'Save As' and choose a file name."
            )
        save_project(self.project, target)
        self.path = target
        self.dirty = False
        self._notify("saved")
        return target

    def replace_project(self, project: Project, path: str | Path | None = None) -> None:
        """Load another project into this document (clears undo history)."""
        self.project = project
        self.path = None if path is None else Path(path)
        self.dirty = False
        self.last_result = None
        self.result_stale = False
        self._undo.clear()
        self._redo.clear()
        self._ensure_positions()
        self._notify("loaded")

    # --- snapshots / undo ------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(project_to_dict(self.project))

    def _restore(self, snap: dict[str, Any]) -> None:
        self.project = project_from_dict(snap, self.registry)

    def _mutate(self, description: str, action: Callable[[], Any]) -> Any:
        before = self._snapshot()
        try:
            value = action()
        except Exception:
            self._restore(before)
            raise
        self._undo.append((description, before))
        del self._undo[:-MAX_UNDO]
        self._redo.clear()
        self.dirty = True
        if self.last_result is not None:
            self.result_stale = True
        self._notify("changed")
        return value

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_text(self) -> str:
        """Description of the edit that undo would revert."""
        return self._undo[-1][0] if self._undo else ""

    @property
    def redo_text(self) -> str:
        return self._redo[-1][0] if self._redo else ""

    def undo(self) -> None:
        """Revert the last edit."""
        if not self._undo:
            return
        description, snap = self._undo.pop()
        self._redo.append((description, self._snapshot()))
        self._restore(snap)
        self.dirty = True
        self.result_stale = self.last_result is not None
        self._notify("changed")

    def redo(self) -> None:
        """Re-apply the last undone edit."""
        if not self._redo:
            return
        description, snap = self._redo.pop()
        self._undo.append((description, self._snapshot()))
        self._restore(snap)
        self.dirty = True
        self.result_stale = self.last_result is not None
        self._notify("changed")

    # --- positions -------------------------------------------------------------------------
    def _positions(self) -> dict[str, list[float]]:
        return self.project.schematic.setdefault("positions", {})

    def _ensure_positions(self) -> None:
        pos = self._positions()
        missing = [n for n in self.project.graph.node_names if n not in pos]
        if missing:
            auto = auto_positions(self.project.graph)
            for n in missing:
                pos[n] = list(auto[n])
        for n in [n for n in pos if n not in self.project.graph]:
            del pos[n]

    def position(self, name: str) -> Position:
        """Schematic position of ``name``."""
        x, y = self._positions()[name]
        return float(x), float(y)

    # --- editing ---------------------------------------------------------------------------
    def unique_name(self, base: str) -> str:
        """``base_1``, ``base_2``, ... not yet used in the graph."""
        i = 1
        while f"{base}_{i}" in self.project.graph:
            i += 1
        return f"{base}_{i}"

    def add_component(
        self,
        type_id: str,
        name: str | None = None,
        position: Position = (0.0, 0.0),
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Add a component (default parameters unless given); returns its name."""
        cls = self.registry.get(type_id)
        name = name or self.unique_name(type_id.rsplit(".", 1)[-1])
        self._check_name(name)

        def action() -> str:
            self.project.graph.add(cls(name, parameters))
            self._positions()[name] = [float(position[0]), float(position[1])]
            return name

        return self._mutate(f"Add {name}", action)

    def remove_components(self, names: list[str]) -> None:
        """Remove components (and their connections)."""

        def action() -> None:
            for n in names:
                self.project.graph.remove(n)
                self._positions().pop(n, None)

        self._mutate(f"Delete {', '.join(names)}", action)

    def rename_component(self, old: str, new: str) -> None:
        """Rename a component, keeping its parameters, connections and position."""
        if old == new:
            return
        self._check_name(new)
        if new in self.project.graph:
            raise InvalidGraphError(f"A component named '{new}' already exists.")

        def action() -> None:
            data = project_to_dict(self.project)
            for c in data["components"]:
                if c["name"] == old:
                    c["name"] = new
            for conn in data["connections"]:
                for end in ("from", "to"):
                    if conn[end][0] == old:
                        conn[end][0] = new
            pos = data["schematic"].get("positions", {})
            if old in pos:
                pos[new] = pos.pop(old)
            self._restore(data)

        self._mutate(f"Rename {old} to {new}", action)

    def move_component(self, name: str, position: Position) -> None:
        """Set the schematic position (undoable)."""
        self.project.graph.node(name)
        self._mutate(
            f"Move {name}",
            lambda: self._positions().__setitem__(name, [float(position[0]), float(position[1])]),
        )

    def connect(self, source: str, source_port: str, target: str, target_port: str) -> None:
        """Connect two ports (graph rules apply)."""
        self._mutate(
            f"Connect {source}.{source_port} -> {target}.{target_port}",
            lambda: self.project.graph.connect(source, source_port, target, target_port),
        )

    def disconnect(self, source: str, source_port: str, target: str, target_port: str) -> None:
        """Remove a connection."""
        graph = self.project.graph
        conn = next(
            (
                c
                for c in graph.connections
                if (c.source.node, c.source.port, c.target.node, c.target.port)
                == (source, source_port, target, target_port)
            ),
            None,
        )
        if conn is None:
            raise InvalidGraphError(
                f"No connection {source}.{source_port} -> {target}.{target_port}."
            )
        self._mutate(f"Disconnect {conn}", lambda: graph.disconnect(conn))

    def set_parameters(self, name: str, **values: Any) -> None:
        """Change parameters (validated by the component)."""
        keys = ", ".join(values)
        self._mutate(
            f"Set {name}.{keys}", lambda: self.project.graph.set_parameters(name, **values)
        )

    def set_simulation(
        self, *, seed: int | None = None, layout: SimulationLayout | None | str = "keep"
    ) -> None:
        """Change the seed and/or the global layout (``None`` removes it)."""
        from optobuild.core.rng import validate_seed

        def action() -> None:
            if seed is not None:
                self.project.seed = validate_seed(seed)
            if layout != "keep":
                self.project.layout = layout  # type: ignore[assignment]

        self._mutate("Simulation settings", action)

    @staticmethod
    def _check_name(name: str) -> None:
        if not _NAME_RE.match(name):
            raise InvalidGraphError(
                f"Invalid component name {name!r}.",
                hint="Start with a letter or '_' and use letters, digits, '_', '-', '.', ' '.",
            )

    # --- results ---------------------------------------------------------------------------
    def set_result(self, result: SimulationResult) -> None:
        """Store a finished run (called by the runner)."""
        self.last_result = result
        self.result_stale = False
        self._notify("result")

    def snapshot_project(self) -> Project:
        """Independent copy of the project (for running in a background thread)."""
        return project_from_dict(self._snapshot(), self.registry)


__all__ = ["ProjectDocument", "auto_positions"]
