"""Enforce the layered architecture of docs/architecture.md by static import analysis.

A module may import optobuild modules from its own layer or lower layers only.
The GUI additionally may not import physics, solvers or numerics (no physics in
the GUI). Imports inside ``if TYPE_CHECKING:`` blocks are still counted, to keep
the rule simple and strict.
"""

from __future__ import annotations

import ast
from pathlib import Path

import optobuild

LAYERS: dict[str, int] = {
    "core": 0,
    "numerics": 1,
    "signals": 2,
    "physics": 3,
    "analysis": 3,
    "solvers": 4,
    "components": 5,
    "graph": 6,
    "engine": 7,
    "persistence": 8,
    "sweeps": 8,
    "optimization": 8,
    "reporting": 8,
    "plugins": 9,
    "cli": 9,
    "gui": 9,
}
EXTRA_FORBIDDEN: dict[str, set[str]] = {"gui": {"physics", "solvers", "numerics"}}
ALLOWED_EXCEPTIONS: dict[str, set[str]] = {"gui": {"optobuild.numerics.layout"}}
"""Modules exempt from EXTRA_FORBIDDEN: the layout is a data definition the GUI
must edit (ADR-0013); it performs no numerical computation."""
GUI_TOOLKITS = ("PySide6", "PyQt5", "PyQt6", "pyqtgraph", "shiboken6")

PKG_ROOT = Path(optobuild.__file__).parent


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            raise AssertionError(f"{path}: use absolute imports (found relative import)")
        found.update(names)
    return found


def _imported_subsystems(path: Path, owner: str = "") -> set[str]:
    found: set[str] = set()
    for name in _imported_modules(path):
        parts = name.split(".")
        if parts[0] == "optobuild" and len(parts) > 1:
            if any(
                name == m or name.startswith(m + ".") for m in ALLOWED_EXCEPTIONS.get(owner, ())
            ):
                continue
            found.add(parts[1])
    return found


def test_every_subsystem_has_a_layer() -> None:
    dirs = {p.name for p in PKG_ROOT.iterdir() if (p / "__init__.py").exists()}
    assert dirs == set(LAYERS), "Add new subsystems to LAYERS and docs/architecture.md"


def test_no_upward_imports() -> None:
    violations = []
    for path in PKG_ROOT.rglob("*.py"):
        rel = path.relative_to(PKG_ROOT)
        if len(rel.parts) < 2:
            continue  # top-level optobuild/__init__.py
        owner = rel.parts[0]
        for dep in _imported_subsystems(path, owner):
            if dep not in LAYERS:
                violations.append(f"{rel}: imports unknown subsystem '{dep}'")
            elif LAYERS[dep] > LAYERS[owner]:
                violations.append(f"{rel}: layer '{owner}' imports higher layer '{dep}'")
            elif dep in EXTRA_FORBIDDEN.get(owner, set()):
                violations.append(f"{rel}: '{owner}' must not import '{dep}'")
    assert not violations, "\n".join(violations)


def test_only_the_gui_imports_gui_toolkits() -> None:
    """The engine must run headless: Qt/pyqtgraph are imported only under optobuild.gui."""
    offenders = []
    for path in PKG_ROOT.rglob("*.py"):
        rel = path.relative_to(PKG_ROOT)
        if rel.parts[0] == "gui":
            continue
        for mod in _imported_modules(path):
            if mod.split(".")[0] in GUI_TOOLKITS:
                offenders.append(f"{rel}: imports {mod}")
    assert not offenders, "\n".join(offenders)


def test_gui_models_are_qt_free() -> None:
    """Qt-free GUI modules (forms, document, plotdata) must not import Qt."""
    for name in ("forms.py", "document.py", "plotdata.py"):
        mods = _imported_modules(PKG_ROOT / "gui" / name)
        assert not [m for m in mods if m.split(".")[0] in GUI_TOOLKITS], name
