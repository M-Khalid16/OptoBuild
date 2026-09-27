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

PKG_ROOT = Path(optobuild.__file__).parent


def _imported_subsystems(path: Path) -> set[str]:
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
        for name in names:
            parts = name.split(".")
            if parts[0] == "optobuild" and len(parts) > 1:
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
        for dep in _imported_subsystems(path):
            if dep not in LAYERS:
                violations.append(f"{rel}: imports unknown subsystem '{dep}'")
            elif LAYERS[dep] > LAYERS[owner]:
                violations.append(f"{rel}: layer '{owner}' imports higher layer '{dep}'")
            elif dep in EXTRA_FORBIDDEN.get(owner, set()):
                violations.append(f"{rel}: '{owner}' must not import '{dep}'")
    assert not violations, "\n".join(violations)
