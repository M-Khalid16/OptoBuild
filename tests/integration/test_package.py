"""Package-level smoke tests: every subsystem imports cleanly."""

from __future__ import annotations

import importlib
import pkgutil

import optobuild

SUBSYSTEMS = (
    "core",
    "numerics",
    "signals",
    "physics",
    "analysis",
    "solvers",
    "components",
    "graph",
    "engine",
    "persistence",
    "sweeps",
    "optimization",
    "reporting",
    "plugins",
    "cli",
    "gui",
)


def test_version() -> None:
    assert optobuild.__version__ == "0.1.0"


def test_all_subsystems_present_and_documented() -> None:
    found = {m.name for m in pkgutil.iter_modules(optobuild.__path__)}
    assert set(SUBSYSTEMS) <= found
    for name in SUBSYSTEMS:
        mod = importlib.import_module(f"optobuild.{name}")
        assert mod.__doc__ and "Layer" in mod.__doc__, name


def test_every_module_imports() -> None:
    for info in pkgutil.walk_packages(optobuild.__path__, prefix="optobuild."):
        if info.name.endswith(".__main__"):
            continue  # entry-point script: importing it runs the CLI
        importlib.import_module(info.name)
