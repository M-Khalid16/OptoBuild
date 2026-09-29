"""Shared pytest configuration.

Tests are auto-marked by directory (unit / integration / validation /
regression), so ``pytest -m validation`` selects all analytical checks.
"""

from __future__ import annotations

import os

# GUI tests run headless; must be set before Qt is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

_LEVELS = ("unit", "integration", "validation", "regression")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = Path(str(item.fspath)).parts
        for level in _LEVELS:
            if level in parts:
                item.add_marker(getattr(pytest.mark, level))


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator for tests that need random *inputs* (not physics)."""
    return np.random.default_rng(20260927)


class _Ctx:
    """Minimal RunContext for exercising a single component outside the engine."""

    def __init__(self, seed: int = 0, name: str = "test") -> None:
        from optobuild.core.log import component_logger
        from optobuild.core.rng import component_generator

        self.rng = component_generator(seed, name)
        self.logger = component_logger(name)
        self.layout = None
        self.results: dict = {}
        self.diagnostics: list = []

    def check_cancelled(self) -> None:
        pass

    def report_progress(self, fraction: float, message: str = "") -> None:
        pass

    def record(self, key: str, value: object) -> None:
        self.results[key] = value

    def warn(self, diagnostic: object) -> None:
        self.diagnostics.append(diagnostic)


@pytest.fixture
def run_component():
    """Run one component on given inputs; returns (outputs, context)."""

    def _run(component, inputs=None, seed: int = 0, layout=None):  # type: ignore[no-untyped-def]
        ctx = _Ctx(seed, component.name)
        ctx.layout = layout
        return component.run(inputs or {}, ctx), ctx

    return _run
