"""Shared pytest configuration.

Tests are auto-marked by directory (unit / integration / validation /
regression), so ``pytest -m validation`` selects all analytical checks.
"""

from __future__ import annotations

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
