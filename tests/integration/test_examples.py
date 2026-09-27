"""The checked-in example project files must load and reproduce the demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from optobuild.cli.demos import reference_project
from optobuild.engine import FeedForwardExecutor
from optobuild.persistence import dumps_project, load_project

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_reference_example_file_matches_demo() -> None:
    loaded = load_project(EXAMPLES / "reference_pipeline.json")
    assert dumps_project(loaded) == dumps_project(reference_project(seed=1234))
    res = FeedForwardExecutor().run(loaded.graph, seed=loaded.seed)
    np.testing.assert_array_equal(res.result("clean", "samples"), 2.0 * np.arange(16) + 10)
