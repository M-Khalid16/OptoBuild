"""Framework reference example: build, save, reload and run a small graph.

No optical physics is involved; every clean output value is known exactly
(v_n = 2 n + 10 V), so this example validates the engine itself.

Run:  python examples/reference_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from optobuild.cli.demos import reference_project
from optobuild.cli.main import print_result
from optobuild.engine import FeedForwardExecutor
from optobuild.persistence import load_project, save_project


def main() -> None:
    project = reference_project(seed=1234)
    path = save_project(project, Path(__file__).with_name("reference_pipeline.json"))
    reloaded = load_project(path)

    result = FeedForwardExecutor().run(reloaded.graph, seed=reloaded.seed)
    print_result(result)

    expected = 2.0 * np.arange(16) + 10.0
    exact = np.array_equal(result.result("clean", "samples"), expected)
    print(f"clean path equals 2 n + 10 exactly: {exact}")
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
