"""The checked-in example project files must load and reproduce the demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from optobuild.cli.demos import reference_project
from optobuild.persistence import dumps_project, load_project, run_project

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_reference_example_file_matches_demo() -> None:
    loaded = load_project(EXAMPLES / "reference_pipeline.json")
    assert dumps_project(loaded) == dumps_project(reference_project(seed=1234))
    res = run_project(loaded)
    np.testing.assert_array_equal(res.result("clean", "samples"), 2.0 * np.arange(16) + 10)


def test_optical_link_example_file_matches_demo() -> None:
    from optobuild.cli.demos import optical_link_project

    loaded = load_project(EXAMPLES / "optical_link.json")
    assert dumps_project(loaded) == dumps_project(optical_link_project(seed=2026))


def test_optical_link_example_script_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "optical_link_example", EXAMPLES / "optical_link.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.main(["--no-save", "--length-km", "0"])
    out = capsys.readouterr().out
    assert "counted errors" in out and "TX power" in out
