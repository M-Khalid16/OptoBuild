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


def test_fso_example_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location("fso_example", EXAMPLES / "fso_link.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.main(["--trials", "5"])
    out = capsys.readouterr().out
    assert "link margin" in out and "analytic" in out


def test_photonic_circuits_example_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    import importlib.util

    path = EXAMPLES / "photonic_circuits.py"
    spec = importlib.util.spec_from_file_location("photonic_example", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.main()
    out = capsys.readouterr().out
    assert "closed form" in out and "Two coupled rings" in out and "TMM" in out


def test_lasers_example_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location("lasers_example", EXAMPLES / "lasers.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.main()
    out = capsys.readouterr().out
    assert "closed form" in out and "EDFA" in out and "DML link" in out


def test_ultrafast_example_runs_quick(capsys) -> None:  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location("ultrafast_example", EXAMPLES / "ultrafast.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.main(["--quick"])
    out = capsys.readouterr().out
    assert "photon-number change" in out and "converged after" in out
