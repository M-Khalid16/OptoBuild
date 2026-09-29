"""HDF5 result storage round trips (skipped if h5py is not installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from optobuild.cli.demos import optical_link_project, reference_project  # noqa: E402
from optobuild.core.errors import ProjectFormatError  # noqa: E402
from optobuild.persistence import run_project  # noqa: E402
from optobuild.persistence.results import (  # noqa: E402
    load_results,
    results_summary,
    save_results,
)
from optobuild.signals import ElectricalQuantity, OpticalSignal  # noqa: E402


@pytest.fixture(scope="module")
def link():  # type: ignore[no-untyped-def]
    p = optical_link_project(seed=4, prbs_order=7)
    return p, run_project(p)


def test_round_trip_is_lossless(tmp_path: Path, link) -> None:  # type: ignore[no-untyped-def]
    project, res = link
    stored = load_results(save_results(res, tmp_path / "r.h5", project=project))
    assert stored.order == res.order and stored.seed == 4 and stored.trial is None
    assert stored.layout == project.layout
    for node, port in [
        ("laser", "out"),
        ("fiber", "out"),
        ("pin", "out"),
        ("prbs", "out"),
        ("decision", "bits"),
    ]:
        a, b = res.signal(node, port), stored.signal(node, port)
        assert type(a) is type(b) and dict(a.metadata) == dict(b.metadata)
        if isinstance(a, OpticalSignal):
            np.testing.assert_array_equal(a.field, b.field)
            assert a.grid == b.grid and a.center_frequency == b.center_frequency
        elif hasattr(a, "samples"):
            np.testing.assert_array_equal(a.samples, b.samples)
            assert a.grid == b.grid and a.quantity is b.quantity
        else:
            np.testing.assert_array_equal(a.bits, b.bits)
            assert a.bit_rate == b.bit_rate
    assert stored.signal("pin", "out").quantity is ElectricalQuantity.CURRENT
    for node in res.order:
        for key, value in res.nodes[node].results.items():
            got = stored.result(node, key)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(got, value)
            else:
                assert got == value and type(got) is type(value), (node, key)
        assert stored.node_diagnostics[node] == res.nodes[node].diagnostics


def test_embedded_project_reproduces_the_run(tmp_path: Path, link) -> None:  # type: ignore[no-untyped-def]
    project, res = link
    stored = load_results(save_results(res, tmp_path / "r.h5", project=project))
    rerun = run_project(stored.project())
    np.testing.assert_array_equal(rerun.result("eye", "traces"), stored.result("eye", "traces"))
    assert rerun.result("ber", "n_errors") == stored.result("ber", "n_errors")


def test_results_only_and_summary(tmp_path: Path, link) -> None:  # type: ignore[no-untyped-def]
    _, res = link
    stored = load_results(save_results(res, tmp_path / "r.h5", include_signals=False))
    assert stored.signals == {}
    with pytest.raises(KeyError):
        stored.signal("laser", "out")
    with pytest.raises(ProjectFormatError, match="does not embed"):
        stored.project()
    summary = results_summary(stored)
    assert summary["ber"]["n_bits"] == 127
    assert "traces" not in summary["eye"]
    assert results_summary(res)["ber"] == summary["ber"]


def test_unusual_component_names(tmp_path: Path) -> None:
    from optobuild.components.reference import RampSource, Recorder
    from optobuild.engine import FeedForwardExecutor
    from optobuild.graph import SimulationGraph

    g = SimulationGraph()
    g.add(RampSource("a/b c"))
    g.add(Recorder("rec.1"))
    g.connect("a/b c", "out", "rec.1", "in")
    stored = load_results(save_results(FeedForwardExecutor().run(g), tmp_path / "r.h5"))
    np.testing.assert_array_equal(stored.result("rec.1", "samples"), np.arange(16.0))


def test_trial_and_reference_results(tmp_path: Path) -> None:
    p = reference_project()
    res = run_project(p, trial=3)
    stored = load_results(save_results(res, tmp_path / "r.h5"))
    assert stored.trial == 3 and stored.layout is None


def test_not_a_results_file(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "something-else"
    with pytest.raises(ProjectFormatError, match="not an OptoBuild results file"):
        load_results(path)


def test_cli_saves_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from optobuild.cli.main import main

    out = tmp_path / "ref.h5"
    assert main(["demo", "reference", "--save-results", str(out)]) == 0
    assert "saved results to" in capsys.readouterr().out
    stored = load_results(out)
    assert stored.result("clean", "mean") == 25.0
    assert stored.project().seed == 1234
