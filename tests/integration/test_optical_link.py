"""Integration tests of the complete Phase 2 reference link."""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.cli.demos import optical_link_project
from optobuild.cli.main import main
from optobuild.core.errors import InvalidGraphError, PortTypeMismatchError
from optobuild.core.units import db_per_km_to_per_m
from optobuild.engine import FeedForwardExecutor, ResultCache
from optobuild.persistence import dumps_project, loads_project

ARRAY_RESULTS = [
    ("eye", "traces"),
    ("tx_spectrum", "psd_w_per_hz"),
    ("decision", "decision_samples"),
]


@pytest.fixture(scope="module")
def link_result():  # type: ignore[no-untyped-def]
    project = optical_link_project(seed=7)
    return project, FeedForwardExecutor().run(project.graph, seed=project.seed)


def test_link_runs_and_measurements_are_consistent(link_result) -> None:  # type: ignore[no-untyped-def]
    project, res = link_result
    tx = res.result("tx_power", "average_power_w")
    rx = res.result("rx_power", "average_power_w")
    alpha = db_per_km_to_per_m(0.2)
    assert rx / tx == pytest.approx(np.exp(-alpha * 50e3), rel=1e-12)
    assert res.result("tx_spectrum", "total_power_w") == pytest.approx(tx, rel=1e-12)
    assert res.result("ber", "n_bits") == 2047
    assert res.result("eye", "traces").shape == (2047, 33)
    # the pattern delivered to the BER analyzer is the transmitted PRBS
    assert res.signal("prbs", "out").metadata["pattern"] == "PRBS11"
    # PIN output has the bit timing needed downstream
    assert res.signal("filter", "out").metadata["samples_per_bit"] == 16


def test_noiseless_back_to_back_link_is_error_free() -> None:
    g = optical_link_project(seed=1, fiber_length_m=0.0).graph
    g.set_parameters("pin", shot_noise=False, thermal_noise=False, dark_current=0.0)
    res = FeedForwardExecutor().run(g, seed=1)
    assert res.result("ber", "n_errors") == 0
    assert res.result("ber", "alignment_shift_bits") == 0


def test_same_seed_reproduces_everything() -> None:
    a = FeedForwardExecutor().run(optical_link_project(seed=5).graph, seed=5)
    b = FeedForwardExecutor().run(optical_link_project(seed=5).graph, seed=5)
    assert a.result("ber", "n_errors") == b.result("ber", "n_errors")
    for node, key in ARRAY_RESULTS:
        np.testing.assert_array_equal(a.result(node, key), b.result(node, key))
    np.testing.assert_array_equal(a.signal("pin", "out").samples, b.signal("pin", "out").samples)


def test_different_seed_changes_only_noise() -> None:
    a = FeedForwardExecutor().run(optical_link_project().graph, seed=5)
    b = FeedForwardExecutor().run(optical_link_project().graph, seed=6)
    np.testing.assert_array_equal(a.signal("fiber", "out").field, b.signal("fiber", "out").field)
    assert not np.array_equal(a.signal("pin", "out").samples, b.signal("pin", "out").samples)


def test_project_round_trip_reproduces_results() -> None:
    project = optical_link_project(seed=11)
    loaded = loads_project(dumps_project(project))
    a = FeedForwardExecutor().run(project.graph, seed=11)
    b = FeedForwardExecutor().run(loaded.graph, seed=11)
    for node, key in ARRAY_RESULTS:
        np.testing.assert_array_equal(a.result(node, key), b.result(node, key))


def test_changing_the_fiber_reuses_transmitter_cache() -> None:
    cache = ResultCache()
    ex = FeedForwardExecutor(cache)
    g = optical_link_project().graph
    ex.run(g, seed=0)
    g.set_parameters("fiber", length=60e3)
    res = ex.run(g, seed=0)
    hits = {n: r.cache_hit for n, r in res.nodes.items()}
    assert all(hits[n] for n in ("prbs", "nrz", "laser", "mzm", "tx_power", "tx_spectrum"))
    assert not any(hits[n] for n in ("fiber", "pin", "filter", "decision", "ber", "eye"))


def test_inconsistent_sampling_is_reported() -> None:
    from optobuild.core.errors import ComponentExecutionError

    g = optical_link_project().graph
    g.set_parameters("laser", n_samples=1000)
    with pytest.raises(ComponentExecutionError, match="Incompatible sampling"):
        FeedForwardExecutor().run(g)


def test_wrong_wiring_is_rejected() -> None:
    g = optical_link_project().graph
    with pytest.raises(PortTypeMismatchError):
        g.connect("pin", "out", "rx_power", "in")
    with pytest.raises(InvalidGraphError, match="already connected"):
        g.connect("mzm", "optical_out", "pin", "in")
    from optobuild.components.fiber import LinearFiber

    g.add(LinearFiber("second_fiber"))
    with pytest.raises(InvalidGraphError, match="duplicate optical power"):
        g.connect("mzm", "optical_out", "second_fiber", "in")


def test_cli_optical_link_demo(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "optical_link", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "ber.n_bits = 2047" in out and "rx_power.average_power_dbm" in out
