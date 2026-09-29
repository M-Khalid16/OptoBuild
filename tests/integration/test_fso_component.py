"""FSO channel component inside graphs, Monte Carlo trials, demo and CLI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.cli.demos import fso_link_project
from optobuild.cli.main import main
from optobuild.components.analyzers import OpticalPowerMeter
from optobuild.components.fso import FSOChannel
from optobuild.components.sources import CWLaser
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.errors import InvalidParameterError
from optobuild.engine import FeedForwardExecutor, ResultCache
from optobuild.graph import SimulationGraph
from optobuild.persistence import run_project
from optobuild.sweeps.monte_carlo import monte_carlo_ber

FSO = {
    "distance": 1500.0,
    "beam_waist": 0.01,
    "divergence": 0.5e-3,
    "aperture_diameter": 0.1,
    "visibility": 4e3,
    "pointing_jitter": 100e-6,
    "turbulence": "gamma_gamma",
    "cn2": 5e-14,
    "beam_wander": True,
}


def _graph(**overrides) -> SimulationGraph:  # type: ignore[no-untyped-def]
    g = SimulationGraph()
    g.add(CWLaser("laser", {"power": 1e-2, "n_samples": 64}))
    g.add(FSOChannel("fso", {**FSO, **overrides}))
    g.add(OpticalPowerMeter("rx"))
    g.connect("laser", "out", "fso", "in")
    g.connect("fso", "out", "rx", "in")
    return g


def test_mean_mode_applies_the_analytic_mean_gain_and_delay() -> None:
    g = _graph(fading="mean")
    res = FeedForwardExecutor().run(g)
    model = g.node("fso").model(1550e-9)
    assert res.result("rx", "average_power_w") == pytest.approx(1e-2 * model.mean_gain(), rel=1e-12)
    assert res.signal("fso", "out").grid.t0 == pytest.approx(1500 / SPEED_OF_LIGHT)
    np.testing.assert_allclose(
        np.angle(res.signal("fso", "out").field),
        np.angle(res.signal("laser", "out").field),
        atol=1e-15,
    )


def test_trial_gains_follow_the_channel_statistics() -> None:
    """Channel gains drawn by the component over 3000 Monte Carlo trials vs the analytic
    mean and outage (5-sigma tolerances)."""
    g = _graph()
    ex = FeedForwardExecutor(ResultCache(max_entries=8))
    gains = np.array(
        [ex.run(g, seed=3, trial=k).result("fso", "channel_gain") for k in range(3000)]
    )
    model = g.node("fso").model(1550e-9)
    n = gains.size
    assert gains.mean() == pytest.approx(model.mean_gain(), abs=5 * gains.std() / math.sqrt(n))
    th = 0.5 * model.mean_gain()
    p = model.outage_probability(th)
    assert (gains < th).mean() == pytest.approx(p, abs=5 * math.sqrt(p * (1 - p) / n))
    assert len(set(gains.round(15))) == n  # independent realizations


def test_reproducible_and_seed_dependent() -> None:
    g = _graph()
    a = FeedForwardExecutor().run(g, seed=1).result("fso", "channel_gain")
    b = FeedForwardExecutor().run(g, seed=1).result("fso", "channel_gain")
    c = FeedForwardExecutor().run(g, seed=2).result("fso", "channel_gain")
    assert a == b and a != c


def test_parameter_and_physics_checks(run_component) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidParameterError, match="needs cn2"):
        FSOChannel("f", {"turbulence": "lognormal"})
    laser = run_component(CWLaser("l", {"n_samples": 64}))[0]["out"]
    with pytest.raises(InvalidParameterError, match="diffraction limit"):
        run_component(FSOChannel("f", {"divergence": 1e-6, "beam_waist": 1e-3}), {"in": laser})
    _, ctx = run_component(
        FSOChannel("f", {"turbulence": "lognormal", "cn2": 1e-13, "distance": 5e3}), {"in": laser}
    )
    assert "fso.lognormal_strong_turbulence" in [d.code for d in ctx.diagnostics]
    long_window = run_component(CWLaser("l", {"n_samples": 64, "sample_rate": 1e5}))[0]["out"]
    _, ctx = run_component(FSOChannel("f", FSO), {"in": long_window})
    assert "fso.quasi_static" in [d.code for d in ctx.diagnostics]


def test_fso_link_demo_and_monte_carlo_ber() -> None:
    p = fso_link_project(prbs_order=7)
    res = run_project(p)
    assert res.result("ber", "n_bits") == 127
    assert res.result("fso", "rytov_variance") > 0.3  # moderate turbulence
    mc = monte_carlo_ber(p, 5)
    assert mc.n_bits == 5 * 127 and len(set(mc.errors_per_trial)) >= 1


def test_cli_fso_budget(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "fso-budget",
                "--distance",
                "2 km",
                "--visibility",
                "5 km",
                "--aperture",
                "10 cm",
                "--divergence",
                "0.5 mrad",
                "--turbulence",
                "gamma_gamma",
                "--cn2",
                "1e-14",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "geometric + static pointing" in out and "-23.02 dB" in out
    assert "outage probability" in out
    assert main(["fso-budget", "--distance", "2 parsecs"]) == 2
    assert "km" in capsys.readouterr().err
