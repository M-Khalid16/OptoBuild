"""Optimizer validation against analytic optima (ADR-0021).

* reference pipeline, clean output v_n = g n + 10 (n = 0..15): RMS minimized at
  g* = -10 sum n / sum n^2 (least squares); tolerance = the optimizer's xatol
  (1e-6 of the bracket = 1e-5);
* two variables with an active bound: v_n = g n + o, g in [0.5, 3]: optimum g = 0.5,
  o = -0.5 mean(n) = -3.75, RMS = 0.5 std(n) (Nelder-Mead xatol 1e-6 of the span);
* add-drop ring detuned by 50 GHz: the drop power of the symmetric NRZ spectrum is
  maximal when the resonance sits on the carrier, i.e. phase_shift = 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.cli.demos import reference_project, ring_filter_project
from optobuild.core.errors import OptoBuildError
from optobuild.optimization.optimizer import Objective, Variable, optimize
from optobuild.persistence import dumps_project

N = np.arange(16)


def test_single_variable_least_squares_optimum() -> None:
    project = reference_project()
    before = dumps_project(project)
    res = optimize(project, [Variable("gain", "gain", -5.0, 5.0)], Objective("clean", "rms", "min"))
    assert res.x["gain.gain"] == pytest.approx(-10 * N.sum() / (N**2).sum(), abs=1e-5)
    assert res.success and res.n_evaluations == len(res.history)
    assert dumps_project(project) == before  # the input project is untouched
    assert res.project.graph.node("gain").parameters["gain"] == res.x["gain.gain"]


def test_two_variables_with_an_active_bound() -> None:
    res = optimize(
        reference_project(),
        [Variable("gain", "gain", 0.5, 3.0, 1.0), Variable("dc", "offset", -5.0, 5.0, 0.0)],
        Objective("clean", "rms", "min"),
    )
    assert res.x["gain.gain"] == pytest.approx(0.5, abs=1e-5)
    assert res.x["dc.offset"] == pytest.approx(-3.75, abs=1e-4)
    assert res.value == pytest.approx(0.5 * np.std(N), rel=1e-8)


def test_ring_tuning_finds_the_carrier_resonance() -> None:
    res = optimize(
        ring_filter_project(detuning_hz=50e9),
        [Variable("ring", "phase_shift", -1.0, 1.0, 0.3)],
        Objective("power_drop", "average_power_w", "max"),
    )
    assert res.x["ring.phase_shift"] == pytest.approx(0.0, abs=1e-5)
    detuned = ring_filter_project(detuning_hz=50e9)
    from optobuild.persistence import run_project

    assert res.value > 5 * run_project(detuned).result("power_drop", "average_power_w")


def test_noisy_objective_uses_common_random_numbers() -> None:
    """With fixed trials the averaged objective is deterministic: repeated optimizations
    return identical results."""
    obj = Objective("noisy", "rms", "min", trials=(0, 1, 2))
    var = [Variable("gain", "gain", -5.0, 5.0)]
    a = optimize(reference_project(), var, obj)
    b = optimize(reference_project(), var, obj)
    assert a.x == b.x and a.value == b.value
    assert a.x["gain.gain"] == pytest.approx(-10 * N.sum() / (N**2).sum(), abs=0.05)


def test_optimizer_input_errors() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        optimize(
            reference_project(), [Variable("gain", "gain", 1.0, 1.0)], Objective("clean", "rms")
        )
    with pytest.raises(ValueError, match="outside"):
        optimize(reference_project(), [Variable("gain", "gain", 0.0, 1.0, 2.0)],
                 Objective("clean", "rms"))  # fmt: skip
    with pytest.raises(ValueError, match="sense"):
        Objective("clean", "rms", "biggest")
    with pytest.raises(KeyError):
        optimize(reference_project(), [Variable("gain", "gain", 0.0, 1.0)], Objective("x", "y"))
    with pytest.raises(ValueError, match="single variable"):
        two = [Variable("gain", "gain", 0, 1), Variable("dc", "offset", 0, 1)]
        optimize(reference_project(), two, Objective("clean", "rms"), method="bounded")
    # a parameter outside the component's schema is rejected by the component
    with pytest.raises(OptoBuildError):
        optimize(reference_project(), [Variable("noise", "sigma", -1.0, -0.5)],
                 Objective("noisy", "rms"))  # fmt: skip
