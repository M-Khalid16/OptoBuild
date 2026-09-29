"""Ultrafast components (Phase 9): GNLSE fiber, mode-locked fiber laser, autocorrelator.

The physics is validated in tests/validation/test_gnlse.py, test_cavity.py and
test_pulses.py; here: wiring, agreement with the SSFM component in the pure-NLSE
limit, and physical plausibility of the mode-locked state (checks, not exact
references: single soliton-like pulse near the sech^2 transform limit).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.analysis.pulses import TRANSFORM_LIMITED_TBP
from optobuild.components.nonlinear import NonlinearFiber
from optobuild.components.ultrafast import (
    Autocorrelator,
    ModeLockedFiberLaser,
    UltrafastFiber,
    count_pulses,
)
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.errors import SignalTypeError
from optobuild.core.units import dispersion_slope_to_beta3
from optobuild.numerics.grid import TimeGrid
from optobuild.signals import OpticalSignal

F0 = SPEED_OF_LIGHT / 1550e-9
GRID = TimeGrid(1024, 20e-15, -512 * 20e-15)
T = GRID.time()


def _pulse(p0: float, t0: float = 300e-15) -> OpticalSignal:
    return OpticalSignal(GRID, (math.sqrt(p0) / np.cosh(T / t0)).astype(complex), F0)


def test_ultrafast_fiber_agrees_with_ssfm_in_the_nlse_limit(run_component) -> None:  # type: ignore[no-untyped-def]
    """N = 2 soliton, 1 dispersion length; SSFM (order 2, max phase 5e-4 rad/step, error
    ~2e-5 of the peak by its O(h^2) scaling from tests/validation/test_ssfm.py) vs RK4IP
    (tolerance 1e-10): tolerance 1e-4 of the peak."""
    beta2, gamma, t0 = -21.7e-27, 1.3e-3, 300e-15
    p1 = abs(beta2) / (gamma * t0**2)
    ld = t0**2 / abs(beta2)
    sig = _pulse(4 * p1, t0)
    slow = NonlinearFiber("n", {"length": ld, "attenuation": 0.0, "dispersion": 17e-6,
                                "gamma": gamma, "max_phase": 5e-4})  # fmt: skip
    beta2_ssfm = run_component(slow, {"in": sig})[1].results["beta2_s2_per_m"]
    # NonlinearFiber derives beta3 from D and the slope (non-zero even for S = 0)
    beta3 = dispersion_slope_to_beta3(17e-6, 0.0, 1550e-9)
    params = {
        "length": ld,
        "attenuation": 0.0,
        "beta2": beta2_ssfm,
        "beta3": beta3,
        "gamma": gamma,
        "raman": False,
        "self_steepening": False,
        "tolerance": 1e-10,
    }
    fast = UltrafastFiber("u", params)
    a = run_component(fast, {"in": sig})[0]["out"]
    b = run_component(slow, {"in": sig})[0]["out"]
    np.testing.assert_allclose(a.field, b.field, atol=1e-4 * math.sqrt(4 * p1))
    assert a.grid.t0 == pytest.approx(b.grid.t0)


def test_ultrafast_fiber_records_and_rejects_dual_polarization(run_component) -> None:  # type: ignore[no-untyped-def]
    fiber = UltrafastFiber("u", {"length": 0.5, "attenuation": 0.0})  # lossless: photons kept
    out, ctx = run_component(fiber, {"in": _pulse(100.0)})
    assert abs(ctx.results["photon_number_change"]) < 1e-5
    assert ctx.results["n_steps"] >= 1
    dual = OpticalSignal(GRID, np.ones((2, 1024), complex), F0)
    with pytest.raises(SignalTypeError, match="scalar"):
        run_component(fiber, {"in": dual})


def test_mode_locked_laser_converges_to_a_single_soliton_like_pulse(run_component) -> None:  # type: ignore[no-untyped-def]
    laser = ModeLockedFiberLaser("ml")
    out, ctx = run_component(laser, seed=5)
    r = ctx.results
    assert r["converged"] and r["n_pulses"] == 1
    assert 100e-15 < r["pulse_fwhm_s"] < 1e-12
    # near the sech^2 transform limit (weak chirp from the lumped elements)
    assert TRANSFORM_LIMITED_TBP["sech2"] < r["time_bandwidth_product"] < 1.3 * 0.3148
    assert r["pulse_energy_j"] == pytest.approx(out["out"].average_power() *
                                                out["out"].grid.duration, rel=1e-12)  # fmt: skip
    again = run_component(laser, seed=5)[0]["out"]
    np.testing.assert_array_equal(again.field, out["out"].field)


def test_mode_locked_laser_diagnoses_multipulsing_and_non_convergence(run_component) -> None:  # type: ignore[no-untyped-def]
    strong = ModeLockedFiberLaser("ml", {"small_signal_gain": math.exp(3.0)})
    _, ctx = run_component(strong, seed=5)
    assert ctx.results["n_pulses"] >= 2
    assert any(d.code == "cavity.multiple_pulses" for d in ctx.diagnostics)
    short = ModeLockedFiberLaser("ml", {"max_round_trips": 3})
    _, ctx = run_component(short, seed=5)
    assert not ctx.results["converged"]
    assert any(d.code == "cavity.not_converged" for d in ctx.diagnostics)


def test_autocorrelator_deconvolves_a_sech_pulse(run_component) -> None:  # type: ignore[no-untyped-def]
    t0 = 300e-15
    _, ctx = run_component(Autocorrelator("ac"), {"in": _pulse(1.0, t0)})
    assert ctx.results["deconvolved_fwhm_s"] == pytest.approx(2 * math.acosh(math.sqrt(2)) * t0,
                                                              rel=1e-3)  # fmt: skip


def test_count_pulses() -> None:
    x = np.zeros(100)
    x[[10, 11, 50, 98, 99, 0]] = 1.0
    assert count_pulses(x) == 3  # lobes 10-11, 50 and 98-99-0 (wrapping around)
    assert count_pulses(np.ones(10)) == 1


def test_supercontinuum_demo_short_fiber_conserves_photons() -> None:
    """1 cm of the Dudley benchmark on a 4096-point grid: broadening with photon number
    conserved to the solver tolerance while energy is lost to the Raman shift."""
    from optobuild.cli.demos import supercontinuum_project
    from optobuild.persistence import run_project

    r = run_project(supercontinuum_project(length_m=0.01, n_samples=2**12))
    fiber = r.nodes["pcf"].results
    assert abs(fiber["photon_number_change"]) < 1e-6
    assert fiber["energy_change"] < 0
    rbw_in = r.result("input_spectrum", "total_power_w")
    assert r.result("output_spectrum", "total_power_w") < rbw_in


def test_mode_locked_laser_demo() -> None:
    from optobuild.cli.demos import mode_locked_laser_project
    from optobuild.persistence import run_project

    r = run_project(mode_locked_laser_project())
    assert r.result("laser", "converged")
    assert r.result("autocorrelator", "deconvolved_fwhm_s") == pytest.approx(
        r.result("laser", "pulse_fwhm_s"),
        rel=0.05,  # the pulse is close to sech^2
    )
