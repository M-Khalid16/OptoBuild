"""Laser and amplifier components (Phase 8): wiring of the validated models into the
signal flow (steady states, chirp reference, noise seeding, gain/ASE bookkeeping)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.components.lasers import (
    DirectlyModulatedLaser,
    ErbiumDopedFiberAmplifier,
    FiberRingLaser,
)
from optobuild.core.constants import PLANCK_CONSTANT, SPEED_OF_LIGHT
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.edfa import ring_laser_closed_form
from optobuild.signals import ElectricalQuantity, ElectricalSignal, OpticalSignal
from optobuild.solvers.edfa import amplify

GRID = TimeGrid(2048, 1 / 160e9)


def _drive(values: np.ndarray, quantity=ElectricalQuantity.VOLTAGE) -> ElectricalSignal:  # type: ignore[no-untyped-def]
    return ElectricalSignal(GRID, values, quantity, {"bit_rate": 10e9})


def test_dml_cw_equals_steady_state_and_reference_frequency(run_component) -> None:  # type: ignore[no-untyped-def]
    dml = DirectlyModulatedLaser("dml", {"bias_current": 0.07})
    out, ctx = run_component(dml, {"drive": _drive(np.zeros(2048))})
    laser = dml.laser()
    expected = laser.output_power(laser.steady_state(0.07)[1])
    assert out["out"].power() == pytest.approx(np.full(2048, expected), rel=1e-9)
    # emission at the bias current defines the reference: no phase ramp
    assert np.ptp(np.unwrap(out["out"].phase()[0])) < 1e-9
    assert out["out"].metadata["bit_rate"] == 10e9
    assert ctx.results["threshold_current_a"] == pytest.approx(laser.threshold_current)


def test_dml_voltage_and_current_drives_agree_and_chirp(run_component) -> None:  # type: ignore[no-untyped-def]
    t = GRID.time()
    i = 0.02 * np.sign(np.sin(2 * np.pi * 2.5e9 * t))
    dml = DirectlyModulatedLaser("dml")
    a = run_component(dml, {"drive": _drive(i, ElectricalQuantity.CURRENT)})
    b = run_component(dml, {"drive": _drive(i * 50.0)})
    np.testing.assert_allclose(a[0]["out"].field, b[0]["out"].field, rtol=1e-12)
    assert a[1].results["chirp_peak_to_peak_hz"] > 1e9  # transient chirp of a DML
    assert a[1].results["frequency_chirp_hz"].shape == (2048,)


def test_dml_noise_is_seeded_and_below_threshold_warns(run_component) -> None:  # type: ignore[no-untyped-def]
    noisy = DirectlyModulatedLaser("dml", {"noise": True})
    z = _drive(np.zeros(2048))
    f1 = run_component(noisy, {"drive": z}, seed=1)[0]["out"].field
    f2 = run_component(noisy, {"drive": z}, seed=1)[0]["out"].field
    f3 = run_component(noisy, {"drive": z}, seed=2)[0]["out"].field
    np.testing.assert_array_equal(f1, f2)
    assert not np.array_equal(f1, f3)
    low = DirectlyModulatedLaser("dml", {"bias_current": 0.01})
    _, ctx = run_component(low, {"drive": _drive(np.full(2048, -1.0))})
    codes = {d.code for d in ctx.diagnostics}
    assert {"laser.negative_current", "laser.below_threshold"} <= codes


def test_edfa_gain_and_ase_follow_the_amplifier_solution(run_component) -> None:  # type: ignore[no-untyped-def]
    p_in = 1e-5
    lam = 1550e-9
    sig = OpticalSignal(GRID, np.full(2048, math.sqrt(p_in), complex), SPEED_OF_LIGHT / lam)
    amp = ErbiumDopedFiberAmplifier("edfa", {"pump_power": 0.05})
    out, ctx = run_component(amp, {"in": sig}, seed=3)
    from optobuild.components.lasers import _edfa_model

    fiber, pump, beam = _edfa_model(amp.parameters)
    sol = amplify(fiber, (pump, beam), (0.05, p_in))
    assert ctx.results["gain_db"] == pytest.approx(10 * math.log10(sol.gains[1]))
    assert 3.0 < ctx.results["noise_figure_db"] < 6.0
    # output power = G P_in + ASE power in the simulated band (co-polarized, one pol)
    expected = sol.gains[1] * p_in + sol.ase_density[1] * GRID.sample_rate
    # std of the sample mean over N samples: signal x ASE beat 2 G P_in P_ase / N plus the
    # ASE x ASE term P_ase^2 / N (P_ase = S fs per polarization)
    p_ase = sol.ase_density[1] * GRID.sample_rate
    n = GRID.n_samples
    tol = 5 * math.sqrt((2 * sol.gains[1] * p_in * p_ase + p_ase**2) / n)
    assert out["out"].average_power() == pytest.approx(expected, abs=tol)
    quiet = ErbiumDopedFiberAmplifier("edfa", {"pump_power": 0.05, "ase_noise": False})
    q = run_component(quiet, {"in": sig})[0]["out"]
    assert q.average_power() == pytest.approx(sol.gains[1] * p_in, rel=1e-12)
    h_nu = PLANCK_CONSTANT * SPEED_OF_LIGHT / lam
    assert ctx.results["ase_psd_per_pol_w_per_hz"] > h_nu  # high-gain ASE


def test_edfa_wavelength_mismatch_diagnostic(run_component) -> None:  # type: ignore[no-untyped-def]
    sig = OpticalSignal(GRID, np.full(2048, 1e-3, complex), SPEED_OF_LIGHT / 1560e-9)
    _, ctx = run_component(ErbiumDopedFiberAmplifier("edfa"), {"in": sig}, seed=1)
    assert any(d.code == "edfa.wavelength_mismatch" for d in ctx.diagnostics)


def test_fiber_ring_laser_power_and_threshold(run_component) -> None:  # type: ignore[no-untyped-def]
    ring = FiberRingLaser("ring", {"pump_power": 0.05, "n_samples": 256})
    out, ctx = run_component(ring)
    from optobuild.components.lasers import _edfa_model

    p = dict(ring.parameters)
    p["signal_wavelength"] = p["wavelength"]
    fiber, pump, beam = _edfa_model(p)
    expected = ring_laser_closed_form(fiber, pump, beam, 0.05, 0.5, 0.8)
    assert out["out"].average_power() == pytest.approx(expected, rel=1e-7)
    assert ctx.results["output_power_w"] == pytest.approx(expected, rel=1e-7)
    dark = FiberRingLaser("ring", {"pump_power": 1e-4, "n_samples": 256})
    out, ctx = run_component(dark)
    assert out["out"].average_power() == 0.0
    assert any(d.code == "laser.below_threshold" for d in ctx.diagnostics)


def test_dml_link_demo_back_to_back_and_reproducible() -> None:
    from optobuild.cli.demos import dml_link_project
    from optobuild.persistence import dumps_project, loads_project, run_project

    p = dml_link_project(fiber_length_m=0.0)
    a = run_project(p)
    b = run_project(loads_project(dumps_project(p)))
    assert a.result("ber", "ber") == 0.0
    assert a.result("eye", "q_factor_decision_directed") > 10
    np.testing.assert_array_equal(a.result("dml", "frequency_chirp_hz"),
                                  b.result("dml", "frequency_chirp_hz"))  # fmt: skip
