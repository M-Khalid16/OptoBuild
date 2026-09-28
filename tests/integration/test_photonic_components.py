"""Photonic-circuit components in signal-flow simulations (Phase 7).

Checks: the applied transfer equals the physics closed form at the FFT bins;
lossless devices conserve energy between their outputs (exact up to
round-off, 1e-10 relative); superposition of the second input; diagnostics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.components.photonic import (
    AddDropRing,
    AllPassRing,
    BraggGrating,
    MachZehnderInterferometer,
)
from optobuild.core.constants import SPEED_OF_LIGHT as C
from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.integrated_optics import all_pass_ring, waveguide_transmission
from optobuild.signals import OpticalSignal

F0 = C / 1550e-9
GRID = TimeGrid(4096, 1 / 400e9)  # 400 GHz band, df = 98 MHz


def _pulse(n_pol: int = 1) -> OpticalSignal:
    t = GRID.time() - GRID.duration / 2
    a = np.exp(-(t**2) / (2 * (5e-12) ** 2)).astype(complex)
    return OpticalSignal(GRID, np.vstack([a] * n_pol), F0)


def _energy(sig: OpticalSignal) -> float:
    return float(np.sum(np.abs(sig.field) ** 2) * sig.grid.dt)


def test_all_pass_ring_applies_the_closed_form(run_component) -> None:  # type: ignore[no-untyped-def]
    ring = AllPassRing("r", {"radius": 20e-6, "power_coupling": 0.1})
    cw = OpticalSignal(GRID, np.ones(GRID.n_samples, complex), F0)
    out, ctx = run_component(ring, {"in": cw})
    p = ring.parameters
    h = waveguide_transmission(
        np.array([F0]), 2 * math.pi * 20e-6, p["n_eff"], p["n_group"], 1550e-9, p["loss"]
    )
    expected = abs(all_pass_ring(h, 0.1)[0]) ** 2
    assert out["through"].average_power() == pytest.approx(expected, rel=1e-12)
    assert ctx.results["fsr_hz"] == pytest.approx(C / (4.2 * 2 * math.pi * 20e-6))
    assert ctx.results["transfer_through"].shape == (GRID.n_samples,)
    assert np.all(np.diff(ctx.results["transfer_frequency_hz"]) > 0)


def test_lossless_add_drop_conserves_energy_and_add_superposes(run_component) -> None:  # type: ignore[no-untyped-def]
    ring = AddDropRing("r", {"loss": 0.0, "power_coupling_in": 0.1, "power_coupling_drop": 0.2})
    sig = _pulse(n_pol=2)
    out, ctx = run_component(ring, {"in": sig})
    total = _energy(out["through"]) + _energy(out["drop"])
    assert total == pytest.approx(_energy(sig), rel=1e-10)
    assert ctx.results["drop_peak_transmission"] == pytest.approx(
        0.1 * 0.2 / (1 - math.sqrt(0.9 * 0.8)) ** 2
    )
    both, _ = run_component(ring, {"in": sig, "add": sig})
    add_only_drop = both["drop"].field - out["drop"].field
    # lossless 2x2 (in, add) -> (through, drop) is unitary: total output energy = 2 E_in
    assert _energy(both["through"]) + _energy(both["drop"]) == pytest.approx(
        2 * _energy(sig), rel=1e-10
    )
    assert np.any(add_only_drop != 0)
    other = OpticalSignal(GRID, np.ones(GRID.n_samples), 1.9e14)
    with pytest.raises(SamplingError, match="same carrier"):
        run_component(ring, {"in": _pulse(), "add": other})


def test_mzi_lossless_split_and_second_input(run_component) -> None:  # type: ignore[no-untyped-def]
    mzi = MachZehnderInterferometer("m", {"loss": 0.0, "delta_length": 200e-6})
    sig = _pulse()
    out, ctx = run_component(mzi, {"in1": sig})
    assert _energy(out["bar"]) + _energy(out["cross"]) == pytest.approx(_energy(sig), rel=1e-10)
    assert ctx.results["fsr_hz"] == pytest.approx(C / (4.2 * 200e-6))
    both, _ = run_component(mzi, {"in1": sig, "in2": sig})
    assert _energy(both["bar"]) + _energy(both["cross"]) == pytest.approx(
        2 * _energy(sig), rel=1e-10
    )


def test_bragg_grating_splits_energy_and_reports_figures(run_component) -> None:  # type: ignore[no-untyped-def]
    g = BraggGrating("g", {"length": 500e-6, "coupling": 5e3})
    sig = _pulse()
    out, ctx = run_component(g, {"in": sig})
    assert _energy(out["transmitted"]) + _energy(out["reflected"]) == pytest.approx(
        _energy(sig), rel=1e-10
    )
    assert ctx.results["peak_reflectance"] == pytest.approx(math.tanh(2.5) ** 2)
    assert ctx.results["period_m"] == pytest.approx(1550e-9 / 4.8)
    refl = ctx.results["transfer_reflected"]
    assert refl.max() == pytest.approx(math.tanh(2.5) ** 2, rel=1e-3)  # df-limited sampling


def test_unresolved_resonance_is_diagnosed(run_component) -> None:  # type: ignore[no-untyped-def]
    short = TimeGrid(64, 1 / 400e9)  # df = 6.25 GHz
    sig = OpticalSignal(short, np.ones(64), F0)
    ring = AllPassRing("r", {"power_coupling": 0.01, "loss": 0.0})
    _, ctx = run_component(ring, {"in": sig})
    assert any(d.code == "photonic.resonance_unresolved" for d in ctx.diagnostics)


def test_db_per_cm_unit() -> None:
    from optobuild.core.units import from_si, parse_to_si

    alpha = parse_to_si("3 dB/cm", expect="attenuation")
    assert alpha == pytest.approx(3 * math.log(10) / 10 * 100)
    assert from_si(alpha, "dB/cm") == pytest.approx(3.0)
