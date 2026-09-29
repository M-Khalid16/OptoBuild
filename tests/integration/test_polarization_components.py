"""Polarization components, receiver impairments and the dual-polarization DSP (Phase 6b)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.components.coherent import (
    CoherentAnalyzer,
    CoherentReceiver,
    DualPolCoherentDSP,
    SymbolMapper,
)
from optobuild.components.polarization import (
    PolarizationBeamCombiner,
    PolarizationBeamSplitter,
    PolarizationController,
    RandomPMD,
)
from optobuild.core.errors import InvalidParameterError, SamplingError, SignalTypeError
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.detection import coherent_detection
from optobuild.signals import DigitalSequence, ElectricalSignal, OpticalSignal, SymbolSequence

F0 = 193.4e12
GRID = TimeGrid(512, 1e-12, -256e-12)


def _dual(seed: int = 0) -> OpticalSignal:
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((2, 512)) + 1j * rng.standard_normal((2, 512))
    return OpticalSignal(GRID, 1e-2 * field, F0)


def test_pbs_pbc_round_trip_and_scalar_split(run_component) -> None:  # type: ignore[no-untyped-def]
    sig = _dual()
    parts = run_component(PolarizationBeamSplitter("pbs"), {"in": sig})[0]
    assert parts["x"].n_pol == parts["y"].n_pol == 1
    back = run_component(PolarizationBeamCombiner("pbc"), parts)[0]["out"]
    np.testing.assert_array_equal(back.field, sig.field)
    scalar = OpticalSignal(GRID, np.full(512, math.sqrt(1e-3), complex), F0)
    out = run_component(PolarizationBeamSplitter("pbs"), {"in": scalar})[0]
    assert out["x"].average_power() == pytest.approx(0.5e-3, rel=1e-12)
    assert out["y"].average_power() == pytest.approx(0.5e-3, rel=1e-12)
    out = run_component(PolarizationBeamSplitter("pbs", {"input_angle": 0.0}), {"in": scalar})[0]
    assert out["y"].average_power() == 0.0


def test_pbc_rejects_mismatched_inputs(run_component) -> None:  # type: ignore[no-untyped-def]
    x = OpticalSignal(GRID, np.ones(512), F0)
    with pytest.raises(SamplingError, match="carriers differ"):
        run_component(
            PolarizationBeamCombiner("pbc"), {"x": x, "y": OpticalSignal(GRID, np.ones(512), 1e14)}
        )
    with pytest.raises(SignalTypeError, match="single-polarization"):
        run_component(PolarizationBeamCombiner("pbc"), {"x": x, "y": _dual()})
    shifted = OpticalSignal(GRID.with_t0(0.0), np.ones(512), F0)
    _, ctx = run_component(PolarizationBeamCombiner("pbc"), {"x": x, "y": shifted})
    assert any(d.code == "pbc.time_origin_mismatch" for d in ctx.diagnostics)


def test_polarization_controller(run_component) -> None:  # type: ignore[no-untyped-def]
    sig = _dual()
    ctl = PolarizationController("pc", {"azimuth": 0.4, "phase": 0.3, "dgd": 5e-12})
    out = run_component(ctl, {"in": sig})[0]["out"]
    assert out.average_power() == pytest.approx(sig.average_power(), rel=1e-12)
    # pure rotation by 90 degrees swaps the axes (x -> y, y -> -x)
    swap = run_component(PolarizationController("pc", {"azimuth": math.pi / 2}), {"in": sig})
    np.testing.assert_allclose(swap[0]["out"].field[1], sig.field[0], atol=1e-15)
    with pytest.raises(SignalTypeError, match="PolarizationBeamCombiner"):
        run_component(ctl, {"in": OpticalSignal(GRID, np.ones(512), F0)})


def test_random_pmd_is_seeded_lossless_and_records_dgd(run_component) -> None:  # type: ignore[no-untyped-def]
    sig = _dual()
    pmd = RandomPMD("pmd", {"mean_dgd": 5e-12, "n_sections": 16})
    (a, ca), (b, _), (c, cc) = (run_component(pmd, {"in": sig}, seed=s) for s in (1, 1, 2))
    np.testing.assert_array_equal(a["out"].field, b["out"].field)
    assert not np.array_equal(a["out"].field, c["out"].field)
    assert a["out"].average_power() == pytest.approx(sig.average_power(), rel=1e-12)
    assert 0 < ca.results["dgd_s"] < 20e-12 and ca.results["dgd_s"] != cc.results["dgd_s"]
    big = RandomPMD("pmd", {"mean_dgd": 200e-12})
    _, ctx = run_component(big, {"in": sig}, seed=1)
    assert any(d.code == "pmd.dgd_vs_window" for d in ctx.diagnostics)


def test_receiver_impairments_follow_physics(run_component) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(3)
    es = 1e-2 * (rng.standard_normal(512) + 1j * rng.standard_normal(512))
    sig = OpticalSignal(GRID, es, F0)
    lo = OpticalSignal(GRID, np.full(512, 0.03 + 0j), F0)
    params = {"shot_noise": False, "thermal_noise": False}
    imp = {"hybrid_phase_error": 0.1, "quadrature_gain": 0.9}
    out = run_component(CoherentReceiver("rx", {**params, **imp}), {"signal": sig, "lo": lo})[0]
    i, q = coherent_detection(es[None, :], lo.field, 0.8, None, 1.0, phase_error=0.1,
                              quadrature_gain=0.9)  # fmt: skip
    np.testing.assert_allclose(out["i"].samples, i, rtol=1e-12)
    np.testing.assert_allclose(out["q"].samples, q, rtol=1e-12)
    skew = CoherentReceiver("rx", {**params, "iq_skew": 2e-12})
    out2 = run_component(skew, {"signal": sig, "lo": lo})[0]
    ref = run_component(CoherentReceiver("rx", params), {"signal": sig, "lo": lo})[0]
    np.testing.assert_array_equal(out2["i"].samples, ref["i"].samples)
    # 2 ps at dt = 1 ps: an exact two-sample (circular) delay of the Q current
    np.testing.assert_allclose(np.roll(ref["q"].samples, 2), out2["q"].samples, atol=1e-12)


def test_symbol_mapper_dual_polarization(run_component) -> None:  # type: ignore[no-untyped-def]
    bits = DigitalSequence(np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1], np.uint8), 1e9)
    out = run_component(SymbolMapper("m", {"polarizations": 2}), {"bits": bits})[0]["symbols"]
    assert out.symbols.shape == (2, 6)
    single = run_component(SymbolMapper("m"), {"bits": bits})[0]["symbols"]
    np.testing.assert_array_equal(out.symbols[0], single.symbols)
    rev = DigitalSequence(bits.bits[::-1].copy(), 1e9)
    np.testing.assert_array_equal(
        out.symbols[1], run_component(SymbolMapper("m"), {"bits": rev})[0]["symbols"].symbols
    )


def test_dual_pol_dsp_validation(run_component) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidParameterError, match="odd"):
        DualPolCoherentDSP("d", {"mimo_taps": 8})
    with pytest.raises(InvalidParameterError, match="smaller"):
        DualPolCoherentDSP("d", {"mimo_epochs": 2, "mimo_x_epochs": 2})
    md = {"symbol_rate": 1e9, "samples_per_symbol": 3, "modulation": "qpsk"}
    e = ElectricalSignal(TimeGrid(48, 1 / 3e9), np.ones(48), metadata=md)
    with pytest.raises(SamplingError, match="even"):
        run_component(DualPolCoherentDSP("d"), {"xi": e, "xq": e, "yi": e, "yq": e})


def test_analyzer_resolves_polarization_swap(run_component) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(4)
    from optobuild.analysis.constellations import map_bits

    s = np.vstack([map_bits(rng.integers(0, 2, 2048), "qpsk") for _ in range(2)])
    ref = SymbolSequence(s, 1e9, {"modulation": "qpsk"})
    rx = SymbolSequence(np.vstack([s[1] * 1j, np.roll(s[0], 5)]), 1e9)
    _, ctx = run_component(CoherentAnalyzer("a"), {"received": rx, "reference": ref})
    assert ctx.results["polarizations_swapped"] is True
    assert ctx.results["bit_errors"] == 0 and ctx.results["n_bits"] == 4096
    with pytest.raises(SamplingError, match="equal shapes"):
        run_component(
            CoherentAnalyzer("a"),
            {"received": SymbolSequence(s[0], 1e9), "reference": ref},
        )
