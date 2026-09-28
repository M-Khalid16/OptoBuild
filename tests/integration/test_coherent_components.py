"""Coherent components: behaviour, delegation, metadata and error handling."""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.analysis.constellations import map_bits
from optobuild.cli.demos import coherent_link_project
from optobuild.components.coherent import (
    ASENoiseLoader,
    CoherentAnalyzer,
    CoherentDSP,
    CoherentReceiver,
    IQModulator,
    PulseShaper,
    SymbolMapper,
)
from optobuild.components.sources import CWLaser
from optobuild.core.errors import ComponentExecutionError, SamplingError, SignalTypeError
from optobuild.numerics.grid import TimeGrid
from optobuild.persistence import dumps_project, loads_project, run_project
from optobuild.physics.modulation import iq_modulator_field
from optobuild.signals import (
    DigitalSequence,
    ElectricalQuantity,
    ElectricalSignal,
    OpticalSignal,
    SymbolSequence,
)


def test_mapper_repeats_odd_length_patterns(run_component) -> None:  # type: ignore[no-untyped-def]
    bits = DigitalSequence(np.array([1, 0, 1, 1, 0, 0, 1]), 64e9, {"pattern": "x"})
    out = run_component(SymbolMapper("m", {"modulation": "qpsk"}), {"bits": bits})[0]["symbols"]
    assert out.n_symbols == 7 and out.symbol_rate == 32e9
    np.testing.assert_allclose(out.symbols, map_bits(np.tile(bits.bits, 2), "qpsk"))
    even = DigitalSequence(np.array([1, 0, 1, 1]), 64e9)
    assert run_component(SymbolMapper("m"), {"bits": even})[0]["symbols"].n_symbols == 2
    assert dict(out.metadata) == {"modulation": "qpsk", "pattern": "x"}


def test_pulse_shaper_outputs_and_metadata(run_component) -> None:  # type: ignore[no-untyped-def]
    sym = SymbolSequence(
        map_bits(np.random.default_rng(0).integers(0, 2, 128), "qpsk"), 32e9, {"modulation": "qpsk"}
    )
    out = run_component(
        PulseShaper(
            "p", {"samples_per_symbol": 4, "rolloff": 0.2, "amplitude": 0.3, "pulse": "rc"}
        ),
        {"symbols": sym},
    )[0]
    i, q = out["i"], out["q"]
    assert i.grid.n_samples == 256 and i.grid.sample_rate == pytest.approx(128e9)
    np.testing.assert_allclose(
        i.samples[::4] + 1j * q.samples[::4], 0.3 * sym.symbols, atol=1e-12
    )  # RC passes through the symbols
    assert i.metadata["samples_per_symbol"] == 4 and i.metadata["pulse_rolloff"] == 0.2
    # dual-polarization symbols: the shaper selects its tributary
    dual = SymbolSequence(np.vstack([np.ones(8), -1j * np.ones(8)]), 1e9)
    params = {"pulse": "rc", "amplitude": 1.0}
    out = run_component(PulseShaper("p", {**params, "polarization": "y"}), {"symbols": dual})[0]
    np.testing.assert_allclose(out["q"].samples[::4], -1.0, atol=1e-12)
    out = run_component(PulseShaper("p", params), {"symbols": dual})[0]
    np.testing.assert_allclose(out["i"].samples[::4], 1.0, atol=1e-12)


def test_iq_modulator_delegates_to_physics(run_component) -> None:  # type: ignore[no-untyped-def]
    g = TimeGrid(64, 1e-12)
    rng = np.random.default_rng(1)
    vi, vq = rng.normal(size=64), rng.normal(size=64)
    opt = OpticalSignal(g, np.full(64, 0.1 + 0.05j), 193e12)
    mod = IQModulator("iq", {"v_pi": 3.0, "insertion_loss": 0.5, "extinction_ratio": 300.0})
    out = run_component(
        mod, {"optical_in": opt, "i": ElectricalSignal(g, vi), "q": ElectricalSignal(g, vq)}
    )[0]["optical_out"]
    np.testing.assert_array_equal(
        out.field[0], opt.field[0] * iq_modulator_field(vi, vq, 3.0, 0.5, 300.0)
    )
    with pytest.raises(SignalTypeError):
        run_component(
            mod,
            {
                "optical_in": opt,
                "i": ElectricalSignal(g, vi),
                "q": ElectricalSignal(g, vq, ElectricalQuantity.CURRENT),
            },
        )


def test_ase_loader_sets_the_osnr(run_component) -> None:  # type: ignore[no-untyped-def]
    g = TimeGrid.from_sample_rate(2**16, 128e9)
    sig = OpticalSignal(g, np.full(2**16, np.sqrt(1e-3), dtype=complex), 193e12)
    out, ctx = run_component(ASENoiseLoader("a", {"osnr": 100.0}), {"in": sig}, seed=3)
    psd = ctx.results["ase_psd_per_pol_w_per_hz"]
    assert psd == pytest.approx(1e-3 / (2 * 100 * 12.5e9))
    noise_power = np.mean(np.abs(out["out"].field - sig.field) ** 2)
    assert noise_power == pytest.approx(psd * 128e9, rel=5 / np.sqrt(2**16))


def test_receiver_rejects_far_lo_and_pol_mismatch(run_component) -> None:  # type: ignore[no-untyped-def]
    g = TimeGrid.from_sample_rate(64, 100e9)
    sig = OpticalSignal(g, np.ones(64), 193.0e12)
    far = OpticalSignal(g, np.ones(64), 193.1e12)
    with pytest.raises(SamplingError, match="beyond"):
        run_component(CoherentReceiver("r"), {"signal": sig, "lo": far})
    with pytest.raises(SamplingError, match="polarizations"):
        run_component(
            CoherentReceiver("r"), {"signal": sig, "lo": OpticalSignal(g, np.ones((2, 64)), 193e12)}
        )


def test_dsp_and_analyzer_explain_missing_inputs(run_component) -> None:  # type: ignore[no-untyped-def]
    g = TimeGrid(64, 1e-12)
    bare = ElectricalSignal(g, np.zeros(64))
    with pytest.raises(SignalTypeError, match="symbol timing"):
        run_component(CoherentDSP("d"), {"i": bare, "q": bare})
    a = SymbolSequence(np.ones(8), 1e9, {"modulation": "qpsk"})
    b = SymbolSequence(np.ones(9), 1e9, {"modulation": "qpsk"})
    with pytest.raises(SamplingError, match="equal shapes"):
        run_component(CoherentAnalyzer("x"), {"received": a, "reference": b})


def test_laser_linewidth_is_seeded(run_component) -> None:  # type: ignore[no-untyped-def]
    laser = CWLaser("l", {"linewidth": 1e6, "n_samples": 256})
    a = run_component(laser, seed=1)[0]["out"].phase()
    b = run_component(laser, seed=1)[0]["out"].phase()
    c = run_component(laser, seed=2)[0]["out"].phase()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(run_component(CWLaser("l", {"n_samples": 8}))[0]["out"].phase() == 0)


def test_coherent_demo_round_trip_and_reproducibility() -> None:
    p = coherent_link_project(prbs_order=11)
    q = loads_project(dumps_project(p))
    a, b = run_project(p), run_project(q)
    np.testing.assert_array_equal(
        a.result("analyzer", "constellation"), b.result("analyzer", "constellation")
    )
    assert a.result("analyzer", "snr_db") > 10
    assert a.result("dsp", "frequency_offset_hz") == pytest.approx(-1e9, rel=2e-3)


def test_ase_loader_gain(run_component) -> None:  # type: ignore[no-untyped-def]
    g = TimeGrid.from_sample_rate(1024, 128e9)
    sig = OpticalSignal(g, np.full(1024, np.sqrt(1e-6), dtype=complex), 193e12)
    out, ctx = run_component(ASENoiseLoader("a", {"osnr": 1e9, "gain": 100.0}), {"in": sig})
    assert out["out"].average_power() == pytest.approx(1e-4, rel=1e-6)
    assert ctx.results["ase_psd_per_pol_w_per_hz"] == pytest.approx(1e-4 / (2e9 * 12.5e9))


def test_analyzer_guard_excludes_window_edge_artefacts() -> None:
    """With non-periodic laser phase noise the first received symbol can be corrupted by
    the circular window (observed: 1 error at OSNR 18 dB, trial 0, symbol 0); the demos
    exclude 16 guard symbols at each edge."""
    p = coherent_link_project(osnr_db=18.0)
    r = run_project(p, trial=0)
    assert r.result("analyzer", "bit_errors") == 0
    assert r.result("analyzer", "n_symbols") == 32767 - 32
    p.graph.set_parameters("analyzer", guard_symbols=0)
    assert run_project(p, trial=0).result("analyzer", "bit_errors") == 1
    p.graph.set_parameters("analyzer", guard_symbols=20000)
    with pytest.raises(ComponentExecutionError, match="guard_symbols"):
        run_project(p)
