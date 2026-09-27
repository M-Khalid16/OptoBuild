"""Validation: NRZ edges and the Mach-Zehnder transfer function."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.components.modulators import MachZehnderModulator, NRZGenerator
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.modulation import (
    mzm_field_transfer,
    mzm_power_transfer,
    nrz_grid,
    nrz_waveform,
)
from optobuild.signals import DigitalSequence, ElectricalSignal, OpticalSignal

VPI = 4.0


def _mzm_on_ramp(run_component, **params):  # type: ignore[no-untyped-def]
    """Drive an MZM with a voltage ramp and return (voltage, output field)."""
    n = 2001
    grid = TimeGrid(n, 1e-12)
    v = np.linspace(-2 * VPI, 2 * VPI, n)
    opt = OpticalSignal(grid, np.full(n, np.sqrt(1e-3)), 193.4e12)
    mzm = MachZehnderModulator("mzm", {"v_pi": VPI, **params})
    out, _ = run_component(mzm, {"optical_in": opt, "drive": ElectricalSignal(grid, v)})
    return v, out["optical_out"].field[0] / np.sqrt(1e-3)


@pytest.mark.parametrize(
    "params",
    [
        {"v_bias": 0.0, "extinction_ratio": 1e30},
        {"v_bias": -2.0, "insertion_loss": 0.5, "extinction_ratio": 100.0},
        {"v_bias": 1.3, "phi0": 0.4, "insertion_loss": 0.8, "extinction_ratio": 20.0},
    ],
)
def test_mzm_transfer_curve_matches_closed_form(run_component, params: dict) -> None:  # type: ignore[no-untyped-def]
    v, h = _mzm_on_ramp(run_component, **params)
    il = params.get("insertion_loss", 1.0)
    er = params["extinction_ratio"]
    half = 0.5 * (np.pi * (v + params["v_bias"]) / VPI + params.get("phi0", 0.0))
    expected_t = il * (np.cos(half) ** 2 + np.sin(half) ** 2 / er)
    np.testing.assert_allclose(np.abs(h) ** 2, expected_t, rtol=1e-12, atol=1e-15)
    # residual-chirp phase arctan(eps tan(dphi/2)) where cos(half) > 0
    ok = np.cos(half) > 1e-3
    np.testing.assert_allclose(
        np.angle(h[ok]), np.arctan(np.tan(half[ok]) / np.sqrt(er)), atol=1e-12
    )


def test_ideal_mzm_is_cos_squared_with_period_2vpi() -> None:
    v = np.linspace(-10, 10, 1001)
    t = mzm_power_transfer(v, VPI)
    np.testing.assert_allclose(t, np.cos(np.pi * v / (2 * VPI)) ** 2, atol=1e-15)
    np.testing.assert_allclose(mzm_power_transfer(v + 2 * VPI, VPI), t, atol=1e-12)
    assert mzm_power_transfer(VPI, VPI) == pytest.approx(0.0, abs=1e-30)


def test_extinction_ratio_and_quadrature_point() -> None:
    er, il = 316.0, 0.7
    t_max = mzm_power_transfer(0.0, VPI, insertion_loss=il, extinction_ratio=er)
    t_min = mzm_power_transfer(VPI, VPI, insertion_loss=il, extinction_ratio=er)
    assert t_max == pytest.approx(il, rel=1e-14)
    assert t_max / t_min == pytest.approx(er, rel=1e-12)
    t_q = mzm_power_transfer(-VPI / 2, VPI, insertion_loss=il, extinction_ratio=er)
    assert t_q == pytest.approx(il * (1 + 1 / er) / 2, rel=1e-12)
    # field and power forms agree
    v = np.linspace(-5, 5, 11)
    np.testing.assert_allclose(
        np.abs(mzm_field_transfer(v, VPI, 0.3, il, er, 0.2)) ** 2,
        mzm_power_transfer(v, VPI, 0.3, il, er, 0.2),
        rtol=1e-13,
    )


def test_nrz_levels_and_timing() -> None:
    bits = np.array([0, 1, 1, 0, 1])
    grid = nrz_grid(5, 10e9, 8)
    v = nrz_waveform(bits, 8, grid, -1.0, 2.0)
    np.testing.assert_array_equal(v, np.repeat([-1.0, 2.0, 2.0, -1.0, 2.0], 8))
    assert grid.dt == pytest.approx(1 / 80e9)


def test_nrz_rise_time_matches_specification() -> None:
    """10-90 % rise time of a single isolated edge vs the Gaussian-edge model.

    Measured by linear interpolation between samples; with 256 samples per bit
    and t_r = 0.3 T_b the interpolation error is < dt/10, far below the
    tolerance of 1 % t_r = 0.77 dt.
    """
    sps, rb, tr = 256, 10e9, 0.3 / 10e9
    bits = np.array([0] * 16 + [1] * 16)
    grid = nrz_grid(bits.size, rb, sps)
    v = nrz_waveform(bits, sps, grid, 0.0, 1.0, rise_time=tr)
    t = grid.time()
    edge = slice(8 * sps, 24 * sps)  # contains only the 0->1 edge at 16 T_b
    t10 = np.interp(0.1, v[edge], t[edge])
    t90 = np.interp(0.9, v[edge], t[edge])
    assert t90 - t10 == pytest.approx(tr, rel=0.01)
    assert np.interp(0.5, v[edge], t[edge]) == pytest.approx(16 / rb, abs=grid.dt)


def test_nrz_component_annotates_timing(run_component) -> None:  # type: ignore[no-untyped-def]
    seq = DigitalSequence(np.array([0, 1, 0, 1]), 10e9, {"pattern": "test"})
    out, ctx = run_component(NRZGenerator("nrz", {"samples_per_bit": 4}), {"bits": seq})
    sig = out["out"]
    assert dict(sig.metadata) == {"bit_rate": 10e9, "samples_per_bit": 4, "pattern": "test"}
    assert sig.grid.n_samples == 16
    # rectangular NRZ of a random-like pattern is not band-limited: the aliasing
    # diagnostic must say so; with finite rise time the edge energy drops below threshold
    from optobuild.physics.prbs import prbs

    prbs_seq = DigitalSequence(prbs(7, 127), 10e9)
    _, ctx = run_component(NRZGenerator("nrz", {"samples_per_bit": 4}), {"bits": prbs_seq})
    assert [d.code for d in ctx.diagnostics] == ["sampling.aliasing_risk"]
    smooth = NRZGenerator("nrz", {"samples_per_bit": 16, "rise_time": 30e-12})
    _, ctx = run_component(smooth, {"bits": prbs_seq})
    assert ctx.diagnostics == []
    assert [d.code for d in NRZGenerator("n2", {"samples_per_bit": 2}).diagnostics] == [
        "sampling.samples_per_symbol"
    ]


def test_mzm_requires_matching_grids_and_voltage(run_component) -> None:  # type: ignore[no-untyped-def]
    from optobuild.core.errors import SamplingError, SignalTypeError
    from optobuild.signals import ElectricalQuantity

    g1, g2 = TimeGrid(8, 1e-12), TimeGrid(16, 1e-12)
    opt = OpticalSignal(g1, np.ones(8), 193e12)
    mzm = MachZehnderModulator("m")
    with pytest.raises(SamplingError):
        run_component(mzm, {"optical_in": opt, "drive": ElectricalSignal(g2, np.zeros(16))})
    with pytest.raises(SignalTypeError):
        run_component(
            mzm,
            {
                "optical_in": opt,
                "drive": ElectricalSignal(g1, np.zeros(8), ElectricalQuantity.CURRENT),
            },
        )
    assert math.isfinite(mzm.parameters["extinction_ratio"])
