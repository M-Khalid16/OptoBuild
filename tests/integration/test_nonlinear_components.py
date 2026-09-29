"""Pulse source and nonlinear-fiber components inside graphs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.cli.demos import soliton_project
from optobuild.components.fiber import LinearFiber
from optobuild.components.nonlinear import NonlinearFiber, OpticalPulseSource
from optobuild.core.errors import InvalidParameterError
from optobuild.persistence import dumps_project, loads_project, run_project


def test_pulse_source_shapes_and_energy(run_component) -> None:  # type: ignore[no-untyped-def]
    for shape, energy_factor in (("gaussian", math.sqrt(math.pi)), ("sech", 2.0)):
        src = OpticalPulseSource(
            "p",
            {
                "shape": shape,
                "peak_power": 0.1,
                "width": 2e-12,
                "n_samples": 4096,
                "sample_rate": 20e12,
            },
        )
        out, ctx = run_component(src)
        sig = out["out"]
        assert sig.power().max() == pytest.approx(0.1, rel=1e-6)
        energy = sig.power().sum() * sig.grid.dt  # analytic: P0 T0 sqrt(pi) or 2 P0 T0
        assert energy == pytest.approx(0.1 * 2e-12 * energy_factor, rel=1e-9)
        assert ctx.diagnostics == []
    short = OpticalPulseSource("s", {"width": 50e-12, "n_samples": 256, "sample_rate": 1e12})
    codes = [d.code for d in run_component(short)[1].diagnostics]
    assert codes == ["sampling.window_too_short"]


def test_nonlinear_fiber_equals_linear_fiber_for_zero_gamma(run_component) -> None:  # type: ignore[no-untyped-def]
    src = OpticalPulseSource("p", {"peak_power": 0.5})
    sig = run_component(src)[0]["out"]
    nl = run_component(NonlinearFiber("f", {"gamma": 0.0, "length": 20e3}), {"in": sig})
    lin = run_component(LinearFiber("f", {"length": 20e3}), {"in": sig})
    np.testing.assert_allclose(nl[0]["out"].field, lin[0]["out"].field, atol=1e-14)
    assert nl[0]["out"].grid == lin[0]["out"].grid  # same group-delay bookkeeping


def test_soliton_demo_preserves_the_pulse() -> None:
    """The component includes beta3 derived from D (non-zero even for S = 0), which
    perturbs the soliton by ~ L / L_D3 with L_D3 = T0^3 / |beta3|; that bound is
    the tolerance (the SSFM splitting error, 4e-7, is far smaller)."""
    from optobuild.core.units import dispersion_slope_to_beta3

    res = run_project(soliton_project())
    a_in = res.signal("pulse", "out").power()
    a_out = res.signal("fiber", "out").power()
    beta3 = dispersion_slope_to_beta3(17e-6, 0.0, 1550e-9)
    length = 5 * (5e-12) ** 2 / abs(res.result("fiber", "beta2_s2_per_m"))  # 5 L_D
    bound = length / ((5e-12) ** 3 / abs(beta3))
    assert 1e-4 < bound < 1e-2
    np.testing.assert_allclose(a_out, a_in, atol=bound * a_in.max())
    assert res.result("output_power", "average_power_w") == pytest.approx(
        res.result("input_power", "average_power_w"), rel=1e-12
    )
    assert res.result("fiber", "max_step_phase_rad") <= 1e-3 * (1 + 1e-9)
    assert res.all_diagnostics() == []


def test_higher_order_soliton_broadens_the_spectrum() -> None:
    p = soliton_project(soliton_order=3.0, length_in_ld=0.5 * math.pi / 4)
    p.graph.set_parameters("fiber", max_phase=0.01)  # qualitative check: coarser steps suffice
    res = run_project(p)

    def rms_width(node: str) -> float:
        f = res.result(node, "frequency_hz")
        s = res.result(node, "psd_w_per_hz")
        m = np.sum(f * s) / np.sum(s)
        return float(np.sqrt(np.sum((f - m) ** 2 * s) / np.sum(s)))

    assert rms_width("output_spectrum") > 2 * rms_width("input_spectrum")


def test_diagnostics_spectral_truncation_and_wraparound(run_component) -> None:  # type: ignore[no-untyped-def]
    coarse = OpticalPulseSource(
        "p",
        {
            "shape": "gaussian",
            "peak_power": 20.0,
            "width": 2e-12,
            "n_samples": 1024,
            "sample_rate": 2e12,
        },
    )
    sig = run_component(coarse)[0]["out"]
    _, ctx = run_component(
        NonlinearFiber("f", {"length": 2e3, "dispersion": 0.0, "attenuation": 0.0}), {"in": sig}
    )
    assert "ssfm.spectral_truncation" in [d.code for d in ctx.diagnostics]
    narrow = OpticalPulseSource("p", {"width": 2e-12, "n_samples": 2048, "sample_rate": 4e12})
    sig = run_component(narrow)[0]["out"]
    _, ctx = run_component(NonlinearFiber("f", {"length": 20e3, "gamma": 0.0}), {"in": sig})
    assert "sampling.window_wraparound" in [d.code for d in ctx.diagnostics]
    _, ctx = run_component(
        NonlinearFiber("f", {"step_mode": "fixed", "n_steps": 1, "length": 50e3}),
        {"in": run_component(OpticalPulseSource("p", {"peak_power": 5.0}))[0]["out"]},
    )
    assert "ssfm.large_step_phase" in [d.code for d in ctx.diagnostics]
    with pytest.raises(InvalidParameterError):
        NonlinearFiber("f", {"max_phase": 0.0})


def test_soliton_project_round_trip() -> None:
    p = soliton_project(length_in_ld=0.5)
    q = loads_project(dumps_project(p))
    a = run_project(p).signal("fiber", "out").field
    b = run_project(q).signal("fiber", "out").field
    np.testing.assert_array_equal(a, b)
