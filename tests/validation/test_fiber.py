"""Validation: fiber attenuation, group delay, Gaussian-pulse dispersion, sign convention."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.components.fiber import LinearFiber
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.units import db_per_km_to_per_m, dispersion_to_beta2
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.fiber import dispersion_length, propagate_linear
from optobuild.signals import OpticalSignal

F0 = SPEED_OF_LIGHT / 1550e-9


def _gaussian(t0: float, n: int = 4096, window: float = 80.0, p0: float = 1e-3):
    grid = TimeGrid(n, window * t0 / n, t0=-(n // 2) * window * t0 / n)
    field = np.sqrt(p0) * np.exp(-(grid.time() ** 2) / (2 * t0**2))
    return OpticalSignal(grid, field, F0)


@pytest.mark.parametrize("length_km", [0.0, 1.0, 50.0, 100.0])
def test_attenuation_exponential_law(run_component, length_km: float) -> None:  # type: ignore[no-untyped-def]
    sig = _gaussian(10e-12)
    fiber = LinearFiber("f", {"length": length_km * 1e3, "dispersion": 17e-6})
    out = run_component(fiber, {"in": sig})[0]["out"]
    alpha = db_per_km_to_per_m(0.2)
    ratio = out.average_power() / sig.average_power()
    assert ratio == pytest.approx(math.exp(-alpha * length_km * 1e3), rel=1e-12)
    assert 10 * math.log10(ratio) == pytest.approx(-0.2 * length_km, abs=1e-10)


def test_lossless_fiber_conserves_energy_and_records_delay(run_component) -> None:  # type: ignore[no-untyped-def]
    sig = _gaussian(5e-12)
    fiber = LinearFiber("f", {"length": 20e3, "attenuation": 0.0, "group_index": 1.5})
    out, ctx = run_component(fiber, {"in": sig})
    assert out["out"].average_power() == pytest.approx(sig.average_power(), rel=1e-12)
    delay = 1.5 * 20e3 / SPEED_OF_LIGHT
    assert out["out"].grid.t0 - sig.grid.t0 == pytest.approx(delay, rel=1e-12)
    assert ctx.results["group_delay_s"] == pytest.approx(delay, rel=1e-12)


@pytest.mark.parametrize("l_over_ld", [0.5, 1.0, 2.0, 4.0])
def test_gaussian_pulse_matches_analytic_field(l_over_ld: float) -> None:
    """Complex field vs A(L,T) = sqrt(P0) T0/sqrt(T0^2 + i b2 L) exp(-T^2/(2(T0^2 + i b2 L))).

    The window is 80 T0 and the grid resolves the spectrum to beyond 12 standard
    deviations, so truncation/aliasing errors are < 1e-12 of the peak; the
    tolerance 1e-9 of the input peak allows for round-off only.
    """
    t0, p0 = 10e-12, 1e-3
    beta2 = dispersion_to_beta2(17e-6, 1550e-9)
    length = l_over_ld * dispersion_length(t0, beta2)
    sig = _gaussian(t0, p0=p0)
    field, _ = propagate_linear(sig.field, sig.grid, length, 0.0, beta2)
    t = sig.grid.time()
    q = t0**2 + 1j * beta2 * length
    expected = np.sqrt(p0) * t0 / np.sqrt(q) * np.exp(-(t**2) / (2 * q))
    np.testing.assert_allclose(field[0], expected, rtol=0, atol=1e-9 * np.sqrt(p0))


@pytest.mark.parametrize("l_over_ld", [0.0, 1.0, 3.0])
def test_gaussian_broadening_factor(l_over_ld: float) -> None:
    """RMS width ratio sqrt(1 + (L/L_D)^2) and peak power reduced by the same factor."""
    t0 = 10e-12
    beta2 = dispersion_to_beta2(17e-6, 1550e-9)
    length = l_over_ld * dispersion_length(t0, beta2)
    sig = _gaussian(t0)
    field, _ = propagate_linear(sig.field, sig.grid, length, 0.0, beta2)
    t = sig.grid.time()

    def rms(p: np.ndarray) -> float:
        return float(np.sqrt(np.sum(t**2 * p) / np.sum(p)))

    factor = math.sqrt(1 + l_over_ld**2)
    p_in, p_out = np.abs(sig.field[0]) ** 2, np.abs(field[0]) ** 2
    assert rms(p_out) / rms(p_in) == pytest.approx(factor, rel=1e-9)
    assert p_in.max() / p_out.max() == pytest.approx(factor, rel=1e-6)


def test_anomalous_dispersion_blue_travels_faster() -> None:
    """Sign convention check: with D > 0 (beta2 < 0) a pulse detuned by +df arrives earlier.

    Expected delay of the pulse centroid: beta2 * (2 pi df) * L (+ beta3 term = 0).
    """
    t0 = 20e-12
    sig = _gaussian(t0, n=8192, window=200.0)
    df = 50e9
    detuned = sig.field * np.exp(2j * np.pi * df * sig.grid.time())
    beta2 = dispersion_to_beta2(17e-6, 1550e-9)
    length = 10e3
    field, _ = propagate_linear(detuned, sig.grid, length, 0.0, beta2)
    t = sig.grid.time()
    p = np.abs(field[0]) ** 2
    centroid = np.sum(t * p) / np.sum(p)
    assert centroid == pytest.approx(beta2 * 2 * np.pi * df * length, rel=1e-9)
    assert centroid < 0


def test_beta3_group_delay_curvature() -> None:
    """With beta2 = 0, a component at w has group delay beta3 w^2 L / 2 (independent of sign)."""
    t0 = 20e-12
    sig = _gaussian(t0, n=8192, window=200.0)
    beta3, length = 0.1e-39 * 1e3, 50e3  # 0.1 ps^3/km in s^3/m
    delays = []
    for df in (-100e9, 100e9):
        detuned = sig.field * np.exp(2j * np.pi * df * sig.grid.time())
        field, _ = propagate_linear(detuned, sig.grid, length, 0.0, 0.0, beta3)
        p = np.abs(field[0]) ** 2
        delays.append(np.sum(sig.grid.time() * p) / np.sum(p))
    w = 2 * np.pi * 100e9
    # centroid delay of a Gaussian spectrum: beta3 L/2 (w^2 + <dw^2>), <dw^2> = 1/(2 T0^2)
    expected = 0.5 * beta3 * length * (w**2 + 1 / (2 * t0**2))
    for d in delays:
        assert d == pytest.approx(expected, rel=1e-6)


def test_wraparound_diagnostic(run_component) -> None:  # type: ignore[no-untyped-def]
    sig = _gaussian(1e-12, n=1024, window=40.0)  # very short window, huge bandwidth
    out, ctx = run_component(LinearFiber("f", {"length": 50e3}), {"in": sig})
    assert "sampling.window_wraparound" in [d.code for d in ctx.diagnostics]
