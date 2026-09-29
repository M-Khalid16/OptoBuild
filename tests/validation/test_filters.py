"""Validation: electrical low-pass filter responses, impulse response and NEB."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.components.electrical import LowPassFilter
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.filters import (
    butterworth_neb,
    gaussian_neb,
    lowpass_on_grid,
    lowpass_response,
    noise_equivalent_bandwidth,
)
from optobuild.numerics.grid import TimeGrid
from optobuild.signals import ElectricalSignal

B = 10e9


@pytest.mark.parametrize(
    ("kind", "order"),
    [
        ("bessel", 2),
        ("bessel", 4),
        ("bessel", 6),
        ("butterworth", 1),
        ("butterworth", 5),
        ("gaussian", 0),
    ],
)
def test_dc_gain_and_3db_point(kind: str, order: int) -> None:
    h = lowpass_response(np.array([0.0, B, -B]), kind, B, order)
    assert abs(h[0]) == pytest.approx(1.0, abs=1e-12)
    assert abs(h[1]) == pytest.approx(1 / math.sqrt(2), rel=1e-9)
    assert h[2] == pytest.approx(np.conj(h[1]), abs=1e-12)  # real impulse response


@pytest.mark.parametrize("order", [1, 2, 4, 8])
def test_butterworth_magnitude(order: int) -> None:
    f = np.linspace(-5 * B, 5 * B, 201)
    h = lowpass_response(f, "butterworth", B, order)
    np.testing.assert_allclose(np.abs(h) ** 2, 1 / (1 + (f / B) ** (2 * order)), rtol=1e-9)


def test_rectangular_is_brick_wall() -> None:
    f = np.array([0.0, 0.999 * B, B, 1.001 * B, -1.001 * B])
    np.testing.assert_array_equal(np.abs(lowpass_response(f, "rectangular", B)), [1, 1, 1, 0, 0])


def test_gaussian_impulse_response_is_analytic_gaussian() -> None:
    """Impulse response: unit-area Gaussian, sigma_t = sqrt(ln 2) / (2 pi B)."""
    n, fs = 4096, 2e12
    grid = TimeGrid.from_sample_rate(n, fs)
    x = np.zeros(n)
    x[0] = 1.0 / grid.dt  # discrete delta of unit area
    y = np.real(apply_transfer_function(x, lowpass_on_grid(grid, "gaussian", B), grid))
    sigma = math.sqrt(math.log(2)) / (2 * math.pi * B)
    t = np.where(np.arange(n) < n // 2, np.arange(n), np.arange(n) - n) * grid.dt
    expected = np.exp(-(t**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-9 * expected.max())


@pytest.mark.parametrize(
    ("kind", "order", "analytic"),
    [
        ("butterworth", 1, butterworth_neb(B, 1)),
        ("butterworth", 4, butterworth_neb(B, 4)),
        ("gaussian", 4, gaussian_neb(B)),
        ("rectangular", 4, B),
    ],
)
def test_noise_equivalent_bandwidth(kind: str, order: int, analytic: float) -> None:
    """Discrete NEB on a wide grid vs closed forms.

    Tolerance: the grid covers +-fs/2 = +-50 B; the neglected tail of a first-order
    Butterworth is B^2/(fs/2) / B ~ 2 %/pi of NEB, higher orders are negligible;
    rectangular is exact up to the bin quantization df/B.
    """
    grid = TimeGrid.from_sample_rate(2**18, 100 * B)
    neb = noise_equivalent_bandwidth(grid, lowpass_on_grid(grid, kind, B, order))
    tol = {1: 0.02}.get(order, 1e-3) if kind == "butterworth" else 1e-3
    assert neb == pytest.approx(analytic, rel=tol)


def test_bessel_group_delay_is_flat_in_passband() -> None:
    """Bessel-Thomson: group delay constant to within 1 % up to 0.5 B (order 4)."""
    f = np.linspace(1e6, 0.5 * B, 200)
    phase = np.unwrap(np.angle(lowpass_response(f, "bessel", B, 4)))
    tau = -np.gradient(phase, 2 * np.pi * f)
    assert np.ptp(tau) / tau.mean() < 0.01


def test_filter_component_preserves_real_signals_and_dc(run_component) -> None:  # type: ignore[no-untyped-def]
    grid = TimeGrid.from_sample_rate(1024, 160e9)
    x = 1e-3 + 1e-4 * np.cos(2 * np.pi * 3 * grid.df * grid.time())
    x[::7] += 5e-4  # broadband content including the Nyquist bin
    sig = ElectricalSignal(grid, x)
    for kind in ("bessel", "butterworth", "gaussian", "rectangular"):
        out, ctx = run_component(LowPassFilter("f", {"kind": kind, "bandwidth": B}), {"in": sig})
        y = out["out"].samples
        assert y.dtype == np.float64
        assert y.mean() == pytest.approx(x.mean(), rel=1e-12)  # DC gain 1
        h = lowpass_on_grid(grid, kind, B, 4)
        np.testing.assert_allclose(y, np.real(apply_transfer_function(x, h, grid)), atol=1e-18)
        assert ctx.results["noise_equivalent_bandwidth_hz"] > 0
