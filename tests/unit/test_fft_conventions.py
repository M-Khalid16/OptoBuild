"""Unit tests pinning the FFT convention of ADR-0002 / docs/numerical_conventions.md."""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.core.errors import SamplingError
from optobuild.numerics import fft
from optobuild.numerics.grid import TimeGrid


@pytest.fixture
def grid() -> TimeGrid:
    return TimeGrid(n_samples=256, dt=1e-12)


def test_forward_scaling_matches_definition(grid: TimeGrid, rng: np.random.Generator) -> None:
    x = rng.normal(size=grid.n_samples) + 1j * rng.normal(size=grid.n_samples)
    n = np.arange(grid.n_samples)
    k = n[:, None]
    direct = grid.dt * np.sum(
        x[None, :] * np.exp(-2j * np.pi * k * n[None, :] / grid.n_samples), axis=1
    )
    np.testing.assert_allclose(fft.spectrum(x, grid), direct, rtol=1e-10, atol=1e-25)


def test_round_trip_is_identity(grid: TimeGrid, rng: np.random.Generator) -> None:
    x = rng.normal(size=(2, grid.n_samples)) + 1j * rng.normal(size=(2, grid.n_samples))
    for g in (grid, grid.with_t0(-3.7e-11)):
        np.testing.assert_allclose(fft.inverse_spectrum(fft.spectrum(x, g), g), x, atol=1e-12)


def test_positive_frequency_tone_lands_in_positive_bin(grid: TimeGrid) -> None:
    """exp(+i 2 pi f t) must appear at +f: fixes the sign convention."""
    k0 = 5
    f0 = k0 * grid.df
    x = np.exp(2j * np.pi * f0 * grid.time())
    spec = np.abs(fft.spectrum(x, grid))
    peak = int(np.argmax(spec))
    assert grid.frequency()[peak] == pytest.approx(f0)
    assert grid.frequency()[peak] > 0


def test_parseval_energy(grid: TimeGrid, rng: np.random.Generator) -> None:
    x = rng.normal(size=grid.n_samples) + 1j * rng.normal(size=grid.n_samples)
    g = grid.with_t0(1.23e-10)  # origin phase must not affect energy
    spec = fft.spectrum(x, g)
    assert fft.energy(x, g) == pytest.approx(fft.spectral_energy(spec, g), rel=1e-12)


def test_psd_integrates_to_mean_power(grid: TimeGrid, rng: np.random.Generator) -> None:
    x = rng.normal(size=grid.n_samples) + 1j * rng.normal(size=grid.n_samples)
    psd = fft.power_spectral_density(x, grid)
    assert np.sum(psd) * grid.df == pytest.approx(fft.mean_power(x, grid), rel=1e-12)


def test_cw_psd_units(grid: TimeGrid) -> None:
    """A CW envelope sqrt(P) puts all power P in the DC bin: S_0 * df == P."""
    power_w = 1e-3
    x = np.full(grid.n_samples, np.sqrt(power_w), dtype=complex)
    psd = fft.power_spectral_density(x, grid)
    assert psd[0] * grid.df == pytest.approx(power_w, rel=1e-12)
    assert np.sum(psd[1:]) * grid.df == pytest.approx(0.0, abs=1e-18)


def test_time_axis_is_last_and_polarizations_independent(
    grid: TimeGrid, rng: np.random.Generator
) -> None:
    x = rng.normal(size=(2, grid.n_samples)) + 0j
    spec = fft.spectrum(x, grid)
    np.testing.assert_allclose(spec[1], fft.spectrum(x[1], grid))


def test_transfer_function_delay(grid: TimeGrid) -> None:
    """H(f) = exp(-i 2 pi f tau) is a delay by tau (circular within the window)."""
    shift = 7
    tau = shift * grid.dt
    x = np.zeros(grid.n_samples, dtype=complex)
    x[10] = 1.0
    y = fft.apply_transfer_function(x, np.exp(-2j * np.pi * grid.frequency() * tau), grid)
    np.testing.assert_allclose(y, np.roll(x, shift), atol=1e-12)


def test_display_order_is_fftshift(grid: TimeGrid) -> None:
    (f,) = fft.to_display_order(grid.frequency())
    assert np.all(np.diff(f) > 0)
    np.testing.assert_allclose(f, grid.frequency(shifted=True))


def test_length_mismatch_is_rejected(grid: TimeGrid) -> None:
    with pytest.raises(SamplingError, match="last axis"):
        fft.spectrum(np.zeros(grid.n_samples + 1), grid)
    with pytest.raises(SamplingError):
        fft.apply_transfer_function(np.zeros(grid.n_samples), np.ones(3), grid)
