"""Validate the discrete FFT convention against analytic continuous Fourier pairs.

Reference pairs (forward kernel exp(-i 2 pi f t)):
    exp(-t^2 / (2 T0^2))            <->  T0 sqrt(2 pi) exp(-2 pi^2 f^2 T0^2)
    g(t - tau)                      <->  G(f) exp(-i 2 pi f tau)
Tolerances: the Gaussian is sampled at >= 20 points per T0 and the window is
+-20 T0, so aliasing and truncation errors are far below 1e-12 relative to the
peak; the tolerance is set to 1e-9 to allow for floating-point round-off only.
"""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.numerics import fft
from optobuild.numerics.grid import TimeGrid

T0 = 10e-12  # 10 ps
N = 1024
DT = 40 * T0 / N  # window = 40 T0


def _centered_grid(n: int = N, dt: float = DT) -> TimeGrid:
    return TimeGrid(n_samples=n, dt=dt, t0=-(n // 2) * dt)


def _gaussian_ft(f: np.ndarray) -> np.ndarray:
    return T0 * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi**2 * f**2 * T0**2)


def test_gaussian_transform_matches_analytic() -> None:
    g = _centered_grid()
    x = np.exp(-(g.time() ** 2) / (2 * T0**2))
    spec = fft.spectrum(x, g)
    ref = _gaussian_ft(g.frequency())
    np.testing.assert_allclose(spec, ref, rtol=0, atol=1e-9 * ref.max())


def test_shifted_gaussian_phase_matches_shift_theorem() -> None:
    g = _centered_grid()
    tau = 3.3 * T0
    x = np.exp(-((g.time() - tau) ** 2) / (2 * T0**2))
    ref = _gaussian_ft(g.frequency()) * np.exp(-2j * np.pi * g.frequency() * tau)
    np.testing.assert_allclose(fft.spectrum(x, g), ref, rtol=0, atol=1e-9 * np.abs(ref).max())


def test_inverse_of_analytic_spectrum_recovers_gaussian() -> None:
    g = _centered_grid()
    x = fft.inverse_spectrum(_gaussian_ft(g.frequency()), g)
    np.testing.assert_allclose(x, np.exp(-(g.time() ** 2) / (2 * T0**2)), atol=1e-9)


def test_gaussian_energy_matches_analytic() -> None:
    """integral exp(-t^2/T0^2) dt = T0 sqrt(pi) [J for a 1 W peak]."""
    g = _centered_grid()
    x = np.exp(-(g.time() ** 2) / (2 * T0**2))
    assert fft.energy(x, g) == pytest.approx(T0 * np.sqrt(np.pi), rel=1e-12)
    assert fft.spectral_energy(fft.spectrum(x, g), g) == pytest.approx(
        T0 * np.sqrt(np.pi), rel=1e-12
    )


@pytest.mark.parametrize("n", [512, 1000, 1023])
def test_result_independent_of_grid_length(n: int) -> None:
    """Even/odd N with the same dt give the same continuous spectrum samples."""
    g = _centered_grid(n=n)
    x = np.exp(-(g.time() ** 2) / (2 * T0**2))
    ref = _gaussian_ft(g.frequency())
    np.testing.assert_allclose(fft.spectrum(x, g), ref, rtol=0, atol=1e-9 * ref.max())
