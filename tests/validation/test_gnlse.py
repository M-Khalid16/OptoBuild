"""Validation of the generalized NLSE solver (RK4IP) against exact results
(docs/physics_models.md 3.27).

* pure NLSE limit: Satsuma-Yajima N = 2 soliton (closed form) and the
  fundamental soliton;
* exact dispersionless self-steepening intensity solution I = I0(T - 3 gamma I z / w0);
* conservation of the photon number with Raman and self-steepening (and energy
  loss by the Raman shift), energy conservation without them;
* Gordon's soliton self-frequency shift;
* fourth-order convergence of RK4IP.
Tolerances are stated per test from the method's error (step-doubling local
tolerance, O(h^4)) or the asymptotic nature of the reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.numerics.fft import spectrum
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.ultrafast import (
    raman_response_spectrum,
    raman_time,
    shock_intensity_delay,
    soliton_self_frequency_shift_rate,
)
from optobuild.solvers.gnlse import propagate_gnlse

F0 = SPEED_OF_LIGHT / 1550e-9
BETA2 = -20e-27
GAMMA = 1e-3
T0 = 100e-15
P1 = abs(BETA2) / (GAMMA * T0**2)
LD = T0**2 / abs(BETA2)
GRID = TimeGrid(1024, 8e-15, -512 * 8e-15)  # 82 T0 window, Nyquist 62 THz
T = GRID.time()
PURE = {"self_steepening": False, "raman_fraction": 0.0}


def _sech(p0: float, t0: float = T0, t: np.ndarray = T) -> np.ndarray:
    return (np.sqrt(p0) / np.cosh(t / t0)).astype(complex)


def test_raman_response_normalization_and_first_moment() -> None:
    assert raman_response_spectrum(np.array([0.0]))[0] == pytest.approx(1.0)
    # T_R from a numerical first moment of h_R(t) (independent of the closed form)
    tau1, tau2 = 12.2e-15, 32e-15
    t = np.linspace(0, 2e-12, 400001)
    h = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-t / tau2) * np.sin(t / tau1)
    assert trapezoid(h, t) == pytest.approx(1.0, rel=1e-6)
    assert 0.18 * trapezoid(t * h, t) == pytest.approx(raman_time(), rel=1e-6)


def test_fundamental_soliton_is_stationary() -> None:
    out, rep = propagate_gnlse(_sech(P1), GRID, 5 * LD, F0, betas=[BETA2], gamma=GAMMA,
                               tolerance=1e-9, **PURE)  # fmt: skip
    expected = _sech(P1) * np.exp(-0.5j * GAMMA * P1 * 5 * LD)
    np.testing.assert_allclose(out, expected, atol=1e-7 * math.sqrt(P1))
    assert rep.energy_change == pytest.approx(0.0, abs=1e-10)


def test_second_order_soliton_matches_satsuma_yajima() -> None:
    """Closed form as in tests/validation/test_ssfm.py (conjugated for our convention).
    Adaptive tolerance 1e-10 per step over ~100 steps: global error < 1e-7 of the peak."""
    xi = math.pi / 4
    tau = T / T0
    u = (4 * (np.cosh(3 * tau) + 3 * np.exp(4j * xi) * np.cosh(tau)) * np.exp(0.5j * xi)
         / (np.cosh(4 * tau) + 4 * np.cosh(2 * tau) + 3 * np.cos(4 * xi)))  # fmt: skip
    out, _ = propagate_gnlse(2 * _sech(P1), GRID, xi * LD, F0, betas=[BETA2], gamma=GAMMA,
                             tolerance=1e-10, **PURE)  # fmt: skip
    np.testing.assert_allclose(out, math.sqrt(P1) * np.conj(u), atol=1e-6 * math.sqrt(P1))


def test_rk4ip_is_fourth_order() -> None:
    a0 = 2 * _sech(P1)
    ref, _ = propagate_gnlse(a0, GRID, LD, F0, betas=[BETA2], gamma=GAMMA, n_steps=800)
    errs = [
        np.max(
            np.abs(
                propagate_gnlse(a0, GRID, LD, F0, betas=[BETA2], gamma=GAMMA, n_steps=n)[0] - ref
            )
        )  # fmt: skip
        for n in (50, 100)
    ]
    assert 12 < errs[0] / errs[1] < 20  # 2^4 = 16 asymptotically


def test_self_steepening_matches_the_exact_intensity_solution() -> None:
    """No dispersion, no Raman: I(z, T) = I0(T - kappa I z), kappa = 3 gamma / w0, before the
    shock distance; at half of it (phi_NL ~ 12 rad) with tolerance 1e-10 per step the
    residual is < 1e-6 of the peak."""
    t0, p0 = 50e-15, 1e3
    grid = TimeGrid(2048, 2e-15, -2048e-15)
    t = grid.time()
    kappa = shock_intensity_delay(GAMMA, 1.0, F0)
    z_shock = t0 / (kappa * p0 * math.sqrt(2 / math.e))

    def i0(x: np.ndarray) -> np.ndarray:
        return p0 * np.exp(-(x**2) / t0**2)

    z = 0.5 * z_shock
    out, _ = propagate_gnlse(np.sqrt(i0(t)).astype(complex), grid, z, F0, betas=[],
                             gamma=GAMMA, raman_fraction=0.0, tolerance=1e-10)  # fmt: skip
    intensity = np.abs(out) ** 2
    assert np.max(np.abs(intensity - i0(t - kappa * intensity * z))) < 1e-6 * p0
    # the peak is delayed (trailing-edge steepening in our time convention)
    assert t[np.argmax(intensity)] > 0


def test_photon_number_is_conserved_with_raman_and_self_steepening() -> None:
    """N = 2 soliton with Raman, shock and beta3: the Raman shift loses energy (red shift)
    but conserves photons exactly; with tolerance 1e-9 the drift is < 1e-7."""
    out, rep = propagate_gnlse(2 * _sech(P1), GRID, 3 * LD, F0, betas=[BETA2, 1e-40],
                               gamma=GAMMA, tolerance=1e-9)  # fmt: skip
    assert abs(rep.photon_number_change) < 1e-7
    assert rep.energy_change < -1e-3  # photons are red-shifted
    assert out.shape == (1024,)


def test_soliton_self_frequency_shift_follows_gordon() -> None:
    """T0 = 500 fs >> tau1, tau2: the linear Raman approximation behind Gordon's formula has
    relative corrections O((tau2/T0)^2) ~ 0.4 %; plus the mean shift includes the
    reshaping transient; tolerance 2 %."""
    t0 = 500e-15
    p0 = abs(BETA2) / (GAMMA * t0**2)
    ld = t0**2 / abs(BETA2)
    grid = TimeGrid(2048, 20e-15, -1024 * 20e-15)
    t = grid.time()
    z = 20 * ld
    out, _ = propagate_gnlse(_sech(p0, t0, t), grid, z, F0, betas=[BETA2], gamma=GAMMA,
                             self_steepening=False, tolerance=1e-9)  # fmt: skip
    w = grid.angular_frequency()
    s = np.abs(spectrum(out, grid)) ** 2
    mean_w = np.sum(w * s) / np.sum(s)
    rate = soliton_self_frequency_shift_rate(BETA2, t0, raman_time())
    assert mean_w / z == pytest.approx(rate, rel=0.02)


def test_linear_propagation_with_loss_is_exact() -> None:
    """gamma = 0: the interaction-picture step is exact (dispersion to 4th order + loss)."""
    a0 = _sech(P1)
    betas = [BETA2, 1e-40, 1e-55]
    out, rep = propagate_gnlse(a0, GRID, 2.0, F0, betas=betas, gamma=0.0, alpha=0.1, n_steps=1)
    w = GRID.angular_frequency()
    h = np.exp((-0.05 - 1j * (betas[0] * w**2 / 2 + betas[1] * w**3 / 6 + betas[2] * w**4 / 24))
               * 2.0)  # fmt: skip
    expected = np.fft.ifft(np.fft.fft(a0) * h)
    np.testing.assert_allclose(out, expected, atol=1e-12 * math.sqrt(P1))
    assert rep.energy_change == pytest.approx(math.exp(-0.2) - 1, rel=1e-9)
