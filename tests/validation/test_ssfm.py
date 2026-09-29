"""Validation of the split-step Fourier NLSE solver against analytic solutions.

Grid: 4096 samples over 80 T0 (sech tails < 1e-17 at the edges; spectrum
resolved far beyond its 1e-12 level), so discretization errors are
dominated by the O(h^2) splitting error, which each test bounds explicitly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.core.units import db_per_km_to_per_m, dispersion_to_beta2
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.fiber import propagate_linear
from optobuild.physics.nonlinear import (
    effective_length,
    gamma_from_n2,
    nonlinear_length,
    soliton_order,
)
from optobuild.solvers.ssfm import propagate_nlse

BETA2 = dispersion_to_beta2(17e-6, 1550e-9)  # anomalous, ~ -21.7 ps^2/km
GAMMA = 1.3e-3
T0 = 5e-12
P1 = abs(BETA2) / (GAMMA * T0**2)  # fundamental-soliton peak power
LD = T0**2 / abs(BETA2)
N = 4096
GRID = TimeGrid(N, 80 * T0 / N, -(N // 2) * 80 * T0 / N)
T = GRID.time()


def _sech(p0: float) -> np.ndarray:
    return (np.sqrt(p0) / np.cosh(T / T0))[None, :].astype(complex)


def test_gamma_formula_matches_standard_fiber_value() -> None:
    """n2 = 2.6e-20 m^2/W, A_eff = 80 um^2, 1550 nm -> gamma ~ 1.32 /(W km) (Agrawal)."""
    g = gamma_from_n2(2.6e-20, 1550e-9, 80e-12)
    assert g == pytest.approx(1.3174e-3, rel=1e-3)
    assert soliton_order(GAMMA, P1, T0, BETA2) == pytest.approx(1.0)
    assert nonlinear_length(GAMMA, P1) == pytest.approx(LD)
    assert effective_length(50e3, 0.0) == 50e3
    alpha = db_per_km_to_per_m(0.2)
    assert effective_length(1e9, alpha) == pytest.approx(1 / alpha, rel=1e-9)


def test_linear_limit_equals_linear_fiber_exactly() -> None:
    a0 = _sech(P1)
    alpha = db_per_km_to_per_m(0.2)
    for kwargs in ({"n_steps": 7}, {"max_step": 900.0}):
        out, rep = propagate_nlse(
            a0, GRID, 5e3, alpha=alpha, beta2=BETA2, beta3=1e-40, gamma=0.0, **kwargs
        )
        ref, _ = propagate_linear(a0, GRID, 5e3, alpha, BETA2, 1e-40)
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * math.sqrt(P1))


@pytest.mark.parametrize("n_steps", [1, 3, 50])
def test_pure_spm_with_loss_is_exact(n_steps: int) -> None:
    """beta2 = 0: A(L) = A0 exp(-alpha L/2) exp(-i gamma |A0|^2 L_eff), for any step count."""
    p0, length = 0.5, 30e3
    alpha = db_per_km_to_per_m(0.2)
    a0 = (np.sqrt(p0) * np.exp(-(T**2) / (2 * T0**2)))[None, :].astype(complex)
    out, _ = propagate_nlse(a0, GRID, length, alpha=alpha, beta2=0.0, gamma=GAMMA, n_steps=n_steps)
    leff = effective_length(length, alpha)
    expected = a0 * math.exp(-alpha * length / 2) * np.exp(-1j * GAMMA * np.abs(a0) ** 2 * leff)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12 * math.sqrt(p0))


def test_spm_red_shifts_the_leading_edge() -> None:
    """Physical sign check: instantaneous frequency < 0 on the rising edge (T < 0)."""
    a0 = (np.sqrt(1.0) * np.exp(-(T**2) / (2 * T0**2)))[None, :].astype(complex)
    out, _ = propagate_nlse(a0, GRID, 1e3, alpha=0.0, beta2=0.0, gamma=GAMMA, n_steps=1)
    inst_freq = np.gradient(np.unwrap(np.angle(out[0])), T) / (2 * np.pi)
    lead, trail = (T > -2 * T0) & (T < -0.5 * T0), (T > 0.5 * T0) & (T < 2 * T0)
    assert np.all(inst_freq[lead] < 0) and np.all(inst_freq[trail] > 0)


def test_fundamental_soliton_is_stationary() -> None:
    """N = 1: A(z) = sqrt(P0) sech(T/T0) exp(-i gamma P0 z / 2) (our sign convention).

    Splitting error with max_phase = 1e-3 rad (5000 steps over 5 L_D) was measured
    at 4e-7 of the peak; the tolerance 1e-5 leaves margin while still being
    ~10^4 times smaller than the 2nd-order soliton reshaping it would miss.
    """
    z = 5 * LD
    out, rep = propagate_nlse(
        _sech(P1), GRID, z, alpha=0.0, beta2=BETA2, gamma=GAMMA, max_phase=1e-3
    )
    expected = _sech(P1) * np.exp(-0.5j * GAMMA * P1 * z)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-5 * math.sqrt(P1))
    assert rep.max_step_phase <= 1e-3 * (1 + 1e-9)


def test_second_order_soliton_period() -> None:
    """N = 2 soliton: |A| returns to its input shape after z0 = (pi/2) L_D."""
    a0 = 2 * _sech(P1)
    out, _ = propagate_nlse(
        a0, GRID, 0.5 * math.pi * LD, alpha=0.0, beta2=BETA2, gamma=GAMMA, n_steps=4096
    )
    peak = np.abs(a0).max()
    np.testing.assert_allclose(np.abs(out), np.abs(a0), rtol=0, atol=1e-4 * peak)
    mid, _ = propagate_nlse(
        a0, GRID, 0.25 * math.pi * LD, alpha=0.0, beta2=BETA2, gamma=GAMMA, n_steps=2048
    )
    assert np.abs(mid).max() > 1.5 * peak  # strong compression half-way (known behaviour)


def test_second_order_convergence() -> None:
    """Symmetric splitting: halving the step divides the error by ~4 (order 2)."""
    a0 = 2 * _sech(P1)
    z = 0.5 * math.pi * LD
    ref, _ = propagate_nlse(a0, GRID, z, alpha=0.0, beta2=BETA2, gamma=GAMMA, n_steps=8192)
    errors = []
    for n in (64, 128, 256):
        out, _ = propagate_nlse(a0, GRID, z, alpha=0.0, beta2=BETA2, gamma=GAMMA, n_steps=n)
        errors.append(np.max(np.abs(out - ref)))
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(1.8 < o < 2.2 for o in orders), orders


def test_energy_conserved_without_loss() -> None:
    a0 = 2 * _sech(P1)
    out, _ = propagate_nlse(
        a0, GRID, 3 * LD, alpha=0.0, beta2=BETA2, beta3=0.1e-39 * 1e3, gamma=GAMMA, max_phase=0.01
    )
    assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(a0) ** 2), rel=1e-12)


def test_step_control_and_errors() -> None:
    a0 = _sech(P1)
    _, rep = propagate_nlse(
        a0, GRID, 10e3, alpha=0.0, beta2=BETA2, gamma=GAMMA, max_phase=0.01, max_step=500.0
    )
    assert rep.max_step <= 500.0 + 1e-9 and rep.max_step_phase <= 0.01 * (1 + 1e-9)
    out, rep0 = propagate_nlse(a0, GRID, 0.0, alpha=0.0, beta2=BETA2, gamma=GAMMA)
    assert rep0.n_steps == 0 and np.array_equal(out, a0)
    with pytest.raises(ValueError):
        propagate_nlse(a0, GRID, -1.0, alpha=0.0, beta2=BETA2, gamma=GAMMA)
    with pytest.raises(ValueError):
        propagate_nlse(a0, GRID, 1.0, alpha=0.0, beta2=BETA2, gamma=GAMMA, n_steps=0)


@pytest.mark.parametrize("xi", [math.pi / 8, math.pi / 4, 0.9])
def test_second_order_soliton_matches_satsuma_yajima_solution(xi: float) -> None:
    """Full complex field vs the closed-form N = 2 soliton (Satsuma & Yajima 1974;
    Agrawal, Nonlinear Fiber Optics, eq. 5.2.16):

        u = 4 [cosh 3t + 3 exp(4 i xi) cosh t] exp(i xi / 2)
            / [cosh 4t + 4 cosh 2t + 3 cos 4 xi],   xi = z / L_D, t = T / T0,

    written for exp(-i w0 t) carriers; in our convention the field is
    sqrt(P1) * conj(u). Observed error with 4000 steps <= 3.7e-6 of the peak
    (and 16x smaller than with 1000 steps: order 2); tolerance 1e-5.
    """
    tau = T / T0
    u = (
        4
        * (np.cosh(3 * tau) + 3 * np.exp(4j * xi) * np.cosh(tau))
        * np.exp(0.5j * xi)
        / (np.cosh(4 * tau) + 4 * np.cosh(2 * tau) + 3 * np.cos(4 * xi))
    )
    expected = math.sqrt(P1) * np.conj(u)
    out, _ = propagate_nlse(
        2 * _sech(P1), GRID, xi * LD, alpha=0.0, beta2=BETA2, gamma=GAMMA, n_steps=4000
    )
    np.testing.assert_allclose(out[0], expected, rtol=0, atol=1e-5 * 4 * math.sqrt(P1))
