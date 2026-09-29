"""Validation of Jones-matrix optics, DGD/PMD and hybrid impairments (physics_models.md 3.20-3.21).

Deterministic checks are exact up to round-off (tolerance 1e-12 relative).
The PMD statistics test compares a Monte Carlo mean of tau^2 with the exact
E[tau^2] = N tau_s^2 using a 5-sigma band from the sample standard deviation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.numerics.grid import TimeGrid
from optobuild.physics.detection import coherent_detection, delay_samples
from optobuild.physics.polarization import (
    MAXWELLIAN_MEAN_TO_RMS,
    apply_jones,
    dgd_element,
    dgd_from_jones,
    pmd_jones,
    random_unitary,
    rotation,
    sop_transform,
    waveplate,
    waveplate_section_dgd,
)

EYE = np.eye(2)


def _is_unitary(m: np.ndarray) -> bool:
    return bool(np.allclose(m @ m.conj().T, EYE, atol=1e-12))


@pytest.mark.parametrize("theta,phi", [(0.3, 0.0), (1.1, -0.7), (math.pi / 4, math.pi / 4)])
def test_sop_transform_is_special_unitary(theta: float, phi: float) -> None:
    u = sop_transform(theta, phi)
    assert _is_unitary(u)
    assert np.linalg.det(u) == pytest.approx(1.0, abs=1e-12)


def test_half_wave_plate_at_22_5_deg_swaps_x_to_45_deg() -> None:
    out = waveplate(math.pi, math.pi / 8) @ np.array([1.0, 0.0])
    assert np.abs(out) ** 2 == pytest.approx([0.5, 0.5], abs=1e-12)


def test_quarter_wave_plate_makes_circular_polarization() -> None:
    out = waveplate(math.pi / 2, math.pi / 4) @ np.array([1.0, 0.0])
    # circular: equal amplitudes, +-90 degree relative phase
    assert np.abs(out) ** 2 == pytest.approx([0.5, 0.5], abs=1e-12)
    assert abs(np.angle(out[1] / out[0])) == pytest.approx(math.pi / 2, abs=1e-12)


def test_haar_unitaries_are_unitary_and_isotropic() -> None:
    """Uniform on the Poincare sphere: the output Stokes s1 = |x|^2 - |y|^2 of x-polarized
    light is uniform on [-1, 1] (mean 0, variance 1/3). 4000 draws: 5 sigma bands."""
    rng = np.random.default_rng(1)
    us = [random_unitary(rng) for _ in range(4000)]
    assert all(_is_unitary(u) for u in us[:50])
    s1 = np.array([abs(u[0, 0]) ** 2 - abs(u[1, 0]) ** 2 for u in us])
    n = s1.size
    assert abs(s1.mean()) < 5 * math.sqrt(1 / 3 / n)
    # var of s1^2 for U(-1, 1): E[s^4] - E[s^2]^2 = 1/5 - 1/9
    assert abs(np.mean(s1**2) - 1 / 3) < 5 * math.sqrt((1 / 5 - 1 / 9) / n)


def test_dgd_element_delays_axes_by_plus_minus_half_tau() -> None:
    """A Gaussian pulse on x arrives tau/2 late, on y tau/2 early (centroid, exact for a
    band-limited pulse well inside the window)."""
    grid = TimeGrid(1024, 1e-12, -512e-12)
    t = grid.time()
    pulse = np.exp(-(t**2) / (2 * (5e-12) ** 2))
    field = np.vstack([pulse, pulse]).astype(complex)
    tau = 8e-12
    out = apply_jones(field, dgd_element(grid.angular_frequency(), tau), grid)
    p = np.abs(out) ** 2
    centroids = (p * t).sum(axis=1) / p.sum(axis=1)
    assert centroids == pytest.approx([tau / 2, -tau / 2], abs=1e-18)


def test_apply_jones_conserves_power_and_matches_constant_matrix() -> None:
    rng = np.random.default_rng(2)
    grid = TimeGrid(256, 1e-12)
    field = rng.standard_normal((2, 256)) + 1j * rng.standard_normal((2, 256))
    u = random_unitary(rng)
    const = apply_jones(field, u, grid)
    freq = apply_jones(field, np.repeat(u[:, :, None], 256, axis=2), grid)
    assert np.allclose(const, freq, atol=1e-12)
    j = pmd_jones(grid.angular_frequency(), [3e-12, 2e-12], [random_unitary(rng)] * 2)
    out = apply_jones(field, j, grid)
    assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(field) ** 2), rel=1e-12)


def _dgd_of(taus, us, dw=1e6):  # type: ignore[no-untyped-def]
    j = pmd_jones(np.array([-dw / 2, dw / 2]), taus, us)
    return dgd_from_jones(j[:, :, 0], j[:, :, 1], dw)


def test_eigenanalysis_recovers_section_dgd_sums() -> None:
    """Aligned sections add (tau1 + tau2); crossed sections (90 deg axes) subtract."""
    u0 = np.eye(2, dtype=complex)
    r90 = rotation(math.pi / 2)
    assert _dgd_of([3e-12], [random_unitary(np.random.default_rng(3))]) == pytest.approx(
        3e-12, rel=1e-9
    )
    assert _dgd_of([3e-12, 2e-12], [u0, u0]) == pytest.approx(5e-12, rel=1e-9)
    assert _dgd_of([3e-12, 2e-12], [u0, r90]) == pytest.approx(1e-12, rel=1e-9)


def test_orthogonal_stokes_axes_add_in_quadrature() -> None:
    """Principal axes 45 deg apart in azimuth are orthogonal on the Poincare sphere (a
    Jones rotation by theta rotates Stokes vectors by 2 theta): |Omega| = sqrt(tau1^2 + tau2^2)
    at w = 0 (frequency-dependent away from it; dw tau << 1 keeps the O(dw^2) error < 1e-6)."""
    u0 = np.eye(2, dtype=complex)
    got = _dgd_of([3e-12, 4e-12], [u0, rotation(math.pi / 4)])
    assert got == pytest.approx(5e-12, rel=1e-6)


def test_random_waveplate_pmd_second_moment_is_exact() -> None:
    """E[tau^2] = N tau_s^2 exactly (independent Haar coupling). 3000 fibers of 20 sections."""
    rng = np.random.default_rng(4)
    n, tau_s = 20, waveplate_section_dgd(10e-12, 20)
    sq = np.array(
        [
            _dgd_of(np.full(n, tau_s), [random_unitary(rng) for _ in range(n)]) ** 2
            for _ in range(3000)
        ]
    )
    expected = n * tau_s**2
    assert abs(sq.mean() - expected) < 5 * sq.std() / math.sqrt(sq.size)
    # The PMD vector performs an isotropic random flight of N equal steps tau_s, so the
    # DGD distribution equals that of the length of a 3-D random walk (Rayleigh's random
    # flight), simulated independently here; two-sample 5-sigma band on the mean.
    tau = np.sqrt(sq)
    steps = rng.standard_normal((20000, n, 3))
    steps *= tau_s / np.linalg.norm(steps, axis=2, keepdims=True)
    flight = np.linalg.norm(steps.sum(axis=1), axis=1)
    sigma = math.hypot(tau.std() / math.sqrt(tau.size), flight.std() / math.sqrt(flight.size))
    assert abs(tau.mean() - flight.mean()) < 5 * sigma
    # Maxwellian limit: the flight's exact moment ratio E[R^4]/E[R^2]^2 = 5/3 - 2/(3N)
    # differs from the Maxwellian 5/3 by 2/(3N) = 3 % at N = 20 and the mean is less
    # sensitive; allow 1 % for the finite-N shape plus 5 sigma of the flight's mean.
    tol = 0.01 * 10e-12 + 5 * flight.std() / math.sqrt(flight.size)
    assert abs(flight.mean() - 10e-12) < tol
    assert MAXWELLIAN_MEAN_TO_RMS == pytest.approx(0.9213, abs=1e-4)


# -- hybrid imbalance and skew ---------------------------------------------------------------


def test_hybrid_phase_error_and_quadrature_gain() -> None:
    """i_Q = g R Im(z exp(-i eps)) exactly (noiseless)."""
    rng = np.random.default_rng(5)
    es = (rng.standard_normal(64) + 1j * rng.standard_normal(64))[None, :] * 1e-3
    elo = np.full((1, 64), math.sqrt(1e-3), dtype=complex)
    eps, g, r = 0.1, 0.9, 0.8
    i, q = coherent_detection(es, elo, r, None, 1.0, phase_error=eps, quadrature_gain=g)
    z = (es * elo.conj())[0]
    assert np.allclose(i, r * z.real, rtol=1e-12, atol=1e-18)
    assert np.allclose(q, g * r * (z * np.exp(-1j * eps)).imag, rtol=1e-12, atol=1e-18)


def test_delay_samples_is_exact_for_band_limited_tone() -> None:
    grid = TimeGrid(128, 1.0)
    t = grid.time()
    f = 5 / 128
    x = np.cos(2 * np.pi * f * t)
    assert np.allclose(delay_samples(x, 0.3, grid), np.cos(2 * np.pi * f * (t - 0.3)), atol=1e-12)
