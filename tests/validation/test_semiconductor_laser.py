"""Validation of the semiconductor-laser rate equations, their integrator and Langevin noise
(docs/physics_models.md 3.25).

References: closed forms for eps = beta = 0 (threshold, L-I, turn-on delay,
small-signal relaxation oscillation), the algebraic chirp identity, the
RK4 order of convergence, and an independent frequency-domain linear-response
calculation of the RIN and FM-noise spectra (whose f -> 0 limit is Henry's
linewidth) compared with periodograms of the time-domain SDE.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from optobuild.core.constants import ELEMENTARY_CHARGE as Q
from optobuild.physics.semiconductor_laser import LaserParameters
from optobuild.solvers.laser_dynamics import simulate

BASE = LaserParameters()
IDEAL = dataclasses.replace(BASE, spontaneous_coupling=0.0, gain_compression=0.0)


def test_threshold_and_light_current_closed_forms() -> None:
    p = IDEAL
    n_th = p.transparency_density + 1 / (p.confinement * p.gain_coefficient * p.photon_lifetime)
    assert p.threshold_density == pytest.approx(n_th, rel=1e-15)
    assert p.threshold_current == pytest.approx(Q * p.volume * n_th / p.carrier_lifetime)
    i1, i2 = 0.05, 0.08
    p1, p2 = (p.output_power(p.steady_state(i)[1]) for i in (i1, i2))
    assert (p2 - p1) / (i2 - i1) == pytest.approx(p.slope_efficiency, rel=1e-12)
    n_below, s_below = p.steady_state(0.02)
    assert s_below == 0 and n_below == pytest.approx(0.02 * p.carrier_lifetime / (Q * p.volume))


@pytest.mark.parametrize("current", [0.01, 0.038, 0.06, 0.12])
def test_general_steady_state_is_a_fixed_point(current: float) -> None:
    n, s = BASE.steady_state(current)
    dn, ds, _ = BASE.rhs(current, n, s)
    assert abs(dn) * BASE.carrier_lifetime < 1e-9 * n
    assert abs(ds) * BASE.photon_lifetime < 1e-9 * max(s, 1.0)


def test_integrator_relaxes_to_the_steady_state() -> None:
    n = 4000
    i = np.full(n, 0.07)
    _, s, _ = simulate(BASE, i, 1e-12, initial=BASE.steady_state(0.05) + (0.0,))
    assert s[-1] == pytest.approx(BASE.steady_state(0.07)[1], rel=1e-6)


def test_turn_on_delay() -> None:
    """From N = 0 the carriers charge as N(t) = N_inf (1 - exp(-t/tau_n)) while the photon
    density is negligible (beta = 1e-8), so N reaches N_th at t_d = tau_n ln(I/(I - I_th)).
    Sampling 0.1 ps with linear interpolation: error << 0.1 ps."""
    p = dataclasses.replace(IDEAL, spontaneous_coupling=1e-8)
    current = 0.08
    dt = 0.1e-12
    n, s, _ = simulate(p, np.full(20000, current), dt, initial=(0.0, 0.0, 0.0))
    k = int(np.argmax(n >= p.threshold_density))
    t_cross = (k - 1 + (p.threshold_density - n[k - 1]) / (n[k] - n[k - 1])) * dt
    assert t_cross == pytest.approx(p.turn_on_delay(current), abs=0.1e-12)
    # the light turns on later (photon build-up from spontaneous emission)
    assert int(np.argmax(p.output_power(s) > 1e-3)) * dt > p.turn_on_delay(current)


def test_small_signal_relaxation_oscillation() -> None:
    """Step 60 -> 60.2 mA from steady state (eps = beta = 0): the response of the linearized
    two-pole system is a damped sinusoid; extrema are pi/Omega_R apart with amplitude ratio
    exp(-gamma_R pi / (2 Omega_R)). Linearization error O(dS/S0) = 0.8 %: tolerance 1 %
    for Omega_R and 2 % for gamma_R."""
    p = IDEAL
    dt = 0.2e-12
    n_samples = 10000
    i = np.full(n_samples, 0.0602)
    _, s, _ = simulate(p, i, dt, initial=p.steady_state(0.060) + (0.0,))
    x = s - p.steady_state(0.0602)[1]
    idx = [
        k
        for k in range(1, n_samples - 1)
        if abs(x[k]) >= abs(x[k - 1]) and abs(x[k]) >= abs(x[k + 1])
    ]
    idx = [k for k in idx if abs(x[k]) > 1e-3 * abs(x).max()][1:6]
    omega, gamma = p.relaxation_oscillation(0.0602)
    half_periods = np.diff(idx) * dt
    assert np.mean(half_periods) == pytest.approx(math.pi / omega, rel=0.01)
    ratios = np.abs(x[idx[1:]] / x[idx[:-1]])
    measured_gamma = -2 * np.log(ratios).mean() / np.mean(half_periods)
    assert measured_gamma == pytest.approx(gamma, rel=0.02)


def test_rk4_converges_with_fourth_order() -> None:
    """Large-signal modulation; error vs a 32-substep reference drops ~16x per halving."""
    dt = 2e-12
    t = np.arange(1500) * dt
    i = 0.06 + 0.02 * np.sign(np.sin(2 * np.pi * 2.5e9 * t))
    ref = simulate(BASE, i, dt, substeps=32)[1]
    errs = [np.max(np.abs(simulate(BASE, i, dt, substeps=m)[1] - ref)) for m in (4, 8)]
    assert 12 < errs[0] / errs[1] < 20


def test_chirp_identity() -> None:
    """dphi/dt = (alpha/2)(S'/S - Gamma beta N/(tau_n S)) exactly for eps = 0."""
    p = dataclasses.replace(BASE, gain_compression=0.0)
    t = np.arange(2000) * 1e-12
    i = 0.06 + 0.02 * np.sin(2 * np.pi * 3e9 * t)
    n, s, _ = simulate(p, i, 1e-12)
    dn, ds, dphi = p.rhs(i, n, s)
    spont = p.confinement * p.spontaneous_coupling * n / p.carrier_lifetime
    expected = 0.5 * p.linewidth_enhancement * (ds / s - spont / s)
    np.testing.assert_allclose(dphi, expected, rtol=1e-9, atol=1e-9 * np.abs(dphi).max())


def test_henry_linewidth_is_the_zero_frequency_fm_noise() -> None:
    p = dataclasses.replace(BASE, gain_compression=0.0)
    _, fm = p.noise_spectra(0.06, [0.0])
    # O(R_sp / n_p) corrections: R_sp/n_p ~ 1e-5 relative here
    assert 2 * math.pi * fm[0] == pytest.approx(p.henry_linewidth(0.06), rel=1e-4)
    no_alpha = dataclasses.replace(p, linewidth_enhancement=0.0)
    n0, s0 = p.steady_state(0.06)
    schawlow_townes = p.spontaneous_rate(n0) / (4 * math.pi * s0 * p.volume / p.confinement)
    assert 2 * math.pi * no_alpha.noise_spectra(0.06, [0.0])[1][0] == pytest.approx(
        schawlow_townes, rel=1e-12
    )


def test_langevin_simulation_matches_linear_response_spectra() -> None:
    """200 realizations x 20 ns, 1 ps samples. The periodogram of a Gaussian process has
    relative standard deviation 1 per bin; band averages over B bins and M realizations
    have 1/sqrt(M B). Bands avoid the relaxation peak (leakage). 5 sigma."""
    current, dt, n, m = 0.06, 1e-12, 20000, 200
    rng = np.random.default_rng(2)
    _, s, phi = simulate(BASE, np.full(n, current), dt, rng=rng, ensemble=m)
    s0 = BASE.steady_state(current)[1]
    nu = np.diff(phi, axis=1) / (2 * math.pi * dt)
    x = s[:, :-1] / s0 - 1
    f = np.fft.rfftfreq(n - 1, dt)

    def periodogram(y: np.ndarray) -> np.ndarray:
        y = y - y.mean(axis=1, keepdims=True)
        return np.mean(np.abs(np.fft.rfft(y, axis=1)) ** 2, axis=0) * dt / (n - 1)

    p_fm, p_rin = periodogram(nu), periodogram(x)
    for lo, hi in ((1, 40), (150, 200)):  # 0.05-2 GHz and 7.5-10 GHz
        rin, fm = BASE.noise_spectra(current, f[lo:hi])
        tol = 5 / math.sqrt(m * (hi - lo))
        assert p_fm[lo:hi].mean() / fm.mean() == pytest.approx(1.0, abs=tol)
        assert p_rin[lo:hi].mean() / rin.mean() == pytest.approx(1.0, abs=tol)
