"""IQ modulator, coherent detection, phase noise, ASE/OSNR and CD compensation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.core.constants import BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE
from optobuild.core.units import dispersion_to_beta2
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.detection import coherent_detection
from optobuild.physics.fiber import cd_compensation_transfer, propagate_linear
from optobuild.physics.modulation import iq_modulator_field
from optobuild.physics.noise import ase_psd_for_osnr, complex_white_noise, osnr_to_snr
from optobuild.physics.sources import wiener_phase_noise

VPI = 3.0


def test_iq_modulator_closed_form() -> None:
    v = np.linspace(-2 * VPI, 2 * VPI, 101)
    vi, vq = v, v[::-1] * 0.7
    h = iq_modulator_field(vi, vq, VPI, insertion_loss=0.5)
    expected = (
        -0.5
        * math.sqrt(0.5)
        * (np.sin(np.pi * vi / (2 * VPI)) + 1j * np.sin(np.pi * vq / (2 * VPI)))
    )
    np.testing.assert_allclose(h, expected, atol=1e-15)
    assert abs(iq_modulator_field(0.0, 0.0, VPI)) == pytest.approx(0.0, abs=1e-16)  # null


def test_iq_modulator_linear_regime() -> None:
    """|v| <= 0.05 V_pi: deviation from the linear map is the sine's cubic term, < 0.5 %."""
    v = 0.05 * VPI * np.array([1.0, -0.3, 0.7])
    h = iq_modulator_field(v, -v, VPI)
    lin = -(np.pi / (4 * VPI)) * (v - 1j * v)
    np.testing.assert_allclose(h, lin, rtol=(np.pi * 0.05 / 2) ** 2 / 6 * 1.01)


def test_finite_extinction_ratio_leaks_carrier() -> None:
    """At null each child MZM transmits eps = 1/sqrt(ER) in quadrature (physics.modulation)."""
    h = iq_modulator_field(0.0, 0.0, VPI, extinction_ratio=100.0)
    assert abs(h) == pytest.approx(0.5 * math.sqrt(2) * 0.1, rel=1e-12)


def test_noiseless_coherent_detection_recovers_field_times_lo() -> None:
    rng = np.random.default_rng(1)
    es = (rng.normal(size=(1, 256)) + 1j * rng.normal(size=(1, 256))) * 1e-2
    elo = np.full((1, 256), math.sqrt(10e-3) * np.exp(0.4j))
    i, q = coherent_detection(es, elo, 0.9, None, 1e11)
    np.testing.assert_allclose(i + 1j * q, 0.9 * es[0] * np.conj(elo[0]), atol=1e-15)


def test_coherent_detection_noise_variance() -> None:
    """Per quadrature: one-sided PSD 2 q R (P_s + P_lo)/2 + 2 * 4kT/R_L; variance = G fs / 2."""
    n, fs, r = 2**17, 1e11, 0.8
    p_lo, p_s = 5e-3, 1e-4
    es = np.full((1, n), math.sqrt(p_s), dtype=complex)
    elo = np.full((1, n), math.sqrt(p_lo), dtype=complex)
    i, q = coherent_detection(es, elo, r, np.random.default_rng(2), fs)
    g = 2 * ELEMENTARY_CHARGE * r * (p_s + p_lo) / 2 + 2 * 4 * BOLTZMANN_CONSTANT * 300 / 50
    tol = 5 * math.sqrt(2 / n)
    assert i.var() == pytest.approx(g * fs / 2, rel=tol)
    assert q.var() == pytest.approx(g * fs / 2, rel=tol)
    assert i.mean() == pytest.approx(r * math.sqrt(p_s * p_lo), rel=1e-2)


def test_wiener_phase_noise_statistics() -> None:
    """Phase increments over m samples: variance 2 pi dv m dt (5-sigma, chi-square)."""
    dv, dt, n = 1e6, 1e-11, 2**20
    phi = wiener_phase_noise(np.random.default_rng(4), dv, dt, n)
    assert phi[0] == 0.0
    for m in (1, 16, 256):
        d = phi[m:] - phi[:-m]
        n_eff = n // m  # independent increments
        assert d.var() == pytest.approx(2 * np.pi * dv * m * dt, rel=5 * math.sqrt(2 / n_eff))
    assert np.all(wiener_phase_noise(np.random.default_rng(0), 0.0, dt, 10) == 0)


def test_ase_osnr_definition() -> None:
    """Noise PSD per pol N = P/(2 OSNR B_ref): measured power in B_ref (x2 pols) = P/OSNR."""
    p, osnr, bref, fs, n = 1e-3, 10**1.5, 12.5e9, 200e9, 2**18
    psd = ase_psd_for_osnr(p, osnr, bref)
    noise = complex_white_noise(np.random.default_rng(5), psd, fs, (n,))
    assert np.mean(np.abs(noise) ** 2) == pytest.approx(psd * fs, rel=5 * math.sqrt(1 / n))
    in_bref = 2 * psd * bref  # both polarizations
    assert p / in_bref == pytest.approx(osnr)
    assert osnr_to_snr(osnr, 32e9) == pytest.approx(2 * bref * osnr / 32e9)
    assert osnr_to_snr(osnr, 32e9, n_pol_signal=2) == pytest.approx(bref * osnr / 32e9)


def test_cd_compensation_inverts_the_fiber() -> None:
    grid = TimeGrid.from_sample_rate(4096, 128e9)
    rng = np.random.default_rng(6)
    a0 = rng.normal(size=(1, 4096)) + 1j * rng.normal(size=(1, 4096))
    d, length, lam = 17e-6, 80e3, 1550e-9
    out, _ = propagate_linear(a0, grid, length, 0.0, dispersion_to_beta2(d, lam))
    back = apply_transfer_function(
        out, cd_compensation_transfer(grid.angular_frequency(), d * length, lam), grid
    )
    np.testing.assert_allclose(back, a0, atol=1e-10)
