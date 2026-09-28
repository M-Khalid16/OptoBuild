"""Coherent DSP blocks on synthetic data with known impairments."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.optimize

from optobuild.analysis.coherent_dsp import (
    align_to_reference,
    best_sampling_phase,
    blind_phase_search,
    estimate_frequency_offset,
    evm_rms,
    normalize_power,
    remove_frequency_offset,
)
from optobuild.analysis.constellations import bits_per_symbol, decide, map_bits, theoretical_ser

RS = 32e9


def _awgn_symbols(fmt: str, n: int, snr_db: float, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    s = map_bits(rng.integers(0, 2, n * bits_per_symbol(fmt)), fmt)
    n0 = 10 ** (-snr_db / 10)
    noise = math.sqrt(n0 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return s, noise, rng


@pytest.mark.parametrize("fmt", ["qpsk", "16qam"])
@pytest.mark.parametrize("df_over_rs", [0.013, -0.1, 0.12])
def test_fourth_power_frequency_offset_estimate(fmt: str, df_over_rs: float) -> None:
    """Accuracy far below one FFT bin (1/N = 5e-5 R_s) thanks to interpolation; tol 1e-6 R_s."""
    s, noise, _ = _awgn_symbols(fmt, 20_000, 15.0, 1)
    k = np.arange(s.size)
    y = (s * np.exp(0.7j) + noise) * np.exp(2j * np.pi * df_over_rs * k)
    assert estimate_frequency_offset(y, RS) == pytest.approx(df_over_rs * RS, abs=1e-6 * RS)


def _penalty_db(fmt: str, ser: float, snr_db: float) -> float:
    eq = scipy.optimize.brentq(lambda d: theoretical_ser(fmt, 10 ** (d / 10)) - ser, 0, 40)
    return snr_db - eq


@pytest.mark.parametrize(("fmt", "snr_db"), [("qpsk", 10.0), ("16qam", 17.0)])
def test_bps_penalty_is_small_versus_ideal_phase(fmt: str, snr_db: float) -> None:
    """BPS (B = 32, N = 16) adds < 0.2 dB over an ideal (known-phase) receiver on the same
    noise realization (Pfau et al. 2009 report < 0.2 dB for 16-QAM); the ideal receiver itself
    must agree with the exact SER within 5 sigma."""
    s, noise, _ = _awgn_symbols(fmt, 100_000, snr_db, 2)
    y = s * np.exp(0.3j) + noise
    genie = decide(y * np.exp(-0.3j), fmt)
    ser_genie = np.mean(genie != s)
    p = float(theoretical_ser(fmt, 10 ** (snr_db / 10)))
    assert ser_genie == pytest.approx(p, abs=5 * math.sqrt(p * (1 - p) / s.size))
    z, _ = blind_phase_search(y, fmt)
    shift, rot, aligned = align_to_reference(z, s, fmt)
    assert shift == 0
    ser_bps = np.mean(decide(aligned, fmt) != s)
    extra = _penalty_db(fmt, ser_bps, snr_db) - _penalty_db(fmt, max(ser_genie, 1e-9), snr_db)
    assert extra < 0.2, extra


def test_bps_tracks_laser_phase_noise_without_cycle_slips() -> None:
    """Wiener phase noise with linewidth x T = 1e-5 (e.g. 320 kHz combined linewidth at 32 GBd):
    the phase estimate follows the true phase (rms error < 0.05 rad) with a constant
    multiple-of-pi/2 offset, i.e. no cycle slips."""
    s, noise, rng = _awgn_symbols("qpsk", 50_000, 14.0, 3)
    phi = np.cumsum(rng.normal(0, math.sqrt(2 * math.pi * 1e-5), s.size)) + 1.0
    z, est = blind_phase_search(s * np.exp(1j * phi) + noise, "qpsk")
    err = phi - est
    offset = np.round(np.mean(err) / (math.pi / 2)) * (math.pi / 2)
    resid = err - offset
    assert np.sqrt(np.mean(resid**2)) < 0.05
    assert np.max(np.abs(resid)) < math.pi / 4  # a slip would jump by pi/2


def test_alignment_recovers_shift_and_rotation() -> None:
    s, _, _ = _awgn_symbols("16qam", 4096, 30.0, 4)
    r = np.roll(s, -123) * np.exp(1j * 3 * math.pi / 2)
    shift, rot, aligned = align_to_reference(r, s, "16qam")
    assert shift == 4096 - 123 and rot == 3
    np.testing.assert_allclose(aligned, s, atol=1e-12)
    assert evm_rms(aligned, s) == pytest.approx(0.0, abs=1e-12)


def test_evm_equals_inverse_sqrt_snr_for_awgn() -> None:
    s, noise, _ = _awgn_symbols("16qam", 200_000, 20.0, 5)
    assert evm_rms(s + noise, s) == pytest.approx(10 ** (-20 / 20), rel=0.01)


def test_helpers() -> None:
    x = np.tile([0.1, 1.0, 0.2, 0.05], 100) * np.exp(0.4j)
    assert best_sampling_phase(x, 4) == 1
    assert np.mean(np.abs(normalize_power(3 * x)) ** 2) == pytest.approx(1.0)
    y = remove_frequency_offset(np.ones(8), RS / 8, RS)
    np.testing.assert_allclose(y, np.exp(-2j * np.pi * np.arange(8) / 8))
