"""End-to-end validation of the single-polarization coherent link against theory.

Statistical bounds: the data-aided SNR estimate from N = 32767 symbols has a
relative standard deviation ~1/sqrt(N) (0.024 dB); tests allow 5 sigma
(0.12 dB). Error counts are compared with the exact AWGN expectation using a
5-sigma Poisson band, widened on the upper side by the documented 0.2 dB
blind-phase-search penalty (tests/validation/test_coherent_dsp.py).
"""

from __future__ import annotations

import math

import pytest

from optobuild.analysis.constellations import theoretical_ber, theoretical_ser
from optobuild.cli.demos import coherent_link_project
from optobuild.core.constants import BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE
from optobuild.persistence import run_project
from optobuild.physics.noise import osnr_to_snr

RS = 32e9
SNR_TOL_DB = 0.12


def _ideal(**kw):  # type: ignore[no-untyped-def]
    """B2B link without laser phase noise, LO offset, receiver noise or modulator impairments;
    small IQ drive (0.05 V_pi) so the sine nonlinearity is negligible."""
    defaults = {"tx_linewidth": 0.0, "lo_linewidth": 0.0, "lo_offset": 0.0, "fiber_length_m": 0.0}
    p = coherent_link_project(**{**defaults, **kw})
    p.graph.set_parameters("receiver", shot_noise=False, thermal_noise=False)
    p.graph.set_parameters("iq_mod", extinction_ratio=1e12)
    p.graph.set_parameters("shaper", amplitude=0.2)
    return p


def _db(x: float) -> float:
    return 10 * math.log10(x)


@pytest.mark.parametrize("osnr_db", [10.0, 15.0, 20.0])
def test_measured_snr_equals_osnr_formula(osnr_db: float) -> None:
    r = run_project(_ideal(osnr_db=osnr_db, seed=5))
    expected = _db(osnr_to_snr(10 ** (osnr_db / 10), RS))
    assert r.result("analyzer", "snr_db") == pytest.approx(expected, abs=SNR_TOL_DB)


def _count_band(expected_fn, snr: float, n: int) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    lo = expected_fn(snr) * n
    hi = expected_fn(snr * 10 ** (-0.02)) * n  # 0.2 dB BPS allowance
    return lo - 5 * math.sqrt(lo) - 1, hi + 5 * math.sqrt(hi) + 1


def test_qpsk_ber_with_full_dsp_matches_theory() -> None:
    r = run_project(_ideal(osnr_db=9.0, seed=6))
    snr = osnr_to_snr(10**0.9, RS)
    lo, hi = _count_band(lambda g: theoretical_ber("qpsk", g), snr, r.result("analyzer", "n_bits"))
    assert lo <= r.result("analyzer", "bit_errors") <= hi
    assert lo > 100  # meaningful statistics


def test_16qam_ser_with_full_dsp_matches_theory() -> None:
    r = run_project(_ideal(osnr_db=19.0, seed=7, modulation="16qam"))
    snr = osnr_to_snr(10**1.9, RS)
    n = r.result("analyzer", "n_symbols")
    lo, hi = _count_band(lambda g: theoretical_ser("16qam", g), snr, n)
    assert lo <= r.result("analyzer", "symbol_errors") <= hi
    assert r.result("analyzer", "snr_db") == pytest.approx(_db(snr), abs=SNR_TOL_DB)


def test_receiver_shot_and_thermal_noise() -> None:
    """ASE-free (OSNR 40 dB), weak LO (0.2 mW): 1/SNR = 1/SNR_ase + 1/SNR_rx with
    SNR_rx = R^2 P_s P_lo / (G R_s), G = 2 q R (P_s + P_lo)/2 + 2 * 4 k T / R_L
    (per quadrature, RRC matched filter noise bandwidth R_s / 2)."""
    p = coherent_link_project(
        tx_linewidth=0.0, lo_linewidth=0.0, lo_offset=0.0, fiber_length_m=0.0, osnr_db=40.0, seed=8
    )
    p.graph.set_parameters("iq_mod", extinction_ratio=1e12)
    p.graph.set_parameters("shaper", amplitude=0.2)
    p.graph.set_parameters("lo", power=0.2e-3)
    r = run_project(p)
    ps, plo, resp = r.result("receiver", "signal_power_w"), r.result("receiver", "lo_power_w"), 0.8
    g = 2 * ELEMENTARY_CHARGE * resp * (ps + plo) / 2 + 2 * 4 * BOLTZMANN_CONSTANT * 300 / 50
    snr_rx = resp**2 * ps * plo / (g * RS)
    total = 1 / (1 / snr_rx + 1 / osnr_to_snr(1e4, RS))
    assert r.result("analyzer", "snr_db") == pytest.approx(_db(total), abs=SNR_TOL_DB)
    assert snr_rx < osnr_to_snr(1e4, RS) / 3  # receiver noise dominates in this test


def test_dispersion_is_fully_compensated_with_lo_offset() -> None:
    """80 km SMF (1360 ps/nm) with a 1 GHz LO offset: after CD compensation and fractional
    timing recovery the SNR equals the OSNR formula; without compensation it collapses."""
    p = _ideal(osnr_db=15.0, seed=9, fiber_length_m=80e3, lo_offset=1e9)
    r = run_project(p)
    assert r.result("analyzer", "snr_db") == pytest.approx(
        _db(osnr_to_snr(10**1.5, RS)), abs=SNR_TOL_DB
    )
    assert r.result("dsp", "frequency_offset_hz") == pytest.approx(-1e9, rel=1e-3)
    p.graph.set_parameters("dsp", cd_compensation=0.0)
    assert run_project(p).result("analyzer", "snr_db") < 3.0


def test_laser_phase_noise_small_penalty_no_cycle_slips() -> None:
    """100 kHz TX and LO linewidths at 32 GBd (combined dnu T = 6e-6): BPS tracks the phase,
    the SNR penalty is below 0.3 dB and the error count stays within the band (no slips,
    which would give ~50 % errors in the affected block)."""
    p = _ideal(osnr_db=9.0, seed=10, tx_linewidth=100e3, lo_linewidth=100e3)
    r = run_project(p)
    snr = osnr_to_snr(10**0.9, RS)
    assert r.result("analyzer", "snr_db") > _db(snr) - 0.3
    lo, hi = _count_band(
        lambda g: theoretical_ber("qpsk", g), snr * 10 ** (-0.01), r.result("analyzer", "n_bits")
    )
    assert r.result("analyzer", "bit_errors") <= hi
