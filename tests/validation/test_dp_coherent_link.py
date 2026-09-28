"""End-to-end validation of the dual-polarization coherent link and of the receiver
impairment compensation (docs/physics_models.md 3.20-3.21).

Reference: with OSNR defined over both polarizations and signal in both,
SNR per symbol = 2 B_ref OSNR / (2 R_s) (physics.noise.osnr_to_snr with
n_pol_signal = 2; Essiambre et al. 2010, eq. 11).

Tolerances: the pooled data-aided SNR estimate from N = 2 x 32767 symbols
has a relative standard deviation ~1/sqrt(N) (0.017 dB); 5 sigma = 0.085 dB,
plus EQ_PENALTY_DB = 0.1 dB for the adaptive equalizer's excess MSE (see
tests/validation/test_dual_pol_dsp.py). 16-QAM additionally carries the
documented 0.2 dB blind-phase-search allowance (test_coherent_dsp.py).
Error counts use 5-sigma Poisson bands around the exact AWGN expectation.
"""

from __future__ import annotations

import math

import pytest

from optobuild.analysis.constellations import theoretical_ber
from optobuild.cli.demos import coherent_link_project, dp_coherent_link_project
from optobuild.persistence import run_project
from optobuild.physics.noise import osnr_to_snr

RS = 32e9
STAT_DB = 5 * 10 / math.log(10) / math.sqrt(2 * 32767)
EQ_PENALTY_DB = 0.1
BPS_PENALTY_DB = 0.2


def _db(x: float) -> float:
    return 10 * math.log10(x)


def _ideal_dp(**kw):  # type: ignore[no-untyped-def]
    """No laser phase noise or receiver noise; linear, carrier-free modulators; the
    impairments under test are enabled per test."""
    defaults = {
        "tx_linewidth": 0.0,
        "lo_linewidth": 0.0,
        "lo_offset": 0.0,
        "fiber_length_m": 0.0,
        "mean_dgd": 0.0,
    }
    p = dp_coherent_link_project(**{**defaults, **kw})
    for pol in "xy":
        p.graph.set_parameters(f"receiver_{pol}", shot_noise=False, thermal_noise=False)
        p.graph.set_parameters(f"iq_{pol}", extinction_ratio=1e12)
        p.graph.set_parameters(f"shaper_{pol}", amplitude=0.2)
    return p


def _theory_db(osnr_db: float) -> float:
    return _db(osnr_to_snr(10 ** (osnr_db / 10), RS, n_pol_signal=2))


@pytest.mark.parametrize(
    "kw",
    [
        {"osnr_db": 12.0, "seed": 5},  # random SOP (Haar coupling), no DGD
        # 80 km CD (compensated), random PMD (mean DGD 15 ps ~ 0.5 symbol), 1 GHz LO offset
        {"osnr_db": 12.0, "seed": 6, "fiber_length_m": 80e3, "mean_dgd": 15e-12, "lo_offset": 1e9},
    ],
)
def test_dp_qpsk_snr_equals_osnr_formula(kw: dict) -> None:
    r = run_project(_ideal_dp(**kw))
    expected = _theory_db(kw["osnr_db"])
    assert r.result("analyzer", "snr_db") == pytest.approx(expected, abs=STAT_DB + EQ_PENALTY_DB)
    for pol in "xy":  # both tributaries recovered (no CMA singularity)
        tol = STAT_DB * math.sqrt(2) + EQ_PENALTY_DB  # one tributary: half the symbols
        assert r.result("analyzer", f"snr_db_{pol}") == pytest.approx(expected, abs=tol)


def test_dp_qpsk_ber_matches_theory() -> None:
    r = run_project(_ideal_dp(osnr_db=12.0, seed=7, fiber_length_m=80e3, mean_dgd=15e-12))
    snr = osnr_to_snr(10**1.2, RS, n_pol_signal=2)
    n = r.result("analyzer", "n_bits")
    lo = theoretical_ber("qpsk", snr) * n
    hi = theoretical_ber("qpsk", snr * 10 ** (-EQ_PENALTY_DB / 10)) * n
    errors = r.result("analyzer", "bit_errors")
    assert lo - 5 * math.sqrt(lo) - 1 <= errors <= hi + 5 * math.sqrt(hi) + 1
    assert lo > 300  # meaningful statistics


def test_dp_16qam_snr_with_pmd_and_offset() -> None:
    kw = {"osnr_db": 22.0, "seed": 8, "fiber_length_m": 80e3, "mean_dgd": 15e-12, "lo_offset": 1e9}
    r = run_project(_ideal_dp(modulation="16qam", **kw))
    expected = _theory_db(22.0)
    snr = r.result("analyzer", "snr_db")
    assert expected - STAT_DB - EQ_PENALTY_DB - BPS_PENALTY_DB <= snr <= expected + STAT_DB


def test_prbs15_64qam_is_flagged() -> None:
    """Documented limitation: a PRBS recurrence spanning < 3 symbols is diagnosed."""
    p = dp_coherent_link_project(modulation="64qam", prbs_order=15)
    p.graph.set_parameters("dsp", mimo_epochs=2, mimo_x_epochs=1)  # outcome irrelevant here
    r = run_project(p)
    assert any(d.code == "dsp.prbs_order_too_low" for d in r.all_diagnostics())


# -- receiver hybrid imbalance and skew --------------------------------------------------------


def _ideal_sp(**kw):  # type: ignore[no-untyped-def]
    defaults = {"tx_linewidth": 0.0, "lo_linewidth": 0.0, "lo_offset": 0.0, "fiber_length_m": 0.0}
    p = coherent_link_project(**{**defaults, **kw})
    p.graph.set_parameters("receiver", shot_noise=False, thermal_noise=False)
    p.graph.set_parameters("iq_mod", extinction_ratio=1e12)
    p.graph.set_parameters("shaper", amplitude=0.2)
    return p


IMPAIRED = {"hybrid_phase_error": math.radians(10.0), "quadrature_gain": 0.8, "iq_skew": 4e-12}


@pytest.mark.parametrize("modulation", ["qpsk", "16qam"])
def test_gsop_and_deskew_undo_hybrid_impairments(modulation: str) -> None:
    """With ASE as the only noise (added before the hybrid), GSOP followed by exact deskew
    inverts the receiver's linear I/Q distortion; the SNR equals that of the unimpaired
    receiver on the same noise realization up to the finite-sample GSOP estimate
    (correlation estimate error ~1/sqrt(N): < 0.01 dB)."""
    ref = run_project(_ideal_sp(osnr_db=15.0, seed=9, modulation=modulation))
    p = _ideal_sp(osnr_db=15.0, seed=9, modulation=modulation)
    p.graph.set_parameters("receiver", **IMPAIRED)
    uncompensated = run_project(p).result("analyzer", "snr_db")
    p.graph.set_parameters("dsp", iq_orthogonalization=True, iq_deskew=4e-12)
    compensated = run_project(p).result("analyzer", "snr_db")
    assert compensated == pytest.approx(ref.result("analyzer", "snr_db"), abs=0.01)
    assert uncompensated < ref.result("analyzer", "snr_db") - 1.0  # the impairment matters
