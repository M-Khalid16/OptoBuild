"""Validation: deterministic BER counting, alignment, confidence bounds, Gaussian theory."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.special

from optobuild.analysis.ber import (
    ber_from_q,
    best_alignment,
    clopper_pearson,
    count_errors,
    gaussian_error_probability,
    q_factor,
)
from optobuild.analysis.decision import decide, max_variance_offset, sample_bits
from optobuild.components.analyzers import BERAnalyzer
from optobuild.components.electrical import DecisionCircuit
from optobuild.core.errors import SamplingError, SignalTypeError
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.prbs import prbs
from optobuild.signals import DigitalSequence, ElectricalSignal

REF = prbs(9, 511)


@pytest.mark.parametrize("positions", [[], [0], [5, 17, 300], list(range(0, 511, 7))])
def test_injected_errors_are_counted_exactly(positions: list[int]) -> None:
    rx = REF.copy()
    rx[positions] ^= 1
    res = count_errors(REF, rx)
    assert res.n_errors == len(positions) and res.n_bits == 511 and res.shift == 0
    assert res.ber == len(positions) / 511


@pytest.mark.parametrize("shift", [1, 13, 250, 510])
def test_alignment_recovers_circular_delay(shift: int) -> None:
    rx = np.roll(REF, shift)
    rx[[3, 99]] ^= 1
    assert best_alignment(REF, rx) == shift
    res = count_errors(REF, rx)
    assert (res.shift, res.n_errors) == (shift, 2)
    assert count_errors(REF, rx, align=False).n_errors > 100


def test_length_mismatch_rejected() -> None:
    with pytest.raises(SamplingError, match="differ in length"):
        count_errors(REF, REF[:-1])


def test_clopper_pearson() -> None:
    lo, hi = clopper_pearson(0, 1000)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 1000), rel=1e-9)
    lo, hi = clopper_pearson(10, 1000)
    assert lo < 0.01 < hi
    assert clopper_pearson(1000, 1000)[1] == 1.0


def test_q_to_ber_reference_values() -> None:
    """Classical values: Q = 6 -> 9.87e-10, Q = 7 -> 1.28e-12."""
    assert ber_from_q(6.0) == pytest.approx(9.8659e-10, rel=1e-4)
    assert ber_from_q(7.0) == pytest.approx(1.2798e-12, rel=1e-4)


def test_gaussian_error_probability_two_level_case_equals_q_formula() -> None:
    mu1, mu0, sigma = 1.0, 0.0, 0.1
    bits = np.array([1, 0] * 50)
    means = np.where(bits == 1, mu1, mu0)
    p = gaussian_error_probability(means, bits, 0.5, sigma)
    assert p.mean() == pytest.approx(ber_from_q((mu1 - mu0) / (2 * sigma)), rel=1e-12)
    # a mean on the wrong side of the threshold gives p > 1/2
    assert gaussian_error_probability([0.4], [1], 0.5, 0.1)[0] > 0.5


def test_q_factor_statistics(rng: np.random.Generator) -> None:
    bits = rng.integers(0, 2, 200_000)
    v = np.where(bits == 1, 1.0, 0.2) + rng.normal(0, 0.05, bits.size)
    mu1, mu0, s1, s0, q = q_factor(v, bits)
    assert (mu1, mu0) == (pytest.approx(1.0, abs=1e-3), pytest.approx(0.2, abs=1e-3))
    assert q == pytest.approx(0.8 / 0.1, rel=0.01)
    assert scipy.special.erfc(q / math.sqrt(2)) / 2 == pytest.approx(ber_from_q(q))


def _waveform(bits: np.ndarray, sps: int, lo: float, hi: float) -> ElectricalSignal:
    grid = TimeGrid(bits.size * sps, 1e-12)
    return ElectricalSignal(
        grid,
        np.repeat(np.where(bits == 1, hi, lo), sps),
        metadata={"bit_rate": 1 / (sps * 1e-12), "samples_per_bit": sps},
    )


def test_decision_on_constructed_waveform_is_exact(run_component) -> None:  # type: ignore[no-untyped-def]
    wf = _waveform(REF, 8, 1e-6, 9e-6)
    for params in (
        {},
        {"timing": "fixed", "sampling_phase": 0.1},
        {"threshold_mode": "fixed", "threshold": 5e-6},
    ):
        out, ctx = run_component(DecisionCircuit("d", params), {"in": wf})
        np.testing.assert_array_equal(out["bits"].bits, REF)
        assert out["bits"].bit_rate == pytest.approx(1.25e11)
    assert ctx.results["threshold"] == 5e-6


def test_decision_requires_timing_metadata(run_component) -> None:  # type: ignore[no-untyped-def]
    sig = ElectricalSignal(TimeGrid(16, 1e-12), np.zeros(16))
    with pytest.raises(SignalTypeError, match="bit timing"):
        run_component(DecisionCircuit("d"), {"in": sig})


def test_decision_primitives() -> None:
    x = np.tile([0.0, 1.0, 3.0, 1.0], 4) * np.repeat([1, -1, 1, -1], 4)
    assert max_variance_offset(x, 4) == 2
    np.testing.assert_array_equal(sample_bits(x, 4, 2), [3, -3, 3, -3])
    np.testing.assert_array_equal(decide([0.1, -0.1, 0.0], 0.0), [1, 0, 0])
    with pytest.raises(SamplingError):
        sample_bits(x, 5, 0)


def test_ber_analyzer_component(run_component) -> None:  # type: ignore[no-untyped-def]
    rx = np.roll(REF, 4)
    rx[[10, 20, 30]] ^= 1
    out, ctx = run_component(
        BERAnalyzer("ber"),
        {"reference": DigitalSequence(REF, 1e9), "received": DigitalSequence(rx, 1e9)},
    )
    assert out == {}
    r = ctx.results
    assert (r["n_bits"], r["n_errors"], r["alignment_shift_bits"]) == (511, 3, 4)
    assert r["ber"] == pytest.approx(3 / 511)
    assert r["ber_lower_95"] < r["ber"] < r["ber_upper_95"]
    assert [d.code for d in ctx.diagnostics] == ["ber.low_error_count"]
