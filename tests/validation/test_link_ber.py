"""Validation: counted BER of a thermal-noise-limited link vs exact Gaussian theory.

Method (no curve fitting):
1. Run the complete link with receiver noise disabled -> noiseless decision
   samples m_k (including the filter's pattern-dependent ISI), the sampling
   offset and the aligned reference bits.
2. Fix timing and threshold to the values from step 1 and run again with
   thermal noise only (shot noise off, so the noise is exactly Gaussian with
   one-sided PSD G = 4 k_B T / R_L).
3. The filtered noise per sample has variance sigma^2 = G * NEB with NEB
   computed exactly on the grid, so the expected error count is
   sum_k 1/2 erfc(|m_k - th| / (sqrt(2) sigma)).
4. The counted errors must lie within 5 Poisson standard deviations of the
   expectation (plus 1 for discreteness). The bit decisions are independent
   up to the filter memory (a few bits), so the Poisson band is a reasonable,
   slightly optimistic bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.analysis.ber import gaussian_error_probability
from optobuild.cli.demos import optical_link_project
from optobuild.core.constants import BOLTZMANN_CONSTANT
from optobuild.engine import FeedForwardExecutor


def _link(power: float):  # type: ignore[no-untyped-def]
    p = optical_link_project(
        seed=31, prbs_order=15, samples_per_bit=8, laser_power_w=power, fiber_length_m=20e3
    )
    return p.graph


@pytest.mark.parametrize("power", [95e-6, 85e-6, 75e-6])  # BER ~ 1.4e-3 ... 9e-3
def test_counted_ber_matches_gaussian_theory(power: float) -> None:
    g = _link(power)
    g.set_parameters("pin", shot_noise=False, thermal_noise=False)
    clean = FeedForwardExecutor().run(g, seed=31)
    assert clean.result("ber", "n_errors") == 0
    offset = clean.result("decision", "sampling_offset")
    means = clean.result("decision", "decision_samples")
    shift = clean.result("ber", "alignment_shift_bits")
    ref = np.roll(clean.signal("prbs", "out").bits, shift)
    threshold = 0.5 * (means[ref == 1].mean() + means[ref == 0].mean())

    g.set_parameters("pin", thermal_noise=True)
    g.set_parameters(
        "decision",
        timing="fixed",
        sampling_phase=(offset + 0.5) / 8,
        threshold_mode="fixed",
        threshold=threshold,
    )
    noisy = FeedForwardExecutor().run(g, seed=31)
    assert noisy.result("decision", "sampling_offset") == offset

    neb = noisy.result("filter", "noise_equivalent_bandwidth_hz")
    sigma = math.sqrt(4 * BOLTZMANN_CONSTANT * 300.0 / 50.0 * neb)
    expected = float(np.sum(gaussian_error_probability(means, ref, threshold, sigma)))
    counted = noisy.result("ber", "n_errors")
    assert noisy.result("ber", "alignment_shift_bits") == shift
    assert expected > 20, "operating point must produce enough errors to be a meaningful test"
    assert abs(counted - expected) <= 5 * math.sqrt(expected) + 1, (counted, expected)
