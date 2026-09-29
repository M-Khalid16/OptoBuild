"""Validation: Monte Carlo BER accumulation vs exact Gaussian theory (method of test_link_ber).

Noise realizations of different trials are independent, the pattern is the
same, so the expected total is n_trials times the per-trial expectation
computed from the noiseless decision samples. Tolerance: 5 Poisson standard
deviations (+1), as in test_link_ber.py.
"""

from __future__ import annotations

import math

import numpy as np

from optobuild.analysis.ber import gaussian_error_probability
from optobuild.cli.demos import optical_link_project
from optobuild.core.constants import BOLTZMANN_CONSTANT
from optobuild.engine import FeedForwardExecutor, ResultCache
from optobuild.persistence import run_project
from optobuild.sweeps.monte_carlo import monte_carlo_ber


def _thermal_limited_link(power: float = 95e-6):  # type: ignore[no-untyped-def]
    p = optical_link_project(
        seed=31, prbs_order=11, samples_per_bit=8, laser_power_w=power, fiber_length_m=20e3
    )
    p.graph.set_parameters("pin", shot_noise=False, thermal_noise=False)
    clean = run_project(p)
    offset = clean.result("decision", "sampling_offset")
    means = clean.result("decision", "decision_samples")
    ref = np.roll(clean.signal("prbs", "out").bits, clean.result("ber", "alignment_shift_bits"))
    threshold = 0.5 * (means[ref == 1].mean() + means[ref == 0].mean())
    p.graph.set_parameters("pin", thermal_noise=True)
    p.graph.set_parameters(
        "decision",
        timing="fixed",
        sampling_phase=(offset + 0.5) / 8,
        threshold_mode="fixed",
        threshold=threshold,
    )
    neb = run_project(p).result("filter", "noise_equivalent_bandwidth_hz")
    sigma = math.sqrt(4 * BOLTZMANN_CONSTANT * 300.0 / 50.0 * neb)
    expected_per_trial = float(np.sum(gaussian_error_probability(means, ref, threshold, sigma)))
    return p, expected_per_trial


def test_accumulated_errors_match_theory() -> None:
    p, per_trial = _thermal_limited_link()
    n_trials = 20
    mc = monte_carlo_ber(p, n_trials)
    expected = n_trials * per_trial
    assert expected > 50
    assert mc.n_bits == n_trials * 2047
    assert abs(mc.n_errors - expected) <= 5 * math.sqrt(expected) + 1, (mc.n_errors, expected)
    assert mc.ber_lower_95 < expected / mc.n_bits < mc.ber_upper_95
    assert len(set(mc.errors_per_trial)) > 1  # trials are different realizations


def test_trials_are_reproducible_and_disjoint_ranges_differ() -> None:
    p, _ = _thermal_limited_link(85e-6)
    a = monte_carlo_ber(p, 3)
    b = monte_carlo_ber(p, 3)
    c = monte_carlo_ber(p, 3, first_trial=3)
    assert a.errors_per_trial == b.errors_per_trial
    assert a.errors_per_trial != c.errors_per_trial
    whole = monte_carlo_ber(p, 6)
    assert whole.errors_per_trial == a.errors_per_trial + c.errors_per_trial


def test_deterministic_part_is_computed_once() -> None:
    p, _ = _thermal_limited_link()
    cache = ResultCache()
    ex = FeedForwardExecutor(cache)
    monte_carlo_ber(p, 4, executor=ex)
    res = run_project(p, executor=ex, trial=2)
    hits = {n: r.cache_hit for n, r in res.nodes.items()}
    assert all(hits[n] for n in ("prbs", "nrz", "laser", "mzm", "fiber", "rx_power"))
    assert all(hits.values())  # trial 2 was already computed
