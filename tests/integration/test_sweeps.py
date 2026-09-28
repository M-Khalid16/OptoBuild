"""Parameter sweeps and generic Monte Carlo (Phase 10, ADR-0021).

Reference pipeline: clean recorder mean = 7.5 g + 10 exactly (ramp 0..15 times gain g
plus 10); noisy recorder mean = clean mean + mean of 16 N(0, sigma^2) samples, so over
independent trials its standard deviation is sigma / 4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.cli.demos import reference_project
from optobuild.persistence import dumps_project
from optobuild.sweeps.parameter_sweep import Axis, Probe, monte_carlo, sweep

CLEAN = Probe("clean", "mean")
NOISY = Probe("noisy", "mean")


def test_sweep_values_are_exact_and_the_project_is_not_modified() -> None:
    project = reference_project()
    before = dumps_project(project)
    gains = (0.5, 1.0, 2.0, 3.0)
    res = sweep(project, [Axis("gain", "gain", gains)], [CLEAN])
    np.testing.assert_allclose(res.mean("clean.mean"), [7.5 * g + 10 for g in gains], rtol=1e-15)
    assert res.values["clean.mean"].shape == (4, 1)
    assert dumps_project(project) == before
    csv = res.to_csv().splitlines()
    assert csv[0] == "gain.gain,trial,clean.mean" and len(csv) == 5


def test_two_axes_grid_and_common_random_numbers() -> None:
    res = sweep(
        reference_project(),
        [Axis("gain", "gain", (1.0, 2.0)), Axis("noise", "sigma", (0.1, 0.2))],
        [CLEAN, NOISY],
        trials=3,
    )
    assert res.values["noisy.mean"].shape == (2, 2, 3)
    # same trial -> same standard-normal draws: the noise part scales exactly with sigma
    noise = res.values["noisy.mean"] - res.values["clean.mean"]
    np.testing.assert_allclose(noise[:, 1, :], 2 * noise[:, 0, :], rtol=1e-12)
    np.testing.assert_allclose(noise[0], noise[1], rtol=1e-12)  # independent of the gain


def test_parallel_sweep_is_bit_identical_to_serial() -> None:
    axes = [Axis("noise", "sigma", (0.1, 0.3, 0.5))]
    serial = sweep(reference_project(), axes, [NOISY], trials=4)
    parallel = sweep(reference_project(), axes, [NOISY], trials=4, workers=2)
    np.testing.assert_array_equal(serial.values["noisy.mean"], parallel.values["noisy.mean"])


def test_monte_carlo_statistics_match_the_noise_model() -> None:
    """2000 trials: the mean is within 5 standard errors of 25; the sample std of the
    trial means estimates sigma/4 = 0.025 with relative std ~1/sqrt(2 (n - 1)) = 1.6 %
    (5 sigma: 8 %)."""
    est = monte_carlo(reference_project(), [NOISY], 2000)["noisy.mean"]
    assert abs(est.mean - 25.0) < 5 * est.standard_error
    assert est.std == pytest.approx(0.1 / 4, rel=5 / math.sqrt(2 * 1999))
    lo, hi = est.interval()
    assert lo < est.mean < hi


def test_sweep_input_errors() -> None:
    with pytest.raises(KeyError, match="no node 'nope'"):
        sweep(reference_project(), [Axis("nope", "gain", (1.0,))], [CLEAN])
    with pytest.raises(ValueError, match="no values"):
        sweep(reference_project(), [Axis("gain", "gain", ())], [CLEAN])
    with pytest.raises(TypeError, match="not a scalar"):
        sweep(reference_project(), [], [Probe("clean", "samples")])
    with pytest.raises(ValueError, match="at least 2"):
        monte_carlo(reference_project(), [NOISY], 1)
