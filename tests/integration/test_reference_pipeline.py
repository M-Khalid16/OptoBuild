"""End-to-end validation of the framework on the reference (non-optical) pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.cli.demos import reference_project
from optobuild.engine import FeedForwardExecutor

N = np.arange(16, dtype=float)


def test_clean_path_is_exact() -> None:
    res = FeedForwardExecutor().run(reference_project().graph, seed=0)
    np.testing.assert_array_equal(res.result("clean", "samples"), 2 * N + 10)
    assert res.result("clean", "mean") == 25.0
    assert res.result("clean", "rms") == pytest.approx(np.sqrt(710.0), rel=1e-15)


def test_noisy_path_statistics_are_consistent_with_sigma() -> None:
    """Residual = noise with sigma 0.1 V: 16 samples, check within a 5-sigma band."""
    res = FeedForwardExecutor().run(reference_project().graph, seed=0)
    resid = res.result("noisy", "samples") - (2 * N + 10)
    assert abs(resid.mean()) < 5 * 0.1 / 4  # std of the mean = sigma/sqrt(16)
    assert 0.02 < resid.std() < 0.2


def test_every_intermediate_signal_is_inspectable() -> None:
    res = FeedForwardExecutor().run(reference_project().graph)
    np.testing.assert_array_equal(res.signal("ramp", "out").samples, N)
    np.testing.assert_array_equal(res.signal("gain", "out").samples, 2 * N)
    np.testing.assert_array_equal(res.signal("adder", "out").samples, 2 * N + 10)
    assert res.signal("adder", "out").grid.sample_rate == pytest.approx(1e9)
