from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid


def test_derived_quantities() -> None:
    g = TimeGrid(n_samples=1024, dt=0.5e-12, t0=-1e-10)
    assert g.sample_rate == pytest.approx(2e12)
    assert g.duration == pytest.approx(512e-12)
    assert g.df == pytest.approx(1 / 512e-12)
    assert g.nyquist_frequency == pytest.approx(1e12)
    t = g.time()
    assert t[0] == pytest.approx(-1e-10)
    assert np.allclose(np.diff(t), g.dt)


def test_frequency_grid_is_fft_order() -> None:
    g = TimeGrid(8, 1.0)
    np.testing.assert_allclose(g.frequency(), [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125])
    np.testing.assert_allclose(g.frequency(shifted=True), np.sort(g.frequency()))
    np.testing.assert_allclose(g.angular_frequency(), 2 * np.pi * g.frequency())


def test_odd_length_grid() -> None:
    g = TimeGrid(5, 1.0)
    np.testing.assert_allclose(g.frequency(shifted=True), [-0.4, -0.2, 0.0, 0.2, 0.4])


def test_from_sample_rate_and_compatibility() -> None:
    a = TimeGrid.from_sample_rate(64, 1e11)
    b = a.with_t0(5e-9)
    assert a.dt == pytest.approx(1e-11)
    assert a.is_compatible(b)
    assert not a.is_compatible(TimeGrid(64, 2e-11))
    assert not a.is_compatible(TimeGrid(32, 1e-11))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 1, "dt": 1.0},
        {"n_samples": 2.5, "dt": 1.0},
        {"n_samples": True, "dt": 1.0},
        {"n_samples": 8, "dt": 0.0},
        {"n_samples": 8, "dt": -1.0},
        {"n_samples": 8, "dt": math.nan},
        {"n_samples": 8, "dt": 1.0, "t0": math.inf},
    ],
)
def test_invalid_grids_rejected(kwargs: dict) -> None:
    with pytest.raises(SamplingError):
        TimeGrid(**kwargs)


def test_invalid_sample_rate() -> None:
    with pytest.raises(SamplingError, match="sample_rate"):
        TimeGrid.from_sample_rate(8, 0.0)


def test_grid_is_immutable() -> None:
    g = TimeGrid(8, 1.0)
    with pytest.raises(AttributeError):
        g.dt = 2.0  # type: ignore[misc]
