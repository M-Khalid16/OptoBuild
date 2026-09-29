"""solvers.ode: RK4 accuracy, ensembles and the noise hook."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.solvers.ode import columns, integrate, rk4_step


def test_rk4_global_error_is_fourth_order() -> None:
    """y' = -y: RK4 per-step amplification 1 - h + h^2/2 - h^3/6 + h^4/24 (exact)."""
    for h in (0.1, 0.05):
        y = rk4_step(lambda t, y: (-y[0],), 0.0, (1.0,), h)[0]
        assert y == pytest.approx(1 - h + h**2 / 2 - h**3 / 6 + h**4 / 24, rel=1e-15)
    errs = []
    for m in (10, 20):
        y = integrate(lambda t, y: (-y[0],), (1.0,), 0.0, 1.0, 2, substeps=m)[-1][0]
        errs.append(abs(y - math.exp(-1.0)))
    assert 15 < errs[0] / errs[1] < 17


def test_integrate_arrays_time_dependent_and_noise() -> None:
    out = integrate(lambda t, y: (np.cos(t) + 0 * y[0],), (np.zeros(3),), 0.0, 0.1, 11, 4)
    assert out[-1][0] == pytest.approx(np.full(3, math.sin(1.0)), rel=1e-7)
    kicks = integrate(lambda t, y: (0.0,), (0.0,), 0.0, 1.0, 4, 2, noise=lambda t, y, h: (1.0,))
    assert columns(kicks)[0] == [0.0, 2.0, 4.0, 6.0]
    with pytest.raises(ValueError):
        integrate(lambda t, y: y, (1.0,), 0.0, 0.0, 3)
