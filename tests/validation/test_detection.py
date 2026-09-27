"""Validation: PIN photocurrent and noise statistics.

Statistical tolerances: for n independent Gaussian samples the relative
standard deviation of the sample variance is sqrt(2/(n-1)); tests allow 5 of
these standard deviations, so a correct implementation fails with probability
< 1e-6 for any seed (the seed only makes the outcome repeatable).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.components.detectors import PINPhotodiode
from optobuild.core.constants import BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.filters import lowpass_on_grid, noise_equivalent_bandwidth
from optobuild.numerics.grid import TimeGrid
from optobuild.signals import ElectricalQuantity, OpticalSignal

N = 2**16
FS = 160e9
GRID = TimeGrid.from_sample_rate(N, FS)


def _cw(power: float) -> OpticalSignal:
    return OpticalSignal(GRID, np.full(N, math.sqrt(power), dtype=complex), 193.4e12)


def _modulated() -> OpticalSignal:
    t = GRID.time()
    p = 1e-4 * (1.0 + 0.8 * np.cos(2 * np.pi * 7 * GRID.df * t))
    return OpticalSignal(GRID, np.sqrt(p), 193.4e12)


@pytest.mark.parametrize("r,i_d", [(1.0, 0.0), (0.8, 10e-9), (0.65, 2e-6)])
def test_noiseless_photocurrent_is_r_times_p_plus_dark(run_component, r: float, i_d: float) -> None:  # type: ignore[no-untyped-def]
    sig = _modulated()
    pin = PINPhotodiode(
        "pin", {"responsivity": r, "dark_current": i_d, "shot_noise": False, "thermal_noise": False}
    )
    out = run_component(pin, {"in": sig})[0]["out"]
    assert out.quantity is ElectricalQuantity.CURRENT
    np.testing.assert_allclose(out.samples, r * sig.power() + i_d, rtol=1e-14)
    assert out.mean() == pytest.approx(r * sig.average_power() + i_d, rel=1e-12)


def test_dual_polarization_detection_sums_powers(run_component) -> None:  # type: ignore[no-untyped-def]
    g = TimeGrid(8, 1e-12)
    sig = OpticalSignal(g, np.vstack([np.full(8, 1e-2), np.full(8, 2e-2j)]), 193e12)
    pin = PINPhotodiode("pin", {"shot_noise": False, "thermal_noise": False})
    out = run_component(pin, {"in": sig})[0]["out"]
    np.testing.assert_allclose(out.samples, 1e-4 + 4e-4, rtol=1e-14)


def _tol(n: int) -> float:
    return 5 * math.sqrt(2 / (n - 1))


def test_shot_noise_variance(run_component) -> None:  # type: ignore[no-untyped-def]
    power, r, i_d = 1e-3, 0.9, 1e-6
    pin = PINPhotodiode("pin", {"responsivity": r, "dark_current": i_d, "thermal_noise": False})
    out = run_component(pin, {"in": _cw(power)}, seed=1)[0]["out"]
    i_mean = r * power + i_d
    expected = 2 * ELEMENTARY_CHARGE * i_mean * FS / 2
    assert out.samples.var() == pytest.approx(expected, rel=_tol(N))
    assert out.samples.mean() == pytest.approx(i_mean, abs=5 * math.sqrt(expected / N))


def test_thermal_noise_variance(run_component) -> None:  # type: ignore[no-untyped-def]
    pin = PINPhotodiode(
        "pin", {"shot_noise": False, "temperature": 290.0, "load_resistance": 100.0}
    )
    out = run_component(pin, {"in": _cw(0.0)}, seed=2)[0]["out"]
    expected = 4 * BOLTZMANN_CONSTANT * 290.0 / 100.0 * FS / 2
    assert out.samples.var() == pytest.approx(expected, rel=_tol(N))


def test_filtered_noise_variance_equals_psd_times_neb(run_component) -> None:  # type: ignore[no-untyped-def]
    """After a filter, sigma^2 = (G_shot + G_th) * NEB, NEB computed on the grid (exact)."""
    power = 5e-4
    pin = PINPhotodiode("pin")
    out = run_component(pin, {"in": _cw(power)}, seed=3)[0]["out"]
    h = lowpass_on_grid(GRID, "butterworth", 10e9, 4)
    y = np.real(apply_transfer_function(out.samples - out.samples.mean(), h, GRID))
    g_total = 2 * ELEMENTARY_CHARGE * power + 4 * BOLTZMANN_CONSTANT * 300.0 / 50.0
    neb = noise_equivalent_bandwidth(GRID, h)
    n_eff = int(N * 2 * neb / FS)  # approx. number of independent filtered samples
    assert y.var() == pytest.approx(g_total * neb, rel=_tol(n_eff))


def test_noise_is_reproducible_and_seed_dependent(run_component) -> None:  # type: ignore[no-untyped-def]
    pin = PINPhotodiode("pin")
    a = run_component(pin, {"in": _cw(1e-4)}, seed=7)[0]["out"].samples
    b = run_component(pin, {"in": _cw(1e-4)}, seed=7)[0]["out"].samples
    c = run_component(pin, {"in": _cw(1e-4)}, seed=8)[0]["out"].samples
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
