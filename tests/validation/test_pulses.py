"""Pulse metrics against closed forms (docs/physics_models.md 3.27).

Grid: 8192 samples, >= 150 samples per FWHM: linear interpolation of the half-maximum
crossings has relative error < 1e-5 (O((dx/FWHM)^2)); the window holds the pulse
tails below 1e-30.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.analysis.pulses import (
    AUTOCORRELATION_FACTORS,
    TRANSFORM_LIMITED_TBP,
    fwhm,
    intensity_autocorrelation,
    rms_width,
    time_bandwidth_product,
)
from optobuild.numerics.grid import TimeGrid

T0 = 100e-15
GRID = TimeGrid(8192, 0.5e-15, -4096 * 0.5e-15)
T = GRID.time()
GAUSS = np.exp(-(T**2) / (2 * T0**2)).astype(complex)  # intensity exp(-t^2/T0^2)
SECH = (1 / np.cosh(T / T0)).astype(complex)
TOL = 1e-5
# long window for spectra: df = 30 GHz, ~90 samples per spectral FWHM (interpolation
# error ~(1/90)^2 ~ 1e-4), 166 samples per temporal FWHM
LONG = TimeGrid(2**15, 1e-15, -(2**14) * 1e-15)
TL = LONG.time()


def test_fwhm_and_rms_closed_forms() -> None:
    assert fwhm(T, np.abs(GAUSS) ** 2) == pytest.approx(2 * math.sqrt(math.log(2)) * T0, rel=TOL)
    assert fwhm(T, np.abs(SECH) ** 2) == pytest.approx(2 * math.acosh(math.sqrt(2)) * T0,
                                                       rel=TOL)  # fmt: skip
    assert rms_width(T, np.abs(GAUSS) ** 2) == pytest.approx(T0 / math.sqrt(2), rel=TOL)
    assert math.isnan(fwhm(T, np.ones_like(T)))  # no half-maximum crossing in the window


@pytest.mark.parametrize("shape", ["gaussian", "sech2"])
def test_transform_limits_and_autocorrelation_factors(shape: str) -> None:
    long = np.exp(-(TL**2) / (2 * T0**2)) if shape == "gaussian" else 1 / np.cosh(TL / T0)
    tbp = time_bandwidth_product(long.astype(complex), LONG)
    assert tbp == pytest.approx(TRANSFORM_LIMITED_TBP[shape], rel=1e-3)
    field = GAUSS if shape == "gaussian" else SECH
    tau, ac = intensity_autocorrelation(np.abs(field) ** 2, GRID)
    ratio = fwhm(tau, ac) / fwhm(T, np.abs(field) ** 2)
    assert ratio == pytest.approx(AUTOCORRELATION_FACTORS[shape], rel=1e-4)
    assert ac.max() == pytest.approx(1.0)


def test_chirp_raises_the_time_bandwidth_product() -> None:
    gauss = np.exp(-(TL**2) / (2 * T0**2))
    chirped = gauss * np.exp(-0.5j * 3.0 * (TL / T0) ** 2)  # C = 3: TBP x sqrt(1 + C^2)
    assert time_bandwidth_product(chirped, LONG) == pytest.approx(
        TRANSFORM_LIMITED_TBP["gaussian"] * math.sqrt(10), rel=1e-3
    )
