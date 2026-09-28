"""Ultrashort-pulse characterization (docs/physics_models.md 3.27).

* FWHM of the main lobe of a sampled intensity profile, with linear
  interpolation of the half-maximum crossings (error O(dx^2 I''); resolve the
  pulse with >= 10 samples per FWHM);
* rms width sqrt(<x^2> - <x>^2) weighted by the profile;
* time-bandwidth product FWHM_t x FWHM_f (transform limits: Gaussian
  2 ln2 / pi = 0.4413, sech^2 0.3148);
* intensity autocorrelation A(tau) = sum I(t) I(t + tau) dt (circular, via FFT)
  and its deconvolution factor FWHM_AC / FWHM_pulse (Gaussian sqrt(2),
  sech^2 1.5427).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.numerics.grid import TimeGrid

AUTOCORRELATION_FACTORS = {"gaussian": math.sqrt(2.0), "sech2": 1.5427}
TRANSFORM_LIMITED_TBP = {"gaussian": 2 * math.log(2) / math.pi, "sech2": 0.3148}


def fwhm(x: ArrayLike, profile: ArrayLike) -> float:
    """Full width at half maximum of the main lobe (interpolated crossings)."""
    xs = np.asarray(x, dtype=float)
    y = np.asarray(profile, dtype=float)
    k = int(np.argmax(y))
    half = 0.5 * y[k]
    if half <= 0:
        return math.nan
    lo = k
    while lo > 0 and y[lo - 1] > half:
        lo -= 1
    hi = k
    while hi < y.size - 1 and y[hi + 1] > half:
        hi += 1
    if lo == 0 or hi == y.size - 1:
        return math.nan  # the half maximum is not reached inside the window
    left = xs[lo - 1] + (half - y[lo - 1]) * (xs[lo] - xs[lo - 1]) / (y[lo] - y[lo - 1])
    right = xs[hi] + (half - y[hi]) * (xs[hi + 1] - xs[hi]) / (y[hi + 1] - y[hi])
    return float(right - left)


def rms_width(x: ArrayLike, profile: ArrayLike) -> float:
    xs = np.asarray(x, dtype=float)
    y = np.asarray(profile, dtype=float)
    m = np.sum(xs * y) / np.sum(y)
    return float(math.sqrt(np.sum((xs - m) ** 2 * y) / np.sum(y)))


def spectral_intensity(field: ArrayLike, grid: TimeGrid) -> tuple[NDArray, NDArray]:
    """(baseband frequency [Hz] ascending, |A~(f)|^2) of a scalar envelope."""
    a = np.asarray(field, dtype=complex).reshape(-1)
    s = np.abs(np.fft.fftshift(np.fft.fft(a))) ** 2
    return grid.frequency(shifted=True), s


def time_bandwidth_product(field: ArrayLike, grid: TimeGrid) -> float:
    """FWHM_t [s] x FWHM_f [Hz] of intensity and power spectrum."""
    a = np.asarray(field, dtype=complex).reshape(-1)
    f, s = spectral_intensity(a, grid)
    return fwhm(grid.time(), np.abs(a) ** 2) * fwhm(f, s)


def intensity_autocorrelation(intensity: ArrayLike, grid: TimeGrid) -> tuple[NDArray, NDArray]:
    """(delay tau [s] centred, A(tau) normalized to 1 at tau = 0)."""
    i = np.asarray(intensity, dtype=float).reshape(-1)
    spec = np.fft.fft(i)
    ac = np.real(np.fft.ifft(np.abs(spec) ** 2))
    ac = np.fft.fftshift(ac) / ac[0]
    n = i.size
    tau = (np.arange(n) - n // 2) * grid.dt
    return tau, ac


__all__ = [
    "AUTOCORRELATION_FACTORS",
    "TRANSFORM_LIMITED_TBP",
    "fwhm",
    "intensity_autocorrelation",
    "rms_width",
    "spectral_intensity",
    "time_bandwidth_product",
]
