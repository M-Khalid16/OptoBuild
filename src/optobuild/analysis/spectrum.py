"""Optical spectrum (numerical_conventions.md sec. 4).

The two-sided envelope PSD ``S_k = |X_k|^2 / T`` [W/Hz] (summed over
polarizations) is computed on the absolute optical frequency axis
``nu_k = f_ref + f_k``. An optical spectrum analyzer with resolution bandwidth
RBW reports the power within RBW around each bin,

    P_RBW(nu_k) = sum_{|f_j - f_k| <= RBW/2} S_j df        [W]

(rectangular RBW filter, bins outside the simulated band contribute zero).
With RBW < df each bin is reported on its own (P = S_k df).
Wavelength axis: lambda_k = c / nu_k (vacuum).

Limitations: single-realization periodogram without averaging or windowing;
frequency resolution is limited to df = 1/T; content outside +-fs/2 is not
represented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.numerics.fft import power_spectral_density, to_display_order
from optobuild.signals.optical import OpticalSignal


@dataclass(frozen=True)
class OpticalSpectrum:
    """Spectrum in display order (increasing frequency)."""

    frequency_hz: NDArray[np.float64]
    """Absolute optical frequency [Hz]."""
    wavelength_m: NDArray[np.float64]
    """Vacuum wavelength [m] (decreasing)."""
    psd_w_per_hz: NDArray[np.float64]
    """Two-sided PSD summed over polarizations [W/Hz]."""
    power_per_rbw_w: NDArray[np.float64]
    """Power within the resolution bandwidth centred on each bin [W]."""
    resolution_bandwidth_hz: float
    """Effective RBW [Hz] (max of requested RBW and df)."""
    df_hz: float

    def total_power(self) -> float:
        """Integral of the PSD [W] (equals the average power, Parseval)."""
        return float(np.sum(self.psd_w_per_hz) * self.df_hz)


def optical_spectrum(signal: OpticalSignal, resolution_bandwidth: float = 0.0) -> OpticalSpectrum:
    """Compute the spectrum of ``signal`` with a rectangular RBW [Hz] (0 = per bin)."""
    grid = signal.grid
    psd = power_spectral_density(signal.field, grid).sum(axis=0)
    f, psd = to_display_order(grid.frequency(), psd)
    df = grid.df
    half_width = int(np.floor(0.5 * resolution_bandwidth / df + 1e-9))
    if half_width > 0:
        kernel = np.ones(2 * half_width + 1)
        per_rbw = np.convolve(psd * df, kernel, mode="same")
    else:
        per_rbw = psd * df
    nu = signal.center_frequency + f
    return OpticalSpectrum(
        frequency_hz=nu,
        wavelength_m=SPEED_OF_LIGHT / nu,
        psd_w_per_hz=psd,
        power_per_rbw_w=per_rbw,
        resolution_bandwidth_hz=max(resolution_bandwidth, df) if half_width else df,
        df_hz=df,
    )


__all__ = ["OpticalSpectrum", "optical_spectrum"]
