"""Noise power spectral densities and white-noise sampling.

Conventions (numerical_conventions.md sec. 4): electrical noise is specified by a
**one-sided** PSD G [A^2/Hz]; a real white sequence sampled at fs represents
the band [0, fs/2], so its per-sample variance is ``G fs / 2``.

Shot noise:    G_shot = 2 q I          [A^2/Hz]   (I = mean current incl. dark current)
Thermal noise: G_th   = 4 k_B T / R_L  [A^2/Hz]   (Johnson-Nyquist, load resistor)

Assumptions: white (frequency-independent) spectra over the simulated band;
Gaussian statistics (shot noise: valid for many photoelectrons per sample
interval, i.e. I dt / q >> 1).
References: Agrawal, Fiber-Optic Communication Systems, 5th ed., sec. 4.4;
Saleh & Teich, Fundamentals of Photonics, 3rd ed., sec. 19.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.constants import BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE


def shot_noise_psd(current: ArrayLike) -> NDArray[np.float64] | float:
    """One-sided shot-noise PSD 2 q I [A^2/Hz] (I >= 0 [A])."""
    return (2.0 * ELEMENTARY_CHARGE * np.asarray(current, dtype=float))[()]


def thermal_noise_psd(temperature: float, load_resistance: float) -> float:
    """One-sided thermal-noise current PSD 4 k_B T / R_L [A^2/Hz]."""
    return 4.0 * BOLTZMANN_CONSTANT * temperature / load_resistance


def white_noise_variance(
    one_sided_psd: ArrayLike, sample_rate: float
) -> NDArray[np.float64] | float:
    """Per-sample variance G fs / 2 of real white noise with one-sided PSD G."""
    return (np.asarray(one_sided_psd, dtype=float) * 0.5 * sample_rate)[()]


def real_white_noise(
    rng: np.random.Generator, one_sided_psd: ArrayLike, sample_rate: float, n_samples: int
) -> NDArray[np.float64]:
    """Zero-mean Gaussian samples with per-sample variance G fs / 2 (G may vary per sample)."""
    std = np.sqrt(white_noise_variance(one_sided_psd, sample_rate))
    return rng.standard_normal(n_samples) * std


def complex_white_noise(
    rng: np.random.Generator, two_sided_psd: float, sample_rate: float, shape: tuple[int, ...]
) -> NDArray[np.complex128]:
    """Circular complex Gaussian noise, two-sided PSD N [W/Hz]: per-sample variance N fs
    (N fs / 2 per quadrature), numerical_conventions.md sec. 4."""
    sigma = np.sqrt(0.5 * two_sided_psd * sample_rate)
    return sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def ase_psd_for_osnr(signal_power: float, osnr: float, reference_bandwidth: float) -> float:
    """ASE PSD per polarization [W/Hz] for a given OSNR.

    OSNR = P_signal / (N_ase,total B_ref) with the total ASE counted in both
    polarizations (standard OSA definition, B_ref = 12.5 GHz ~ 0.1 nm):
    N_per_pol = P_signal / (2 OSNR B_ref).
    """
    return signal_power / (2.0 * osnr * reference_bandwidth)


def osnr_to_snr(
    osnr: float, symbol_rate: float, reference_bandwidth: float = 12.5e9, n_pol_signal: int = 1
) -> float:
    """SNR per symbol after ideal matched filtering: SNR = 2 B_ref OSNR / (p R_s).

    p = number of polarizations carrying signal (Essiambre et al., J. Lightwave
    Technol. 28, 662 (2010), eq. 11). Assumes ASE-limited, ideal receiver.
    """
    return 2.0 * reference_bandwidth * osnr / (n_pol_signal * symbol_rate)


__all__ = [
    "ase_psd_for_osnr",
    "complex_white_noise",
    "osnr_to_snr",
    "real_white_noise",
    "shot_noise_psd",
    "thermal_noise_psd",
    "white_noise_variance",
]
