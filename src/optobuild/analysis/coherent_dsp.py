"""Coherent-receiver DSP building blocks (pure functions on complex samples).

Blocks (applied by ``components.coherent.CoherentDSP`` in this order):

1. chromatic-dispersion compensation (``physics.fiber.cd_compensation_transfer``)
2. RRC matched filter (``numerics.pulse_shaping.matched_filter``)
3. timing: Oerder-Meyr square-law estimate of the fractional symbol-timing
   offset, tau/T = -arg(sum_n |y_n|^2 exp(-i 2 pi n / sps)) / (2 pi)
   (Oerder & Meyr, IEEE Trans. Commun. 36, 605 (1988); needs sps >= 3),
   followed by an exact band-limited fractional delay in the frequency
   domain; alternatively the integer offset maximizing mean |y|^2
4. frequency-offset estimation (4th power): for constellations with
   E[s^4] != 0 (QPSK, square QAM), y_k^4 contains a tone at 4 df; df is the
   peak of |FFT(y^4)| (zero-padded x8, parabolic interpolation) / 4.
   Unambiguous for |df| < R_s / 8.
5. carrier phase: blind phase search (BPS, Pfau et al., J. Lightwave Technol.
   27, 989 (2009)): for B test phases in [-pi/4, pi/4) (square QAM symmetry)
   choose, per symbol, the phase minimizing the summed squared distance to
   the nearest constellation point over a sliding window of 2N+1 symbols;
   the phase track is unwrapped modulo the symmetry angle.
6. power normalization to unit average energy.

The residual rotation ambiguity (multiples of pi/2 for square QAM, pi for
BPSK) is resolved in analysis against the transmitted reference
(``align_to_reference``), as is customary in simulation; real receivers use
pilots or differential coding.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.analysis.constellations import decide

SYMMETRY = {"bpsk": math.pi, "qpsk": math.pi / 2, "16qam": math.pi / 2, "64qam": math.pi / 2}


def best_sampling_phase(samples: ArrayLike, samples_per_symbol: int) -> int:
    """Integer offset o in [0, sps) maximizing mean |y[k sps + o]|^2."""
    y = np.asarray(samples, dtype=complex)
    m = y.reshape(-1, samples_per_symbol)
    return int(np.argmax(np.mean(np.abs(m) ** 2, axis=0)))


def estimate_timing_offset(samples: ArrayLike, samples_per_symbol: int) -> float:
    """Oerder-Meyr estimate of the symbol-timing offset, in samples, in [0, sps).

    The symbol instants of ``samples`` are at n = tau + k sps.
    """
    y = np.asarray(samples, dtype=complex)
    n = np.arange(y.size)
    x = np.sum(np.abs(y) ** 2 * np.exp(-2j * np.pi * n / samples_per_symbol))
    return float((-np.angle(x) / (2 * np.pi) * samples_per_symbol) % samples_per_symbol)


def fractional_advance(samples: ArrayLike, shift_samples: float) -> NDArray[np.complex128]:
    """y(t + shift) for a band-limited periodic sequence (exact via linear phase)."""
    y = np.asarray(samples, dtype=complex)
    f = np.fft.fftfreq(y.size)
    return np.fft.ifft(np.fft.fft(y) * np.exp(2j * np.pi * f * shift_samples))


def estimate_frequency_offset(symbols: ArrayLike, symbol_rate: float, zero_pad: int = 8) -> float:
    """4th-power frequency-offset estimate [Hz] (|df| < R_s/8)."""
    y = np.asarray(symbols, dtype=complex)
    n = y.size * zero_pad
    spec = np.abs(np.fft.fft(y**4, n))
    k = int(np.argmax(spec))
    # parabolic interpolation of the peak (in bins)
    a, b, c = spec[k - 1], spec[k], spec[(k + 1) % n]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    f = np.fft.fftfreq(n, d=1.0 / symbol_rate)
    return float((f[k] + delta * symbol_rate / n) / 4.0)


def remove_frequency_offset(
    symbols: ArrayLike, offset: float, symbol_rate: float
) -> NDArray[np.complex128]:
    """y_k exp(-i 2 pi df k T)."""
    y = np.asarray(symbols, dtype=complex)
    return y * np.exp(-2j * np.pi * offset * np.arange(y.size) / symbol_rate)


def blind_phase_search(
    symbols: ArrayLike, fmt: str, n_test: int = 32, half_window: int = 16
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """BPS carrier-phase recovery; returns (corrected symbols, unwrapped phase estimate)."""
    y = np.asarray(symbols, dtype=complex)
    sym = SYMMETRY[fmt]
    phases = -sym / 2 + sym * np.arange(n_test) / n_test
    rotated = y[None, :] * np.exp(-1j * phases)[:, None]
    dist = np.abs(rotated - decide(rotated.ravel(), fmt).reshape(rotated.shape)) ** 2
    # circular moving sum over 2N+1 symbols
    kernel_len = 2 * half_window + 1
    csum = np.concatenate([dist[:, -half_window:], dist, dist[:, :half_window]], axis=1)
    csum = np.cumsum(np.pad(csum, ((0, 0), (1, 0))), axis=1)
    window = csum[:, kernel_len:] - csum[:, :-kernel_len]
    raw = phases[np.argmin(window, axis=0)]
    k = 2 * math.pi / sym  # 4 for square QAM
    phi = np.unwrap(k * raw) / k
    return y * np.exp(-1j * phi), phi


def normalize_power(symbols: ArrayLike) -> NDArray[np.complex128]:
    """Scale to unit mean |y|^2."""
    y = np.asarray(symbols, dtype=complex)
    return y / math.sqrt(float(np.mean(np.abs(y) ** 2)))


def align_to_reference(
    received: ArrayLike, reference: ArrayLike, fmt: str
) -> tuple[int, int, NDArray[np.complex128]]:
    """Circular shift s and rotation k * symmetry that best align ``received`` to ``reference``.

    Returns (shift, rotation index, aligned received symbols) with
    aligned[j] = received[j + shift] * exp(-i k symmetry) compared to reference[j].
    """
    r = np.asarray(received, dtype=complex)
    s = np.asarray(reference, dtype=complex)
    corr = np.fft.ifft(np.fft.fft(r) * np.conj(np.fft.fft(s)))
    shift = int(np.argmax(np.abs(corr)))
    sym = SYMMETRY[fmt]
    k = int(np.round(np.angle(corr[shift]) / sym)) % int(round(2 * math.pi / sym))
    aligned = np.roll(r, -shift) * np.exp(-1j * k * sym)
    return shift, k, aligned


def evm_rms(received: ArrayLike, reference: ArrayLike) -> float:
    """EVM_rms = sqrt(mean |r - s|^2 / mean |s|^2) (data-aided)."""
    r = np.asarray(received, dtype=complex)
    s = np.asarray(reference, dtype=complex)
    return float(np.sqrt(np.mean(np.abs(r - s) ** 2) / np.mean(np.abs(s) ** 2)))


__all__ = [
    "SYMMETRY",
    "align_to_reference",
    "best_sampling_phase",
    "estimate_timing_offset",
    "fractional_advance",
    "blind_phase_search",
    "estimate_frequency_offset",
    "evm_rms",
    "normalize_power",
    "remove_frequency_offset",
]
