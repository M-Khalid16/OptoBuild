"""Low-pass filter frequency responses evaluated on a simulation grid.

All filters are specified by their -3 dB (half-power) bandwidth ``B`` [Hz]
and evaluated as H(f_k) on the FFT-order baseband grid, then applied with
``numerics.fft.apply_transfer_function`` (circular convolution).

Kinds
-----
* ``gaussian``:  H(f) = exp(-(ln 2 / 2) (f/B)^2)         (zero phase, |H(B)|^2 = 1/2)
  Impulse response: a Gaussian in time with standard deviation
  sigma_t = sqrt(ln 2) / (2 pi B), unit area.
* ``rectangular``: H(f) = 1 for |f| <= B, else 0          (ideal brick wall, zero phase)
* ``butterworth``: analog prototype of order n, |H|^2 = 1 / (1 + (f/B)^(2n))
* ``bessel``: analog Bessel-Thomson of order n, normalized so |H(B)| = 1/sqrt(2)
  (scipy.signal.bessel(norm="mag")); maximally flat group delay.

Butterworth and Bessel are causal analog responses H(s) evaluated at
s = i 2 pi f (scipy.signal.freqs). They include their group delay, which
shifts the waveform in time.
At the Nyquist bin of an even-length grid (f = -fs/2, which has no +fs/2
partner) the response is replaced by its real part so that real inputs stay
real; this affects only content at exactly fs/2.

Noise-equivalent bandwidth (one-sided) ``NEB = integral_0^inf |H(f)|^2 df``:
gaussian: B sqrt(pi / (4 ln 2)) ~= 1.0645 B; rectangular: B;
butterworth order n: B (pi / 2n) / sin(pi / 2n).
Reference: A. V. Oppenheim, A. S. Willsky, Signals and Systems, 2nd ed.;
W. E. Thomson, Proc. IEE 96 (1949) for Bessel filters.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from optobuild.core.errors import InvalidParameterError
from optobuild.numerics.grid import TimeGrid

FILTER_KINDS = ("bessel", "butterworth", "gaussian", "rectangular")


def lowpass_response(
    frequency: NDArray[np.float64], kind: str, bandwidth: float, order: int = 4
) -> NDArray[np.complex128]:
    """Complex response H(f) of a low-pass filter with -3 dB bandwidth ``bandwidth`` [Hz]."""
    if bandwidth <= 0:
        raise InvalidParameterError(f"Filter bandwidth must be > 0 Hz, got {bandwidth}.")
    f = np.asarray(frequency, dtype=float)
    if kind == "gaussian":
        return np.exp(-0.5 * math.log(2.0) * (f / bandwidth) ** 2).astype(complex)
    if kind == "rectangular":
        return (np.abs(f) <= bandwidth).astype(complex)
    if kind in ("butterworth", "bessel"):
        if not 1 <= order <= 12:
            raise InvalidParameterError(f"Filter order must be in [1, 12], got {order}.")
        wc = 2.0 * math.pi * bandwidth
        if kind == "butterworth":
            b, a = scipy.signal.butter(order, wc, btype="low", analog=True)
        else:
            b, a = scipy.signal.bessel(order, wc, btype="low", analog=True, norm="mag")
        _, h = scipy.signal.freqs(b, a, worN=2.0 * math.pi * f)
        return np.asarray(h, dtype=complex)
    raise InvalidParameterError(
        f"Unknown filter kind {kind!r}.", hint=f"Use one of {list(FILTER_KINDS)}."
    )


def lowpass_on_grid(
    grid: TimeGrid, kind: str, bandwidth: float, order: int = 4
) -> NDArray[np.complex128]:
    """H(f_k) on ``grid`` (FFT order) with the Nyquist bin made real for even N."""
    h = lowpass_response(grid.frequency(), kind, bandwidth, order)
    if grid.n_samples % 2 == 0:
        h[grid.n_samples // 2] = h[grid.n_samples // 2].real
    return h


def noise_equivalent_bandwidth(grid: TimeGrid, transfer: NDArray[np.complexfloating]) -> float:
    """Discrete one-sided NEB = (fs/2) * mean_k |H_k|^2 [Hz] (normalized to |H(0)| = 1).

    For real white noise of per-sample variance s^2 = G fs / 2, the filtered
    per-sample variance is s^2 mean|H_k|^2 = G * NEB, exactly on the grid.
    """
    h0 = abs(transfer[0])
    return 0.5 * grid.sample_rate * float(np.mean(np.abs(transfer) ** 2)) / h0**2


def butterworth_neb(bandwidth: float, order: int) -> float:
    """Analytic one-sided NEB of an order-n Butterworth low-pass [Hz]."""
    x = math.pi / (2 * order)
    return bandwidth * x / math.sin(x)


def gaussian_neb(bandwidth: float) -> float:
    """Analytic one-sided NEB of the Gaussian low-pass [Hz]."""
    return bandwidth * math.sqrt(math.pi / (4.0 * math.log(2.0)))


__all__ = [
    "FILTER_KINDS",
    "butterworth_neb",
    "gaussian_neb",
    "lowpass_on_grid",
    "lowpass_response",
    "noise_equivalent_bandwidth",
]
