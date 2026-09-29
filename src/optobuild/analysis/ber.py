"""Bit-error counting, confidence bounds and Gaussian-noise BER references.

Error counting
--------------
After aligning the received sequence to the reference (circular shift that
maximizes agreement, found by FFT cross-correlation of +-1 sequences):

    N_err = sum_k [r_k != b_k],   BER = N_err / N_bits

Confidence interval: exact (Clopper-Pearson) two-sided 95 % interval for a
binomial proportion; with N_err = 0 the upper bound is 1 - 0.025^(1/N)
(~3.7/N). The one-sided 95 % "rule of three" bound is 3/N.

Gaussian reference
------------------
For decision samples x_k = m_k + n_k with Gaussian noise of standard deviation
sigma_k and threshold th, the error probability of bit k is

    p_k = 1/2 erfc(|m_k - th| / (sqrt(2) sigma_k))     (if m_k is on the correct side)

and the expected BER is mean_k p_k (exact for Gaussian noise, including
pattern-dependent ISI in m_k). The classical Q-factor form
BER = 1/2 erfc(Q / sqrt(2)), Q = (mu1 - mu0)/(sigma1 + sigma0) is the special
case of two levels at the optimum threshold (Agrawal, Fiber-Optic
Communication Systems, 5th ed., sec. 4.5; Proakis & Salehi, Digital
Communications, 5th ed.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import SamplingError


@dataclass(frozen=True)
class BERResult:
    """Outcome of direct error counting."""

    n_bits: int
    n_errors: int
    shift: int
    """Circular shift (bits) applied to the reference to align it with the received bits."""
    ber_lower_95: float
    ber_upper_95: float

    @property
    def ber(self) -> float:
        """Counted bit-error ratio N_err / N_bits."""
        return self.n_errors / self.n_bits


def clopper_pearson(n_errors: int, n_bits: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for the error probability."""
    a = 1.0 - confidence
    lo, hi = 0.0, 1.0
    if n_errors > 0:
        lo = float(scipy.stats.beta.ppf(a / 2, n_errors, n_bits - n_errors + 1))
    if n_errors < n_bits:
        hi = float(scipy.stats.beta.ppf(1 - a / 2, n_errors + 1, n_bits - n_errors))
    return lo, hi


def best_alignment(reference: ArrayLike, received: ArrayLike) -> int:
    """Circular shift s maximizing agreement of roll(reference, s) with received."""
    r = 2.0 * np.asarray(reference, dtype=float) - 1.0
    x = 2.0 * np.asarray(received, dtype=float) - 1.0
    corr = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(r))).real
    return int(np.argmax(np.round(corr, 6)))


def count_errors(reference: ArrayLike, received: ArrayLike, *, align: bool = True) -> BERResult:
    """Count bit errors between two equally long bit sequences."""
    ref = np.asarray(reference, dtype=np.uint8)
    rx = np.asarray(received, dtype=np.uint8)
    if ref.shape != rx.shape or ref.ndim != 1:
        raise SamplingError(
            f"Reference ({ref.shape}) and received ({rx.shape}) sequences differ in length.",
            hint="Both must come from the same pattern length (same PRBS n_bits).",
        )
    shift = best_alignment(ref, rx) if align else 0
    n_err = int(np.count_nonzero(np.roll(ref, shift) != rx))
    lo, hi = clopper_pearson(n_err, ref.size)
    return BERResult(int(ref.size), n_err, shift, lo, hi)


def gaussian_error_probability(
    means: ArrayLike, bits: ArrayLike, threshold: float, sigma: ArrayLike
) -> NDArray[np.float64]:
    """Per-bit error probability for Gaussian noise around noiseless means m_k."""
    m = np.asarray(means, dtype=float)
    b = np.asarray(bits)
    s = np.asarray(sigma, dtype=float)
    # signed distance to the threshold on the correct side
    d = np.where(b == 1, m - threshold, threshold - m)
    return 0.5 * scipy.special.erfc(d / (np.sqrt(2.0) * s))


def q_factor(values: ArrayLike, bits: ArrayLike) -> tuple[float, float, float, float, float]:
    """(mu1, mu0, sigma1, sigma0, Q) of decision samples grouped by the true bits."""
    v = np.asarray(values, dtype=float)
    b = np.asarray(bits)
    ones, zeros = v[b == 1], v[b == 0]
    mu1, mu0 = float(ones.mean()), float(zeros.mean())
    s1, s0 = float(ones.std()), float(zeros.std())
    q = (mu1 - mu0) / (s1 + s0) if (s1 + s0) > 0 else float("inf")
    return mu1, mu0, s1, s0, q


def ber_from_q(q: float) -> float:
    """BER = 1/2 erfc(Q / sqrt(2))."""
    return float(0.5 * scipy.special.erfc(q / np.sqrt(2.0)))


__all__ = [
    "BERResult",
    "ber_from_q",
    "best_alignment",
    "clopper_pearson",
    "count_errors",
    "gaussian_error_probability",
    "q_factor",
]
