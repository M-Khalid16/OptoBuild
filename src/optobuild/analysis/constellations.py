"""Gray-coded constellations and exact AWGN error-rate references.

Formats: ``bpsk``, ``qpsk``, ``16qam``, ``64qam``. Square M-QAM is built as
two independent Gray-coded sqrt(M)-PAM axes: level index i (ascending
amplitude) carries the Gray code i XOR (i >> 1); the first half of each
symbol's bits select the I level, the second half the Q level. Unnormalized
levels are 2i - (L - 1); constellations are scaled to unit average energy
E_s = 1 (for equiprobable symbols). QPSK = 4-QAM: (+-1 +- 1j)/sqrt(2).

AWGN references (E_s/N0 = SNR per symbol, Q(x) = erfc(x / sqrt 2) / 2):

* BPSK:  BER = SER = Q(sqrt(2 SNR))
* QPSK (Gray):  BER = Q(sqrt(SNR)),  SER = 1 - (1 - Q(sqrt(SNR)))^2
* square M-QAM:  SER = 1 - (1 - P_L)^2,  P_L = 2 (1 - 1/sqrt(M)) Q(sqrt(3 SNR / (M - 1)))

(Proakis & Salehi, Digital Communications, 5th ed., sec. 4.3.) The square-QAM
SER is exact; for M >= 16 the Gray BER has no equally simple exact form and
is not provided (use SER, or count bits in simulation).
Hard decisions are nearest-neighbour (per-axis slicing, exact for square QAM).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import InvalidParameterError

FORMATS: dict[str, int] = {"bpsk": 2, "qpsk": 4, "16qam": 16, "64qam": 64}


def _check(fmt: str) -> int:
    if fmt not in FORMATS:
        raise InvalidParameterError(
            f"Unknown modulation format {fmt!r}.", hint=f"Use one of {list(FORMATS)}."
        )
    return FORMATS[fmt]


def bits_per_symbol(fmt: str) -> int:
    """log2(M)."""
    return int(math.log2(_check(fmt)))


def _gray(i: NDArray[np.int64]) -> NDArray[np.int64]:
    return i ^ (i >> 1)


def _axis(fmt: str) -> tuple[int, int, float]:
    """(levels per axis L, bits per axis, scale to unit energy)."""
    m = _check(fmt)
    lvl = int(round(math.sqrt(m)))
    return lvl, int(math.log2(lvl)), math.sqrt(2.0 * (m - 1) / 3.0)


def constellation(fmt: str) -> NDArray[np.complex128]:
    """Constellation points indexed by the integer value of their bit label (MSB first)."""
    m = _check(fmt)
    labels = np.arange(m)
    bits = ((labels[:, None] >> np.arange(bits_per_symbol(fmt) - 1, -1, -1)) & 1).astype(np.uint8)
    return map_bits(bits.ravel(), fmt)


def map_bits(bits: ArrayLike, fmt: str) -> NDArray[np.complex128]:
    """Map a bit stream (length multiple of log2 M) to unit-energy symbols."""
    b = np.asarray(bits, dtype=np.int64)
    k = bits_per_symbol(fmt)
    if b.size % k:
        raise InvalidParameterError(f"{b.size} bits is not a multiple of {k} bits per symbol.")
    b = b.reshape(-1, k)
    if fmt == "bpsk":
        return (2.0 * b[:, 0] - 1.0).astype(complex)
    lvl, kb, scale = _axis(fmt)
    weights = 1 << np.arange(kb - 1, -1, -1)
    gi, gq = b[:, :kb] @ weights, b[:, kb:] @ weights
    inv = np.argsort(_gray(np.arange(lvl)))  # Gray label -> level index
    ii, iq = inv[gi], inv[gq]
    return ((2 * ii - (lvl - 1)) + 1j * (2 * iq - (lvl - 1))) / scale


def demap_hard(symbols: ArrayLike, fmt: str) -> NDArray[np.uint8]:
    """Nearest-neighbour hard decision -> bits (inverse of :func:`map_bits`)."""
    s = np.asarray(symbols, dtype=complex).ravel()
    if fmt == "bpsk":
        return (s.real > 0).astype(np.uint8)
    lvl, kb, scale = _axis(fmt)

    def axis_bits(x: NDArray[np.float64]) -> NDArray[np.uint8]:
        idx = np.clip(np.round((x * scale + (lvl - 1)) / 2.0), 0, lvl - 1).astype(np.int64)
        g = _gray(idx)
        return ((g[:, None] >> np.arange(kb - 1, -1, -1)) & 1).astype(np.uint8)

    return np.hstack([axis_bits(s.real), axis_bits(s.imag)]).ravel()


def decide(symbols: ArrayLike, fmt: str) -> NDArray[np.complex128]:
    """Nearest constellation point of each symbol."""
    return map_bits(demap_hard(symbols, fmt), fmt)


def _q(x: ArrayLike) -> NDArray[np.float64] | float:
    return (0.5 * scipy.special.erfc(np.asarray(x, dtype=float) / math.sqrt(2.0)))[()]


def theoretical_ser(fmt: str, snr: ArrayLike) -> NDArray[np.float64] | float:
    """Exact AWGN symbol-error rate at SNR = E_s/N0 (linear)."""
    g = np.asarray(snr, dtype=float)
    if fmt == "bpsk":
        return _q(np.sqrt(2 * g))
    m = _check(fmt)
    p_l = 2 * (1 - 1 / math.sqrt(m)) * np.asarray(_q(np.sqrt(3 * g / (m - 1))))
    return (1 - (1 - p_l) ** 2)[()]


def theoretical_ber(fmt: str, snr: ArrayLike) -> NDArray[np.float64] | float:
    """Exact AWGN bit-error rate for BPSK and Gray QPSK."""
    g = np.asarray(snr, dtype=float)
    if fmt == "bpsk":
        return _q(np.sqrt(2 * g))
    if fmt == "qpsk":
        return _q(np.sqrt(g))
    raise InvalidParameterError(
        f"No exact BER expression implemented for {fmt!r}.",
        hint="Use theoretical_ser, or count bit errors by simulation.",
    )


__all__ = [
    "FORMATS",
    "bits_per_symbol",
    "constellation",
    "decide",
    "demap_hard",
    "map_bits",
    "theoretical_ber",
    "theoretical_ser",
]
