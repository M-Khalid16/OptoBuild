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

Dual polarization (``components.coherent.DualPolCoherentDSP``): after
matched filtering and timing, the two received polarizations at 2 samples
per symbol pass a 2x2 butterfly FIR equalizer (``mimo_equalize``) that
inverts polarization rotation, PMD and residual linear ISI, adapted blindly
by the constant-modulus algorithm (CMA, Godard 1980; Kikuchi, J. Lightwave
Technol. 34, 157 (2016)), optionally followed by the radius-directed
equalizer (RDE) for multi-ring QAM. FOE (4th power, spectra of both outputs
summed) and BPS follow per polarization. I/Q imbalance of the hybrid is
removed by Gram-Schmidt orthogonalization (``gram_schmidt_orthogonalize``,
Fatadin et al., IEEE Photon. Technol. Lett. 20, 1733 (2008)) and a known
I/Q skew by a fractional delay.

The residual rotation ambiguity (multiples of pi/2 for square QAM, pi for
BPSK) is resolved in analysis against the transmitted reference
(``align_to_reference``), as is customary in simulation; real receivers use
pilots or differential coding.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.analysis.constellations import constellation, decide

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
    """4th-power frequency-offset estimate [Hz] (|df| < R_s/8).

    ``symbols`` of shape (n,) or (n_pol, n); for several rows the magnitude
    spectra are summed (common carrier offset).
    """
    y = np.atleast_2d(np.asarray(symbols, dtype=complex))
    n = y.shape[-1] * zero_pad
    spec = np.sum(np.abs(np.fft.fft(y**4, n, axis=-1)), axis=0)
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


def gram_schmidt_orthogonalize(
    in_phase: ArrayLike, quadrature: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GSOP: I' = I / sqrt(P_I), Q' = (Q - rho I / P_I) / sqrt(P_Q'), rho = <I Q>.

    Removes hybrid phase error and I/Q amplitude imbalance (both outputs unit power).
    """
    i = np.asarray(in_phase, dtype=float)
    q = np.asarray(quadrature, dtype=float)
    p_i = float(np.mean(i**2))
    q = q - float(np.mean(i * q)) / p_i * i
    return i / math.sqrt(p_i), q / math.sqrt(float(np.mean(q**2)))


def constant_modulus_radii(fmt: str) -> NDArray[np.float64]:
    """Squared ring radii |s|^2 of a unit-energy constellation (ascending)."""
    return np.unique(np.round(np.abs(constellation(fmt)) ** 2, 12))


def cma_radius(fmt: str) -> float:
    """Godard radius R_2 = E|s|^4 / E|s|^2 of a unit-energy constellation."""
    a = np.abs(constellation(fmt)) ** 2
    return float(np.mean(a**2) / np.mean(a))


def mimo_equalize(
    samples: ArrayLike,
    fmt: str,
    n_taps: int = 15,
    step: float = 2.5e-4,
    epochs: int = 6,
    radius_directed: bool = False,
    x_epochs: int = 2,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Blind 2x2 butterfly equalizer, T/2-spaced; returns (symbols (2, n), taps (2, 2, n_taps)).

    ``samples``: shape (2, 2 n), two samples per symbol, symbol instants at even
    indices, power-normalized. Output z_p[k] = sum_q sum_m W[p, q, m] y_q[2k + m - h]
    (h = n_taps // 2, circular). Stochastic-gradient update per symbol:

        W_p <- W_p - mu e_p z_p conj(y),   e_p = |z_p|^2 - R

    with R = R_2 (CMA) or the ring radius nearest to |z_p|^2 (RDE). The first
    ``x_epochs`` passes adapt only output x from centre-tap initialization;
    output y is then initialized orthogonally, W_yy[m] = conj(W_xx[-m]),
    W_yx[m] = -conj(W_xy[-m]) (the inverse of a unitary channel has this form),
    which prevents both outputs converging to the same source (CMA
    singularity). The remaining passes adapt both outputs; RDE (if requested)
    replaces CMA from the second pass; the last pass uses mu/4 to reduce the
    steady-state misadjustment. Two x-only passes are needed for 16-QAM: after
    one, the x taps are still noisy and the derived y initialization can fail.

    Blind equalization relies on 4th-order statistics of i.i.d. symbols. A
    short PRBS mapped to many bits per symbol violates this (its linear
    recurrence spans m / log2(M) symbols; PRBS15 with 64-QAM fails, a PRBS23
    segment of the same length converges); see docs/physics_models.md 3.21. The returned symbols are
    computed with the final (frozen) taps over the whole block: offline
    training for a channel that is static over the window.
    """
    y = np.asarray(samples, dtype=complex)
    if y.ndim != 2 or y.shape[0] != 2 or y.shape[1] % 2:
        raise ValueError(f"need shape (2, 2 n) at 2 samples per symbol, got {y.shape}")
    if n_taps % 2 == 0 or n_taps < 1:
        raise ValueError(f"n_taps must be odd and positive, got {n_taps}")
    if not 1 <= x_epochs < epochs:
        raise ValueError(f"need 1 <= x_epochs < epochs, got {x_epochs} and {epochs}")
    h = n_taps // 2
    n_sym = y.shape[1] // 2
    padded = np.concatenate([y[:, y.shape[1] - h :], y, y[:, :h]], axis=1)
    win = np.lib.stride_tricks.sliding_window_view(padded, n_taps, axis=1)[:, ::2][:, :n_sym]
    windows = np.ascontiguousarray(win.transpose(1, 0, 2))  # (n_sym, 2, n_taps)
    w = np.zeros((2, 2, n_taps), dtype=complex)
    w[0, 0, h] = w[1, 1, h] = 1.0
    r2 = cma_radius(fmt)
    radii = constant_modulus_radii(fmt)
    for epoch in range(epochs):
        mu = step / 4 if epoch == epochs - 1 else step
        rows = 1 if epoch < x_epochs else 2
        rde = radius_directed and epoch >= 1
        for x in windows:
            z = np.einsum("pqm,qm->p", w[:rows], x)
            m2 = z.real**2 + z.imag**2
            if rde:
                target = radii[np.argmin(np.abs(radii[None, :] - m2[:, None]), axis=1)]
            else:
                target = r2
            w[:rows] -= (mu * (m2 - target) * z)[:, None, None] * x.conj()[None, :, :]
        if epoch == x_epochs - 1:
            w[1, 1] = np.conj(w[0, 0, ::-1])
            w[1, 0] = -np.conj(w[0, 1, ::-1])
    return np.einsum("pqm,kqm->pk", w, windows), w


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
    "cma_radius",
    "constant_modulus_radii",
    "gram_schmidt_orthogonalize",
    "mimo_equalize",
    "estimate_frequency_offset",
    "evm_rms",
    "normalize_power",
    "remove_frequency_offset",
]
