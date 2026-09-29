"""Validation of the dual-polarization DSP blocks (docs/physics_models.md 3.21).

Synthetic signals: independent Gray-mapped symbols on x and y, RRC shaping
(roll-off 0.1) at 2 samples/symbol, a known Jones channel, complex AWGN of
exactly known per-polarization SNR (Es/N0), RRC matched filter.

Tolerances: the data-aided SNR estimate from N symbols has a relative
standard deviation ~1/sqrt(N), i.e. 4.34/sqrt(N) dB (0.048 dB for N = 8192);
we allow 5 sigma plus the documented
equalizer excess-MSE allowance EQ_PENALTY_DB. The LMS misadjustment estimate
M ~ mu tr(R) / 2 = 2.5e-4 * 30 / 2 = 0.4 % (0.02 dB) at the final step size;
0.1 dB covers the stochastic-gradient approximation of CMA/RDE.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.analysis.coherent_dsp import (
    align_to_reference,
    cma_radius,
    constant_modulus_radii,
    estimate_frequency_offset,
    gram_schmidt_orthogonalize,
    mimo_equalize,
)
from optobuild.analysis.constellations import bits_per_symbol, map_bits
from optobuild.numerics.grid import TimeGrid
from optobuild.numerics.pulse_shaping import matched_filter, shape
from optobuild.physics.polarization import apply_jones, dgd_element, sop_transform

N_SYM = 8192
EQ_PENALTY_DB = 0.1


def _stat_tol_db(n: int) -> float:
    """5 sigma of a dB-scaled SNR estimate from n symbols (relative std ~ 1/sqrt(n))."""
    return 5 * 10 / math.log(10) / math.sqrt(n)


def _channel(fmt: str, snr_db: float, theta: float, phi: float, dgd_symbols: float, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    k = bits_per_symbol(fmt)
    s = np.vstack([map_bits(rng.integers(0, 2, N_SYM * k), fmt) for _ in range(2)])
    grid = TimeGrid(2 * N_SYM, 0.5)  # R_s = 1 Bd, 2 samples/symbol
    tx = np.vstack([shape(s[p], 2, grid, 0.1, "rrc") for p in range(2)])
    j = np.einsum(
        "ij,jkn->ikn",
        sop_transform(theta, phi),
        np.einsum("ijn,jk->ikn", dgd_element(grid.angular_frequency(), dgd_symbols),
                  sop_transform(0.7, 0.4)),
    )  # fmt: skip
    rx = apply_jones(tx, j, grid)
    # Es/N0 per polarization: Es = mean|tx|^2 T, T = 1 s; noise variance N0 fs per sample
    n0 = np.mean(np.abs(tx) ** 2) / 10 ** (snr_db / 10)
    rx = rx + math.sqrt(n0 * grid.sample_rate / 2) * (
        rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)
    )
    y = np.vstack([matched_filter(rx[p], 2, grid, 0.1) for p in range(2)])
    return s, y / math.sqrt(np.mean(np.abs(y) ** 2))


def _snr_db(r: np.ndarray, s: np.ndarray) -> float:
    c = np.vdot(s, r) / np.vdot(s, s)
    return float(
        10 * np.log10(abs(c) ** 2 * np.mean(np.abs(s) ** 2) / np.mean(np.abs(r - c * s) ** 2))
    )


def _assign(z: np.ndarray, s: np.ndarray, fmt: str) -> list[float]:
    """SNR of each output against the source it recovered; asserts distinct sources."""
    out, used = [], set()
    for p in range(2):
        snrs = [_snr_db(align_to_reference(z[p], s[q], fmt)[2], s[q]) for q in range(2)]
        q = int(np.argmax(snrs))
        used.add(q)
        out.append(snrs[q])
    assert used == {0, 1}, "both equalizer outputs converged to the same source"
    return out


@pytest.mark.parametrize(
    "theta,phi,dgd",
    [(0.3, 0.2, 0.0), (math.pi / 4, math.pi / 4, 0.3), (1.2, -1.0, 1.5)],
)
def test_cma_inverts_rotation_and_dgd_qpsk(theta: float, phi: float, dgd: float) -> None:
    """Includes the state (45 deg, 45 deg) where each received polarization is an equal
    mixture (worst case for the CMA singularity) and a DGD of 1.5 symbols."""
    s, y = _channel("qpsk", 12.0, theta, phi, dgd, seed=1)
    z, taps = mimo_equalize(y, "qpsk")
    assert taps.shape == (2, 2, 15)
    for snr in _assign(z, s, "qpsk"):
        assert snr == pytest.approx(12.0, abs=_stat_tol_db(N_SYM) + EQ_PENALTY_DB)


def test_rde_16qam() -> None:
    s, y = _channel("16qam", 20.0, 0.9, 0.5, 0.3, seed=2)
    z, _ = mimo_equalize(y, "16qam", radius_directed=True)
    for snr in _assign(z, s, "16qam"):
        assert snr == pytest.approx(20.0, abs=_stat_tol_db(N_SYM) + EQ_PENALTY_DB)


def test_mimo_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2 samples per symbol"):
        mimo_equalize(np.zeros((2, 7), complex), "qpsk")
    with pytest.raises(ValueError, match="odd"):
        mimo_equalize(np.zeros((2, 8), complex), "qpsk", n_taps=4)


def test_cm_radii() -> None:
    assert cma_radius("qpsk") == pytest.approx(1.0)
    assert cma_radius("16qam") == pytest.approx(1.32)  # E|s|^4 = 1.32 for unit-energy 16-QAM
    assert constant_modulus_radii("16qam") == pytest.approx([0.2, 1.0, 1.8])


def test_gram_schmidt_removes_hybrid_imbalance_exactly() -> None:
    rng = np.random.default_rng(3)
    z = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    eps, g = 0.2, 0.7
    i, q = z.real, g * (z * np.exp(-1j * eps)).imag
    io, qo = gram_schmidt_orthogonalize(i, q)
    # the orthogonalized pair is a rotation-free rescaling of (Re z, Im z) up to the
    # finite-sample correlation of Re z and Im z, which GSOP also removes:
    i_ref, q_ref = gram_schmidt_orthogonalize(z.real, z.imag)
    assert np.allclose(io, i_ref, atol=1e-12)
    assert np.allclose(qo, q_ref, atol=1e-12)


def test_joint_frequency_offset_estimate_two_polarizations() -> None:
    """Common offset on both rows of a random mixture; resolution-limited estimate
    (zero-padded x8 plus parabolic interpolation: error << R_s / (8 n))."""
    rng = np.random.default_rng(4)
    s = np.vstack([map_bits(rng.integers(0, 2, 2 * 4096), "qpsk") for _ in range(2)])
    df = 0.0123
    k = np.arange(4096)
    y = sop_transform(0.5, 0.3) @ s * np.exp(2j * np.pi * df * k)
    assert estimate_frequency_offset(y, 1.0) == pytest.approx(df, abs=1 / (8 * 4096))
