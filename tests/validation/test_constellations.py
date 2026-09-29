"""Constellations: Gray property, energy, mapping and exact AWGN error rates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.analysis.constellations import (
    FORMATS,
    bits_per_symbol,
    constellation,
    decide,
    demap_hard,
    map_bits,
    theoretical_ber,
    theoretical_ser,
)
from optobuild.core.errors import InvalidParameterError


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_unit_energy_and_round_trip(fmt: str) -> None:
    pts = constellation(fmt)
    assert pts.size == FORMATS[fmt] and len(set(np.round(pts, 12))) == FORMATS[fmt]
    assert np.mean(np.abs(pts) ** 2) == pytest.approx(1.0, rel=1e-12)
    bits = np.random.default_rng(0).integers(0, 2, 600 * bits_per_symbol(fmt))
    np.testing.assert_array_equal(demap_hard(map_bits(bits, fmt), fmt), bits)


@pytest.mark.parametrize("fmt", ["qpsk", "16qam", "64qam"])
def test_gray_property(fmt: str) -> None:
    """Nearest neighbours differ in exactly one bit."""
    pts = constellation(fmt)
    k = bits_per_symbol(fmt)
    dmin = np.min(np.abs(pts[:, None] - pts[None, :]) + 10 * np.eye(pts.size))
    for a in range(pts.size):
        for b in range(pts.size):
            if a != b and abs(pts[a] - pts[b]) < dmin * (1 + 1e-9):
                assert bin(a ^ b).count("1") == 1, (a, b)
    assert k == int(math.log2(pts.size))


def test_qpsk_points() -> None:
    np.testing.assert_allclose(
        sorted(constellation("qpsk"), key=lambda z: (z.real, z.imag)),
        np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / math.sqrt(2),
    )


def test_decisions_and_errors() -> None:
    noisy = constellation("16qam") + 0.02
    np.testing.assert_allclose(decide(noisy, "16qam"), constellation("16qam"))
    with pytest.raises(InvalidParameterError):
        map_bits([1, 0, 1], "qpsk")
    with pytest.raises(InvalidParameterError):
        constellation("8psk")
    with pytest.raises(InvalidParameterError):
        theoretical_ber("16qam", 10.0)


def test_theory_reference_values() -> None:
    """QPSK: BER = Q(sqrt(SNR)); at SNR = 9.8 dB, BER ~ 1e-3 (textbook: Eb/N0 = 6.8 dB)."""
    snr = 10 ** (0.98)
    assert theoretical_ber("qpsk", snr) == pytest.approx(1.0e-3, rel=0.05)
    assert theoretical_ber("bpsk", 10**0.68) == pytest.approx(1.0e-3, rel=0.05)
    assert theoretical_ser("qpsk", snr) == pytest.approx(
        1 - (1 - theoretical_ber("qpsk", snr)) ** 2
    )


@pytest.mark.parametrize(
    ("fmt", "snr_db"), [("bpsk", 6.0), ("qpsk", 9.0), ("16qam", 15.0), ("64qam", 21.0)]
)
def test_monte_carlo_ser_matches_exact_theory(fmt: str, snr_db: float) -> None:
    """Complex AWGN with per-dimension variance N0/2, N0 = 1/SNR (E_s = 1). 5-sigma band."""
    rng = np.random.default_rng(7)
    n = 400_000
    k = bits_per_symbol(fmt)
    bits = rng.integers(0, 2, n * k)
    s = map_bits(bits, fmt)
    n0 = 10 ** (-snr_db / 10)
    r = s + math.sqrt(n0 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ser_sim = np.mean(decide(r, fmt) != s)
    p = float(theoretical_ser(fmt, 10 ** (snr_db / 10)))
    assert ser_sim == pytest.approx(p, abs=5 * math.sqrt(p * (1 - p) / n))
    if fmt in ("bpsk", "qpsk"):
        ber_sim = np.mean(demap_hard(r, fmt) != bits)
        pb = float(theoretical_ber(fmt, 10 ** (snr_db / 10)))
        assert ber_sim == pytest.approx(pb, abs=5 * math.sqrt(pb * (1 - pb) / (n * k)))
