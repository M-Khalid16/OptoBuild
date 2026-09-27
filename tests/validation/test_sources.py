"""Validation: CW power normalization; PRBS maximal-length properties."""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.analysis.power import measure_power
from optobuild.analysis.spectrum import optical_spectrum
from optobuild.components.sources import CWLaser, PRBSGenerator
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.errors import InvalidParameterError
from optobuild.core.units import dbm_to_watt
from optobuild.physics.prbs import PRBS_POLYNOMIALS, prbs, prbs_period


@pytest.mark.parametrize("power_dbm", [-30.0, 0.0, 13.0])
def test_cw_power_normalization(run_component, power_dbm: float) -> None:  # type: ignore[no-untyped-def]
    """mean |A|^2 must equal the set power exactly (float round-off only)."""
    p0 = float(dbm_to_watt(power_dbm))
    laser = CWLaser("laser", {"power": p0, "phase": 0.7, "n_samples": 1024})
    out, _ = run_component(laser)
    sig = out["out"]
    np.testing.assert_allclose(sig.power(), p0, rtol=1e-14)
    assert measure_power(sig).average_w == pytest.approx(p0, rel=1e-14)
    assert measure_power(sig).average_dbm == pytest.approx(power_dbm, abs=1e-12)
    np.testing.assert_allclose(sig.phase(), 0.7, atol=1e-14)
    assert sig.center_frequency == pytest.approx(SPEED_OF_LIGHT / 1550e-9, rel=1e-15)


def test_cw_spectrum_is_a_single_line_with_full_power(run_component) -> None:  # type: ignore[no-untyped-def]
    laser = CWLaser(
        "laser",
        {
            "power": 2e-3,
            "n_samples": 512,
            "sample_rate": 100e9,
            "frequency_offset": 5 * 100e9 / 512,
        },
    )
    sig = run_component(laser)[0]["out"]
    spec = optical_spectrum(sig)
    k = int(np.argmax(spec.psd_w_per_hz))
    assert spec.frequency_hz[k] == pytest.approx(sig.center_frequency + 5 * 100e9 / 512)
    assert spec.psd_w_per_hz[k] * spec.df_hz == pytest.approx(2e-3, rel=1e-12)
    assert spec.total_power() == pytest.approx(2e-3, rel=1e-12)


def test_laser_rejects_out_of_band_offset_and_warns_on_leakage() -> None:
    with pytest.raises(InvalidParameterError, match="outside"):
        CWLaser("l", {"frequency_offset": 90e9, "sample_rate": 160e9})
    laser = CWLaser("l", {"frequency_offset": 1.234e9})
    assert [d.code for d in laser.diagnostics] == ["laser.offset_not_periodic"]


def _prime_factors(n: int) -> set[int]:
    out, p = set(), 2
    while p * p <= n:
        while n % p == 0:
            out.add(p)
            n //= p
        p += 1
    if n > 1:
        out.add(n)
    return out


def _reference_lfsr(m: int, k: int, n: int) -> np.ndarray:
    """Independent bit-by-bit Fibonacci LFSR (taps m, k), register initialized to all ones."""
    reg = [1] * m  # reg[0] is the oldest bit (next output)
    out = []
    for _ in range(n):
        out.append(reg[0])
        new = reg[m - k] ^ reg[0]  # s[n] = s[n-k] ^ s[n-m]
        reg = reg[1:] + [new]
    return np.array(out, dtype=np.uint8)


@pytest.mark.parametrize("order", [7, 9, 10, 11, 15])
def test_prbs_is_maximal_length_and_balanced(order: int) -> None:
    period = prbs_period(order)
    s = prbs(order, 2 * period)
    np.testing.assert_array_equal(s[:period], s[period:])
    for p in _prime_factors(period):  # no shorter period divides 2^m - 1
        sub = period // p
        assert not np.array_equal(s[:period], np.roll(s[:period], sub))
    assert int(s[:period].sum()) == 2 ** (order - 1)


@pytest.mark.parametrize("order", sorted(PRBS_POLYNOMIALS))
def test_prbs_matches_independent_lfsr(order: int) -> None:
    m, k = PRBS_POLYNOMIALS[order]
    np.testing.assert_array_equal(prbs(order, 3000), _reference_lfsr(m, k, 3000))


def test_prbs_component_defaults_to_one_period(run_component) -> None:  # type: ignore[no-untyped-def]
    gen = PRBSGenerator("p", {"order": 9, "bit_rate": 25e9})
    seq = run_component(gen)[0]["out"]
    assert seq.n_bits == 511 and seq.bit_rate == 25e9
    assert seq.metadata["pattern"] == "PRBS9"
    assert gen.diagnostics == ()
    assert [d.code for d in PRBSGenerator("q", {"n_bits": 100}).diagnostics] == [
        "sampling.pattern_periodicity"
    ]
    with pytest.raises(InvalidParameterError):
        PRBSGenerator("r", {"order": 7, "seed": 128})
