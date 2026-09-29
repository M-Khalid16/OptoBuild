"""RC/RRC pulse shaping: Nyquist zero-ISI, matched-filter cascade, power, spectra."""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.analysis.constellations import map_bits
from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid
from optobuild.numerics.pulse_shaping import (
    matched_filter,
    raised_cosine,
    root_raised_cosine,
    shape,
)

RS = 32e9


def _symbols(n: int, fmt: str = "16qam") -> np.ndarray:
    rng = np.random.default_rng(3)
    k = {"qpsk": 2, "16qam": 4}[fmt]
    return map_bits(rng.integers(0, 2, n * k), fmt)


@pytest.mark.parametrize("rolloff", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("sps", [2, 4, 8])
def test_rrc_cascade_has_zero_isi(rolloff: float, sps: int) -> None:
    s = _symbols(512)
    grid = TimeGrid.from_sample_rate(512 * sps, RS * sps)
    z = matched_filter(shape(s, sps, grid, rolloff), sps, grid, rolloff)
    np.testing.assert_allclose(z[::sps], s, atol=1e-12)


@pytest.mark.parametrize("rolloff", [0.0, 0.25, 1.0])
def test_rc_waveform_passes_through_the_symbols(rolloff: float) -> None:
    s = _symbols(256)
    grid = TimeGrid.from_sample_rate(256 * 4, RS * 4)
    np.testing.assert_allclose(shape(s, 4, grid, rolloff, "rc")[::4], s, atol=1e-12)


def test_spectrum_definition() -> None:
    f = np.array([0.0, 0.4 * RS, 0.5 * RS, 0.55 * RS, 0.6 * RS, 0.7 * RS])
    h = raised_cosine(f, RS, 0.2)
    np.testing.assert_allclose(
        h, [1, 1, 0.5, 0.5 * (1 + np.cos(np.pi / 0.2 * 0.15)), 0, 0], atol=1e-15
    )
    np.testing.assert_allclose(root_raised_cosine(f, RS, 0.2) ** 2, h, atol=1e-15)
    # Nyquist criterion in the frequency domain: folded spectrum is flat
    ff = np.linspace(-0.5 * RS, 0.5 * RS, 101)
    folded = sum(raised_cosine(ff + k * RS, RS, 0.35) for k in (-1, 0, 1))
    np.testing.assert_allclose(folded, 1.0, atol=1e-12)


def test_rrc_waveform_power_equals_symbol_energy() -> None:
    """For i.i.d. symbols E|y|^2 = E|s|^2; with 2^14 symbols the sample mean agrees to ~1 %."""
    s = _symbols(2**14)
    grid = TimeGrid.from_sample_rate(2**14 * 4, RS * 4)
    y = shape(s, 4, grid, 0.1)
    assert np.mean(np.abs(y) ** 2) == pytest.approx(np.mean(np.abs(s) ** 2), rel=0.03)


def test_bandwidth_and_length_checks() -> None:
    grid = TimeGrid.from_sample_rate(64, RS * 1.5)
    with pytest.raises(SamplingError, match="exceeds fs/2"):
        shape(np.ones(64), 1, TimeGrid.from_sample_rate(64, RS), 0.1)
    with pytest.raises(SamplingError):
        shape(np.ones(10), 4, grid, 0.1)
