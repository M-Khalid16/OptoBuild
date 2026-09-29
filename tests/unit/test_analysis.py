from __future__ import annotations

import numpy as np
import pytest

from optobuild.analysis.eye import eye_diagram
from optobuild.analysis.power import PowerMeasurement, measure_power
from optobuild.analysis.spectrum import optical_spectrum
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.numerics.grid import TimeGrid
from optobuild.numerics.sampling import (
    band_edge_energy_fraction,
    occupied_bandwidth,
    samples_per_symbol_diagnostic,
)
from optobuild.signals import OpticalSignal

G = TimeGrid.from_sample_rate(1000, 100e9)  # df = 100 MHz
F0 = 193.1e12


def _tones(powers: dict[int, float]) -> OpticalSignal:
    t = G.time()
    field = sum(np.sqrt(p) * np.exp(2j * np.pi * k * G.df * t) for k, p in powers.items())
    return OpticalSignal(G, field, F0)


def test_power_measurement() -> None:
    m = measure_power(_tones({0: 1e-3, 10: 1e-3}))
    assert m.average_w == pytest.approx(2e-3, rel=1e-12)
    assert m.peak_w == pytest.approx(4e-3, rel=1e-9)
    assert m.average_dbm == pytest.approx(10 * np.log10(2), abs=1e-9)
    assert PowerMeasurement(0.0, 0.0).average_dbm == float("-inf")


def test_spectrum_axes_and_parseval() -> None:
    spec = optical_spectrum(_tones({0: 1e-3, 20: 2e-3, -30: 5e-4}))
    assert np.all(np.diff(spec.frequency_hz) > 0)
    np.testing.assert_allclose(spec.wavelength_m, SPEED_OF_LIGHT / spec.frequency_hz)
    assert spec.total_power() == pytest.approx(3.5e-3, rel=1e-12)
    k = int(np.argmin(np.abs(spec.frequency_hz - (F0 + 20 * G.df))))
    assert spec.power_per_rbw_w[k] == pytest.approx(2e-3, rel=1e-12)


def test_rbw_integrates_neighbouring_lines() -> None:
    sig = _tones({0: 1e-3, 3: 1e-3, 20: 1e-3})
    spec = optical_spectrum(sig, resolution_bandwidth=7 * G.df)  # +-3 bins
    k0 = int(np.argmin(np.abs(spec.frequency_hz - F0)))
    assert spec.power_per_rbw_w[k0] == pytest.approx(2e-3, rel=1e-12)
    assert spec.resolution_bandwidth_hz == pytest.approx(7 * G.df)
    assert optical_spectrum(sig, 0.5 * G.df).resolution_bandwidth_hz == pytest.approx(G.df)


def test_eye_traces() -> None:
    bits = np.array([0, 1, 1, 0])
    x = np.repeat(bits.astype(float), 4)
    eye = eye_diagram(x, 4, 1e-10, symbols_per_trace=2, offset=0)
    assert eye.traces.shape == (4, 9)
    np.testing.assert_array_equal(eye.traces[0], [0, 0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(eye.traces[3], [0, 0, 0, 0, 0, 0, 0, 0, 1])  # wraps around
    np.testing.assert_allclose(eye.time_s, np.arange(9) * 0.25e-10)
    assert set(np.unique(eye.traces)) == {0.0, 1.0}  # no interpolation artefacts


def test_sampling_helpers() -> None:
    assert band_edge_energy_fraction(np.zeros(1000), G) == 0.0
    sig = _tones({0: 1.0, 480: 1.0})  # 48 GHz tone, beyond 0.9 * 50 GHz
    assert band_edge_energy_fraction(sig.field, G) == pytest.approx(0.5)
    assert occupied_bandwidth(_tones({5: 1.0}).field, G) == pytest.approx(2 * 5 * G.df)
    assert samples_per_symbol_diagnostic(8) is None
    assert samples_per_symbol_diagnostic(3) is not None
