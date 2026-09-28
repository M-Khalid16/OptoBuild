from __future__ import annotations

import numpy as np
import pytest

from optobuild.cli.demos import optical_link_project
from optobuild.gui.plotdata import (
    SPECTRUM_FLOOR_DBM,
    engineering_scale,
    eye_curve,
    scalar_rows,
    spectrum_curve,
    waveform_curve,
)
from optobuild.persistence import run_project


@pytest.fixture(scope="module")
def result():  # type: ignore[no-untyped-def]
    return run_project(optical_link_project(prbs_order=7))


def test_engineering_scale() -> None:
    assert engineering_scale(np.array([2e-5, -3e-5])) == (1e-6, "µ")
    assert engineering_scale(0.0) == (1.0, "")
    assert engineering_scale(np.array([1500.0])) == (1e3, "k")


def test_eye_curve_preserves_samples(result) -> None:  # type: ignore[no-untyped-def]
    r = dict(result.nodes["eye"].results)
    c = eye_curve(r)
    traces = r["traces"]
    finite = c.y[np.isfinite(c.y)]
    factor = traces.max() / finite.max()
    np.testing.assert_allclose(finite * factor, traces.ravel(), rtol=1e-12)
    assert np.isnan(c.y).sum() == traces.shape[0]
    assert c.x_label == "time [ps]" and "[µA]" in c.y_label


def test_spectrum_curve_units_and_floor(result) -> None:  # type: ignore[no-untyped-def]
    r = dict(result.nodes["tx_spectrum"].results)
    c = spectrum_curve(r)
    assert c.y.min() >= SPECTRUM_FLOOR_DBM
    k = int(np.argmax(r["power_per_rbw_w"]))
    assert c.y[k] == pytest.approx(10 * np.log10(r["power_per_rbw_w"][k] / 1e-3))
    assert c.x[k] == pytest.approx(r["wavelength_m"][k] * 1e9)
    assert "12.5 GHz RBW" in c.y_label
    f = spectrum_curve(r, axis="frequency")
    assert f.x_label == "frequency [THz]" and f.x[k] == pytest.approx(r["frequency_hz"][k] / 1e12)


def test_waveforms_and_scalars(result) -> None:  # type: ignore[no-untyped-def]
    opt = waveform_curve(result.signal("mzm", "optical_out"))
    assert opt.y_label.startswith("optical power")
    el = waveform_curve(result.signal("pin", "out"))
    assert el.y_label.startswith("current")
    bits = waveform_curve(result.signal("prbs", "out"))
    assert bits.x.size == 128
    with pytest.raises(TypeError):
        waveform_curve(42)
    rows = scalar_rows(result)
    assert ("ber", "n_bits", "127") in rows
    assert not any(k == "traces" for _, k, _ in rows)


def test_constellation_curves() -> None:
    from optobuild.gui.plotdata import constellation_curve
    from optobuild.signals import SymbolSequence

    c = constellation_curve(np.array([1 + 1j, -1 - 1j]))
    assert c.scatter and c.x.tolist() == [1.0, -1.0] and c.y.tolist() == [1.0, -1.0]
    w = waveform_curve(SymbolSequence(np.array([1j, 1.0]), 1e9))
    assert w.scatter and w.x.size == 2
