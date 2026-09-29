from __future__ import annotations

import numpy as np
import pytest

from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid
from optobuild.signals import (
    DigitalSequence,
    ElectricalQuantity,
    ElectricalSignal,
    OpticalSignal,
    SignalKind,
    require_same_grid,
)

G = TimeGrid(8, 1e-12)
F0 = 193.4e12


def test_optical_basic_properties() -> None:
    field = np.sqrt(2e-3) * np.exp(1j * 0.3) * np.ones(8)
    s = OpticalSignal(G, field, F0, {"src": "laser"})
    assert s.kind is SignalKind.OPTICAL
    assert s.field.shape == (1, 8) and s.n_pol == 1
    assert s.average_power() == pytest.approx(2e-3)
    np.testing.assert_allclose(s.phase(), 0.3)
    assert s.wavelength == pytest.approx(SPEED_OF_LIGHT / F0)
    np.testing.assert_allclose(s.absolute_frequency(), F0 + G.frequency())


def test_dual_polarization_power_sums() -> None:
    field = np.vstack([np.full(8, 1.0), np.full(8, 1j * np.sqrt(2.0))])
    s = OpticalSignal(G, field, F0)
    assert s.n_pol == 2
    np.testing.assert_allclose(s.power(), 3.0)


def test_signals_are_immutable_and_copy_input() -> None:
    src = np.ones(8, dtype=complex)
    s = OpticalSignal(G, src, F0)
    src[0] = 5.0
    assert s.field[0, 0] == 1.0
    with pytest.raises(ValueError):
        s.field[0, 0] = 2.0
    with pytest.raises(AttributeError):
        s.center_frequency = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        s.metadata["x"] = 1  # type: ignore[index]
    e = ElectricalSignal(G, np.zeros(8))
    with pytest.raises(ValueError):
        e.samples[0] = 1.0


def test_replace_merges_metadata() -> None:
    s = OpticalSignal(G, np.ones(8), F0, {"a": 1})
    t = s.replace(field=2 * np.ones(8), metadata={"b": 2})
    assert dict(t.metadata) == {"a": 1, "b": 2}
    assert t.average_power() == pytest.approx(4.0)
    assert s.average_power() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "f0"),
    [
        (np.ones(7), F0),  # length mismatch
        (np.ones((3, 8)), F0),  # 3 polarizations
        (np.ones((1, 1, 8)), F0),  # 3-D
        (np.full(8, np.nan), F0),  # NaN
        (np.ones(8), 0.0),  # non-positive carrier
        (np.ones(8), float("inf")),
    ],
)
def test_optical_rejects_invalid(field: np.ndarray, f0: float) -> None:
    with pytest.raises(SamplingError):
        OpticalSignal(G, field, f0)


def test_electrical() -> None:
    e = ElectricalSignal(G, np.arange(8), ElectricalQuantity.CURRENT)
    assert e.unit == "A" and e.mean() == pytest.approx(3.5)
    assert e.samples.dtype == np.float64
    with pytest.raises(SamplingError, match="real"):
        ElectricalSignal(G, np.ones(8) * 1j)
    with pytest.raises(SamplingError):
        ElectricalSignal(G, np.ones(9))
    with pytest.raises(SamplingError):
        ElectricalSignal(G, np.ones((2, 8)))
    with pytest.raises(SamplingError):
        ElectricalSignal(G, np.ones(8), quantity="V")  # type: ignore[arg-type]


def test_digital() -> None:
    d = DigitalSequence(np.array([1, 0, 1, 1]), 10e9)
    assert d.n_bits == 4 and d.bit_period == pytest.approx(1e-10)
    assert d.duration == pytest.approx(4e-10)
    assert d.bits.dtype == np.uint8
    with pytest.raises(ValueError):
        d.bits[0] = 0
    for bad in ([0, 2], [0.5], [], [[0, 1]]):
        with pytest.raises(SamplingError):
            DigitalSequence(np.array(bad), 1e9)
    with pytest.raises(SamplingError):
        DigitalSequence(np.array([0, 1]), 0.0)


def test_require_same_grid() -> None:
    a = ElectricalSignal(G, np.zeros(8))
    require_same_grid(a, ElectricalSignal(G.with_t0(1e-9), np.zeros(8)))
    with pytest.raises(SamplingError, match="Incompatible"):
        require_same_grid(a, ElectricalSignal(TimeGrid(8, 2e-12), np.zeros(8)))


def test_symbol_sequence() -> None:
    from optobuild.signals import SymbolSequence

    s = SymbolSequence(np.array([1 + 1j, -1 - 1j]) / np.sqrt(2), 32e9, {"modulation": "qpsk"})
    assert s.kind is SignalKind.SYMBOLS and s.n_symbols == 2
    assert s.symbol_period == pytest.approx(1 / 32e9)
    assert s.average_energy() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        s.symbols[0] = 0
    assert "qpsk" in repr(s)
    for bad in (np.array([]), np.ones((3, 4)), np.array([np.nan])):
        with pytest.raises(SamplingError):
            SymbolSequence(bad, 1e9)
    with pytest.raises(SamplingError):
        SymbolSequence(np.ones(3), 0.0)
    assert SymbolSequence(np.ones((2, 5)), 1e9).n_symbols == 5
