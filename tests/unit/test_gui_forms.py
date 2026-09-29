from __future__ import annotations

import math

import pytest

from optobuild.components.registry import builtin_registry
from optobuild.components.spec import ParameterSpec, ParameterType
from optobuild.core.errors import InvalidParameterError
from optobuild.gui.forms import FieldModel, WidgetKind, fields_for

REG = builtin_registry()


def _field(type_id: str, name: str) -> FieldModel:
    return FieldModel(REG.get(type_id).parameter_spec(name))


@pytest.mark.parametrize("type_id", list(REG))
def test_every_builtin_default_round_trips_through_its_form(type_id: str) -> None:
    """Displaying and re-entering each default must reproduce the SI value."""
    cls = REG.get(type_id)
    comp = cls("x")
    for f in fields_for(cls.parameter_specs):
        shown = f.to_display(comp.parameters[f.name])
        back = f.from_display(shown)
        value = comp.parameters[f.name]
        if isinstance(value, float):
            assert back == pytest.approx(value, rel=1e-11, abs=1e-300), (type_id, f.name)
        else:
            assert back == value
        if f.spec.display_unit:
            assert f.unit == f.spec.display_unit, f"{type_id}.{f.name}: display unit unusable"


def test_display_units_and_labels() -> None:
    power = _field("optobuild.source.cw_laser", "power")
    assert power.widget is WidgetKind.TEXT_NUMBER and power.unit == "dBm"
    assert power.to_display(1e-3) == "0" and power.to_display(0.0) == "-inf"
    assert power.from_display("3") == pytest.approx(1.9952623e-3)
    assert power.from_display("-inf") == 0.0
    assert power.from_display("2 mW") == pytest.approx(2e-3)
    assert power.label == "power (P0) [dBm]"
    assert "Stored in SI unit 'W'" in power.tooltip
    wl = _field("optobuild.source.cw_laser", "wavelength")
    assert wl.to_display(1550e-9) == "1550"
    assert wl.from_display("1.31 um") == pytest.approx(1310e-9)
    il = _field("optobuild.modulator.mzm", "insertion_loss")
    assert il.to_display(10**-0.5) == "5"  # shown as a positive loss
    assert il.from_display("3") == pytest.approx(10**-0.3)
    d = _field("optobuild.channel.linear_fiber", "dispersion")
    assert d.from_display("17") == pytest.approx(17e-6)
    thr = _field("optobuild.electrical.decision", "threshold")
    assert thr.label.endswith("[A or V]") and thr.from_display("1e-5") == 1e-5


@pytest.mark.parametrize("text", ["abc", "1550 parsecs", "2*3", "__import__('os')", "nan", "5 W"])
def test_bad_input_is_rejected_with_message(text: str) -> None:
    wl = _field("optobuild.source.cw_laser", "wavelength")
    with pytest.raises(InvalidParameterError):
        wl.from_display(text)


def test_other_widget_kinds() -> None:
    order = _field("optobuild.source.prbs", "order")
    assert order.widget is WidgetKind.CHOICE and 31 in order.choices
    n = _field("optobuild.source.prbs", "n_bits")
    assert n.widget is WidgetKind.INTEGER and n.from_display(" 42 ") == 42
    with pytest.raises(InvalidParameterError):
        n.from_display("4.5")
    assert _field("optobuild.detector.pin", "shot_noise").widget is WidgetKind.CHECKBOX
    s = FieldModel(ParameterSpec("label", ParameterType.STRING, default="a"))
    assert s.widget is WidgetKind.TEXT and s.from_display("b") == "b"
    assert FieldModel(ParameterSpec("x", ParameterType.FLOAT)).to_display(-math.inf) == "-inf"
