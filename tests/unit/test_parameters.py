"""Schema-driven parameter validation of components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import InvalidParameterError
from optobuild.signals.kinds import SignalKind


class _Dev(Component):
    type_id = "tests.dev"
    version = "1.0.0"
    display_name = "dev"
    category = ComponentCategory.ELECTRICAL
    output_ports = (PortSpec("out", SignalKind.ELECTRICAL),)
    parameter_specs = (
        ParameterSpec(
            "v_pi",
            ParameterType.FLOAT,
            default=None,
            unit="V",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        ParameterSpec("loss", ParameterType.FLOAT, default=1.0, minimum=0.0, maximum=1.0),
        ParameterSpec("order", ParameterType.INT, default=4, minimum=1, maximum=10),
        ParameterSpec("enabled", ParameterType.BOOL, default=True),
        ParameterSpec(
            "shape", ParameterType.CHOICE, default="bessel", choices=("bessel", "butterworth")
        ),
        ParameterSpec("label", ParameterType.STRING, default="x"),
        ParameterSpec("lo", ParameterType.FLOAT, default=0.0, unit="V"),
        ParameterSpec("hi", ParameterType.FLOAT, default=1.0, unit="V"),
    )

    def validate(self) -> list[Diagnostic]:
        diags = super().validate()
        p = self.parameters
        if p["lo"] >= p["hi"]:
            diags.append(
                Diagnostic(
                    Severity.ERROR, "dev.levels", "lo must be < hi.", hint="Swap the levels."
                )
            )
        if p["order"] > 8:
            diags.append(Diagnostic(Severity.WARNING, "dev.order", "High filter order."))
        return diags

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        return {}


def test_defaults_and_coercion() -> None:
    d = _Dev("d", {"v_pi": 4, "order": 3.0})
    assert d.parameters["v_pi"] == 4.0 and isinstance(d.parameters["v_pi"], float)
    assert d.parameters["order"] == 3 and isinstance(d.parameters["order"], int)
    assert d.parameters["loss"] == 1.0 and d.parameters["shape"] == "bessel"


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        ({}, "'v_pi' is required"),
        ({"v_pi": 0.0}, "'v_pi' must be > 0 V"),
        ({"v_pi": -1.0}, "must be > 0"),
        ({"v_pi": 1.0, "loss": 1.5}, "'loss' must be <= 1"),
        ({"v_pi": 1.0, "order": 2.5}, "must be an integer"),
        ({"v_pi": 1.0, "order": 0}, "'order' must be >= 1"),
        ({"v_pi": True}, "must be a number"),
        ({"v_pi": "4 V"}, "must be a number"),
        ({"v_pi": float("nan")}, "must be finite"),
        ({"v_pi": 1.0, "enabled": 1}, "must be a boolean"),
        ({"v_pi": 1.0, "shape": "chebyshev"}, "must be one of"),
        ({"v_pi": 1.0, "label": 3}, "must be a string"),
        ({"v_pi": 1.0, "vpi": 2.0}, "has no parameter"),
        ({"v_pi": 1.0, "lo": 2.0}, "lo must be < hi"),
    ],
)
def test_invalid_parameters_raise_with_explanation(params: dict, fragment: str) -> None:
    with pytest.raises(InvalidParameterError) as info:
        _Dev("d", params)
    assert fragment in str(info.value)
    assert "Hint" in str(info.value)


def test_multiple_errors_reported_together() -> None:
    with pytest.raises(InvalidParameterError) as info:
        _Dev("d", {"loss": 2.0, "order": 0})
    msg = str(info.value)
    assert "'v_pi' is required" in msg and "'loss'" in msg and "'order'" in msg


def test_warnings_are_kept() -> None:
    d = _Dev("d", {"v_pi": 1.0, "order": 9})
    assert [x.code for x in d.diagnostics] == ["dev.order"]


def test_with_parameters_revalidates() -> None:
    d = _Dev("d", {"v_pi": 1.0})
    e = d.with_parameters(v_pi=2.0)
    assert e.name == "d" and e.parameters["v_pi"] == 2.0 and d.parameters["v_pi"] == 1.0
    with pytest.raises(InvalidParameterError):
        d.with_parameters(v_pi=-2.0)


def test_parameter_spec_lookup() -> None:
    assert _Dev.parameter_spec("v_pi").unit == "V"
    with pytest.raises(InvalidParameterError):
        _Dev.parameter_spec("nope")
