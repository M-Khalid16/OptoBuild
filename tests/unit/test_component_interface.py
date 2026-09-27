"""Phase 0 checks of the component *interface* (no concrete physics components yet)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.errors import InvalidGraphError
from optobuild.signals.kinds import SignalKind


class _Ctx:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.logger = logging.getLogger("test")

    def check_cancelled(self) -> None:
        pass

    def report_progress(self, fraction: float, message: str = "") -> None:
        pass

    def record(self, key: str, value: Any) -> None:
        pass

    def warn(self, diagnostic: Any) -> None:
        pass


class _Gain(Component):
    """Test-only electrical gain: out = g * in."""

    type_id = "tests.electrical.gain"
    version = "0.1.0"
    display_name = "Test gain"
    category = ComponentCategory.ELECTRICAL
    input_ports = (PortSpec("in", SignalKind.ELECTRICAL),)
    output_ports = (PortSpec("out", SignalKind.ELECTRICAL),)
    parameter_specs = (ParameterSpec("gain", ParameterType.FLOAT, default=1.0, unit="1"),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        return {"out": self.parameters["gain"] * inputs["in"]}


def test_component_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        Component("x")  # type: ignore[abstract]


def test_minimal_component_runs() -> None:
    comp = _Gain("g1", {"gain": 2.0})
    ctx = _Ctx()
    assert isinstance(ctx, RunContext)
    out = comp.run({"in": np.array([1.0, 2.0])}, ctx)
    np.testing.assert_allclose(out["out"], [2.0, 4.0])
    assert comp.inputs()[0].kind is SignalKind.ELECTRICAL
    assert comp.validate() == []
    assert "out = g * in" in comp.documentation()


def test_parameters_are_read_only() -> None:
    comp = _Gain("g1", {"gain": 2.0})
    with pytest.raises(TypeError):
        comp.parameters["gain"] = 3.0  # type: ignore[index]


def test_spec_records_are_immutable() -> None:
    p = PortSpec("in", SignalKind.OPTICAL)
    with pytest.raises(AttributeError):
        p.kind = SignalKind.ELECTRICAL  # type: ignore[misc]
    assert ParameterSpec("v_pi", ParameterType.FLOAT).required


def test_empty_name_rejected() -> None:
    with pytest.raises(InvalidGraphError):
        _Gain("")


def test_invalid_type_id_rejected() -> None:
    with pytest.raises(TypeError, match="type_id"):

        class _Bad(_Gain):
            type_id = "Bad Id"


def test_duplicate_port_names_rejected() -> None:
    with pytest.raises(TypeError, match="duplicate"):

        class _Dup(_Gain):
            type_id = "tests.electrical.dup"
            input_ports = (
                PortSpec("in", SignalKind.ELECTRICAL),
                PortSpec("in", SignalKind.ELECTRICAL),
            )


def test_missing_metadata_rejected() -> None:
    with pytest.raises(TypeError, match="must define"):

        class _NoMeta(Component):
            def run(self, inputs, context):  # type: ignore[no-untyped-def]
                return {}
