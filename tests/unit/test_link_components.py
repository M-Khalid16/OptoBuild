"""Metadata and delegation tests: components must call the physics layer, not re-implement it."""

from __future__ import annotations

import numpy as np
import pytest

from optobuild.components.library import OPTICAL_LINK_COMPONENTS
from optobuild.components.modulators import MachZehnderModulator
from optobuild.components.registry import builtin_registry
from optobuild.components.spec import ParameterType
from optobuild.core.errors import InvalidParameterError
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.modulation import mzm_field_transfer
from optobuild.signals import ElectricalSignal, OpticalSignal, SignalKind


@pytest.mark.parametrize("cls", OPTICAL_LINK_COMPONENTS, ids=lambda c: c.type_id)
def test_metadata_is_complete_and_registered(cls: type) -> None:
    assert cls.type_id in builtin_registry()
    assert cls.documentation()
    comp = cls("x")  # defaults must be valid
    for spec in cls.parameter_specs:
        if spec.type in (ParameterType.FLOAT, ParameterType.INT):
            assert spec.unit, f"{cls.type_id}.{spec.name} lacks a unit"
    for port in comp.inputs():
        if cls.category.value == "analyzer":
            assert port.tap, "analyzer inputs must be non-intrusive taps"


def test_mzm_delegates_to_physics(run_component) -> None:  # type: ignore[no-untyped-def]
    grid = TimeGrid(64, 1e-12)
    rng = np.random.default_rng(0)
    field = rng.normal(size=64) + 1j * rng.normal(size=64)
    v = rng.normal(size=64)
    mzm = MachZehnderModulator(
        "m",
        {"v_pi": 3.0, "v_bias": 0.2, "insertion_loss": 0.6, "extinction_ratio": 50.0, "phi0": 0.1},
    )
    out = run_component(
        mzm, {"optical_in": OpticalSignal(grid, field, 193e12), "drive": ElectricalSignal(grid, v)}
    )[0]["optical_out"]
    np.testing.assert_array_equal(
        out.field[0], field * mzm_field_transfer(v, 3.0, 0.2, 0.6, 50.0, 0.1)
    )


def test_port_kinds_of_the_link() -> None:
    reg = builtin_registry()
    mzm = reg.get("optobuild.modulator.mzm")
    assert [p.kind for p in mzm.input_ports] == [SignalKind.OPTICAL, SignalKind.ELECTRICAL]
    pin = reg.get("optobuild.detector.pin")
    assert pin.output_ports[0].kind is SignalKind.ELECTRICAL


@pytest.mark.parametrize(
    ("type_id", "params"),
    [
        ("optobuild.modulator.mzm", {"insertion_loss": 1.5}),
        ("optobuild.modulator.mzm", {"extinction_ratio": 0.5}),
        ("optobuild.modulator.mzm", {"v_pi": 0.0}),
        ("optobuild.channel.linear_fiber", {"length": -1.0}),
        ("optobuild.channel.linear_fiber", {"group_index": 0.9}),
        ("optobuild.detector.pin", {"load_resistance": 0.0}),
        ("optobuild.electrical.lowpass_filter", {"kind": "chebyshev"}),
        ("optobuild.electrical.decision", {"sampling_phase": 1.0}),
        ("optobuild.source.prbs", {"order": 8}),
        ("optobuild.source.cw_laser", {"power": -1e-3}),
    ],
)
def test_invalid_physical_parameters_rejected(type_id: str, params: dict) -> None:
    with pytest.raises(InvalidParameterError):
        builtin_registry().create(type_id, "x", params)
