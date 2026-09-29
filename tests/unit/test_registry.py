from __future__ import annotations

import pytest

from optobuild.components.reference import Gain, RampSource
from optobuild.components.registry import ComponentRegistry, builtin_registry
from optobuild.core.errors import ProjectFormatError


def test_builtin_registry_contains_reference_components() -> None:
    reg = builtin_registry()
    assert "optobuild.reference.gain" in reg
    assert reg.get("optobuild.reference.gain") is Gain
    comp = reg.create("optobuild.reference.gain", "g", {"gain": 3.0})
    assert comp.parameters["gain"] == 3.0
    info = reg.describe("optobuild.reference.ramp_source")
    assert info["outputs"] == [("out", "electrical")]
    assert {p["name"] for p in info["parameters"]} >= {"n_samples", "sample_rate"}


def test_registries_are_independent() -> None:
    a = builtin_registry()
    b = ComponentRegistry()
    assert len(b) == 0 and len(a) > 0


def test_unknown_type() -> None:
    with pytest.raises(ProjectFormatError, match="Unknown component type"):
        builtin_registry().get("optobuild.nope")


def test_register_rules() -> None:
    reg = ComponentRegistry([Gain])
    reg.register(Gain)  # idempotent

    class Other(RampSource):
        type_id = "optobuild.reference.gain"

    with pytest.raises(ValueError, match="already registered"):
        reg.register(Other)
    with pytest.raises(TypeError):
        reg.register(int)  # type: ignore[arg-type]
