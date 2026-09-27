"""Reference components for validating the framework independently of optics.

These blocks have trivially predictable outputs, so graph construction,
execution order, caching, serialization and seeding can be tested exactly.
They are real, documented components (usable from project files), not mocks.
All signals are electrical voltages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.numerics.grid import TimeGrid
from optobuild.signals import ElectricalSignal, SignalKind, require_same_grid

_E = SignalKind.ELECTRICAL


class RampSource(Component):
    """Deterministic ramp v_n = offset + step * n [V] on a grid of n_samples at sample_rate."""

    type_id = "optobuild.reference.ramp_source"
    version = "1.0.0"
    display_name = "Ramp source (reference)"
    category = ComponentCategory.SOURCE
    output_ports = (PortSpec("out", _E, "ramp voltage [V]"),)
    parameter_specs = (
        ParameterSpec("n_samples", ParameterType.INT, default=16, minimum=2, description="N"),
        ParameterSpec(
            "sample_rate",
            ParameterType.FLOAT,
            default=1e9,
            unit="Hz",
            display_unit="GHz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        ParameterSpec("offset", ParameterType.FLOAT, default=0.0, unit="V"),
        ParameterSpec("step", ParameterType.FLOAT, default=1.0, unit="V"),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        p = self.parameters
        grid = TimeGrid.from_sample_rate(p["n_samples"], p["sample_rate"])
        values = p["offset"] + p["step"] * np.arange(p["n_samples"], dtype=float)
        return {"out": ElectricalSignal(grid, values, metadata={"source": self.name})}


class Gain(Component):
    """Linear gain: out = gain * in."""

    type_id = "optobuild.reference.gain"
    version = "1.0.0"
    display_name = "Gain (reference)"
    category = ComponentCategory.ELECTRICAL
    input_ports = (PortSpec("in", _E),)
    output_ports = (PortSpec("out", _E),)
    parameter_specs = (ParameterSpec("gain", ParameterType.FLOAT, default=1.0, unit="1"),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: ElectricalSignal = inputs["in"]
        return {"out": sig.replace(samples=self.parameters["gain"] * sig.samples)}


class Adder(Component):
    """Sum of two waveforms on identical grids: out = a + b (multi-input example)."""

    type_id = "optobuild.reference.adder"
    version = "1.0.0"
    display_name = "Adder (reference)"
    category = ComponentCategory.ELECTRICAL
    input_ports = (PortSpec("a", _E), PortSpec("b", _E))
    output_ports = (PortSpec("out", _E),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        a: ElectricalSignal = inputs["a"]
        b: ElectricalSignal = inputs["b"]
        require_same_grid(a, b, what=f"inputs of '{self.name}'")
        return {"out": a.replace(samples=a.samples + b.samples)}


class GaussianNoise(Component):
    """Adds zero-mean Gaussian noise of standard deviation sigma [V] (seeded, ADR-0008)."""

    type_id = "optobuild.reference.gaussian_noise"
    version = "1.0.0"
    display_name = "Gaussian noise (reference)"
    category = ComponentCategory.ELECTRICAL
    input_ports = (PortSpec("in", _E),)
    output_ports = (PortSpec("out", _E),)
    parameter_specs = (
        ParameterSpec("sigma", ParameterType.FLOAT, default=1.0, unit="V", minimum=0.0),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: ElectricalSignal = inputs["in"]
        noise = context.rng.normal(0.0, self.parameters["sigma"], size=sig.grid.n_samples)
        return {"out": sig.replace(samples=sig.samples + noise)}


class Recorder(Component):
    """Sink recording the samples and their mean and RMS value [V]."""

    type_id = "optobuild.reference.recorder"
    version = "1.0.0"
    display_name = "Recorder (reference)"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("in", _E, tap=True),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: ElectricalSignal = inputs["in"]
        context.record("samples", np.array(sig.samples))
        context.record("mean", float(np.mean(sig.samples)))
        context.record("rms", float(np.sqrt(np.mean(sig.samples**2))))
        return {}


REFERENCE_COMPONENTS: tuple[type[Component], ...] = (
    RampSource,
    Gain,
    Adder,
    GaussianNoise,
    Recorder,
)

__all__ = ["REFERENCE_COMPONENTS", "Adder", "Gain", "GaussianNoise", "RampSource", "Recorder"]
