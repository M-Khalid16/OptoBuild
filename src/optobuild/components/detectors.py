"""PIN photodiode. Equations: optobuild.physics.detection, physics_models.md sec. 3.6."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.physics.detection import photocurrent, pin_noise
from optobuild.signals import ElectricalQuantity, ElectricalSignal, OpticalSignal, SignalKind


class PINPhotodiode(Component):
    """Square-law PIN detector with signal-dependent shot noise, dark current and thermal noise."""

    type_id = "optobuild.detector.pin"
    version = "1.0.0"
    display_name = "PIN photodiode"
    category = ComponentCategory.DETECTOR
    stochastic = True
    input_ports = (PortSpec("in", SignalKind.OPTICAL),)
    output_ports = (PortSpec("out", SignalKind.ELECTRICAL, "photocurrent [A]"),)
    parameter_specs = (
        ParameterSpec(
            "responsivity", ParameterType.FLOAT, default=1.0, unit="A/W", minimum=0.0, symbol="R"
        ),
        ParameterSpec(
            "dark_current",
            ParameterType.FLOAT,
            default=0.0,
            unit="A",
            display_unit="nA",
            minimum=0.0,
            symbol="I_d",
        ),
        ParameterSpec(
            "temperature",
            ParameterType.FLOAT,
            default=300.0,
            unit="K",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="T",
        ),
        ParameterSpec(
            "load_resistance",
            ParameterType.FLOAT,
            default=50.0,
            unit="ohm",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="R_L",
        ),
        ParameterSpec("shot_noise", ParameterType.BOOL, default=True),
        ParameterSpec("thermal_noise", ParameterType.BOOL, default=True),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        p = self.parameters
        mean = photocurrent(sig.power(), p["responsivity"], p["dark_current"])
        noise = pin_noise(
            context.rng,
            mean,
            sig.grid.sample_rate,
            shot=p["shot_noise"],
            thermal=p["thermal_noise"],
            temperature=p["temperature"],
            load_resistance=p["load_resistance"],
        )
        return {
            "out": ElectricalSignal(
                sig.grid, mean + noise, ElectricalQuantity.CURRENT, dict(sig.metadata)
            )
        }


__all__ = ["PINPhotodiode"]
