"""Linear optical fiber. Equations: optobuild.physics.fiber, physics_models.md sec. 3.5."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.units import (
    db_per_km_to_per_m,
    dispersion_slope_to_beta3,
    dispersion_to_beta2,
)
from optobuild.numerics.sampling import occupied_bandwidth
from optobuild.physics.fiber import dispersive_spread, group_delay, propagate_linear
from optobuild.signals import OpticalSignal, SignalKind

WRAP_WARNING_FRACTION = 0.25
"""Dispersive spread (fraction of the window) above which wrap-around is flagged."""


class LinearFiber(Component):
    """Single-mode fiber with attenuation, group delay and chromatic dispersion (linear).

    D and S are specified at the signal's carrier wavelength; beta2 and beta3
    are derived from them for each input signal.
    """

    type_id = "optobuild.channel.linear_fiber"
    version = "1.0.0"
    display_name = "Optical fiber (linear)"
    category = ComponentCategory.CHANNEL
    input_ports = (PortSpec("in", SignalKind.OPTICAL),)
    output_ports = (PortSpec("out", SignalKind.OPTICAL),)
    parameter_specs = (
        ParameterSpec(
            "length",
            ParameterType.FLOAT,
            default=10e3,
            unit="m",
            display_unit="km",
            minimum=0.0,
            symbol="L",
        ),
        ParameterSpec(
            "attenuation",
            ParameterType.FLOAT,
            default=float(db_per_km_to_per_m(0.2)),
            unit="1/m",
            display_unit="dB/km",
            minimum=0.0,
            symbol="alpha",
            description="Power attenuation coefficient",
        ),
        ParameterSpec(
            "dispersion",
            ParameterType.FLOAT,
            default=17e-6,
            unit="s/m^2",
            display_unit="ps/(nm km)",
            symbol="D",
        ),
        ParameterSpec(
            "dispersion_slope",
            ParameterType.FLOAT,
            default=0.0,
            unit="s/m^3",
            display_unit="ps/(nm^2 km)",
            symbol="S",
        ),
        ParameterSpec(
            "group_index",
            ParameterType.FLOAT,
            default=1.4682,
            minimum=1.0,
            symbol="n_g",
            description="Group index (sets the propagation delay)",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        p = self.parameters
        lam = sig.wavelength
        beta2 = dispersion_to_beta2(p["dispersion"], lam)
        beta3 = dispersion_slope_to_beta3(p["dispersion"], p["dispersion_slope"], lam)
        field, grid = propagate_linear(
            sig.field, sig.grid, p["length"], p["attenuation"], beta2, beta3, p["group_index"]
        )
        spread = dispersive_spread(
            occupied_bandwidth(sig.field, sig.grid), p["length"], beta2, beta3
        )
        if spread > WRAP_WARNING_FRACTION * sig.grid.duration:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "sampling.window_wraparound",
                    f"Dispersive spread {spread:.3g} s is {spread / sig.grid.duration:.0%} of the "
                    f"{sig.grid.duration:.3g} s window; energy wraps around the window edges "
                    "(correct only for a pattern that is periodic in the window).",
                    hint="Use a longer window (more bits) or a periodic pattern.",
                    source=self.name,
                )
            )
        context.record("beta2_s2_per_m", beta2)
        context.record("beta3_s3_per_m", beta3)
        context.record("group_delay_s", group_delay(p["length"], p["group_index"]))
        context.record("dispersive_spread_s", spread)
        return {"out": sig.replace(field=field, grid=grid)}


__all__ = ["LinearFiber"]
