"""Optical pulse source and nonlinear fiber (Phase 4).

Equations: optobuild.physics.sources.pulse_field, optobuild.physics.nonlinear,
optobuild.solvers.ssfm; docs/physics_models.md sec. 3.10-3.11.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optobuild.components.base import TIMING_SOURCE_SPEC, Component, RunContext, require_layout
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.units import (
    db_per_km_to_per_m,
    dispersion_slope_to_beta3,
    dispersion_to_beta2,
    wavelength_to_frequency,
)
from optobuild.numerics.grid import TimeGrid
from optobuild.numerics.sampling import band_edge_energy_fraction, time_edge_energy_fraction
from optobuild.physics.fiber import group_delay
from optobuild.physics.nonlinear import effective_length
from optobuild.physics.sources import PULSE_SHAPES, pulse_field
from optobuild.signals import OpticalSignal, SignalKind
from optobuild.solvers.ssfm import propagate_nlse

EDGE_THRESHOLD = 1e-6
"""Energy fraction near the window/band edges above which a warning is issued."""


class OpticalPulseSource(Component):
    """Single unchirped Gaussian or sech pulse centred in the simulation window."""

    type_id = "optobuild.source.optical_pulse"
    version = "1.0.0"
    display_name = "Optical pulse source"
    category = ComponentCategory.SOURCE
    output_ports = (PortSpec("out", SignalKind.OPTICAL, "optical pulse"),)
    parameter_specs = (
        ParameterSpec("shape", ParameterType.CHOICE, default="sech", choices=PULSE_SHAPES),
        ParameterSpec(
            "peak_power",
            ParameterType.FLOAT,
            default=1e-3,
            unit="W",
            display_unit="mW",
            minimum=0.0,
            symbol="P0",
        ),
        ParameterSpec(
            "width",
            ParameterType.FLOAT,
            default=5e-12,
            unit="s",
            display_unit="ps",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="T0",
            description="T0: 1/e-intensity half width (gaussian) or sech width",
        ),
        ParameterSpec(
            "wavelength",
            ParameterType.FLOAT,
            default=1550e-9,
            unit="m",
            display_unit="nm",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        ParameterSpec("n_samples", ParameterType.INT, default=4096, minimum=2),
        ParameterSpec(
            "sample_rate",
            ParameterType.FLOAT,
            default=10e12,
            unit="Hz",
            display_unit="GHz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        TIMING_SOURCE_SPEC,
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        p = self.parameters
        if p["timing_source"] == "layout":
            grid = require_layout(context, self.name).grid()
        else:
            grid = TimeGrid.from_sample_rate(p["n_samples"], p["sample_rate"])
        field = pulse_field(grid, p["shape"], p["peak_power"], p["width"])
        if p["width"] < 3 * grid.dt:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "sampling.pulse_undersampled",
                    f"Pulse width {p['width']:.3g} s spans fewer than 3 samples "
                    f"(dt = {grid.dt:.3g} s).",
                    hint="Increase the sample rate.",
                    source=self.name,
                )
            )
        edge = time_edge_energy_fraction(field)
        if edge > EDGE_THRESHOLD:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "sampling.window_too_short",
                    f"{edge:.2e} of the pulse energy lies in the outer 5 % of the window.",
                    hint="Increase n_samples or reduce the pulse width.",
                    source=self.name,
                )
            )
        freq = float(wavelength_to_frequency(p["wavelength"]))
        return {
            "out": OpticalSignal(
                grid, field, freq, {"source": self.name, "pulse_shape": p["shape"]}
            )
        }


class NonlinearFiber(Component):
    """Single-mode fiber with loss, dispersion (beta2, beta3) and Kerr SPM, solved by SSFM.

    Step size: adaptive (peak nonlinear phase per step <= ``max_phase``, step
    <= ``max_step``) or a fixed number of steps. With gamma = 0 the result
    equals the linear fiber exactly.
    """

    type_id = "optobuild.channel.nonlinear_fiber"
    version = "1.0.0"
    display_name = "Optical fiber (nonlinear, SSFM)"
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
            "gamma",
            ParameterType.FLOAT,
            default=1.3e-3,
            unit="1/(W m)",
            display_unit="1/(W km)",
            minimum=0.0,
            symbol="gamma",
            description="Kerr coefficient 2 pi n2 / (lambda A_eff)",
        ),
        ParameterSpec("group_index", ParameterType.FLOAT, default=1.4682, minimum=1.0),
        ParameterSpec(
            "step_mode", ParameterType.CHOICE, default="adaptive", choices=("adaptive", "fixed")
        ),
        ParameterSpec(
            "max_phase",
            ParameterType.FLOAT,
            default=0.005,
            unit="rad",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
            description="Adaptive mode: max peak nonlinear phase per step",
        ),
        ParameterSpec(
            "max_step",
            ParameterType.FLOAT,
            default=1e3,
            unit="m",
            display_unit="km",
            minimum=0.0,
            minimum_inclusive=False,
            description="Adaptive mode: upper limit of the step length",
        ),
        ParameterSpec(
            "n_steps",
            ParameterType.INT,
            default=100,
            minimum=1,
            description="Fixed mode: number of equal steps",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        p = self.parameters
        lam = sig.wavelength
        beta2 = dispersion_to_beta2(p["dispersion"], lam)
        beta3 = dispersion_slope_to_beta3(p["dispersion"], p["dispersion_slope"], lam)
        fixed = p["step_mode"] == "fixed"
        field, report = propagate_nlse(
            sig.field,
            sig.grid,
            p["length"],
            alpha=p["attenuation"],
            beta2=beta2,
            beta3=beta3,
            gamma=p["gamma"],
            max_phase=p["max_phase"],
            max_step=p["max_step"],
            n_steps=p["n_steps"] if fixed else None,
            check_cancelled=context.check_cancelled,
        )
        grid = sig.grid.with_t0(sig.grid.t0 + group_delay(p["length"], p["group_index"]))
        out = sig.replace(field=field, grid=grid)
        peak_in = float(sig.power().max())
        phi_nl = p["gamma"] * peak_in * effective_length(p["length"], p["attenuation"])
        context.record("beta2_s2_per_m", beta2)
        context.record("n_steps", report.n_steps)
        context.record("max_step_phase_rad", report.max_step_phase)
        context.record("nonlinear_phase_bound_rad", phi_nl)
        context.record("min_step_m", report.min_step)
        context.record("max_step_m", report.max_step)
        if fixed and report.max_step_phase > 0.05:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "ssfm.large_step_phase",
                    f"Peak nonlinear phase per step is {report.max_step_phase:.3g} rad "
                    "(> 0.05 rad); "
                    "the splitting error may be significant.",
                    hint="Increase n_steps or use step_mode='adaptive'.",
                    source=self.name,
                )
            )
        spec_edge = band_edge_energy_fraction(field, sig.grid)
        if spec_edge > EDGE_THRESHOLD:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "ssfm.spectral_truncation",
                    f"{spec_edge:.2e} of the output energy lies in the outer 10 % of the simulated "
                    "band; spectral broadening is reaching +-fs/2 (aliasing).",
                    hint="Increase the sample rate.",
                    source=self.name,
                )
            )
        t_in = time_edge_energy_fraction(sig.field)
        t_out = time_edge_energy_fraction(field)
        if t_in < EDGE_THRESHOLD < t_out:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "sampling.window_wraparound",
                    f"{t_out:.2e} of the pulse energy reached the window edges "
                    f"(input: {t_in:.1e}); "
                    "it wraps around the periodic window.",
                    hint="Use a longer window (more samples).",
                    source=self.name,
                )
            )
        return {"out": out}


__all__ = ["NonlinearFiber", "OpticalPulseSource"]
