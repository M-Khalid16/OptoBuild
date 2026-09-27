"""NRZ driver and Mach-Zehnder modulator.

Equations: optobuild.physics.modulation, docs/physics_models.md sec. 3.3-3.4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic
from optobuild.core.errors import SignalTypeError
from optobuild.numerics.sampling import aliasing_diagnostic, samples_per_symbol_diagnostic
from optobuild.physics.modulation import mzm_field_transfer, nrz_grid, nrz_waveform
from optobuild.signals import (
    DigitalSequence,
    ElectricalQuantity,
    ElectricalSignal,
    OpticalSignal,
    SignalKind,
    require_same_grid,
)
from optobuild.signals import metadata as meta


class NRZGenerator(Component):
    """Non-return-to-zero voltage waveform from a bit sequence (optional Gaussian edges)."""

    type_id = "optobuild.modulator.nrz_generator"
    version = "1.0.0"
    display_name = "NRZ pulse generator"
    category = ComponentCategory.MODULATOR
    input_ports = (PortSpec("bits", SignalKind.DIGITAL),)
    output_ports = (PortSpec("out", SignalKind.ELECTRICAL, "drive voltage [V]"),)
    parameter_specs = (
        ParameterSpec("samples_per_bit", ParameterType.INT, default=16, minimum=2, symbol="sps"),
        ParameterSpec(
            "low",
            ParameterType.FLOAT,
            default=-1.0,
            unit="V",
            symbol="V0",
            description="Level of bit 0",
        ),
        ParameterSpec(
            "high",
            ParameterType.FLOAT,
            default=1.0,
            unit="V",
            symbol="V1",
            description="Level of bit 1",
        ),
        ParameterSpec(
            "rise_time",
            ParameterType.FLOAT,
            default=0.0,
            unit="s",
            display_unit="ps",
            minimum=0.0,
            symbol="t_r",
            description="10-90 % rise time of Gaussian-filtered edges (0 = rectangular)",
        ),
    )

    def validate(self) -> list[Diagnostic]:
        diags = super().validate()
        d = samples_per_symbol_diagnostic(self.parameters["samples_per_bit"])
        return diags + ([d] if d else [])

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        seq: DigitalSequence = inputs["bits"]
        p = self.parameters
        sps = p["samples_per_bit"]
        grid = nrz_grid(seq.n_bits, seq.bit_rate, sps)
        v = nrz_waveform(seq.bits, sps, grid, p["low"], p["high"], p["rise_time"])
        md = {meta.BIT_RATE: seq.bit_rate, meta.SAMPLES_PER_BIT: sps}
        if meta.PATTERN in seq.metadata:
            md[meta.PATTERN] = seq.metadata[meta.PATTERN]
        diag = aliasing_diagnostic(v - v.mean(), grid, self.name)
        if diag:
            context.warn(diag)
        return {"out": ElectricalSignal(grid, v, ElectricalQuantity.VOLTAGE, md)}


class MachZehnderModulator(Component):
    """Push-pull chirp-free MZM with insertion loss and finite extinction ratio.

    Field transfer sqrt(IL) [cos(dphi/2) + i eps sin(dphi/2)],
    dphi = pi (V + V_bias)/V_pi + phi0, eps = 1/sqrt(ER). With the default
    V_bias = -V_pi/2 the device is at quadrature, and an NRZ drive of
    +-V_pi/2 swings between minimum and maximum transmission.
    """

    type_id = "optobuild.modulator.mzm"
    version = "1.0.0"
    display_name = "Mach-Zehnder modulator"
    category = ComponentCategory.MODULATOR
    input_ports = (
        PortSpec("optical_in", SignalKind.OPTICAL, "optical carrier"),
        PortSpec("drive", SignalKind.ELECTRICAL, "drive voltage [V]"),
    )
    output_ports = (PortSpec("optical_out", SignalKind.OPTICAL),)
    parameter_specs = (
        ParameterSpec(
            "v_pi",
            ParameterType.FLOAT,
            default=4.0,
            unit="V",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="V_pi",
        ),
        ParameterSpec("v_bias", ParameterType.FLOAT, default=-2.0, unit="V", symbol="V_bias"),
        ParameterSpec(
            "insertion_loss",
            ParameterType.FLOAT,
            default=1.0,
            unit="1",
            display_unit="dB",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
            symbol="IL",
            description="Linear power transmission at maximum (<= 1)",
        ),
        ParameterSpec(
            "extinction_ratio",
            ParameterType.FLOAT,
            default=1e4,
            unit="1",
            display_unit="dB",
            minimum=1.0,
            symbol="ER",
            description="Linear static extinction ratio (>= 1)",
        ),
        ParameterSpec(
            "phi0",
            ParameterType.FLOAT,
            default=0.0,
            unit="rad",
            symbol="phi0",
            description="Static arm phase difference",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        opt: OpticalSignal = inputs["optical_in"]
        drive: ElectricalSignal = inputs["drive"]
        if drive.quantity is not ElectricalQuantity.VOLTAGE:
            raise SignalTypeError(
                f"MZM '{self.name}' needs a voltage drive, got {drive.quantity.name}.",
                hint="Drive the modulator from a voltage source such as the NRZ generator.",
            )
        require_same_grid(opt, drive, what=f"inputs of MZM '{self.name}'")
        p = self.parameters
        h = mzm_field_transfer(
            drive.samples,
            p["v_pi"],
            p["v_bias"],
            p["insertion_loss"],
            p["extinction_ratio"],
            p["phi0"],
        )
        out = opt.replace(field=opt.field * h[None, :], metadata=dict(drive.metadata))
        diag = aliasing_diagnostic(
            out.field - out.field.mean(axis=-1, keepdims=True), out.grid, self.name
        )
        if diag:
            context.warn(diag)
        return {"optical_out": out}


__all__ = ["MachZehnderModulator", "NRZGenerator"]
