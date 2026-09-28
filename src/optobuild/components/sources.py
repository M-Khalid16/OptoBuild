"""Source components: CW laser and PRBS generator.

Equations: optobuild.physics.sources, optobuild.physics.prbs,
docs/physics_models.md sec. 3.1-3.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.components.base import TIMING_SOURCE_SPEC, Component, RunContext, require_layout
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import SamplingError
from optobuild.core.units import wavelength_to_frequency
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.prbs import PRBS_POLYNOMIALS, prbs, prbs_period
from optobuild.physics.sources import (
    cw_field,
    frequency_offset_is_periodic,
    wiener_phase_noise,
)
from optobuild.signals import DigitalSequence, OpticalSignal, SignalKind
from optobuild.signals import metadata as meta


class CWLaser(Component):
    """Ideal continuous-wave laser: A = sqrt(P0) exp(i (phi0 + 2 pi df t)).

    The signal reference frequency is c / wavelength; ``frequency_offset``
    detunes the laser from it. The sampling grid comes from ``n_samples`` and
    ``sample_rate``, or from the global layout when ``timing_source="layout"``;
    it must match the electrical drive of the modulator it feeds.
    A non-zero ``linewidth`` adds Wiener phase noise (Lorentzian line,
    physics.sources.wiener_phase_noise).
    """

    type_id = "optobuild.source.cw_laser"
    version = "1.2.0"
    display_name = "CW laser"
    category = ComponentCategory.SOURCE
    stochastic = True
    output_ports = (PortSpec("out", SignalKind.OPTICAL, "CW optical field"),)
    parameter_specs = (
        ParameterSpec(
            "power",
            ParameterType.FLOAT,
            default=1e-3,
            unit="W",
            display_unit="dBm",
            minimum=0.0,
            symbol="P0",
            description="Output power",
        ),
        ParameterSpec(
            "wavelength",
            ParameterType.FLOAT,
            default=1550e-9,
            unit="m",
            display_unit="nm",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="lambda0",
            description="Vacuum wavelength of the signal reference carrier",
        ),
        ParameterSpec("phase", ParameterType.FLOAT, default=0.0, unit="rad", symbol="phi0"),
        ParameterSpec(
            "frequency_offset",
            ParameterType.FLOAT,
            default=0.0,
            unit="Hz",
            display_unit="GHz",
            symbol="df",
            description="Laser frequency minus reference frequency",
        ),
        ParameterSpec(
            "n_samples",
            ParameterType.INT,
            default=2048,
            minimum=2,
            description="Number of samples of the simulation window",
        ),
        ParameterSpec(
            "sample_rate",
            ParameterType.FLOAT,
            default=160e9,
            unit="Hz",
            display_unit="GHz",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        TIMING_SOURCE_SPEC,
        ParameterSpec(
            "linewidth",
            ParameterType.FLOAT,
            default=0.0,
            unit="Hz",
            display_unit="kHz",
            minimum=0.0,
            symbol="dnu",
            description="Lorentzian FWHM linewidth (0 = ideal)",
        ),
    )

    @property
    def frequency(self) -> float:
        """Reference optical frequency c / wavelength [Hz]."""
        return float(wavelength_to_frequency(self.parameters["wavelength"]))

    def validate(self) -> list[Diagnostic]:
        diags = super().validate()
        p = self.parameters
        if p["timing_source"] == "layout":
            return diags  # grid unknown until run time; checked in run()
        grid = TimeGrid.from_sample_rate(p["n_samples"], p["sample_rate"])
        if abs(p["frequency_offset"]) >= grid.nyquist_frequency:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "laser.offset_out_of_band",
                    f"frequency_offset {p['frequency_offset']:g} Hz is outside +-fs/2 = "
                    f"+-{grid.nyquist_frequency:g} Hz.",
                    hint="Reduce the offset or increase the sample rate.",
                )
            )
        elif not frequency_offset_is_periodic(grid, p["frequency_offset"]):
            diags.append(
                Diagnostic(
                    Severity.WARNING,
                    "laser.offset_not_periodic",
                    "frequency_offset is not a multiple of 1/T; the periodic window has a phase "
                    "jump at its edge (spectral leakage).",
                    hint=f"Use a multiple of df = {grid.df:g} Hz.",
                )
            )
        return diags

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        p = self.parameters
        if p["timing_source"] == "layout":
            grid = require_layout(context, self.name).grid()
            if abs(p["frequency_offset"]) >= grid.nyquist_frequency:
                raise SamplingError(
                    f"frequency_offset {p['frequency_offset']:g} Hz is outside the layout's "
                    f"+-fs/2 = +-{grid.nyquist_frequency:g} Hz.",
                    hint="Reduce the offset or increase samples_per_bit.",
                )
            if not frequency_offset_is_periodic(grid, p["frequency_offset"]):
                context.warn(
                    Diagnostic(
                        Severity.WARNING,
                        "laser.offset_not_periodic",
                        "frequency_offset is not a multiple of 1/T of the layout window "
                        "(spectral leakage).",
                        hint=f"Use a multiple of df = {grid.df:g} Hz.",
                        source=self.name,
                    )
                )
        else:
            grid = TimeGrid.from_sample_rate(p["n_samples"], p["sample_rate"])
        field = cw_field(grid, p["power"], p["phase"], p["frequency_offset"])
        if p["linewidth"] > 0:
            phi = wiener_phase_noise(context.rng, p["linewidth"], grid.dt, grid.n_samples)
            field = field * np.exp(1j * phi)[None, :]
        return {"out": OpticalSignal(grid, field, self.frequency, {"source": self.name})}


class PRBSGenerator(Component):
    """Maximal-length pseudo-random bit sequence (Fibonacci LFSR, ITU-T O.150 polynomials).

    With ``timing_source="layout"`` the number of bits and the bit rate come
    from the global layout and ``n_bits``/``bit_rate`` are ignored.
    """

    type_id = "optobuild.source.prbs"
    version = "1.1.0"
    display_name = "PRBS generator"
    category = ComponentCategory.SOURCE
    output_ports = (PortSpec("out", SignalKind.DIGITAL, "bit sequence"),)
    parameter_specs = (
        ParameterSpec(
            "order",
            ParameterType.CHOICE,
            default=7,
            choices=tuple(sorted(PRBS_POLYNOMIALS)),
            symbol="m",
            description="Register length; period 2^m - 1",
        ),
        ParameterSpec(
            "n_bits",
            ParameterType.INT,
            default=0,
            minimum=0,
            description="Number of bits (0 = one full period 2^m - 1)",
        ),
        ParameterSpec(
            "seed",
            ParameterType.INT,
            default=0,
            minimum=0,
            description="Initial register state (0 = all ones)",
        ),
        ParameterSpec(
            "bit_rate",
            ParameterType.FLOAT,
            default=10e9,
            unit="bit/s",
            display_unit="Gb/s",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="R_b",
        ),
        TIMING_SOURCE_SPEC,
    )

    @property
    def n_bits(self) -> int:
        """Effective number of output bits."""
        return self.parameters["n_bits"] or prbs_period(self.parameters["order"])

    def validate(self) -> list[Diagnostic]:
        diags = super().validate()
        m = self.parameters["order"]
        if self.parameters["seed"] > prbs_period(m):
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "prbs.seed",
                    f"seed must be <= 2^{m} - 1.",
                    hint="Use 0 (all ones) or a register state that fits in m bits.",
                )
            )
        if self.n_bits % prbs_period(m):
            diags.append(
                Diagnostic(
                    Severity.INFO,
                    "sampling.pattern_periodicity",
                    f"n_bits={self.n_bits} is not a multiple of the PRBS period {prbs_period(m)}; "
                    "the periodic simulation window then repeats a truncated pattern.",
                    hint="Use n_bits = 0 (one period) or a multiple of 2^m - 1.",
                )
            )
        return diags

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        p = self.parameters
        n_bits, bit_rate = self.n_bits, p["bit_rate"]
        if p["timing_source"] == "layout":
            layout = require_layout(context, self.name)
            n_bits, bit_rate = layout.n_bits, layout.bit_rate
            if n_bits % prbs_period(p["order"]):
                context.warn(
                    Diagnostic(
                        Severity.INFO,
                        "sampling.pattern_periodicity",
                        f"Layout n_bits={n_bits} is not a multiple of the PRBS{p['order']} "
                        f"period {prbs_period(p['order'])}.",
                        hint="Choose n_bits as a multiple of 2^m - 1.",
                        source=self.name,
                    )
                )
        bits = prbs(p["order"], n_bits, p["seed"] or None)
        md = {meta.PATTERN: f"PRBS{p['order']}", "source": self.name}
        return {"out": DigitalSequence(bits, bit_rate, md)}


__all__ = ["CWLaser", "PRBSGenerator"]
