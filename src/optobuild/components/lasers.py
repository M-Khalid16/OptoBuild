"""Laser and amplifier device components (Phase 8).

Equations: physics.semiconductor_laser (rate equations), physics.edfa
(two-level erbium model); integration in solvers.laser_dynamics and
solvers.edfa; docs/physics_models.md sec. 3.25-3.26.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.components.base import Component, RunContext
from optobuild.components.sources import CWLaser
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import SignalTypeError
from optobuild.core.units import linear_to_db
from optobuild.physics.edfa import Beam, ErbiumFiber
from optobuild.physics.noise import complex_white_noise
from optobuild.physics.semiconductor_laser import LaserParameters
from optobuild.signals import (
    ElectricalQuantity,
    ElectricalSignal,
    OpticalSignal,
    SignalKind,
)
from optobuild.solvers.edfa import amplify, ring_laser
from optobuild.solvers.laser_dynamics import simulate

OPT, E = SignalKind.OPTICAL, SignalKind.ELECTRICAL
_DEF = LaserParameters()


def _f(name: str, default: float, unit: str, **kw: Any) -> ParameterSpec:
    return ParameterSpec(name, ParameterType.FLOAT, default=default, unit=unit, **kw)


def _pos(name: str, default: float, unit: str, **kw: Any) -> ParameterSpec:
    return _f(name, default, unit, minimum=0.0, minimum_inclusive=False, **kw)


_LASER_SPECS = (
    _pos("volume", _DEF.volume, "m^3", display_unit="um^3", symbol="V"),
    _f("confinement", _DEF.confinement, "1", minimum=0.0, maximum=1.0, symbol="Gamma"),
    _pos("group_index", _DEF.group_index, "1", symbol="n_g"),
    _pos("differential_gain", _DEF.differential_gain, "m^2", display_unit="cm^2", symbol="a"),
    _f(
        "transparency_density",
        _DEF.transparency_density,
        "m^-3",
        display_unit="cm^-3",
        minimum=0.0,
        symbol="N_tr",
    ),
    _pos("carrier_lifetime", _DEF.carrier_lifetime, "s", display_unit="ns", symbol="tau_n"),
    _pos("photon_lifetime", _DEF.photon_lifetime, "s", display_unit="ps", symbol="tau_p"),
    _f(
        "gain_compression",
        _DEF.gain_compression,
        "m^3",
        display_unit="cm^3",
        minimum=0.0,
        symbol="eps",
    ),
    _f("spontaneous_coupling", _DEF.spontaneous_coupling, "1", minimum=0.0, symbol="beta"),
    _f("linewidth_enhancement", _DEF.linewidth_enhancement, "1", symbol="alpha"),
    _f("output_efficiency", _DEF.output_efficiency, "1", minimum=0.0, maximum=1.0, symbol="eta"),
)


class DirectlyModulatedLaser(Component):
    """Single-mode semiconductor laser driven by a current (rate equations).

    Injection current I(t) = bias + drive, with drive = the input current or
    input voltage / ``drive_resistance``. The emitted field is
    A = sqrt(P_out) exp(i (phi - w_bias t)): the signal reference frequency
    c / wavelength is the emission frequency at the bias current, so the
    phase carries the transient and adiabatic chirp relative to it. With
    ``noise`` the Langevin forces add intensity and phase noise (seeded).
    The initial state is the steady state at the first current sample.
    """

    type_id = "optobuild.laser.directly_modulated"
    version = "1.0.0"
    display_name = "Directly modulated laser"
    category = ComponentCategory.LASER
    stochastic = True
    input_ports = (PortSpec("drive", E, "modulation current or voltage"),)
    output_ports = (PortSpec("out", OPT),)
    parameter_specs = (
        _f("bias_current", 0.06, "A", display_unit="mA", minimum=0.0, symbol="I_b"),
        _pos(
            "drive_resistance",
            50.0,
            "ohm",
            description="Converts a voltage drive into current (I = v / R)",
        ),
        _pos("wavelength", _DEF.wavelength, "m", display_unit="nm"),
        ParameterSpec("noise", ParameterType.BOOL, default=False, description="Langevin noise"),
        *_LASER_SPECS,
    )

    def laser(self) -> LaserParameters:
        p = self.parameters
        names = [f.name for f in dataclasses.fields(LaserParameters)]
        return LaserParameters(**{n: p[n] for n in names})

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        drive: ElectricalSignal = inputs["drive"]
        p = self.parameters
        laser = self.laser()
        if drive.quantity is ElectricalQuantity.CURRENT:
            i_drive = drive.samples
        else:
            i_drive = drive.samples / p["drive_resistance"]
        current = p["bias_current"] + i_drive
        if np.any(current < 0):
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "laser.negative_current",
                    "The drive takes the injection current below zero; clipped to 0 A.",
                    hint="Increase the bias or reduce the drive amplitude.",
                    source=self.name,
                )
            )
            current = np.clip(current, 0.0, None)
        if np.min(current) < laser.threshold_current:
            context.warn(
                Diagnostic(
                    Severity.INFO,
                    "laser.below_threshold",
                    f"The current drops below threshold ({laser.threshold_current:.4g} A): "
                    "turn-on delay and strong transient chirp.",
                    source=self.name,
                )
            )
        grid = drive.grid
        n, s, phi = simulate(
            laser,
            current,
            grid.dt,
            rng=context.rng if p["noise"] else None,
            check_cancelled=context.check_cancelled,
        )
        power = laser.output_power(np.clip(s, 0.0, None))
        # reference frequency = steady-state emission frequency at the bias current
        n_b, s_b = laser.steady_state(p["bias_current"])
        w_bias = laser.rhs(p["bias_current"], n_b, s_b)[2]
        t = np.arange(grid.n_samples) * grid.dt
        field = np.sqrt(power) * np.exp(1j * (phi - w_bias * t))
        chirp = (laser.rhs(current, n, s)[2] - w_bias) / (2 * math.pi)
        context.record("threshold_current_a", laser.threshold_current)
        context.record("slope_efficiency_w_per_a", laser.slope_efficiency)
        context.record("bias_power_w", float(laser.output_power(s_b)))
        context.record("average_power_w", float(np.mean(power)))
        context.record("chirp_peak_to_peak_hz", float(np.ptp(chirp)))
        context.record("frequency_chirp_hz", chirp)
        freq = SPEED_OF_LIGHT / p["wavelength"]
        return {"out": OpticalSignal(grid, field, freq, dict(drive.metadata))}


_EDF = ErbiumFiber()


def _edfa_specs() -> tuple[ParameterSpec, ...]:
    return (
        _pos("fiber_length", _EDF.length, "m", symbol="L"),
        _pos("erbium_density", _EDF.erbium_density, "m^-3", display_unit="cm^-3", symbol="n_t"),
        _pos("doped_area", _EDF.doped_area, "m^2", display_unit="um^2", symbol="A"),
        _pos("lifetime", _EDF.lifetime, "s", display_unit="ms", symbol="tau"),
        _f("background_loss", 0.0, "1/m", display_unit="dB/km", minimum=0.0),
        _pos("pump_wavelength", 980e-9, "m", display_unit="nm"),
        _f("pump_absorption_cross_section", 2.2e-25, "m^2", minimum=0.0),
        _f("pump_emission_cross_section", 0.0, "m^2", minimum=0.0),
        _f("pump_overlap", 0.6, "1", minimum=0.0, maximum=1.0),
        _pos(
            "signal_wavelength",
            1550e-9,
            "m",
            display_unit="nm",
            description="Wavelength at which the signal cross sections are given",
        ),  # fmt: skip
        _f("signal_absorption_cross_section", 2.6e-25, "m^2", minimum=0.0),
        _f("signal_emission_cross_section", 3.4e-25, "m^2", minimum=0.0),
        _f("signal_overlap", 0.4, "1", minimum=0.0, maximum=1.0),
    )


def _edfa_model(p: Mapping[str, Any]) -> tuple[ErbiumFiber, Beam, Beam]:
    fiber = ErbiumFiber(
        p["fiber_length"], p["erbium_density"], p["doped_area"], p["lifetime"],
        p["background_loss"],
    )  # fmt: skip
    pump = Beam(
        p["pump_wavelength"], p["pump_absorption_cross_section"],
        p["pump_emission_cross_section"], p["pump_overlap"],
    )  # fmt: skip
    signal = Beam(
        p["signal_wavelength"], p["signal_absorption_cross_section"],
        p["signal_emission_cross_section"], p["signal_overlap"],
    )  # fmt: skip
    return fiber, pump, signal


class ErbiumDopedFiberAmplifier(Component):
    """EDFA (two-level, co-pumped): saturated gain and ASE from the physical model.

    The gain is computed for the input's average power (the ~10 ms erbium
    lifetime makes the gain quasi-static for data signals) at the signal
    wavelength parameter; the co-polarized ASE density S computed along the
    fiber is added as white noise to every polarization present (flat over the
    simulated band). Output OSNR (0.1 nm) = P_out / (2 S B_ref).
    """

    type_id = "optobuild.amplifier.edfa"
    version = "1.0.0"
    display_name = "EDFA (erbium-doped fiber amplifier)"
    category = ComponentCategory.AMPLIFIER
    stochastic = True
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("out", OPT),)
    parameter_specs = (
        _f("pump_power", 0.1, "W", display_unit="dBm", minimum=0.0),
        ParameterSpec("ase_noise", ParameterType.BOOL, default=True),
        *_edfa_specs(),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        p = self.parameters
        fiber, pump, beam = _edfa_model(p)
        p_in = sig.average_power()
        if p_in <= 0:
            raise SignalTypeError(f"'{self.name}': the input carries no power.")
        if abs(sig.wavelength - p["signal_wavelength"]) > 1e-9:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "edfa.wavelength_mismatch",
                    f"Signal at {sig.wavelength * 1e9:.2f} nm but cross sections given at "
                    f"{p['signal_wavelength'] * 1e9:.2f} nm.",
                    hint="Set signal_wavelength and the cross sections for this channel.",
                    source=self.name,
                )
            )
        sol = amplify(fiber, (pump, beam), (p["pump_power"], p_in))
        gain, ase, nf = sol.gains[1], sol.ase_density[1], sol.noise_figures[1]
        field = math.sqrt(gain) * sig.field
        if p["ase_noise"]:
            field = field + complex_white_noise(
                context.rng, ase, sig.grid.sample_rate, sig.field.shape
            )
        b_ref = 12.5e9
        context.record("gain_db", float(linear_to_db(gain)))
        context.record("noise_figure_db", float(linear_to_db(nf)))
        context.record("ase_psd_per_pol_w_per_hz", ase)
        context.record("output_osnr_db", float(linear_to_db(gain * p_in / (2 * ase * b_ref))))
        context.record("residual_pump_w", sol.powers_out[0])
        context.record("mean_upper_fraction", sol.mean_upper_fraction)
        return {"out": sig.replace(field=field)}


class FiberRingLaser(CWLaser):
    """Erbium fiber ring laser (CW): output power from the two-level ring steady state.

    Unidirectional ring of the doped fiber, an output coupler (``output_coupling``)
    and passive loss; lasing at ``wavelength`` with the signal cross sections given
    for it. Output P = T G P_c with G(P_c) (1 - T) eta = 1 (solvers.edfa.ring_laser).
    Linewidth, phase and grid behave as for the CW laser.
    """

    type_id = "optobuild.laser.fiber_ring"
    version = "1.0.0"
    display_name = "Fiber ring laser (erbium)"
    category = ComponentCategory.LASER
    parameter_specs = (
        _f("pump_power", 0.1, "W", display_unit="dBm", minimum=0.0),
        _f(
            "output_coupling",
            0.5,
            "1",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
            maximum_inclusive=False,
            symbol="T",
        ),  # fmt: skip
        _f(
            "passive_transmission",
            0.8,
            "1",
            display_unit="dB loss",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
            symbol="eta",
        ),  # fmt: skip
        *[s for s in CWLaser.parameter_specs if s.name != "power"],
        *[s for s in _edfa_specs() if s.name != "signal_wavelength"],
    )

    def output_power(self, context: RunContext) -> float:
        p = dict(self.parameters)
        p["signal_wavelength"] = p["wavelength"]
        fiber, pump, beam = _edfa_model(p)
        power, intracavity = ring_laser(
            fiber, pump, beam, p["pump_power"], p["output_coupling"], p["passive_transmission"]
        )
        context.record("output_power_w", power)
        context.record("intracavity_power_w", intracavity)
        if power == 0.0:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "laser.below_threshold",
                    "The pump is below the lasing threshold: no output.",
                    hint="Increase the pump power or reduce the cavity loss.",
                    source=self.name,
                )
            )
        return power


__all__ = ["DirectlyModulatedLaser", "ErbiumDopedFiberAmplifier", "FiberRingLaser"]
