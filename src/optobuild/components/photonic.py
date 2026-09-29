"""Photonic-circuit components for signal-flow simulations (Phase 7).

Each component applies the frequency response of a device model
(physics.integrated_optics; docs/physics_models.md 3.22-3.23) to the optical
envelope, evaluated at the absolute frequencies nu = f_ref + f of the FFT
bins, identically for both polarizations (polarization-independent devices).
Signal-flow assumptions: unidirectional excitation (waves reflected back
into upstream components are not re-injected; a grating's reflection is an
output port), linear and time-invariant, periodic window (see the window
diagnostic). Each component records its power transfer over the simulated
band (``transfer_frequency_hz`` absolute, ``transfer_<port>`` linear) and
the device figures of merit. Arbitrary netlists are analysed in the
frequency domain with ``solvers.circuit``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import SamplingError
from optobuild.numerics.fft import apply_transfer_function
from optobuild.physics.integrated_optics import (
    add_drop_ring,
    all_pass_loaded_q,
    all_pass_ring,
    bragg_bandwidth,
    bragg_grating,
    bragg_peak_reflectance,
    mzi,
    waveguide_transmission,
)
from optobuild.signals import OpticalSignal, SignalKind, require_same_grid

OPT = SignalKind.OPTICAL


def _f(**kw: Any) -> ParameterSpec:
    return ParameterSpec(type=ParameterType.FLOAT, **kw)


def _waveguide_specs(default_loss_db_cm: float = 2.0) -> tuple[ParameterSpec, ...]:
    return (
        _f(name="n_eff", default=2.4, unit="1", minimum=1.0, description="Phase index at lambda_0"),
        _f(name="n_group", default=4.2, unit="1", minimum=1.0, description="Group index"),
        _f(
            name="design_wavelength",
            default=1550e-9,
            unit="m",
            display_unit="nm",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="lambda_0",
        ),
        _f(
            name="loss",
            default=default_loss_db_cm * math.log(10) / 10 * 100,
            unit="1/m",
            display_unit="dB/cm",
            minimum=0.0,
            symbol="alpha",
            description="Waveguide propagation loss",
        ),
    )


def _coupling(name: str, default: float, description: str) -> ParameterSpec:
    return _f(
        name=name, default=default, unit="1", minimum=0.0, maximum=1.0, description=description
    )


def _abs_frequency(sig: OpticalSignal) -> np.ndarray:
    return sig.center_frequency + sig.grid.frequency()


def _apply(sig: OpticalSignal, h: np.ndarray) -> np.ndarray:
    return apply_transfer_function(sig.field, h[None, :], sig.grid)


def _record_transfer(context: RunContext, nu: np.ndarray, **ports: np.ndarray) -> None:
    order = np.argsort(nu)
    context.record("transfer_frequency_hz", nu[order])
    for port, h in ports.items():
        context.record(f"transfer_{port}", np.abs(h[order]) ** 2)


def _resolution_check(
    context: RunContext, name: str, sig: OpticalSignal, linewidth_hz: float
) -> None:
    """Resolving a resonance needs several FFT bins across it (df = 1 / window)."""
    if linewidth_hz < 4 * sig.grid.df:
        context.warn(
            Diagnostic(
                Severity.WARNING,
                "photonic.resonance_unresolved",
                f"Device linewidth {linewidth_hz:.3g} Hz spans < 4 frequency bins "
                f"(df = {sig.grid.df:.3g} Hz); its impulse response exceeds the window and "
                "wraps around.",
                hint="Use a longer time window (more samples or bits).",
                source=name,
            )
        )


class AllPassRing(Component):
    """All-pass microring: bus waveguide coupled to a ring (through port only).

    H = (t - A)/(1 - t A), A = a exp(-i (beta L + phase_shift)), L = 2 pi R.
    """

    type_id = "optobuild.photonic.all_pass_ring"
    version = "1.0.0"
    display_name = "Ring resonator (all-pass)"
    category = ComponentCategory.PHOTONIC_CIRCUIT
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("through", OPT),)
    parameter_specs = (
        _f(
            name="radius",
            default=10e-6,
            unit="m",
            display_unit="um",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        _coupling("power_coupling", 0.05, "Bus-ring power coupling kappa^2"),
        _f(
            name="phase_shift",
            default=0.0,
            unit="rad",
            description="Extra round-trip phase (tuning)",
        ),
        *_waveguide_specs(),
    )

    def _round_trip(self, nu: np.ndarray) -> np.ndarray:
        p = self.parameters
        length = 2 * math.pi * p["radius"]
        h = waveguide_transmission(
            nu, length, p["n_eff"], p["n_group"], p["design_wavelength"], p["loss"]
        )
        return h * np.exp(-1j * p["phase_shift"])

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        p = self.parameters
        nu = _abs_frequency(sig)
        h = all_pass_ring(self._round_trip(nu), p["power_coupling"])
        length = 2 * math.pi * p["radius"]
        fsr = SPEED_OF_LIGHT / (p["n_group"] * length)
        lam = sig.wavelength
        q = all_pass_loaded_q(lam, p["n_group"], length, p["power_coupling"], p["loss"])
        context.record("fsr_hz", fsr)
        context.record("fsr_m", lam**2 / (p["n_group"] * length))
        context.record("loaded_q", q)
        _record_transfer(context, nu, through=h)
        _resolution_check(context, self.name, sig, sig.center_frequency / q)
        return {"through": sig.replace(field=_apply(sig, h))}


class AddDropRing(Component):
    """Add-drop microring: two bus waveguides; ports in/add -> through/drop.

    in -> through: (t1 - t2 A)/(1 - t1 t2 A);  in -> drop: -k1 k2 sqrt(A)/(1 - t1 t2 A);
    add -> drop: (t2 - t1 A)/(1 - t1 t2 A);   add -> through: as in -> drop (reciprocity).
    """

    type_id = "optobuild.photonic.add_drop_ring"
    version = "1.0.0"
    display_name = "Ring resonator (add-drop)"
    category = ComponentCategory.PHOTONIC_CIRCUIT
    input_ports = (PortSpec("in", OPT), PortSpec("add", OPT, optional=True))
    output_ports = (PortSpec("through", OPT), PortSpec("drop", OPT))
    parameter_specs = (
        _f(
            name="radius",
            default=10e-6,
            unit="m",
            display_unit="um",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        _coupling("power_coupling_in", 0.05, "Input bus power coupling kappa_1^2"),
        _coupling("power_coupling_drop", 0.05, "Drop bus power coupling kappa_2^2"),
        _f(
            name="phase_shift",
            default=0.0,
            unit="rad",
            description="Extra round-trip phase (tuning)",
        ),
        *_waveguide_specs(),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        p = self.parameters
        nu = _abs_frequency(sig)
        length = 2 * math.pi * p["radius"]
        half = waveguide_transmission(
            nu, length / 2, p["n_eff"], p["n_group"], p["design_wavelength"], p["loss"]
        ) * np.exp(-0.5j * p["phase_shift"])
        k1, k2 = p["power_coupling_in"], p["power_coupling_drop"]
        through, drop = add_drop_ring(half, k1, k2)
        out_t, out_d = _apply(sig, through), _apply(sig, drop)
        add = inputs.get("add")
        if add is not None:
            require_same_grid(sig, add, what=f"inputs of '{self.name}'")
            if add.center_frequency != sig.center_frequency or add.n_pol != sig.n_pol:
                raise SamplingError(
                    f"'{self.name}': the add signal needs the same carrier and polarizations.",
                    hint="Generate both from the same grid/carrier.",
                )
            add_drop, add_through = add_drop_ring(half, k2, k1)
            out_t = out_t + _apply(add, add_through)
            out_d = out_d + _apply(add, add_drop)
        a = math.exp(-0.5 * p["loss"] * length)
        t1t2 = math.sqrt((1 - k1) * (1 - k2))
        fsr = SPEED_OF_LIGHT / (p["n_group"] * length)
        # loaded linewidth (Lorentzian limit) with round-trip loss t1 t2 a
        fwhm = fsr * (1 - t1t2 * a) / (math.pi * math.sqrt(t1t2 * a))
        drop_peak = k1 * k2 * a / (1 - t1t2 * a) ** 2
        context.record("fsr_hz", fsr)
        context.record("loaded_q", sig.center_frequency / fwhm)
        context.record("drop_peak_transmission", drop_peak)
        _record_transfer(context, nu, through=through, drop=drop)
        _resolution_check(context, self.name, sig, fwhm)
        return {"through": sig.replace(field=out_t), "drop": sig.replace(field=out_d)}


class MachZehnderInterferometer(Component):
    """Unbalanced MZI filter: coupler, arms (length L and L + dL), coupler.

    [bar, cross]^T = C_out diag(h1, h2 exp(-i phase_shift)) C_in [in1, in2]^T.
    """

    type_id = "optobuild.photonic.mzi"
    version = "1.0.0"
    display_name = "Mach-Zehnder interferometer"
    category = ComponentCategory.PHOTONIC_CIRCUIT
    input_ports = (PortSpec("in1", OPT), PortSpec("in2", OPT, optional=True))
    output_ports = (PortSpec("bar", OPT), PortSpec("cross", OPT))
    parameter_specs = (
        _f(
            name="arm_length",
            default=1e-3,
            unit="m",
            display_unit="um",
            minimum=0.0,
        ),
        _f(
            name="delta_length",
            default=100e-6,
            unit="m",
            display_unit="um",
            description="Extra length of arm 2 (sets the FSR c / (n_g dL))",
        ),
        _coupling("coupling_in", 0.5, "Input coupler power ratio"),
        _coupling("coupling_out", 0.5, "Output coupler power ratio"),
        _f(name="phase_shift", default=0.0, unit="rad", description="Extra phase in arm 2"),
        *_waveguide_specs(),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in1"]
        p = self.parameters
        nu = _abs_frequency(sig)
        wg = (p["n_eff"], p["n_group"], p["design_wavelength"], p["loss"])
        h1 = waveguide_transmission(nu, p["arm_length"], *wg)
        h2 = waveguide_transmission(nu, p["arm_length"] + p["delta_length"], *wg)
        m = mzi(h1, h2 * np.exp(-1j * p["phase_shift"]), p["coupling_in"], p["coupling_out"])
        bar, cross = _apply(sig, m[0, 0]), _apply(sig, m[1, 0])
        other = inputs.get("in2")
        if other is not None:
            require_same_grid(sig, other, what=f"inputs of '{self.name}'")
            if other.center_frequency != sig.center_frequency or other.n_pol != sig.n_pol:
                raise SamplingError(f"'{self.name}': in2 needs the same carrier and polarizations.")
            bar = bar + _apply(other, m[0, 1])
            cross = cross + _apply(other, m[1, 1])
        if p["delta_length"] != 0.0:
            context.record("fsr_hz", SPEED_OF_LIGHT / (p["n_group"] * abs(p["delta_length"])))
        _record_transfer(context, nu, bar=m[0, 0], cross=m[1, 0])
        return {"bar": sig.replace(field=bar), "cross": sig.replace(field=cross)}


class BraggGrating(Component):
    """Uniform Bragg grating (coupled-mode theory): transmitted and reflected outputs.

    Period Lambda = lambda_B / (2 n_eff). The reflected wave is presented as an output
    (unidirectional excitation); its time origin is that of the input.
    """

    type_id = "optobuild.photonic.bragg_grating"
    version = "1.0.0"
    display_name = "Bragg grating"
    category = ComponentCategory.PHOTONIC_CIRCUIT
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("transmitted", OPT), PortSpec("reflected", OPT))
    parameter_specs = (
        _f(
            name="length",
            default=300e-6,
            unit="m",
            display_unit="um",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        _f(
            name="coupling",
            default=1e4,
            unit="1/m",
            minimum=0.0,
            symbol="kappa_g",
            description="Coupling coefficient (2 dn / lambda for a square profile)",
        ),
        _f(
            name="bragg_wavelength",
            default=1550e-9,
            unit="m",
            display_unit="nm",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="lambda_B",
        ),
        _f(name="n_eff", default=2.4, unit="1", minimum=1.0),
        _f(name="n_group", default=4.2, unit="1", minimum=1.0),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        p = self.parameters
        nu = _abs_frequency(sig)
        lam_b = p["bragg_wavelength"]
        period = lam_b / (2 * p["n_eff"])
        r, t = bragg_grating(
            nu, p["length"], period, p["coupling"], p["n_eff"], p["n_group"], lam_b
        )
        bw = bragg_bandwidth(lam_b, p["n_group"], p["coupling"], p["length"])
        context.record("peak_reflectance", bragg_peak_reflectance(p["coupling"], p["length"]))
        context.record("bandwidth_m", bw)
        context.record("period_m", period)
        _record_transfer(context, nu, transmitted=t, reflected=r)
        _resolution_check(context, self.name, sig, SPEED_OF_LIGHT * bw / lam_b**2)
        return {
            "transmitted": sig.replace(field=_apply(sig, t)),
            "reflected": sig.replace(field=_apply(sig, r)),
        }


__all__ = ["AddDropRing", "AllPassRing", "BraggGrating", "MachZehnderInterferometer"]
