"""Polarization components (Phase 6b): PBS, PBC, polarization controller, random PMD.

Equations: optobuild.physics.polarization; docs/physics_models.md sec. 3.20.
A scalar (n_pol = 1) signal is a single, fully polarized state; where its
orientation matters (the PBS) it is a parameter of the receiving component.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import SamplingError, SignalTypeError
from optobuild.physics.polarization import (
    apply_jones,
    dgd_element,
    dgd_from_jones,
    pmd_jones,
    random_unitary,
    sop_transform,
    waveplate_section_dgd,
)
from optobuild.signals import OpticalSignal, SignalKind, require_same_grid

OPT = SignalKind.OPTICAL


def _angle(name: str, default: float, description: str) -> ParameterSpec:
    return ParameterSpec(
        name,
        ParameterType.FLOAT,
        default=default,
        unit="rad",
        display_unit="deg",
        description=description,
    )


def _require_dual(sig: OpticalSignal, who: str) -> None:
    if sig.n_pol != 2:
        raise SignalTypeError(
            f"'{who}' acts on dual-polarization fields; the input has n_pol = {sig.n_pol}.",
            hint="Combine two polarizations with a PolarizationBeamCombiner first.",
        )


class PolarizationBeamSplitter(Component):
    """Ideal PBS: x and y components of the input to two scalar outputs.

    A scalar input is taken as linearly polarized at azimuth ``input_angle``
    to the PBS x axis: A_x = cos(theta) A, A_y = sin(theta) A (45 deg splits the
    power equally, as for the LO of a polarization-diverse receiver).
    """

    type_id = "optobuild.passive.pbs"
    version = "1.0.0"
    display_name = "Polarization beam splitter"
    category = ComponentCategory.PASSIVE
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("x", OPT), PortSpec("y", OPT))
    parameter_specs = (
        _angle("input_angle", math.pi / 4, "Azimuth of a scalar input's polarization"),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        if sig.n_pol == 2:
            fx, fy = sig.field[0], sig.field[1]
        else:
            th = self.parameters["input_angle"]
            fx, fy = math.cos(th) * sig.field[0], math.sin(th) * sig.field[0]
        return {"x": sig.replace(field=fx[None, :]), "y": sig.replace(field=fy[None, :])}


class PolarizationBeamCombiner(Component):
    """Ideal PBC: two scalar inputs become the x and y components of one field."""

    type_id = "optobuild.passive.pbc"
    version = "1.0.0"
    display_name = "Polarization beam combiner"
    category = ComponentCategory.PASSIVE
    input_ports = (PortSpec("x", OPT), PortSpec("y", OPT))
    output_ports = (PortSpec("out", OPT),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        x: OpticalSignal = inputs["x"]
        y: OpticalSignal = inputs["y"]
        require_same_grid(x, y, what=f"inputs of '{self.name}'")
        if x.n_pol != 1 or y.n_pol != 1:
            raise SignalTypeError(f"'{self.name}' combines two single-polarization signals.")
        if x.center_frequency != y.center_frequency:
            raise SamplingError(
                f"'{self.name}': carriers differ ({x.center_frequency:g} vs "
                f"{y.center_frequency:g} Hz).",
                hint="Derive both polarizations from the same laser.",
            )
        if abs(x.grid.t0 - y.grid.t0) > 1e-3 * x.grid.dt:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "pbc.time_origin_mismatch",
                    "The x and y inputs have different time origins; the x origin is used.",
                    source=self.name,
                )
            )
        return {"out": x.replace(field=np.vstack([x.field[0], y.field[0]]))}


class PolarizationController(Component):
    """Deterministic polarization transformation with optional first-order PMD.

    J(w) = U(theta, phi) D(w, tau): a DGD tau between the input x (slow) and y
    (fast) axes, followed by the SOP transformation U (physics.polarization).
    """

    type_id = "optobuild.passive.polarization_controller"
    version = "1.0.0"
    display_name = "Polarization controller / DGD"
    category = ComponentCategory.PASSIVE
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("out", OPT),)
    parameter_specs = (
        _angle("azimuth", 0.0, "Rotation angle theta of U(theta, phi)"),
        _angle("phase", 0.0, "Phase phi of U(theta, phi)"),
        ParameterSpec(
            "dgd",
            ParameterType.FLOAT,
            default=0.0,
            unit="s",
            display_unit="ps",
            minimum=0.0,
            symbol="tau",
            description="Differential group delay between the input axes",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        _require_dual(sig, self.name)
        p = self.parameters
        u = sop_transform(p["azimuth"], p["phase"])
        if p["dgd"] == 0.0:
            return {"out": sig.replace(field=apply_jones(sig.field, u, sig.grid))}
        j = np.einsum("ij,jkn->ikn", u, dgd_element(sig.grid.angular_frequency(), p["dgd"]))
        return {"out": sig.replace(field=apply_jones(sig.field, j, sig.grid))}


class RandomPMD(Component):
    """Fiber PMD, waveplate model: N sections with Haar-random coupling (one draw per trial).

    Equal section DGDs tau_s are chosen so that the mean DGD equals ``mean_dgd``
    in the Maxwellian limit (physics.polarization.waveplate_section_dgd). The
    realized DGD at the carrier is recorded as ``dgd_s``. Lossless and
    dispersion-free: place it next to a (linear) fiber component.
    """

    type_id = "optobuild.channel.random_pmd"
    version = "1.0.0"
    display_name = "Random PMD (waveplate model)"
    category = ComponentCategory.CHANNEL
    stochastic = True
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("out", OPT),)
    parameter_specs = (
        ParameterSpec(
            "mean_dgd",
            ParameterType.FLOAT,
            default=10e-12,
            unit="s",
            display_unit="ps",
            minimum=0.0,
            symbol="<tau>",
            description="PMD coefficient times sqrt(length)",
        ),
        ParameterSpec("n_sections", ParameterType.INT, default=32, minimum=1),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        _require_dual(sig, self.name)
        n = self.parameters["n_sections"]
        taus = np.full(n, waveplate_section_dgd(self.parameters["mean_dgd"], n))
        us = np.array([random_unitary(context.rng) for _ in range(n)])
        j = pmd_jones(sig.grid.angular_frequency(), taus, us)
        dw = 1e-3 / max(float(taus.sum()), 1e-30)  # dw * total delay << 1: O(dw^2) error
        j0 = pmd_jones(np.array([-dw / 2, dw / 2]), taus, us)
        dgd = dgd_from_jones(j0[:, :, 0], j0[:, :, 1], dw)
        context.record("dgd_s", dgd)
        if dgd > 0.1 * sig.grid.duration:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "pmd.dgd_vs_window",
                    f"DGD {dgd:.3g} s exceeds 10 % of the time window (circular wrap-around).",
                    hint="Use a longer window (more symbols).",
                    source=self.name,
                )
            )
        return {"out": sig.replace(field=apply_jones(sig.field, j, sig.grid))}


__all__ = [
    "PolarizationBeamCombiner",
    "PolarizationBeamSplitter",
    "PolarizationController",
    "RandomPMD",
]
