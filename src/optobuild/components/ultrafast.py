"""Ultrafast components (Phase 9): GNLSE fiber, mode-locked fiber laser, autocorrelator.

Equations: physics.ultrafast (GNLSE), physics.cavity (lumped cavity
elements), solvers.gnlse (RK4IP), solvers.cavity (round-trip iteration),
analysis.pulses; docs/physics_models.md sec. 3.27-3.28.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.analysis.pulses import (
    AUTOCORRELATION_FACTORS,
    fwhm,
    intensity_autocorrelation,
    spectral_intensity,
)
from optobuild.components.base import Component, RunContext
from optobuild.components.nonlinear import EDGE_THRESHOLD
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import SignalTypeError
from optobuild.core.units import db_per_km_to_per_m
from optobuild.numerics.grid import TimeGrid
from optobuild.numerics.sampling import band_edge_energy_fraction, time_edge_energy_fraction
from optobuild.physics.cavity import (
    gaussian_filter,
    output_coupler,
    saturable_absorber,
    saturated_gain,
)
from optobuild.physics.fiber import group_delay
from optobuild.signals import OpticalSignal, SignalKind
from optobuild.solvers.cavity import run_cavity
from optobuild.solvers.gnlse import propagate_gnlse

OPT = SignalKind.OPTICAL


def _f(name: str, default: float, unit: str, **kw: Any) -> ParameterSpec:
    return ParameterSpec(name, ParameterType.FLOAT, default=default, unit=unit, **kw)


def _beta(k: int, default: float = 0.0) -> ParameterSpec:
    return _f(f"beta{k}", default, f"s^{k}/m", display_unit=f"ps^{k}/km", symbol=f"beta_{k}")


class UltrafastFiber(Component):
    """Fiber for ultrashort pulses: generalized NLSE (dispersion to beta10, Kerr,
    self-steepening, Raman), RK4IP with step-doubling error control (or fixed steps).

    Scalar field only (single polarization). The group delay n_g L / c is added to t0.
    """

    type_id = "optobuild.channel.ultrafast_fiber"
    version = "1.0.0"
    display_name = "Optical fiber (GNLSE, ultrafast)"
    category = ComponentCategory.CHANNEL
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("out", OPT),)
    parameter_specs = (
        _f("length", 1.0, "m", minimum=0.0, symbol="L"),
        _f(
            "attenuation", float(db_per_km_to_per_m(0.2)), "1/m", display_unit="dB/km", minimum=0.0
        ),  # fmt: skip
        _beta(2, -21.7e-27),
        *(_beta(k) for k in range(3, 11)),
        _f("gamma", 1.3e-3, "1/(W m)", display_unit="1/(W km)", minimum=0.0, symbol="gamma"),
        ParameterSpec("self_steepening", ParameterType.BOOL, default=True),
        ParameterSpec("raman", ParameterType.BOOL, default=True),
        _f("raman_fraction", 0.18, "1", minimum=0.0, maximum=1.0, symbol="f_R"),
        _f("raman_tau1", 12.2e-15, "s", display_unit="fs", minimum=0.0, minimum_inclusive=False),
        _f("raman_tau2", 32e-15, "s", display_unit="fs", minimum=0.0, minimum_inclusive=False),
        _f("group_index", 1.4682, "1", minimum=1.0),
        ParameterSpec(
            "step_mode", ParameterType.CHOICE, default="adaptive", choices=("adaptive", "fixed")
        ),
        _f(
            "tolerance",
            1e-6,
            "1",
            minimum=0.0,
            minimum_inclusive=False,
            description="Adaptive mode: relative local error per step",
        ),  # fmt: skip
        ParameterSpec("n_steps", ParameterType.INT, default=100, minimum=1),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        if sig.n_pol != 1:
            raise SignalTypeError(
                f"'{self.name}' propagates a scalar field; the input has two polarizations.",
                hint="Split the polarizations with a PBS (vector GNLSE is not implemented).",
            )
        p = self.parameters
        betas = [p[f"beta{k}"] for k in range(2, 11)]
        field, rep = propagate_gnlse(
            sig.field[0],
            sig.grid,
            p["length"],
            sig.center_frequency,
            betas=betas,
            gamma=p["gamma"],
            alpha=p["attenuation"],
            self_steepening=p["self_steepening"],
            raman_fraction=p["raman_fraction"] if p["raman"] else 0.0,
            tau1=p["raman_tau1"],
            tau2=p["raman_tau2"],
            n_steps=p["n_steps"] if p["step_mode"] == "fixed" else None,
            tolerance=p["tolerance"],
            check_cancelled=context.check_cancelled,
        )
        context.record("n_steps", rep.n_steps)
        context.record("rejected_steps", rep.rejected_steps)
        context.record("photon_number_change", rep.photon_number_change)
        context.record("energy_change", rep.energy_change)
        edge = band_edge_energy_fraction(field[None, :], sig.grid)
        if edge > EDGE_THRESHOLD:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "gnlse.spectral_truncation",
                    f"{edge:.2e} of the output energy lies in the outer 10 % of the band "
                    "(spectral broadening reaches +-fs/2: aliasing).",
                    hint="Increase the sample rate.",
                    source=self.name,
                )
            )
        if time_edge_energy_fraction(field[None, :]) > EDGE_THRESHOLD:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "gnlse.window_wraparound",
                    "Energy reaches the edges of the time window (circular wrap-around).",
                    hint="Use a longer window (more samples).",
                    source=self.name,
                )
            )
        grid = sig.grid.with_t0(sig.grid.t0 + group_delay(p["length"], p["group_index"]))
        return {"out": sig.replace(field=field[None, :], grid=grid)}


def count_pulses(intensity: np.ndarray, threshold: float = 0.1) -> int:
    """Number of separate intensity lobes above ``threshold`` x peak (circular window)."""
    above = intensity > threshold * intensity.max()
    if above.all():
        return 1
    rolled = np.roll(above, -int(np.argmin(above)))
    return int(np.count_nonzero(rolled[1:] & ~rolled[:-1]) + (1 if rolled[0] else 0))


class ModeLockedFiberLaser(Component):
    """Passively mode-locked soliton fiber ring laser, iterated to steady state.

    One round trip: saturated gain (energy-saturated, Gaussian gain filter) ->
    anomalous-dispersion fiber (GNLSE, Kerr only) -> fast saturable absorber ->
    output coupler. Started from a weak Gaussian seed plus seeded noise, the
    round trip is repeated (solvers.cavity) until the circulating intensity
    reproduces itself; the output is the field extracted in the last round trip
    (one pulse per window = one repetition period of the model).
    """

    type_id = "optobuild.laser.mode_locked_fiber"
    version = "1.0.0"
    display_name = "Mode-locked fiber laser (soliton)"
    category = ComponentCategory.LASER
    stochastic = True
    output_ports = (PortSpec("out", OPT, "output pulse (one round trip)"),)
    parameter_specs = (
        ParameterSpec("n_samples", ParameterType.INT, default=1024, minimum=64),
        _f("sample_rate", 50e12, "Hz", display_unit="THz", minimum=0.0, minimum_inclusive=False),
        _f("wavelength", 1560e-9, "m", display_unit="nm", minimum=0.0, minimum_inclusive=False),
        _f("fiber_length", 5.0, "m", minimum=0.0, minimum_inclusive=False),
        _beta(2, -22e-27),
        _f("gamma", 1.3e-3, "1/(W m)", display_unit="1/(W km)", minimum=0.0),
        ParameterSpec("fiber_steps", ParameterType.INT, default=20, minimum=1),
        _f(
            "small_signal_gain",
            math.exp(2.0),
            "1",
            display_unit="dB",
            minimum=1.0,
            description="Unsaturated round-trip power gain exp(g0)",
        ),  # fmt: skip
        _f(
            "saturation_energy",
            60e-12,
            "J",
            display_unit="pJ",
            minimum=0.0,
            minimum_inclusive=False,
        ),  # fmt: skip
        _f(
            "filter_bandwidth",
            4.7e12,
            "Hz",
            display_unit="THz",
            minimum=0.0,
            minimum_inclusive=False,
            description="FWHM of the Gaussian gain/filter",
        ),  # fmt: skip
        _f("absorber_modulation_depth", 0.5, "1", minimum=0.0, maximum=1.0, symbol="q0"),
        _f("absorber_saturation_power", 30.0, "W", minimum=0.0, minimum_inclusive=False),
        _f(
            "output_coupling",
            0.3,
            "1",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
            maximum_inclusive=False,
        ),  # fmt: skip
        ParameterSpec("max_round_trips", ParameterType.INT, default=3000, minimum=1),
        _f("tolerance", 1e-7, "1", minimum=0.0, minimum_inclusive=False),
        _f("seed_peak_power", 0.09, "W", minimum=0.0),
        _f("seed_width", 1e-12, "s", display_unit="ps", minimum=0.0, minimum_inclusive=False),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        p = self.parameters
        n = p["n_samples"]
        dt = 1.0 / p["sample_rate"]
        grid = TimeGrid(n, dt, -(n // 2) * dt)
        f0 = SPEED_OF_LIGHT / p["wavelength"]
        # field filter exp(-w^2/W^2) has power FWHM (in Hz) = W sqrt(ln2 / 2) / pi
        half_width = math.pi * p["filter_bandwidth"] / math.sqrt(math.log(2) / 2)
        g0 = math.log(p["small_signal_gain"])

        def fiber(a: np.ndarray) -> np.ndarray:
            return propagate_gnlse(
                a, grid, p["fiber_length"], f0, betas=[p["beta2"]], gamma=p["gamma"],
                self_steepening=False, raman_fraction=0.0, n_steps=p["fiber_steps"],
            )[0]  # fmt: skip

        elements = [
            lambda a: gaussian_filter(
                saturated_gain(a, grid, g0, p["saturation_energy"]), grid, half_width
            ),
            fiber,
            lambda a: saturable_absorber(
                a, p["absorber_modulation_depth"], p["absorber_saturation_power"]
            ),
            lambda a: output_coupler(a, p["output_coupling"]),
        ]
        t = grid.time()
        rng = context.rng
        seed = math.sqrt(p["seed_peak_power"]) * np.exp(-(t**2) / (2 * p["seed_width"] ** 2))
        seed = seed + 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        res = run_cavity(
            elements,
            seed.astype(complex),
            grid,
            max_round_trips=p["max_round_trips"],
            tolerance=p["tolerance"],
            recenter=True,
            check_cancelled=context.check_cancelled,
            progress=context.report_progress,
        )
        out = res.output if res.output is not None else np.zeros(n, complex)
        intensity = np.abs(out) ** 2
        f, s = spectral_intensity(out, grid)
        t_fwhm = fwhm(t, intensity)
        f_fwhm = fwhm(f, s)
        pulses = count_pulses(intensity)
        context.record("round_trips", res.round_trips)
        context.record("converged", res.converged)
        context.record("residual", res.residual)
        context.record("pulse_fwhm_s", t_fwhm)
        context.record("spectral_fwhm_hz", f_fwhm)
        context.record("time_bandwidth_product", t_fwhm * f_fwhm)
        context.record("pulse_energy_j", float(np.sum(intensity) * dt))
        context.record("peak_power_w", float(intensity.max()))
        context.record("n_pulses", pulses)
        context.record("intracavity_energy_history_j", res.energies)
        if not res.converged:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "cavity.not_converged",
                    f"No steady state after {res.round_trips} round trips (residual "
                    f"{res.residual:.2e}); the output is the last round trip.",
                    hint="Increase max_round_trips, or the state may be unstable "
                    "(Q-switching, breathing, multi-pulsing).",
                    source=self.name,
                )
            )
        if pulses > 1:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "cavity.multiple_pulses",
                    f"{pulses} pulses circulate (multi-pulsing); pulse metrics refer to the "
                    "strongest.",
                    hint="Reduce the small-signal gain (pump) or increase the saturation energy.",
                    source=self.name,
                )
            )
        return {"out": OpticalSignal(grid, out[None, :], f0, {"source": self.name})}


class Autocorrelator(Component):
    """Intensity autocorrelator (background-free, second harmonic), with deconvolution."""

    type_id = "optobuild.analyzer.autocorrelator"
    version = "1.0.0"
    display_name = "Autocorrelator (intensity)"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("in", OPT, tap=True),)
    parameter_specs = (
        ParameterSpec(
            "pulse_shape",
            ParameterType.CHOICE,
            default="sech2",
            choices=tuple(AUTOCORRELATION_FACTORS),
            description="Assumed shape for the deconvolution factor",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        tau, ac = intensity_autocorrelation(sig.power(), sig.grid)
        width = fwhm(tau, ac)
        context.record("delay_s", tau)
        context.record("autocorrelation", ac)
        context.record("autocorrelation_fwhm_s", width)
        context.record(
            "deconvolved_fwhm_s", width / AUTOCORRELATION_FACTORS[self.parameters["pulse_shape"]]
        )
        if math.isnan(width):
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "autocorrelation.no_fwhm",
                    "The autocorrelation does not fall to half maximum inside the window.",
                    source=self.name,
                )
            )
        return {}


__all__ = ["Autocorrelator", "ModeLockedFiberLaser", "UltrafastFiber", "count_pulses"]
