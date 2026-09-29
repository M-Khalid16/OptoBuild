"""Free-space optical channel component (Phase 5).

Equations: optobuild.physics.fso_channel (and free_space, atmospheric,
turbulence); docs/physics_models.md sec. 3.12-3.15.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import InvalidParameterError
from optobuild.core.units import linear_to_db
from optobuild.physics.free_space import diffraction_limited_divergence
from optobuild.physics.fso_channel import TURBULENCE_MODELS, FSOChannelModel
from optobuild.signals import OpticalSignal, SignalKind

QUASI_STATIC_LIMIT = 1e-4
"""Window duration [s] above which the constant-channel assumption is flagged
(atmospheric fading coherence times are typically 1-10 ms)."""


def _f(**kw: Any) -> ParameterSpec:
    return ParameterSpec(type=ParameterType.FLOAT, **kw)


class FSOChannel(Component):
    """Horizontal free-space optical link: beam spreading, aperture, pointing,
    fog/haze/rain attenuation and turbulence fading.

    Quasi-static: one channel state per run (per Monte Carlo trial). With
    ``fading = "mean"`` the deterministic mean gain E[h] is applied instead.
    Only the power is scaled (intensity modulation); phase distortion by
    turbulence is not modelled. Propagation delay L / c is added to t0.
    """

    type_id = "optobuild.channel.fso"
    version = "1.0.0"
    display_name = "Free-space optical channel"
    category = ComponentCategory.CHANNEL
    stochastic = True
    input_ports = (PortSpec("in", SignalKind.OPTICAL),)
    output_ports = (PortSpec("out", SignalKind.OPTICAL),)
    parameter_specs = (
        _f(
            name="distance",
            default=1e3,
            unit="m",
            display_unit="km",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="L",
        ),
        _f(
            name="beam_waist",
            default=0.01,
            unit="m",
            display_unit="mm",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="w0",
            description="Transmitted 1/e^2 beam radius",
        ),
        _f(
            name="divergence",
            default=1e-3,
            unit="rad",
            display_unit="mrad",
            minimum=0.0,
            symbol="theta",
            description="Far-field 1/e^2 half angle (0 = diffraction limited)",
        ),
        _f(
            name="aperture_diameter",
            default=0.08,
            unit="m",
            display_unit="cm",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="D",
        ),
        _f(
            name="tx_efficiency",
            default=1.0,
            unit="1",
            display_unit="dB loss",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        ),
        _f(
            name="rx_efficiency",
            default=1.0,
            unit="1",
            display_unit="dB loss",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        ),
        _f(
            name="visibility",
            default=0.0,
            unit="m",
            display_unit="km",
            minimum=0.0,
            symbol="V",
            description="Meteorological visibility (0 = no fog/haze attenuation)",
        ),
        ParameterSpec(
            "attenuation_model", ParameterType.CHOICE, default="kim", choices=("kim", "kruse")
        ),
        _f(name="rain_rate", default=0.0, unit="m/s", display_unit="mm/h", minimum=0.0, symbol="R"),
        _f(
            name="pointing_offset",
            default=0.0,
            unit="rad",
            display_unit="urad",
            minimum=0.0,
            description="Static misalignment angle",
        ),
        _f(
            name="pointing_jitter",
            default=0.0,
            unit="rad",
            display_unit="urad",
            minimum=0.0,
            description="Random jitter, std per axis",
        ),
        ParameterSpec(
            "turbulence", ParameterType.CHOICE, default="none", choices=TURBULENCE_MODELS
        ),
        _f(
            name="cn2",
            default=0.0,
            unit="m^-2/3",
            minimum=0.0,
            symbol="Cn2",
            description="Refractive-index structure parameter (e.g. 1e-15 weak ... 1e-13 strong)",
        ),
        ParameterSpec("beam_wander", ParameterType.BOOL, default=False),
        ParameterSpec(
            "aperture_averaging",
            ParameterType.BOOL,
            default=True,
            description="Log-normal model only",
        ),
        ParameterSpec("fading", ParameterType.CHOICE, default="random", choices=("random", "mean")),
    )

    def validate(self) -> list[Diagnostic]:
        diags = super().validate()
        p = self.parameters
        if p["turbulence"] != "none" and p["cn2"] == 0.0:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "fso.cn2_missing",
                    f"turbulence='{p['turbulence']}' needs cn2 > 0.",
                    hint="Typical: 1e-15 (weak) to 1e-13 m^-2/3 (strong).",
                )
            )
        return diags

    def model(self, wavelength: float) -> FSOChannelModel:
        """Channel model for a carrier wavelength [m]."""
        p = self.parameters
        dl = diffraction_limited_divergence(wavelength, p["beam_waist"])
        if 0.0 < p["divergence"] < dl * (1 - 1e-9):
            raise InvalidParameterError(
                f"divergence {p['divergence']:.3g} rad is below the diffraction limit "
                f"lambda/(pi w0) = {dl:.3g} rad.",
                hint="Increase the divergence, or set 0 for a diffraction-limited beam.",
            )
        return FSOChannelModel(
            distance=p["distance"],
            wavelength=wavelength,
            beam_waist=p["beam_waist"],
            divergence=p["divergence"],
            aperture_diameter=p["aperture_diameter"],
            tx_efficiency=p["tx_efficiency"],
            rx_efficiency=p["rx_efficiency"],
            visibility=p["visibility"] if p["visibility"] > 0 else math.inf,
            attenuation_model=p["attenuation_model"],
            rain_rate=p["rain_rate"],
            pointing_offset=p["pointing_offset"],
            pointing_jitter=p["pointing_jitter"],
            turbulence=p["turbulence"],
            cn2=p["cn2"],
            beam_wander=p["beam_wander"],
            aperture_averaging=p["aperture_averaging"],
        )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        m = self.model(sig.wavelength)
        if self.parameters["fading"] == "random":
            state = m.sample(context.rng, 1)
            gain = float(state["gain"][0])
            context.record("pointing_fraction", float(state["pointing"][0]))
            context.record("turbulence_gain", float(state["turbulence"][0]))
        else:
            gain = m.mean_gain()
        mean = m.mean_gain()
        context.record("channel_gain", gain)
        context.record("channel_gain_db", float(linear_to_db(gain)) if gain > 0 else -math.inf)
        context.record("mean_gain_db", float(linear_to_db(mean)))
        context.record("atmospheric_loss_db", -float(linear_to_db(m.atmospheric_transmission)))
        context.record("geometric_loss_db", -float(linear_to_db(m.geometric_fraction())))
        context.record("beam_radius_rx_m", m.beam_radius_rx)
        context.record("rytov_variance", m.rytov_variance)
        context.record("scintillation_index", m.scintillation_index)
        if sig.grid.duration > QUASI_STATIC_LIMIT and m.sigma_axis + m.scintillation_index > 0:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "fso.quasi_static",
                    f"Window {sig.grid.duration:.3g} s exceeds {QUASI_STATIC_LIMIT:g} s; "
                    "the channel "
                    "is held constant over it, which may exceed the fading coherence time.",
                    hint="Use shorter windows and more Monte Carlo trials.",
                    source=self.name,
                )
            )
        if self.parameters["turbulence"] == "lognormal" and m.rytov_variance > 1.0:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "fso.lognormal_strong_turbulence",
                    f"Rytov variance {m.rytov_variance:.3g} > 1: the log-normal model is a "
                    "weak-turbulence model.",
                    hint="Use turbulence='gamma_gamma'.",
                    source=self.name,
                )
            )
        grid = sig.grid.with_t0(sig.grid.t0 + self.parameters["distance"] / SPEED_OF_LIGHT)
        return {"out": sig.replace(field=sig.field * math.sqrt(gain), grid=grid)}


__all__ = ["FSOChannel"]
