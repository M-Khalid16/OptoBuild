"""Erbium-doped fiber amplifier and fiber ring laser, two-level model (physics_models.md 3.26).

Beams k (pump, signals) with powers P_k [W], frequencies nu_k, all propagating
forward (+z). Photon fluxes Q_k = P_k / (h nu_k) [1/s]. Upper-level fraction
(steady state, homogeneous broadening; Giles & Desurvire, J. Lightwave
Technol. 9, 271 (1991)):

    n2 = sum_k (sigma_a,k Gamma_k Q_k / A) / (1/tau + sum_k (sigma_a,k + sigma_e,k) Gamma_k Q_k / A)

    dQ_k/dz = n_t Gamma_k [(sigma_a,k + sigma_e,k) n2 - sigma_a,k] Q_k - l Q_k

Symbols: n_t [1/m^3] erbium density; A [m^2] doped (core) area; Gamma_k
overlap; sigma_a/e [m^2] absorption/emission cross sections at nu_k; tau [s]
metastable lifetime; l [1/m] background loss.

Saleh-Jopson solution (Saleh et al., IEEE Photon. Technol. Lett. 2, 714
(1990)) for l = 0 and forward beams: with alpha_k = Gamma_k sigma_a,k n_t and
Q_k^sat = A / (Gamma_k (sigma_a,k + sigma_e,k) tau),

    Q_k(L) = Q_k(0) exp(-alpha_k L + (Q_in - Q_out) / Q_k^sat),
    Q_in/out = sum_k Q_k(0 / L),

a scalar implicit equation for Q_out (monotone; solved by bisection).

ASE (not included in the population: valid while the ASE power is small
compared with the signals), co-polarized forward spectral density
S [W/Hz] per polarization at the signal frequency:

    dS/dz = n_t Gamma [(sigma_a + sigma_e) n2 - sigma_a] S - l S + n_t Gamma sigma_e n2 h nu

Noise figure (signal-spontaneous beat limited, both polarizations of ASE
reaching the detector are irrelevant for the co-polarized beat):
NF = 1/G + 2 S(L) / (G h nu). For uniform inversion S = n_sp h nu (G - 1),
n_sp = sigma_e n2 / (sigma_e n2 - sigma_a (1 - n2)) (Desurvire, *Erbium-Doped
Fiber Amplifiers*, Wiley 1994, ch. 2).

Fiber ring laser (unidirectional ring: EDF, output coupler T, passive
transmission eta, lasing at the signal wavelength): steady state
G(P_c) (1 - T) eta = 1 with P_c the intracavity power entering the EDF;
output P_out = T G P_c. With the Saleh-Jopson relation (l = 0) this has the
closed form: K = Q_s^sat (alpha_s L - ln((1 - T) eta)) = Q_in - Q_out,
Q_p(L) = Q_p(0) exp(-alpha_p L + K / Q_p^sat), and
Q_c = [K - Q_p(0) + Q_p(L)] / (1 - 1/((1 - T) eta)).

Assumptions: two-level (980 nm pumping: fast 4I11/2 decay), homogeneous
broadening, no excited-state absorption or ion pairs, uniform radial
overlap factors, CW (the ~10 ms lifetime averages any modulation above
~kHz), cross sections given at the operating wavelengths.
Validation: tests/validation/test_edfa.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scipy.optimize import brentq

from optobuild.core.constants import PLANCK_CONSTANT, SPEED_OF_LIGHT


@dataclass(frozen=True)
class Beam:
    """Optical beam in the doped fiber: wavelength [m], cross sections [m^2], overlap."""

    wavelength: float
    absorption_cross_section: float
    emission_cross_section: float
    overlap: float

    @property
    def photon_energy(self) -> float:
        return PLANCK_CONSTANT * SPEED_OF_LIGHT / self.wavelength


@dataclass(frozen=True)
class ErbiumFiber:
    """Doped-fiber parameters (SI)."""

    length: float = 10.0
    erbium_density: float = 5e24
    doped_area: float = math.pi * (1.4e-6) ** 2
    lifetime: float = 10e-3
    background_loss: float = 0.0

    def absorption(self, beam: Beam) -> float:
        """alpha_k = Gamma_k sigma_a,k n_t [1/m]."""
        return beam.overlap * beam.absorption_cross_section * self.erbium_density

    def saturation_flux(self, beam: Beam) -> float:
        """Q_k^sat = A / (Gamma_k (sigma_a + sigma_e) tau) [photons/s]."""
        s = beam.absorption_cross_section + beam.emission_cross_section
        return self.doped_area / (beam.overlap * s * self.lifetime)

    def upper_fraction(self, beams: tuple[Beam, ...], fluxes: tuple[Any, ...]) -> Any:
        """n2 for photon fluxes Q_k [1/s] (floats or arrays)."""
        num = 0.0
        den = 1.0 / self.lifetime
        for b, q in zip(beams, fluxes, strict=True):
            w = b.overlap * q / self.doped_area
            num = num + b.absorption_cross_section * w
            den = den + (b.absorption_cross_section + b.emission_cross_section) * w
        return num / den

    def gain_coefficient(self, beam: Beam, n2: Any) -> Any:
        """Net field-power gain per length [1/m] at upper fraction n2 (incl. background loss)."""
        s = beam.absorption_cross_section + beam.emission_cross_section
        return (
            self.erbium_density * beam.overlap * (s * n2 - beam.absorption_cross_section)
            - self.background_loss
        )

    def flux_derivatives(self, beams: tuple[Beam, ...], fluxes: tuple[Any, ...]) -> tuple:
        n2 = self.upper_fraction(beams, fluxes)
        return tuple(self.gain_coefficient(b, n2) * q for b, q in zip(beams, fluxes, strict=True))

    def ase_derivative(self, beam: Beam, n2: Any, density: Any) -> Any:
        """dS/dz of the co-polarized forward ASE density S [W/Hz] at ``beam``."""
        spont = self.erbium_density * beam.overlap * beam.emission_cross_section * n2
        return self.gain_coefficient(beam, n2) * density + spont * beam.photon_energy

    # -- Saleh-Jopson (background loss 0) ---------------------------------------------------
    def saleh_jopson(self, beams: tuple[Beam, ...], powers_in: tuple[float, ...]) -> tuple:
        """Output powers [W] from the implicit two-level solution (requires l = 0)."""
        if self.background_loss != 0.0:
            raise ValueError("the Saleh-Jopson solution assumes zero background loss")
        q_in = [p / b.photon_energy for b, p in zip(beams, powers_in, strict=True)]
        total_in = sum(q_in)
        el = self.length

        def outputs(q_out_total: float) -> list[float]:
            return [
                q
                * math.exp(
                    -self.absorption(b) * el + (total_in - q_out_total) / self.saturation_flux(b)
                )  # fmt: skip
                for b, q in zip(beams, q_in, strict=True)
            ]

        def residual(x: float) -> float:
            return sum(outputs(x)) - x

        # residual(0) > 0 and residual(Q_in) = sum Q_k exp(-alpha_k L) - Q_in < 0 (photons
        # are not created: Q_out <= Q_in); the residual is strictly decreasing in between
        x = brentq(residual, 0.0, total_in, xtol=1e-12 * total_in, rtol=1e-15)
        return tuple(q * b.photon_energy for q, b in zip(outputs(x), beams, strict=True))


def noise_figure(gain: float, ase_density: float, photon_energy: float) -> float:
    """NF = 1/G + 2 S / (G h nu) (linear)."""
    return 1.0 / gain + 2.0 * ase_density / (gain * photon_energy)


def spontaneous_emission_factor(beam: Beam, n2: float) -> float:
    """n_sp = sigma_e n2 / (sigma_e n2 - sigma_a (1 - n2)) (requires net gain)."""
    e = beam.emission_cross_section * n2
    return e / (e - beam.absorption_cross_section * (1 - n2))


def ring_laser_closed_form(
    fiber: ErbiumFiber,
    pump: Beam,
    signal: Beam,
    pump_power: float,
    output_coupling: float,
    passive_transmission: float,
) -> float:
    """Output power [W] of the unidirectional ring laser (l = 0), 0 below threshold."""
    rt = (1 - output_coupling) * passive_transmission
    # ln G_s = -alpha_s L + K / Q_s^sat = -ln(rt)
    k = fiber.saturation_flux(signal) * (fiber.absorption(signal) * fiber.length - math.log(rt))
    qp_in = pump_power / pump.photon_energy
    exponent = -fiber.absorption(pump) * fiber.length + k / fiber.saturation_flux(pump)
    qp_out = qp_in * math.exp(exponent)
    qc = (k - qp_in + qp_out) / (1 - 1 / rt)
    if qc <= 0:
        return 0.0
    return output_coupling * qc / rt * signal.photon_energy


def _ring_exponent(fiber: ErbiumFiber, pump: Beam, signal: Beam, rt: float) -> tuple[float, float]:
    k = fiber.saturation_flux(signal) * (fiber.absorption(signal) * fiber.length - math.log(rt))
    x = -fiber.absorption(pump) * fiber.length + k / fiber.saturation_flux(pump)
    return k, x


def ring_laser_threshold(
    fiber: ErbiumFiber,
    pump: Beam,
    signal: Beam,
    output_coupling: float,
    passive_transmission: float,
) -> float:
    """Threshold pump power [W]: K = Q_p (1 - exp(x)), x = -alpha_p L + K / Q_p^sat (l = 0)."""
    k, x = _ring_exponent(fiber, pump, signal, (1 - output_coupling) * passive_transmission)
    return k * pump.photon_energy / (1 - math.exp(x))


def ring_laser_slope_efficiency(
    fiber: ErbiumFiber,
    pump: Beam,
    signal: Beam,
    output_coupling: float,
    passive_transmission: float,
) -> float:
    """dP_out/dP_pump = T (lambda_p / lambda_s) (1 - exp(x)) / (1 - (1 - T) eta) (l = 0)."""
    rt = (1 - output_coupling) * passive_transmission
    _, x = _ring_exponent(fiber, pump, signal, rt)
    return output_coupling * pump.wavelength / signal.wavelength * (1 - math.exp(x)) / (1 - rt)


__all__ = [
    "Beam",
    "ErbiumFiber",
    "noise_figure",
    "ring_laser_closed_form",
    "ring_laser_slope_efficiency",
    "ring_laser_threshold",
    "spontaneous_emission_factor",
]
