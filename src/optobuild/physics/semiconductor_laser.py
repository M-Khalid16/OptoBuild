"""Single-mode semiconductor laser rate equations (physics_models.md 3.25).

Carrier density N [1/m^3], photon density S [1/m^3], optical phase phi [rad]:

    dN/dt   = I/(q V) - N/tau_n - g(N, S) S
    dS/dt   = Gamma g(N, S) S - S/tau_p + Gamma beta N/tau_n
    dphi/dt = (alpha/2) (Gamma v_g a (N - N_tr) - 1/tau_p)
    g(N, S) = v_g a (N - N_tr) / (1 + eps S)

Symbols: I [A] injection current; q elementary charge; V [m^3] active
volume; tau_n [s] carrier lifetime (linear recombination); tau_p [s] photon
lifetime; Gamma confinement factor; v_g = c/n_g group velocity; a [m^2]
differential gain; N_tr [1/m^3] transparency density; eps [m^3] gain
compression; beta spontaneous-emission coupling; alpha linewidth-enhancement
factor. Output power through the facet P_out = eta h nu n_p / tau_p with
n_p = S V / Gamma the photon number in the mode and eta the fraction of the
cavity loss that is useful output. phi is the phase of the field relative to
the threshold (cold-cavity) frequency; its sign follows our envelope
convention E = Re{A e^{+i w t}}: dphi/dt is the instantaneous frequency
deviation times 2 pi.

Closed forms (eps = beta = 0; Agrawal, *Fiber-Optic Communication Systems*,
5th ed., sec. 3.5; Coldren, Corzine & Mashanovitch, *Diode Lasers and
Photonic Integrated Circuits*, 2nd ed., ch. 2 and 5):

* threshold N_th = N_tr + 1/(Gamma v_g a tau_p), I_th = q V N_th / tau_n;
* above threshold N = N_th, S = Gamma tau_p (I - I_th)/(q V);
  slope dP_out/dI = eta h nu / q;
* turn-on delay from N = 0 (step to I > I_th): t_d = tau_n ln(I / (I - I_th));
* small signal about (N_th, S0): s^2 + gamma_R s + v_g a S0 / tau_p = 0,
  gamma_R = 1/tau_n + v_g a S0, i.e. relaxation (angular) frequency
  Omega_R = sqrt(v_g a S0/tau_p - gamma_R^2/4);
* chirp identity (exact for the equations above with eps = 0):
  dphi/dt = (alpha/2) (dS/dt / S - Gamma beta N / (tau_n S)).

Langevin noise (photon number n_p, carrier number n_c = N V; Henry, IEEE J.
Quantum Electron. 18, 259 (1982); Agrawal & Dutta, *Semiconductor Lasers*,
2nd ed., sec. 6.2), R_sp = beta n_c / tau_n spontaneous photons per second
into the mode; white Gaussian forces with

    <F_p F_p> = 2 R_sp n_p,  <F_c F_c> = 2 (R_sp n_p + n_c / tau_n),
    <F_p F_c> = -2 R_sp n_p,  <F_phi F_phi> = R_sp / (2 n_p)

(delta-correlated). Long-time phase diffusion gives Henry's linewidth
Delta nu = (1 + alpha^2) R_sp / (4 pi n_p) (for eps = 0).

Assumptions: single longitudinal mode, spatially uniform densities, linear
gain in N, linear recombination (no B N^2 / C N^3 terms), no thermal effects,
no parasitics (the current is the junction current), eps small.
Validation: tests/validation/test_semiconductor_laser.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from optobuild.core.constants import ELEMENTARY_CHARGE, PLANCK_CONSTANT, SPEED_OF_LIGHT

Q = ELEMENTARY_CHARGE


@dataclass(frozen=True)
class LaserParameters:
    """Rate-equation parameters (SI). Defaults: an InGaAsP 1.55 um DFB-like laser."""

    volume: float = 1.5e-16
    confinement: float = 0.3
    group_index: float = 4.0
    differential_gain: float = 2.5e-20
    transparency_density: float = 1.0e24
    carrier_lifetime: float = 1.0e-9
    photon_lifetime: float = 3.0e-12
    gain_compression: float = 1.5e-23
    spontaneous_coupling: float = 1.0e-5
    linewidth_enhancement: float = 5.0
    output_efficiency: float = 0.25
    wavelength: float = 1550e-9

    def __post_init__(self) -> None:
        for name in (
            "volume",
            "confinement",
            "group_index",
            "differential_gain",
            "carrier_lifetime",
            "photon_lifetime",
            "wavelength",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        if not 0 < self.confinement <= 1 or not 0 <= self.output_efficiency <= 1:
            raise ValueError("confinement in (0, 1] and output_efficiency in [0, 1] required")
        for name in ("gain_compression", "spontaneous_coupling", "transparency_density"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    # -- derived quantities ---------------------------------------------------------------
    @property
    def gain_coefficient(self) -> float:
        """v_g a [m^3/s]."""
        return SPEED_OF_LIGHT / self.group_index * self.differential_gain

    @property
    def photon_energy(self) -> float:
        return PLANCK_CONSTANT * SPEED_OF_LIGHT / self.wavelength

    @property
    def threshold_density(self) -> float:
        """N_th = N_tr + 1/(Gamma v_g a tau_p)."""
        return self.transparency_density + 1.0 / (
            self.confinement * self.gain_coefficient * self.photon_lifetime
        )

    @property
    def threshold_current(self) -> float:
        """I_th = q V N_th / tau_n (eps- and beta-independent)."""
        return Q * self.volume * self.threshold_density / self.carrier_lifetime

    @property
    def slope_efficiency(self) -> float:
        """dP_out/dI above threshold = eta h nu / q [W/A] (beta = 0)."""
        return self.output_efficiency * self.photon_energy / Q

    def output_power(self, photon_density: Any) -> Any:
        """P_out = eta h nu (S V / Gamma) / tau_p [W]."""
        return (
            self.output_efficiency
            * self.photon_energy
            * photon_density
            * self.volume
            / (self.confinement * self.photon_lifetime)
        )

    def photon_density_for_power(self, power: float) -> float:
        return power / self.output_power(1.0)

    # -- equations --------------------------------------------------------------------------
    def rhs(self, current: Any, n: Any, s: Any) -> tuple[Any, Any, Any]:
        """(dN/dt, dS/dt, dphi/dt); arithmetic only (floats or arrays)."""
        g_lin = self.gain_coefficient * (n - self.transparency_density)
        g = g_lin / (1.0 + self.gain_compression * s)
        spont = self.confinement * self.spontaneous_coupling * n / self.carrier_lifetime
        dn = current / (Q * self.volume) - n / self.carrier_lifetime - g * s
        ds = self.confinement * g * s - s / self.photon_lifetime + spont
        dphi = (
            0.5
            * self.linewidth_enhancement
            * (self.confinement * g_lin - 1.0 / self.photon_lifetime)
        )
        return dn, ds, dphi

    def _photons_at(self, n: float) -> float:
        """S >= 0 solving dS/dt = 0 at carrier density n (quadratic through eps S)."""
        p = self
        x = p.gain_coefficient * (n - p.transparency_density)
        r = p.confinement * p.spontaneous_coupling * n / p.carrier_lifetime
        # (1 + eps S) S / tau_p - Gamma x S = r (1 + eps S)
        a2 = p.gain_compression / p.photon_lifetime
        a1 = 1.0 / p.photon_lifetime - p.confinement * x - r * p.gain_compression
        if a2 == 0.0:
            return r / a1 if a1 > 0 else math.inf
        root = math.sqrt(a1 * a1 + 4 * a2 * r)
        # cancellation-free form of (-a1 + root) / (2 a2) for a1 > 0 (below threshold)
        return 2 * r / (a1 + root) if a1 > 0 else (-a1 + root) / (2 * a2)

    def _carrier_balance(self, current: float, n: float) -> float:
        """dN/dt at density n with S on the dS/dt = 0 curve (decreasing in n)."""
        s = self._photons_at(n)
        if not math.isfinite(s):
            return -math.inf
        g = self.gain_coefficient * (n - self.transparency_density)
        g /= 1 + self.gain_compression * s
        return current / (Q * self.volume) - n / self.carrier_lifetime - g * s

    def steady_state(self, current: float) -> tuple[float, float]:
        """(N, S) with dN/dt = dS/dt = 0 for a constant current (unique non-negative root).

        Closed form for beta = eps = 0; otherwise Brent's method on the carrier balance
        along the photon steady-state curve, bracketed in [0, n_hi].
        """
        p = self
        if current < 0:
            raise ValueError("current must be >= 0")
        if current == 0.0:
            return 0.0, 0.0
        if p.spontaneous_coupling == 0.0:
            if current <= p.threshold_current:
                return current * p.carrier_lifetime / (Q * p.volume), 0.0
            if p.gain_compression == 0.0:
                s0 = p.confinement * p.photon_lifetime * (current - p.threshold_current)
                return p.threshold_density, s0 / (Q * p.volume)
        if p.gain_compression == 0.0:
            hi = p.threshold_density * (1 - 1e-12)  # S -> inf as N -> N_th (beta > 0)
        else:
            hi = p.threshold_density
            while self._carrier_balance(current, hi) > 0:
                hi *= 1.5
        n = brentq(lambda x: self._carrier_balance(current, x), 0.0, hi, xtol=1e-9, rtol=1e-15)
        return n, self._photons_at(n)

    # -- small signal (eps = beta = 0) --------------------------------------------------------
    def relaxation_oscillation(self, current: float) -> tuple[float, float]:
        """(Omega_R [rad/s], gamma_R [1/s]) of the linearized equations for eps = beta = 0."""
        s0 = (
            self.confinement
            * self.photon_lifetime
            * (current - self.threshold_current)
            / (Q * self.volume)
        )
        if s0 <= 0:
            raise ValueError("relaxation oscillations need I > I_th")
        gamma = 1.0 / self.carrier_lifetime + self.gain_coefficient * s0
        w2 = self.gain_coefficient * s0 / self.photon_lifetime - gamma**2 / 4
        return math.sqrt(max(w2, 0.0)), gamma

    def turn_on_delay(self, current: float) -> float:
        """t_d = tau_n ln(I / (I - I_th)) for a step from N = 0 (beta = 0 limit)."""
        if current <= self.threshold_current:
            return math.inf
        return self.carrier_lifetime * math.log(current / (current - self.threshold_current))

    # -- noise ----------------------------------------------------------------------------------
    def spontaneous_rate(self, n: Any) -> Any:
        """R_sp = beta n_c / tau_n [photons/s into the mode]."""
        return self.spontaneous_coupling * n * self.volume / self.carrier_lifetime

    def henry_linewidth(self, current: float) -> float:
        """Delta nu = (1 + alpha^2) R_sp / (4 pi n_p) [Hz] at the steady state of ``current``."""
        n, s = self.steady_state(current)
        n_p = s * self.volume / self.confinement
        return (1 + self.linewidth_enhancement**2) * self.spontaneous_rate(n) / (4 * math.pi * n_p)

    def noise_spectra(
        self, current: float, frequency: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Small-signal (RIN, FM) two-sided spectra at the steady state of ``current``.

        Linearized Langevin equations x' = J x + F, x = (dN, dS), F with the diffusion
        matrix of the module docstring (in densities); phase rate
        phi' = (alpha/2) Gamma v_g a dN + F_phi. With H = (i w I - J)^-1:
        RIN(f) = [H D H^H]_SS / S0^2 [1/Hz] and FM(f) = (c^2 [H D H^H]_NN + D_phiphi)
        / (2 pi)^2 [Hz^2/Hz], c = (alpha/2) Gamma v_g a. Lorentzian-equivalent
        linewidth 2 pi FM(0) (equals Henry's formula for eps = 0 up to O(R_sp/n_p)).
        J is the analytic Jacobian of ``rhs``.
        """
        n0, s0 = self.steady_state(current)
        if s0 <= 0:
            raise ValueError("noise spectra need a lasing steady state (S > 0)")
        p = self
        ga, e, gam = p.gain_coefficient, p.gain_compression, p.confinement
        x = ga * (n0 - p.transparency_density)
        g = x / (1 + e * s0)
        dg_dn = ga / (1 + e * s0)
        dg_ds = -x * e / (1 + e * s0) ** 2
        j = np.array(
            [
                [-1 / p.carrier_lifetime - dg_dn * s0, -(g + dg_ds * s0)],
                [
                    gam * dg_dn * s0 + gam * p.spontaneous_coupling / p.carrier_lifetime,
                    gam * (g + dg_ds * s0) - 1 / p.photon_lifetime,
                ],
            ]
        )
        v = p.volume
        n_p, n_c = s0 * v / gam, n0 * v
        r = p.spontaneous_rate(n0)
        d = np.array(
            [
                [2 * (r * n_p + n_c / p.carrier_lifetime) / v**2, -2 * r * n_p * gam / v**2],
                [-2 * r * n_p * gam / v**2, 2 * r * n_p * (gam / v) ** 2],
            ]
        )
        w = 2 * math.pi * np.atleast_1d(np.asarray(frequency, dtype=float))
        h = np.linalg.inv(1j * w[:, None, None] * np.eye(2)[None] - j[None])
        cov = h @ d[None] @ np.conj(np.swapaxes(h, 1, 2))
        c = 0.5 * p.linewidth_enhancement * gam * ga
        rin = np.real(cov[:, 1, 1]) / s0**2
        fm = (c**2 * np.real(cov[:, 0, 0]) + r / (2 * n_p)) / (2 * math.pi) ** 2
        return rin, fm


__all__ = ["LaserParameters"]
