"""PIN photodiode (physics_models.md 3.6).

    I(t) = R P(t) + I_d + n_shot(t) + n_th(t)

* ``P(t) = sum_p |A_p(t)|^2`` [W]: square-law detection of the total power.
* ``R`` [A/W] responsivity, ``I_d`` [A] dark current.
* ``n_shot``: Gaussian, one-sided PSD ``2 q (R P(t) + I_d)`` evaluated with the
  *instantaneous* noiseless current (signal-dependent shot noise).
* ``n_th``: Gaussian, one-sided PSD ``4 k_B T / R_L``.
Both are white over the simulated band [0, fs/2]; the receiver bandwidth is
set by the electrical filter that follows.

Assumptions: flat responsivity over the signal band, unlimited detector
bandwidth (modelled by the following filter), no avalanche gain, no
saturation, Gaussian approximation for shot noise.
Validation: tests/validation/test_detection.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.noise import real_white_noise, shot_noise_psd, thermal_noise_psd


def photocurrent(
    power: ArrayLike, responsivity: float, dark_current: float = 0.0
) -> NDArray[np.float64]:
    """Noiseless photocurrent R P + I_d [A] for optical power P [W]."""
    return responsivity * np.asarray(power, dtype=float) + dark_current


def pin_noise(
    rng: np.random.Generator,
    mean_current: ArrayLike,
    sample_rate: float,
    *,
    shot: bool = True,
    thermal: bool = True,
    temperature: float = 300.0,
    load_resistance: float = 50.0,
) -> NDArray[np.float64]:
    """Realized shot + thermal noise current [A] for the given noiseless current samples."""
    i = np.asarray(mean_current, dtype=float)
    psd = np.zeros_like(i)
    if shot:
        psd = psd + shot_noise_psd(np.clip(i, 0.0, None))
    if thermal:
        psd = psd + thermal_noise_psd(temperature, load_resistance)
    if not (shot or thermal):
        return np.zeros_like(i)
    return real_white_noise(rng, psd, sample_rate, i.size)


def coherent_detection(
    signal_field: ArrayLike,
    lo_field: ArrayLike,
    responsivity: float,
    rng: np.random.Generator | None,
    sample_rate: float,
    *,
    shot: bool = True,
    thermal: bool = True,
    temperature: float = 300.0,
    load_resistance: float = 50.0,
    phase_error: float = 0.0,
    quadrature_gain: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """90-degree hybrid + two balanced photodiode pairs.

    Hybrid outputs (E_s + E_lo)/2, (E_s - E_lo)/2, (E_s + c E_lo)/2, (E_s - c E_lo)/2
    with c = i exp(i eps) (eps = ``phase_error`` [rad], deviation from 90 degrees);
    each photodiode sees P = sum_pol |.|^2 (square law, ``photocurrent``) and
    its own shot and thermal noise (``pin_noise``). The quadrature pair has
    responsivity g R (g = ``quadrature_gain``, amplitude imbalance).
    Balanced differences (z = E_s . conj(E_lo), dot = sum over polarizations):

        i_I = R Re(z),   i_Q = g R Im(z exp(-i eps)) = g R (cos(eps) Im z - sin(eps) Re z)

    Noise per quadrature: one-sided PSD 2 q R (P_s + P_lo)/2 + 2 * 4 k T / R_L
    (times g for the shot term of Q). ``rng=None`` gives noiseless currents.
    Assumptions: identical diodes within a pair (infinite CMRR), otherwise
    ideal couplers. Reference: Kikuchi, J. Lightwave Technol. 34, 157 (2016).
    """
    es = np.asarray(signal_field, dtype=complex)
    elo = np.asarray(lo_field, dtype=complex)
    if es.shape != elo.shape:
        raise ValueError(f"signal {es.shape} and LO {elo.shape} fields must have equal shapes.")
    currents = []
    cq = 1j * np.exp(1j * phase_error)
    for c, r in ((1.0, 1.0), (-1.0, 1.0), (cq, quadrature_gain), (-cq, quadrature_gain)):
        p = np.sum(np.abs(0.5 * (es + c * elo)) ** 2, axis=0)
        i = photocurrent(p, r * responsivity)
        if rng is not None:
            i = i + pin_noise(
                rng,
                i,
                sample_rate,
                shot=shot,
                thermal=thermal,
                temperature=temperature,
                load_resistance=load_resistance,
            )
        currents.append(i)
    return currents[0] - currents[1], currents[2] - currents[3]


def delay_samples(samples: ArrayLike, delay: float, grid: TimeGrid) -> NDArray[np.float64]:
    """Real signal delayed by ``delay`` [s] (band-limited, circular): F^-1{X exp(-i w delay)}.

    Models I/Q skew (a path-length difference after the photodiodes).
    """
    h = np.exp(-1j * grid.angular_frequency() * delay)
    return np.real(apply_transfer_function(np.asarray(samples, dtype=float), h, grid))


__all__ = ["coherent_detection", "delay_samples", "photocurrent", "pin_noise"]
