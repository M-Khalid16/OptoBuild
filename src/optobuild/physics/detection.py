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


__all__ = ["photocurrent", "pin_noise"]
