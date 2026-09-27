"""Atmospheric attenuation for free-space optics (physics_models.md 3.13).

Visibility (fog, haze, clear air) - Kim model:

    beta(lambda) = (3.912 / V) (lambda / 550 nm)^(-q)          [1/m, V in m]
    q = 1.6 (V > 50 km), 1.3 (6 < V <= 50 km), 0.16 V_km + 0.34 (1 < V <= 6 km),
        V_km - 0.5 (0.5 < V <= 1 km), 0 (V <= 0.5 km)
    Kruse model: q = 1.6, 1.3, 0.585 V_km^(1/3) (V <= 6 km)

3.912 = -ln(0.02): visibility V is the distance at which the transmission of
550 nm light drops to 2 % (Koschmieder). Power attenuation coefficient
alpha = beta [1/m]; in dB/km: 10 log10(e) * 1000 * beta.
References: I. I. Kim, B. McArthur, E. Korevaar, Proc. SPIE 4214, 26 (2001);
P. W. Kruse et al., Elements of Infrared Technology (1962).

Rain (independent of wavelength in the near IR):

    alpha_rain [dB/km] = 1.076 R^0.67,    R [mm/h]

Reference: Kaushal & Kaddoum, IEEE Commun. Surv. Tutor. 19, 57 (2017), sec. IV
(after Carbonneau et al.). Snow and scattering by specific aerosols are not
modelled.

Validity: empirical fits for 0.5-2 um and visibilities from ~50 m to > 50 km;
uncertainties of several dB/km are typical in dense fog (the Kim model's
q = 0 in dense fog makes attenuation wavelength independent).
"""

from __future__ import annotations

import math

LN_2_PERCENT = -math.log(0.02)  # 3.912...
_DB_PER_NEPER = 10.0 / math.log(10.0)


def visibility_exponent(visibility: float, model: str = "kim") -> float:
    """Size-distribution exponent q for visibility ``visibility`` [m]."""
    v = visibility / 1e3
    if visibility <= 0:
        raise ValueError(f"visibility must be > 0 m, got {visibility}.")
    if v > 50:
        return 1.6
    if v > 6:
        return 1.3
    if model == "kruse":
        return 0.585 * v ** (1.0 / 3.0)
    if model != "kim":
        raise ValueError(f"Unknown attenuation model {model!r}; use 'kim' or 'kruse'.")
    if v > 1:
        return 0.16 * v + 0.34
    if v > 0.5:
        return v - 0.5
    return 0.0


def visibility_attenuation(visibility: float, wavelength: float, model: str = "kim") -> float:
    """Power attenuation coefficient beta [1/m] from visibility [m] at wavelength [m]."""
    q = visibility_exponent(visibility, model)
    return LN_2_PERCENT / visibility * (wavelength / 550e-9) ** (-q)


def rain_attenuation(rain_rate: float) -> float:
    """Power attenuation coefficient [1/m] for rain rate [m/s] (1.076 R^0.67 dB/km, R in mm/h)."""
    if rain_rate < 0:
        raise ValueError(f"rain_rate must be >= 0, got {rain_rate}.")
    r_mm_h = rain_rate * 3.6e6
    return 1.076 * r_mm_h**0.67 / _DB_PER_NEPER / 1e3


__all__ = [
    "LN_2_PERCENT",
    "rain_attenuation",
    "visibility_attenuation",
    "visibility_exponent",
]
