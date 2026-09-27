"""Physical constants in SI units.

Only constants that are *exact* by definition of the 2019 SI are provided here,
so no uncertainty bookkeeping is required. Values are identical to CODATA 2018
(E. Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021)).

Material- or model-specific quantities (fiber refractive index, responsivity,
...) are *parameters*, never module-level constants.
"""

from __future__ import annotations

import math
from typing import Final

SPEED_OF_LIGHT: Final[float] = 299_792_458.0
"""Speed of light in vacuum, c [m/s] (exact)."""

PLANCK_CONSTANT: Final[float] = 6.626_070_15e-34
"""Planck constant, h [J s] (exact)."""

REDUCED_PLANCK_CONSTANT: Final[float] = PLANCK_CONSTANT / (2.0 * math.pi)
"""Reduced Planck constant, hbar = h / (2 pi) [J s]."""

ELEMENTARY_CHARGE: Final[float] = 1.602_176_634e-19
"""Elementary charge, q [C] (exact)."""

BOLTZMANN_CONSTANT: Final[float] = 1.380_649e-23
"""Boltzmann constant, k_B [J/K] (exact)."""

__all__ = [
    "BOLTZMANN_CONSTANT",
    "ELEMENTARY_CHARGE",
    "PLANCK_CONSTANT",
    "REDUCED_PLANCK_CONSTANT",
    "SPEED_OF_LIGHT",
]
