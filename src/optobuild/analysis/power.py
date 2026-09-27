"""Optical power measurement.

    P_avg = (1/N) sum_n sum_p |A_p(t_n)|^2      [W]
    P_peak = max_n sum_p |A_p(t_n)|^2           [W]

An ideal, wavelength-flat, polarization-insensitive power meter that averages
over the whole simulation window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optobuild.core.units import watt_to_dbm
from optobuild.signals.optical import OpticalSignal


@dataclass(frozen=True)
class PowerMeasurement:
    """Average and peak optical power."""

    average_w: float
    peak_w: float

    @property
    def average_dbm(self) -> float:
        """Average power [dBm] (-inf for zero power)."""
        return float(watt_to_dbm(self.average_w)) if self.average_w > 0 else float("-inf")


def measure_power(signal: OpticalSignal) -> PowerMeasurement:
    """Window-averaged and peak power of an optical signal."""
    p = signal.power()
    return PowerMeasurement(float(np.mean(p)), float(np.max(p)))


__all__ = ["PowerMeasurement", "measure_power"]
