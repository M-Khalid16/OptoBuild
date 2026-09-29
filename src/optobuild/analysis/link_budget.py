"""FSO link budget: mean received power, margin and outage (physics in physics.fso_channel).

    P_rx,mean [dBm] = P_tx [dBm] + 10 log10(E[h])
    margin [dB]     = P_rx,mean - S          (S: receiver sensitivity [dBm])
    P_out           = P(P_tx h < S) = P(h < S / P_tx)

The budget table splits E[h] into optics, atmospheric, geometric (static
pointing) and random-pointing contributions; turbulence has unit mean and
appears only in the outage probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from optobuild.core.units import dbm_to_watt, linear_to_db, watt_to_dbm
from optobuild.physics.fso_channel import FSOChannelModel


@dataclass(frozen=True)
class FSOBudget:
    """Link budget of a horizontal FSO link."""

    tx_power_dbm: float
    rows: tuple[tuple[str, float], ...]
    """(term, gain in dB) - negative values are losses."""
    received_mean_dbm: float
    sensitivity_dbm: float
    margin_db: float
    outage_probability: float
    rytov_variance: float
    scintillation_index: float

    def table(self) -> str:
        """Human-readable budget."""
        lines = [f"{'transmitted power':34s} {self.tx_power_dbm:9.2f} dBm"]
        lines += [f"{name:34s} {db:9.2f} dB" for name, db in self.rows]
        lines += [
            f"{'mean received power':34s} {self.received_mean_dbm:9.2f} dBm",
            f"{'receiver sensitivity':34s} {self.sensitivity_dbm:9.2f} dBm",
            f"{'link margin':34s} {self.margin_db:9.2f} dB",
            f"{'Rytov variance (plane wave)':34s} {self.rytov_variance:9.3g}",
            f"{'scintillation index':34s} {self.scintillation_index:9.3g}",
            f"{'outage probability':34s} {self.outage_probability:9.3g}",
        ]
        return "\n".join(lines)


def _db(x: float) -> float:
    return float(linear_to_db(x)) if x > 0 else -math.inf


def fso_link_budget(model: FSOChannelModel, tx_power: float, sensitivity: float) -> FSOBudget:
    """Budget for transmit power [W] and receiver sensitivity [W]."""
    geo = model.geometric_fraction()
    mean_p = model.mean_pointing_fraction()
    rows = (
        ("transmitter optics", _db(model.tx_efficiency)),
        ("receiver optics", _db(model.rx_efficiency)),
        ("atmospheric attenuation", _db(model.atmospheric_transmission)),
        ("geometric + static pointing", _db(geo)),
        ("random pointing / beam wander (mean)", _db(mean_p / geo) if geo > 0 else -math.inf),
    )
    mean_gain = model.mean_gain()
    rx = float(watt_to_dbm(tx_power)) + _db(mean_gain)
    s_dbm = float(watt_to_dbm(sensitivity))
    return FSOBudget(
        tx_power_dbm=float(watt_to_dbm(tx_power)),
        rows=rows,
        received_mean_dbm=rx,
        sensitivity_dbm=s_dbm,
        margin_db=rx - s_dbm,
        outage_probability=model.outage_probability(float(dbm_to_watt(s_dbm)) / tx_power),
        rytov_variance=model.rytov_variance,
        scintillation_index=model.scintillation_index,
    )


__all__ = ["FSOBudget", "fso_link_budget"]
