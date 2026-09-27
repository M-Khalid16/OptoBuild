"""Layer 2 - immutable signal representations.

OpticalSignal, ElectricalSignal and DigitalSequence (see docs/signal_model.md
and ADR-0001). Signals are frozen dataclasses with read-only arrays; time is
the last array axis.
"""

from optobuild.signals.base import SampledSignal, Signal, require_same_grid
from optobuild.signals.digital import DigitalSequence
from optobuild.signals.electrical import ElectricalQuantity, ElectricalSignal
from optobuild.signals.kinds import SignalKind
from optobuild.signals.optical import OpticalSignal

__all__ = [
    "DigitalSequence",
    "ElectricalQuantity",
    "ElectricalSignal",
    "OpticalSignal",
    "SampledSignal",
    "Signal",
    "SignalKind",
    "require_same_grid",
]
