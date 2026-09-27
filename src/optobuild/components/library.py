"""The list of built-in component classes (consumed by ``builtin_registry``)."""

from __future__ import annotations

from optobuild.components.analyzers import (
    BERAnalyzer,
    EyeDiagramAnalyzer,
    OpticalPowerMeter,
    OpticalSpectrumAnalyzer,
)
from optobuild.components.base import Component
from optobuild.components.detectors import PINPhotodiode
from optobuild.components.electrical import DecisionCircuit, LowPassFilter
from optobuild.components.fiber import LinearFiber
from optobuild.components.fso import FSOChannel
from optobuild.components.modulators import MachZehnderModulator, NRZGenerator
from optobuild.components.nonlinear import NonlinearFiber, OpticalPulseSource
from optobuild.components.reference import REFERENCE_COMPONENTS
from optobuild.components.sources import CWLaser, PRBSGenerator

OPTICAL_LINK_COMPONENTS: tuple[type[Component], ...] = (
    PRBSGenerator,
    NRZGenerator,
    CWLaser,
    MachZehnderModulator,
    LinearFiber,
    PINPhotodiode,
    LowPassFilter,
    DecisionCircuit,
    BERAnalyzer,
    OpticalPowerMeter,
    OpticalSpectrumAnalyzer,
    EyeDiagramAnalyzer,
)

NONLINEAR_COMPONENTS: tuple[type[Component], ...] = (OpticalPulseSource, NonlinearFiber)
FSO_COMPONENTS: tuple[type[Component], ...] = (FSOChannel,)

BUILTIN_COMPONENTS: tuple[type[Component], ...] = (
    *REFERENCE_COMPONENTS,
    *OPTICAL_LINK_COMPONENTS,
    *NONLINEAR_COMPONENTS,
    *FSO_COMPONENTS,
)

__all__ = [
    "BUILTIN_COMPONENTS",
    "FSO_COMPONENTS",
    "NONLINEAR_COMPONENTS",
    "OPTICAL_LINK_COMPONENTS",
]
