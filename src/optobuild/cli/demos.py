"""Built-in demonstration projects used by the CLI and examples."""

from __future__ import annotations

from optobuild.components.reference import Adder, Gain, GaussianNoise, RampSource, Recorder
from optobuild.graph.model import SimulationGraph
from optobuild.persistence.project import Project


def reference_project(seed: int = 1234) -> Project:
    """Framework reference pipeline (no optical physics).

    ::

        ramp (v_n = n) --> gain x2 --> adder.a
        dc   (v_n = 10) ------------> adder.b
        adder --> recorder "clean"                  (expected v_n = 2 n + 10 exactly)
        adder --> noise (sigma 0.1 V) --> recorder "noisy"
    """
    g = SimulationGraph()
    g.add(RampSource("ramp", {"n_samples": 16, "sample_rate": 1e9, "offset": 0.0, "step": 1.0}))
    g.add(RampSource("dc", {"n_samples": 16, "sample_rate": 1e9, "offset": 10.0, "step": 0.0}))
    g.add(Gain("gain", {"gain": 2.0}))
    g.add(Adder("adder"))
    g.add(GaussianNoise("noise", {"sigma": 0.1}))
    g.add(Recorder("clean"))
    g.add(Recorder("noisy"))
    g.connect("ramp", "out", "gain", "in")
    g.connect("gain", "out", "adder", "a")
    g.connect("dc", "out", "adder", "b")
    g.connect("adder", "out", "clean", "in")
    g.connect("adder", "out", "noise", "in")
    g.connect("noise", "out", "noisy", "in")
    return Project(graph=g, seed=seed, metadata={"title": "Framework reference pipeline"})


DEMOS = {"reference": reference_project}


def optical_link_project(
    seed: int = 2026,
    *,
    prbs_order: int = 11,
    bit_rate: float = 10e9,
    samples_per_bit: int = 16,
    laser_power_w: float = 1e-3,
    fiber_length_m: float = 50e3,
    filter_bandwidth_hz: float = 7.5e9,
) -> Project:
    """Reference IM/DD link (Phase 2)::

        PRBS -> NRZ -> MZM <- CW laser
        MZM -> fiber -> PIN -> low-pass filter -> decision -> BER analyzer (ref: PRBS)
        taps: power meters (TX, RX), optical spectrum (TX), eye diagram (after filter)

    All sources share one sampling grid derived from the arguments.
    """
    from optobuild.components.analyzers import (
        BERAnalyzer,
        EyeDiagramAnalyzer,
        OpticalPowerMeter,
        OpticalSpectrumAnalyzer,
    )
    from optobuild.components.detectors import PINPhotodiode
    from optobuild.components.electrical import DecisionCircuit, LowPassFilter
    from optobuild.components.fiber import LinearFiber
    from optobuild.components.modulators import MachZehnderModulator, NRZGenerator
    from optobuild.components.sources import CWLaser, PRBSGenerator
    from optobuild.physics.prbs import prbs_period

    n_bits = prbs_period(prbs_order)
    v_pi = 4.0
    g = SimulationGraph()
    g.add(PRBSGenerator("prbs", {"order": prbs_order, "bit_rate": bit_rate}))
    g.add(
        NRZGenerator(
            "nrz",
            {
                "samples_per_bit": samples_per_bit,
                "low": -v_pi / 2,
                "high": v_pi / 2,
                "rise_time": 0.3 / bit_rate,
            },
        )
    )
    g.add(
        CWLaser(
            "laser",
            {
                "power": laser_power_w,
                "wavelength": 1550e-9,
                "n_samples": n_bits * samples_per_bit,
                "sample_rate": bit_rate * samples_per_bit,
            },
        )
    )
    g.add(
        MachZehnderModulator(
            "mzm",
            {
                "v_pi": v_pi,
                "v_bias": -v_pi / 2,
                "insertion_loss": 10 ** (-0.5),
                "extinction_ratio": 1e3,
            },
        )
    )
    g.add(LinearFiber("fiber", {"length": fiber_length_m}))
    g.add(PINPhotodiode("pin", {"responsivity": 0.8, "dark_current": 10e-9}))
    g.add(LowPassFilter("filter", {"kind": "bessel", "order": 4, "bandwidth": filter_bandwidth_hz}))
    g.add(DecisionCircuit("decision"))
    g.add(BERAnalyzer("ber"))
    g.add(OpticalPowerMeter("tx_power"))
    g.add(OpticalPowerMeter("rx_power"))
    g.add(OpticalSpectrumAnalyzer("tx_spectrum", {"resolution_bandwidth": 12.5e9}))
    g.add(EyeDiagramAnalyzer("eye"))
    g.connect("prbs", "out", "nrz", "bits")
    g.connect("nrz", "out", "mzm", "drive")
    g.connect("laser", "out", "mzm", "optical_in")
    g.connect("mzm", "optical_out", "fiber", "in")
    g.connect("mzm", "optical_out", "tx_power", "in")
    g.connect("mzm", "optical_out", "tx_spectrum", "in")
    g.connect("fiber", "out", "pin", "in")
    g.connect("fiber", "out", "rx_power", "in")
    g.connect("pin", "out", "filter", "in")
    g.connect("filter", "out", "decision", "in")
    g.connect("filter", "out", "eye", "in")
    g.connect("decision", "bits", "ber", "received")
    g.connect("prbs", "out", "ber", "reference")
    return Project(graph=g, seed=seed, metadata={"title": "10 Gb/s NRZ-OOK reference link"})


DEMOS["optical_link"] = optical_link_project

__all__ = ["DEMOS", "optical_link_project", "reference_project"]
