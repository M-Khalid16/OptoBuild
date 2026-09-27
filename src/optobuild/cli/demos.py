"""Built-in demonstration projects used by the CLI and examples."""

from __future__ import annotations

from optobuild.components.reference import Adder, Gain, GaussianNoise, RampSource, Recorder
from optobuild.graph.model import SimulationGraph
from optobuild.numerics.layout import SimulationLayout
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

    All sources follow one global simulation layout (ADR-0011):
    ``bit_rate``, ``n_bits = 2^prbs_order - 1`` and ``samples_per_bit``.
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

    layout = SimulationLayout(bit_rate, prbs_period(prbs_order), samples_per_bit)
    v_pi = 4.0
    g = SimulationGraph()
    g.add(PRBSGenerator("prbs", {"order": prbs_order, "timing_source": "layout"}))
    g.add(
        NRZGenerator(
            "nrz",
            {
                "timing_source": "layout",
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
                "timing_source": "layout",
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
    title = f"{bit_rate / 1e9:g} Gb/s NRZ-OOK reference link"
    return Project(graph=g, seed=seed, layout=layout, metadata={"title": title})


DEMOS["optical_link"] = optical_link_project


def soliton_project(
    seed: int = 0, *, soliton_order: float = 1.0, length_in_ld: float = 5.0
) -> Project:
    """Sech pulse in anomalous-dispersion fiber (Phase 4)::

        pulse (sech, T0 = 5 ps, P0 = N^2 |beta2| / (gamma T0^2)) -> nonlinear fiber (lossless)
        taps: input/output spectra and power meters

    N = 1 propagates unchanged (fundamental soliton); N = 2 breathes with
    period (pi/2) L_D. Standard SMF at 1550 nm: D = 17 ps/(nm km),
    gamma = 1.3 /(W km). The fiber also applies the third-order dispersion
    implied by D at 1550 nm (beta3 = 0.036 ps^3/km even with zero slope),
    which perturbs the soliton by about L / (T0^3/|beta3|) ~ 1e-3.
    """
    from optobuild.components.analyzers import OpticalPowerMeter, OpticalSpectrumAnalyzer
    from optobuild.components.nonlinear import NonlinearFiber, OpticalPulseSource
    from optobuild.core.units import dispersion_to_beta2

    t0, gamma, d = 5e-12, 1.3e-3, 17e-6
    beta2 = dispersion_to_beta2(d, 1550e-9)
    p0 = soliton_order**2 * abs(beta2) / (gamma * t0**2)
    ld = t0**2 / abs(beta2)
    g = SimulationGraph()
    g.add(
        OpticalPulseSource(
            "pulse",
            {
                "shape": "sech",
                "peak_power": p0,
                "width": t0,
                "n_samples": 4096,
                "sample_rate": 4096 / (80 * t0),
            },
        )
    )
    g.add(
        NonlinearFiber(
            "fiber",
            {
                "length": length_in_ld * ld,
                "attenuation": 0.0,
                "dispersion": d,
                "gamma": gamma,
                "max_phase": 1e-3,
                "max_step": ld / 20,
            },
        )
    )
    g.add(OpticalSpectrumAnalyzer("input_spectrum", {"resolution_bandwidth": 0.0}))
    g.add(OpticalSpectrumAnalyzer("output_spectrum", {"resolution_bandwidth": 0.0}))
    g.add(OpticalPowerMeter("input_power"))
    g.add(OpticalPowerMeter("output_power"))
    g.connect("pulse", "out", "fiber", "in")
    g.connect("pulse", "out", "input_spectrum", "in")
    g.connect("pulse", "out", "input_power", "in")
    g.connect("fiber", "out", "output_spectrum", "in")
    g.connect("fiber", "out", "output_power", "in")
    title = f"Order-{soliton_order:g} soliton over {length_in_ld:g} dispersion lengths"
    return Project(graph=g, seed=seed, metadata={"title": title})


DEMOS["soliton"] = soliton_project


def fso_link_project(
    seed: int = 7,
    *,
    distance_m: float = 1.5e3,
    cn2: float = 5e-14,
    visibility_m: float = 4e3,
    prbs_order: int = 11,
) -> Project:
    """10 Gb/s NRZ-OOK over a free-space link (Phase 5): the reference link with the
    fiber replaced by an FSO channel (Gamma-Gamma turbulence, haze, pointing jitter).

    Each Monte Carlo trial draws one quasi-static channel state
    (``optobuild ber demo:fso_link --trials N``).
    """
    from optobuild.components.fso import FSOChannel

    p = optical_link_project(seed=seed, prbs_order=prbs_order, laser_power_w=10e-3)
    g = p.graph
    g.remove("fiber")
    g.add(
        FSOChannel(
            "fso",
            {
                "distance": distance_m,
                "beam_waist": 0.01,
                "divergence": 0.5e-3,
                "aperture_diameter": 0.1,
                "tx_efficiency": 10**-0.1,
                "rx_efficiency": 10**-0.1,
                "visibility": visibility_m,
                "pointing_jitter": 50e-6,
                "turbulence": "gamma_gamma",
                "cn2": cn2,
                "beam_wander": True,
            },
        )
    )
    g.connect("mzm", "optical_out", "fso", "in")
    g.connect("fso", "out", "pin", "in")
    g.connect("fso", "out", "rx_power", "in")
    p.metadata["title"] = "10 Gb/s NRZ-OOK free-space optical link"
    return p


DEMOS["fso_link"] = fso_link_project

__all__ = [
    "DEMOS",
    "fso_link_project",
    "optical_link_project",
    "reference_project",
    "soliton_project",
]
