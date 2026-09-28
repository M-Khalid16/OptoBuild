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


def coherent_link_project(
    seed: int = 11,
    *,
    modulation: str = "qpsk",
    symbol_rate: float = 32e9,
    prbs_order: int = 15,
    samples_per_symbol: int = 4,
    fiber_length_m: float = 80e3,
    osnr_db: float = 15.0,
    tx_linewidth: float = 100e3,
    lo_linewidth: float = 100e3,
    lo_offset: float = 1e9,
    rolloff: float = 0.1,
) -> Project:
    """Single-polarization coherent link (Phase 6)::

        PRBS -> symbol mapper -> RRC pulse shaper -> IQ modulator <- CW laser (linewidth)
        IQ modulator -> fiber -> amplifier (gain = span loss, output OSNR)
                     -> coherent receiver <- LO laser (offset)
        receiver -> DSP (CD compensation, matched filter, FOE, BPS) -> analyzer (ref: mapper)

    The DSP compensates the fiber's accumulated dispersion D L.
    """
    from optobuild.analysis.constellations import bits_per_symbol
    from optobuild.components.coherent import (
        ASENoiseLoader,
        CoherentAnalyzer,
        CoherentDSP,
        CoherentReceiver,
        IQModulator,
        PulseShaper,
        SymbolMapper,
    )
    from optobuild.components.fiber import LinearFiber
    from optobuild.components.sources import CWLaser, PRBSGenerator
    from optobuild.physics.prbs import prbs_period

    k = bits_per_symbol(modulation)
    n_bits = prbs_period(prbs_order)
    n_symbols = n_bits if n_bits % k else n_bits // k
    n_samples = n_symbols * samples_per_symbol
    fs = symbol_rate * samples_per_symbol
    df_grid = fs / n_samples
    lo_offset = round(lo_offset / df_grid) * df_grid  # periodic in the window (no leakage)
    d = 17e-6
    g = SimulationGraph()
    g.add(PRBSGenerator("prbs", {"order": prbs_order, "bit_rate": k * symbol_rate}))
    g.add(SymbolMapper("mapper", {"modulation": modulation}))
    g.add(
        PulseShaper(
            "shaper",
            {"samples_per_symbol": samples_per_symbol, "rolloff": rolloff, "amplitude": 1.0},
        )
    )
    g.add(
        CWLaser(
            "tx_laser",
            {"power": 10e-3, "n_samples": n_samples, "sample_rate": fs, "linewidth": tx_linewidth},
        )
    )
    g.add(IQModulator("iq_mod", {"v_pi": 4.0, "insertion_loss": 10**-0.5}))
    g.add(LinearFiber("fiber", {"length": fiber_length_m, "dispersion": d}))
    span_loss = 10 ** (0.2 * fiber_length_m / 1e3 / 10)  # 0.2 dB/km, compensated by the amplifier
    g.add(ASENoiseLoader("ase", {"osnr": 10 ** (osnr_db / 10), "gain": span_loss}))
    g.add(
        CWLaser(
            "lo",
            {
                "power": 10e-3,
                "n_samples": n_samples,
                "sample_rate": fs,
                "linewidth": lo_linewidth,
                "frequency_offset": lo_offset,
            },
        )
    )
    g.add(CoherentReceiver("receiver"))
    g.add(CoherentDSP("dsp", {"cd_compensation": d * fiber_length_m}))
    g.add(CoherentAnalyzer("analyzer"))
    for src, sp, dst, dp in (
        ("prbs", "out", "mapper", "bits"),
        ("mapper", "symbols", "shaper", "symbols"),
        ("shaper", "i", "iq_mod", "i"),
        ("shaper", "q", "iq_mod", "q"),
        ("tx_laser", "out", "iq_mod", "optical_in"),
        ("iq_mod", "optical_out", "fiber", "in"),
        ("fiber", "out", "ase", "in"),
        ("ase", "out", "receiver", "signal"),
        ("lo", "out", "receiver", "lo"),
        ("receiver", "i", "dsp", "i"),
        ("receiver", "q", "dsp", "q"),
        ("dsp", "symbols", "analyzer", "received"),
        ("mapper", "symbols", "analyzer", "reference"),
    ):
        g.connect(src, sp, dst, dp)
    title = f"{symbol_rate / 1e9:g} GBd {modulation.upper()} coherent link"
    return Project(graph=g, seed=seed, metadata={"title": title})


DEMOS["coherent_link"] = coherent_link_project

__all__ = [
    "DEMOS",
    "coherent_link_project",
    "fso_link_project",
    "optical_link_project",
    "reference_project",
    "soliton_project",
]


def dp_coherent_link_project(
    seed: int = 12,
    *,
    modulation: str = "qpsk",
    symbol_rate: float = 32e9,
    prbs_order: int = 15,
    n_bits: int = 0,
    samples_per_symbol: int = 4,
    fiber_length_m: float = 80e3,
    mean_dgd: float = 10e-12,
    osnr_db: float = 18.0,
    tx_linewidth: float = 100e3,
    lo_linewidth: float = 100e3,
    lo_offset: float = 1e9,
    rolloff: float = 0.1,
) -> Project:
    """Dual-polarization coherent link (Phase 6b)::

    PRBS -> symbol mapper (x, y) -> RRC shapers (x, y) -> IQ modulators (x, y)
    CW laser -> PBS (45 deg) -> IQ modulators;  IQ modulators -> PBC
    PBC -> fiber -> random PMD -> amplifier (gain = span loss, output OSNR) -> PBS
    LO laser (offset) -> PBS (45 deg);  PBS x/y -> coherent receivers (x, y)
    receivers -> dual-pol DSP (CD comp., matched filter, timing, 2x2 CMA/RDE,
    FOE, BPS) -> analyzer (reference: mapper)

    ``n_bits = 0`` uses one PRBS period. For 64-QAM use a PRBS23 segment (e.g.
    ``prbs_order=23, n_bits=6 * 2**15``): a PRBS15 recurrence spans too few
    symbols for blind equalization (docs/physics_models.md 3.21).
    """
    from optobuild.analysis.constellations import bits_per_symbol
    from optobuild.components.coherent import (
        ASENoiseLoader,
        CoherentAnalyzer,
        CoherentReceiver,
        DualPolCoherentDSP,
        IQModulator,
        PulseShaper,
        SymbolMapper,
    )
    from optobuild.components.fiber import LinearFiber
    from optobuild.components.polarization import (
        PolarizationBeamCombiner,
        PolarizationBeamSplitter,
        RandomPMD,
    )
    from optobuild.components.sources import CWLaser, PRBSGenerator
    from optobuild.physics.prbs import prbs_period

    k = bits_per_symbol(modulation)
    n_bits = n_bits or prbs_period(prbs_order)
    n_symbols = n_bits if n_bits % k else n_bits // k
    n_samples = n_symbols * samples_per_symbol
    fs = symbol_rate * samples_per_symbol
    df_grid = fs / n_samples
    lo_offset = round(lo_offset / df_grid) * df_grid  # periodic in the window (no leakage)
    d = 17e-6
    g = SimulationGraph()
    g.add(
        PRBSGenerator("prbs", {"order": prbs_order, "bit_rate": k * symbol_rate, "n_bits": n_bits})
    )
    g.add(SymbolMapper("mapper", {"modulation": modulation, "polarizations": 2}))
    laser = {"power": 10e-3, "n_samples": n_samples, "sample_rate": fs}
    g.add(CWLaser("tx_laser", {**laser, "linewidth": tx_linewidth}))
    g.add(PolarizationBeamSplitter("tx_pbs"))
    for pol in ("x", "y"):
        g.add(
            PulseShaper(
                f"shaper_{pol}",
                {
                    "samples_per_symbol": samples_per_symbol,
                    "rolloff": rolloff,
                    "amplitude": 1.0,
                    "polarization": pol,
                },
            )
        )
        g.add(IQModulator(f"iq_{pol}", {"v_pi": 4.0, "insertion_loss": 10**-0.5}))
        g.add(CoherentReceiver(f"receiver_{pol}"))
    g.add(PolarizationBeamCombiner("pbc"))
    g.add(LinearFiber("fiber", {"length": fiber_length_m, "dispersion": d}))
    g.add(RandomPMD("pmd", {"mean_dgd": mean_dgd}))
    span_loss = 10 ** (0.2 * fiber_length_m / 1e3 / 10)  # 0.2 dB/km, compensated by the amplifier
    g.add(ASENoiseLoader("ase", {"osnr": 10 ** (osnr_db / 10), "gain": span_loss}))
    g.add(PolarizationBeamSplitter("rx_pbs"))
    g.add(CWLaser("lo", {**laser, "linewidth": lo_linewidth, "frequency_offset": lo_offset}))
    g.add(PolarizationBeamSplitter("lo_pbs"))
    g.add(DualPolCoherentDSP("dsp", {"cd_compensation": d * fiber_length_m}))
    g.add(CoherentAnalyzer("analyzer"))
    links = [
        ("prbs", "out", "mapper", "bits"),
        ("tx_laser", "out", "tx_pbs", "in"),
        ("iq_x", "optical_out", "pbc", "x"),
        ("iq_y", "optical_out", "pbc", "y"),
        ("pbc", "out", "fiber", "in"),
        ("fiber", "out", "pmd", "in"),
        ("pmd", "out", "ase", "in"),
        ("ase", "out", "rx_pbs", "in"),
        ("lo", "out", "lo_pbs", "in"),
        ("dsp", "symbols", "analyzer", "received"),
        ("mapper", "symbols", "analyzer", "reference"),
    ]
    for pol in ("x", "y"):
        links += [
            ("mapper", "symbols", f"shaper_{pol}", "symbols"),
            (f"shaper_{pol}", "i", f"iq_{pol}", "i"),
            (f"shaper_{pol}", "q", f"iq_{pol}", "q"),
            ("tx_pbs", pol, f"iq_{pol}", "optical_in"),
            ("rx_pbs", pol, f"receiver_{pol}", "signal"),
            ("lo_pbs", pol, f"receiver_{pol}", "lo"),
            (f"receiver_{pol}", "i", "dsp", f"{pol}i"),
            (f"receiver_{pol}", "q", "dsp", f"{pol}q"),
        ]
    for src, sp, dst, dp in links:
        g.connect(src, sp, dst, dp)
    title = f"{symbol_rate / 1e9:g} GBd DP-{modulation.upper()} coherent link"
    return Project(graph=g, seed=seed, metadata={"title": title})


DEMOS["dp_coherent_link"] = dp_coherent_link_project


def ring_filter_project(
    seed: int = 7,
    *,
    bit_rate: float = 10e9,
    prbs_order: int = 9,
    samples_per_bit: int = 32,
    power_coupling: float = 0.1,
    detuning_hz: float = 0.0,
) -> Project:
    """Add-drop microring filtering a 10 Gb/s NRZ-OOK signal (Phase 7)::

        PRBS -> NRZ -> MZM <- CW laser (1550 nm) -> add-drop ring -> through, drop
        taps: power meters and optical spectra at the input, through and drop ports

    The ring circumference is an integer number of guided wavelengths at 1550 nm
    (resonance on the carrier); ``detuning_hz`` shifts the resonance by adding the
    round-trip phase 2 pi detuning n_g L / c.
    """
    import math

    from optobuild.components.analyzers import OpticalPowerMeter, OpticalSpectrumAnalyzer
    from optobuild.components.modulators import MachZehnderModulator, NRZGenerator
    from optobuild.components.photonic import AddDropRing
    from optobuild.components.sources import CWLaser, PRBSGenerator
    from optobuild.core.constants import SPEED_OF_LIGHT
    from optobuild.physics.prbs import prbs_period

    layout = SimulationLayout(bit_rate, prbs_period(prbs_order), samples_per_bit)
    lam0, n_eff, n_g = 1550e-9, 2.4, 4.2
    m = round(2 * math.pi * 10e-6 * n_eff / lam0)  # ~10 um radius, resonant at lam0
    radius = m * lam0 / (2 * math.pi * n_eff)
    tuning = 2 * math.pi * detuning_hz * n_g * 2 * math.pi * radius / SPEED_OF_LIGHT
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
        )  # fmt: skip
    )
    g.add(CWLaser("laser", {"power": 1e-3, "wavelength": lam0, "timing_source": "layout"}))
    g.add(MachZehnderModulator("mzm", {"v_pi": v_pi, "v_bias": -v_pi / 2}))
    g.add(
        AddDropRing(
            "ring",
            {
                "radius": radius,
                "power_coupling_in": power_coupling,
                "power_coupling_drop": power_coupling,
                "phase_shift": -tuning,
                "n_eff": n_eff,
                "n_group": n_g,
                "design_wavelength": lam0,
            },
        )
    )
    for port in ("in", "through", "drop"):
        g.add(OpticalPowerMeter(f"power_{port}"))
        g.add(OpticalSpectrumAnalyzer(f"spectrum_{port}", {"resolution_bandwidth": 1e9}))
    g.connect("prbs", "out", "nrz", "bits")
    g.connect("nrz", "out", "mzm", "drive")
    g.connect("laser", "out", "mzm", "optical_in")
    g.connect("mzm", "optical_out", "ring", "in")
    for port, (src, sp) in {
        "in": ("mzm", "optical_out"),
        "through": ("ring", "through"),
        "drop": ("ring", "drop"),
    }.items():
        g.connect(src, sp, f"power_{port}", "in")
        g.connect(src, sp, f"spectrum_{port}", "in")
    title = f"Add-drop ring filtering {bit_rate / 1e9:g} Gb/s NRZ-OOK"
    return Project(graph=g, seed=seed, layout=layout, metadata={"title": title})


DEMOS["ring_filter"] = ring_filter_project


def dml_link_project(
    seed: int = 8,
    *,
    bit_rate: float = 10e9,
    prbs_order: int = 9,
    samples_per_bit: int = 16,
    fiber_length_m: float = 10e3,
    bias_current: float = 0.045,
    drive_voltage: float = 1.5,
    pump_power: float = 0.02,
    laser_noise: bool = True,
) -> Project:
    """Directly modulated laser link with an EDFA preamplifier (Phase 8)::

        PRBS -> NRZ (+-drive/2 V into 50 ohm) -> directly modulated laser (rate equations)
        DML -> SMF (17 ps/(nm km)) -> EDFA (pump) -> PIN -> low-pass -> decision -> BER
        taps: power meters (TX, after EDFA), optical spectrum (TX), eye diagram

    The DML's transient and adiabatic chirp interacts with fiber dispersion
    (compare fiber_length_m = 0 with 20-40 km). The laser is a small-volume,
    high-differential-gain 10G-class DML (V = 3e-17 m^3, a = 5e-20 m^2: I_th = 6.2 mA,
    relaxation frequency ~15 GHz at the bias); other rate-equation parameters are the
    physics defaults.
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
    from optobuild.components.lasers import DirectlyModulatedLaser, ErbiumDopedFiberAmplifier
    from optobuild.components.modulators import NRZGenerator
    from optobuild.components.sources import PRBSGenerator
    from optobuild.physics.prbs import prbs_period

    layout = SimulationLayout(bit_rate, prbs_period(prbs_order), samples_per_bit)
    g = SimulationGraph()
    g.add(PRBSGenerator("prbs", {"order": prbs_order, "timing_source": "layout"}))
    g.add(
        NRZGenerator(
            "nrz",
            {
                "timing_source": "layout",
                "low": -drive_voltage / 2,
                "high": drive_voltage / 2,
                "rise_time": 0.3 / bit_rate,
            },
        )
    )
    g.add(
        DirectlyModulatedLaser(
            "dml",
            {
                "bias_current": bias_current,
                "noise": laser_noise,
                "volume": 3e-17,
                "differential_gain": 5e-20,
            },
        )
    )
    g.add(LinearFiber("fiber", {"length": fiber_length_m}))
    g.add(ErbiumDopedFiberAmplifier("edfa", {"pump_power": pump_power}))
    g.add(PINPhotodiode("pin", {"responsivity": 0.8}))
    g.add(LowPassFilter("filter", {"kind": "bessel", "order": 4, "bandwidth": 0.75 * bit_rate}))
    g.add(DecisionCircuit("decision"))
    g.add(BERAnalyzer("ber"))
    g.add(OpticalPowerMeter("tx_power"))
    g.add(OpticalPowerMeter("rx_power"))
    g.add(OpticalSpectrumAnalyzer("tx_spectrum", {"resolution_bandwidth": 1e9}))
    g.add(EyeDiagramAnalyzer("eye"))
    for src, sp, dst, dp in (
        ("prbs", "out", "nrz", "bits"),
        ("nrz", "out", "dml", "drive"),
        ("dml", "out", "fiber", "in"),
        ("dml", "out", "tx_power", "in"),
        ("dml", "out", "tx_spectrum", "in"),
        ("fiber", "out", "edfa", "in"),
        ("edfa", "out", "pin", "in"),
        ("edfa", "out", "rx_power", "in"),
        ("pin", "out", "filter", "in"),
        ("filter", "out", "decision", "in"),
        ("filter", "out", "eye", "in"),
        ("decision", "bits", "ber", "received"),
        ("prbs", "out", "ber", "reference"),
    ):
        g.connect(src, sp, dst, dp)
    title = f"{bit_rate / 1e9:g} Gb/s directly modulated laser link, {fiber_length_m / 1e3:g} km"
    return Project(graph=g, seed=seed, layout=layout, metadata={"title": title})


DEMOS["dml_link"] = dml_link_project
