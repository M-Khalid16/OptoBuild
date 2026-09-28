"""Coherent transmission components (Phase 6, single polarization).

Equations: analysis.constellations (mapping), numerics.pulse_shaping,
physics.modulation.iq_modulator_field, physics.noise (ASE/OSNR),
physics.detection.coherent_detection, physics.fiber.cd_compensation_transfer,
analysis.coherent_dsp; docs/physics_models.md sec. 3.16-3.19.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.analysis.ber import clopper_pearson
from optobuild.analysis.coherent_dsp import (
    SYMMETRY,
    align_to_reference,
    best_sampling_phase,
    blind_phase_search,
    estimate_frequency_offset,
    estimate_timing_offset,
    fractional_advance,
    normalize_power,
    remove_frequency_offset,
)
from optobuild.analysis.constellations import (
    FORMATS,
    bits_per_symbol,
    decide,
    demap_hard,
    map_bits,
)
from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import SamplingError, SignalTypeError
from optobuild.core.units import linear_to_db
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid
from optobuild.numerics.pulse_shaping import matched_filter, shape
from optobuild.physics.detection import coherent_detection
from optobuild.physics.fiber import cd_compensation_transfer
from optobuild.physics.modulation import iq_modulator_field
from optobuild.physics.noise import ase_psd_for_osnr, complex_white_noise
from optobuild.signals import (
    DigitalSequence,
    ElectricalQuantity,
    ElectricalSignal,
    OpticalSignal,
    SignalKind,
    SymbolSequence,
    require_same_grid,
)
from optobuild.signals import metadata as meta

_FMT = tuple(FORMATS)
E, OPT, S = SignalKind.ELECTRICAL, SignalKind.OPTICAL, SignalKind.SYMBOLS


class SymbolMapper(Component):
    """Gray-maps bits to unit-energy symbols (BPSK, QPSK, 16-QAM, 64-QAM).

    If the number of bits is not a multiple of log2(M) (e.g. a PRBS of odd
    length), the pattern is repeated log2(M) times, giving one symbol per
    original bit period count (n_symbols = n_bits) and a periodic symbol
    stream. Symbol rate = bit rate / log2(M).
    """

    type_id = "optobuild.dsp.symbol_mapper"
    version = "1.0.0"
    display_name = "Symbol mapper"
    category = ComponentCategory.DSP
    input_ports = (PortSpec("bits", SignalKind.DIGITAL),)
    output_ports = (PortSpec("symbols", S),)
    parameter_specs = (
        ParameterSpec("modulation", ParameterType.CHOICE, default="qpsk", choices=_FMT),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        seq: DigitalSequence = inputs["bits"]
        fmt = self.parameters["modulation"]
        k = bits_per_symbol(fmt)
        bits = seq.bits if seq.n_bits % k == 0 else np.tile(seq.bits, k)
        md = {meta.MODULATION: fmt}
        if meta.PATTERN in seq.metadata:
            md[meta.PATTERN] = seq.metadata[meta.PATTERN]
        return {"symbols": SymbolSequence(map_bits(bits, fmt), seq.bit_rate / k, md)}


class PulseShaper(Component):
    """Symbols -> in-phase and quadrature drive voltages with (root-)raised-cosine pulses.

    v_I + i v_Q = amplitude * sum_k s_k p(t - k T_s); symbol k centred at sample k*sps.
    """

    type_id = "optobuild.dsp.pulse_shaper"
    version = "1.0.0"
    display_name = "Pulse shaper (RRC/RC)"
    category = ComponentCategory.DSP
    input_ports = (PortSpec("symbols", S),)
    output_ports = (
        PortSpec("i", E, "in-phase drive [V]"),
        PortSpec("q", E, "quadrature drive [V]"),
    )
    parameter_specs = (
        ParameterSpec("samples_per_symbol", ParameterType.INT, default=4, minimum=2),
        ParameterSpec(
            "rolloff", ParameterType.FLOAT, default=0.1, minimum=0.0, maximum=1.0, symbol="beta"
        ),
        ParameterSpec("pulse", ParameterType.CHOICE, default="rrc", choices=("rrc", "rc")),
        ParameterSpec(
            "amplitude",
            ParameterType.FLOAT,
            default=0.5,
            unit="V",
            minimum=0.0,
            minimum_inclusive=False,
            description="Drive voltage per unit symbol amplitude",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sym: SymbolSequence = inputs["symbols"]
        p = self.parameters
        sps = p["samples_per_symbol"]
        if sym.symbols.ndim != 1:
            raise SignalTypeError("PulseShaper supports single-polarization symbols only.")
        grid = TimeGrid(sym.n_symbols * sps, 1.0 / (sym.symbol_rate * sps))
        y = p["amplitude"] * shape(sym.symbols, sps, grid, p["rolloff"], p["pulse"])
        md = dict(sym.metadata)
        md.update(
            {
                meta.SYMBOL_RATE: sym.symbol_rate,
                meta.SAMPLES_PER_SYMBOL: sps,
                meta.PULSE_ROLLOFF: p["rolloff"],
            }
        )
        return {
            "i": ElectricalSignal(grid, y.real, ElectricalQuantity.VOLTAGE, md),
            "q": ElectricalSignal(grid, y.imag, ElectricalQuantity.VOLTAGE, md),
        }


class IQModulator(Component):
    """Nested Mach-Zehnder (IQ) modulator: two child MZMs at null, 90-degree combiner."""

    type_id = "optobuild.modulator.iq"
    version = "1.0.0"
    display_name = "IQ modulator"
    category = ComponentCategory.MODULATOR
    input_ports = (PortSpec("optical_in", OPT), PortSpec("i", E), PortSpec("q", E))
    output_ports = (PortSpec("optical_out", OPT),)
    parameter_specs = (
        ParameterSpec(
            "v_pi",
            ParameterType.FLOAT,
            default=4.0,
            unit="V",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="V_pi",
        ),
        ParameterSpec(
            "insertion_loss",
            ParameterType.FLOAT,
            default=1.0,
            unit="1",
            display_unit="dB loss",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        ),
        ParameterSpec(
            "extinction_ratio",
            ParameterType.FLOAT,
            default=1e4,
            unit="1",
            display_unit="dB",
            minimum=1.0,
            description="Of each child MZM",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        opt: OpticalSignal = inputs["optical_in"]
        vi: ElectricalSignal = inputs["i"]
        vq: ElectricalSignal = inputs["q"]
        for v in (vi, vq):
            if v.quantity is not ElectricalQuantity.VOLTAGE:
                raise SignalTypeError(f"IQ modulator '{self.name}' needs voltage drives.")
            require_same_grid(opt, v, what=f"inputs of '{self.name}'")
        p = self.parameters
        h = iq_modulator_field(
            vi.samples, vq.samples, p["v_pi"], p["insertion_loss"], p["extinction_ratio"]
        )
        return {
            "optical_out": opt.replace(field=opt.field * h[None, :], metadata=dict(vi.metadata))
        }


class ASENoiseLoader(Component):
    """Ideal amplifier with a specified output OSNR.

    The field is amplified by sqrt(G) (``gain``, 1 = no gain), then white ASE is
    added so that OSNR = P_out / (N_ase,total B_ref) (both polarizations): the
    co-polarized half N = P_out / (2 OSNR B_ref) is added to each polarization
    present. P_out is the measured average output signal power. The OSNR is a
    specification here; deriving it from a noise figure belongs to an amplifier
    model (Phase 8).
    """

    type_id = "optobuild.amplifier.ase_noise_loader"
    version = "1.1.0"
    display_name = "ASE noise loader (OSNR)"
    category = ComponentCategory.AMPLIFIER
    stochastic = True
    input_ports = (PortSpec("in", OPT),)
    output_ports = (PortSpec("out", OPT),)
    parameter_specs = (
        ParameterSpec(
            "osnr",
            ParameterType.FLOAT,
            default=10**1.5,
            unit="1",
            display_unit="dB",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        ParameterSpec(
            "reference_bandwidth",
            ParameterType.FLOAT,
            default=12.5e9,
            unit="Hz",
            display_unit="GHz",
            minimum=0.0,
            minimum_inclusive=False,
            description="12.5 GHz = 0.1 nm at 1550 nm",
        ),
        ParameterSpec(
            "gain",
            ParameterType.FLOAT,
            default=1.0,
            unit="1",
            display_unit="dB",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="G",
            description="Power gain applied before the noise (1 = none)",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        p = self.parameters
        sig: OpticalSignal = inputs["in"]
        sig = sig.replace(field=math.sqrt(p["gain"]) * sig.field)
        psd = ase_psd_for_osnr(sig.average_power(), p["osnr"], p["reference_bandwidth"])
        noise = complex_white_noise(context.rng, psd, sig.grid.sample_rate, sig.field.shape)
        context.record("ase_psd_per_pol_w_per_hz", psd)
        context.record("osnr_db", float(linear_to_db(p["osnr"])))
        if p["reference_bandwidth"] > sig.grid.sample_rate:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "ase.reference_band_exceeds_simulation",
                    "The OSNR reference bandwidth exceeds the simulated band.",
                    source=self.name,
                )
            )
        return {"out": sig.replace(field=sig.field + noise)}


class CoherentReceiver(Component):
    """Intradyne receiver: 90-degree hybrid with LO and two balanced photodiode pairs.

    Outputs photocurrents i_I + i i_Q = R E_s conj(E_LO) + noise. The LO is
    expressed on the signal's reference frequency (a frequency offset between
    the two carriers appears as a rotation).
    """

    type_id = "optobuild.detector.coherent_receiver"
    version = "1.0.0"
    display_name = "Coherent receiver"
    category = ComponentCategory.DETECTOR
    stochastic = True
    input_ports = (PortSpec("signal", OPT), PortSpec("lo", OPT, "local oscillator"))
    output_ports = (
        PortSpec("i", E, "in-phase current [A]"),
        PortSpec("q", E, "quadrature current [A]"),
    )
    parameter_specs = (
        ParameterSpec("responsivity", ParameterType.FLOAT, default=0.8, unit="A/W", minimum=0.0),
        ParameterSpec("shot_noise", ParameterType.BOOL, default=True),
        ParameterSpec("thermal_noise", ParameterType.BOOL, default=True),
        ParameterSpec(
            "temperature",
            ParameterType.FLOAT,
            default=300.0,
            unit="K",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        ParameterSpec(
            "load_resistance",
            ParameterType.FLOAT,
            default=50.0,
            unit="ohm",
            minimum=0.0,
            minimum_inclusive=False,
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["signal"]
        lo: OpticalSignal = inputs["lo"]
        require_same_grid(sig, lo, what=f"inputs of '{self.name}'")
        if sig.n_pol != lo.n_pol:
            raise SamplingError("Signal and LO must have the same number of polarizations.")
        df = lo.center_frequency - sig.center_frequency
        if abs(df) >= sig.grid.nyquist_frequency:
            raise SamplingError(
                f"LO and signal carriers differ by {df:g} Hz, beyond +-fs/2.",
                hint="Tune the LO close to the signal or increase the sample rate.",
            )
        lo_field = lo.field * np.exp(2j * np.pi * df * sig.grid.time())[None, :]
        p = self.parameters
        noisy = p["shot_noise"] or p["thermal_noise"]
        i, q = coherent_detection(
            sig.field,
            lo_field,
            p["responsivity"],
            context.rng if noisy else None,
            sig.grid.sample_rate,
            shot=p["shot_noise"],
            thermal=p["thermal_noise"],
            temperature=p["temperature"],
            load_resistance=p["load_resistance"],
        )
        md = dict(sig.metadata)
        md[meta.CARRIER_WAVELENGTH] = sig.wavelength
        context.record("lo_power_w", lo.average_power())
        context.record("signal_power_w", sig.average_power())
        return {
            "i": ElectricalSignal(sig.grid, i, ElectricalQuantity.CURRENT, md),
            "q": ElectricalSignal(sig.grid, q, ElectricalQuantity.CURRENT, md),
        }


class CoherentDSP(Component):
    """Receiver DSP: CD compensation, RRC matched filter, timing, FOE, BPS, normalization."""

    type_id = "optobuild.dsp.coherent_dsp"
    version = "1.0.0"
    display_name = "Coherent DSP"
    category = ComponentCategory.DSP
    input_ports = (PortSpec("i", E), PortSpec("q", E))
    output_ports = (PortSpec("symbols", S),)
    parameter_specs = (
        ParameterSpec(
            "cd_compensation",
            ParameterType.FLOAT,
            default=0.0,
            unit="s/m",
            display_unit="ps/nm",
            symbol="D L",
            description="Accumulated dispersion to compensate (0 = none)",
        ),
        ParameterSpec(
            "matched_filter",
            ParameterType.BOOL,
            default=True,
            description="RRC matched filter with the transmitter's roll-off",
        ),
        ParameterSpec(
            "timing",
            ParameterType.CHOICE,
            default="oerder_meyr",
            choices=("oerder_meyr", "integer"),
            description="Fractional (Oerder-Meyr) or integer-sample timing",
        ),
        ParameterSpec("frequency_offset_estimation", ParameterType.BOOL, default=True),
        ParameterSpec(
            "phase_recovery", ParameterType.CHOICE, default="bps", choices=("bps", "none")
        ),
        ParameterSpec("bps_test_phases", ParameterType.INT, default=32, minimum=4),
        ParameterSpec("bps_half_window", ParameterType.INT, default=16, minimum=0),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        si: ElectricalSignal = inputs["i"]
        sq: ElectricalSignal = inputs["q"]
        require_same_grid(si, sq, what=f"inputs of '{self.name}'")
        md = dict(si.metadata)
        rs, sps = meta.require_symbol_timing(md, self.name)
        fmt = md.get(meta.MODULATION)
        if fmt not in FORMATS:
            raise SignalTypeError(
                f"'{self.name}' needs the '{meta.MODULATION}' metadata.",
                hint="Generate symbols with the SymbolMapper.",
            )
        p = self.parameters
        grid = si.grid
        y = si.samples + 1j * sq.samples
        if p["cd_compensation"] != 0.0:
            lam = md.get(meta.CARRIER_WAVELENGTH)
            if lam is None:
                raise SignalTypeError(
                    "CD compensation needs the carrier wavelength metadata.",
                    hint="Feed the DSP from a CoherentReceiver.",
                )
            h = cd_compensation_transfer(grid.angular_frequency(), p["cd_compensation"], lam)
            y = apply_transfer_function(y, h, grid)
        rolloff = float(md.get(meta.PULSE_ROLLOFF, 0.0))

        def to_symbols(wave: np.ndarray) -> tuple[np.ndarray, int, float]:
            """Matched filter, timing, decimation: (symbols, integer offset, tau [s])."""
            if p["matched_filter"]:
                wave = matched_filter(wave, sps, grid, rolloff)
            if p["timing"] == "oerder_meyr":
                if sps < 3:
                    raise SamplingError(
                        "Oerder-Meyr timing needs >= 3 samples per symbol.",
                        hint="Use timing='integer' or more samples per symbol.",
                    )
                tau = estimate_timing_offset(wave, sps)
                return normalize_power(fractional_advance(wave, tau)[::sps]), 0, tau * grid.dt
            off = best_sampling_phase(wave, sps)
            return normalize_power(wave[off::sps]), off, off * grid.dt

        sym, offset, timing = to_symbols(y)
        df = 0.0
        if p["frequency_offset_estimation"]:
            # pass 1: coarse estimate; pass 2: remove it from the full-rate waveform so the
            # matched filter is centred on the signal spectrum, then estimate the residual
            df = estimate_frequency_offset(sym, rs)
            y = y * np.exp(-2j * np.pi * df * grid.time())
            sym, offset, timing = to_symbols(y)
            residual = estimate_frequency_offset(sym, rs)
            sym = remove_frequency_offset(sym, residual, rs)
            df += residual
            if abs(df) > 0.1 * rs:
                context.warn(
                    Diagnostic(
                        Severity.WARNING,
                        "dsp.frequency_offset_near_limit",
                        f"Estimated offset {df:.3g} Hz is close to the 4th-power limit R_s/8.",
                        hint="Tune the LO closer to the signal.",
                        source=self.name,
                    )
                )
        context.record("timing_offset_s", timing)
        if p["phase_recovery"] == "bps":
            sym, _ = blind_phase_search(sym, fmt, p["bps_test_phases"], p["bps_half_window"])
        sym = normalize_power(sym)
        context.record("sampling_offset", offset)
        context.record("frequency_offset_hz", df)
        out_md = {k: md[k] for k in (meta.MODULATION, meta.PATTERN) if k in md}
        return {"symbols": SymbolSequence(sym, rs, out_md)}


class CoherentAnalyzer(Component):
    """EVM, SNR, symbol and bit errors against the transmitted symbols; constellation data.

    Alignment (circular shift) and the constellation's rotation ambiguity are
    resolved against the reference (simulation convention). SNR is estimated
    data-aided with a least-squares complex gain c = <r, s>/<s, s>:
    SNR = |c|^2 mean|s|^2 / mean|r - c s|^2 and EVM_rms = 1/sqrt(SNR)
    (unbiased by noise in the power normalization).
    """

    type_id = "optobuild.analyzer.coherent"
    version = "1.0.0"
    display_name = "Coherent analyzer (EVM/BER)"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("received", S, tap=True), PortSpec("reference", S, tap=True))

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        rx: SymbolSequence = inputs["received"]
        ref: SymbolSequence = inputs["reference"]
        fmt = ref.metadata.get(meta.MODULATION)
        if fmt not in FORMATS or rx.n_symbols != ref.n_symbols:
            raise SamplingError(
                f"Received ({rx.n_symbols}) and reference ({ref.n_symbols}) symbol sequences "
                f"must have equal length and a known modulation (got {fmt!r})."
            )
        shift, rot, r = align_to_reference(rx.symbols, ref.symbols, fmt)
        s = ref.symbols
        c = np.vdot(s, r) / np.vdot(s, s)
        noise = r - c * s
        snr = float(abs(c) ** 2 * np.mean(np.abs(s) ** 2) / np.mean(np.abs(noise) ** 2))
        sym_err = int(np.count_nonzero(np.abs(decide(r, fmt) - s) > 1e-9))
        bits_rx, bits_ref = demap_hard(r, fmt), demap_hard(s, fmt)
        bit_err = int(np.count_nonzero(bits_rx != bits_ref))
        lo, hi = clopper_pearson(bit_err, bits_ref.size)
        context.record("n_symbols", rx.n_symbols)
        context.record("symbol_errors", sym_err)
        context.record("ser", sym_err / rx.n_symbols)
        context.record("n_bits", int(bits_ref.size))
        context.record("bit_errors", bit_err)
        context.record("ber", bit_err / bits_ref.size)
        context.record("ber_lower_95", lo)
        context.record("ber_upper_95", hi)
        context.record("snr_db", float(linear_to_db(snr)))
        context.record("evm_rms_percent", 100.0 / math.sqrt(snr))
        context.record("alignment_shift", shift)
        context.record("rotation_deg", math.degrees(rot * SYMMETRY[fmt]))
        context.record("constellation", r)
        if bit_err < 10:
            context.warn(
                Diagnostic(
                    Severity.INFO,
                    "ber.low_error_count",
                    f"Only {bit_err} bit errors in {bits_ref.size} bits (95 % interval "
                    f"[{lo:.2e}, {hi:.2e}]).",
                    hint="Use more symbols or Monte Carlo trials.",
                    source=self.name,
                )
            )
        return {}


__all__ = [
    "ASENoiseLoader",
    "CoherentAnalyzer",
    "CoherentDSP",
    "CoherentReceiver",
    "IQModulator",
    "PulseShaper",
    "SymbolMapper",
]
