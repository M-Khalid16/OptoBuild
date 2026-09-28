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
    gram_schmidt_orthogonalize,
    mimo_equalize,
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
from optobuild.physics.detection import coherent_detection, delay_samples
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

    With ``polarizations = 2`` the output has shape (2, n): x carries the
    pattern as above, y the time-reversed pattern. For a PRBS this is the
    m-sequence of the reciprocal polynomial, not a cyclic shift of x (a
    shifted copy, e.g. of an odd-length pattern repeated for QPSK, would make
    the tributaries indistinguishable to a cyclic alignment).
    """

    type_id = "optobuild.dsp.symbol_mapper"
    version = "1.1.0"
    display_name = "Symbol mapper"
    category = ComponentCategory.DSP
    input_ports = (PortSpec("bits", SignalKind.DIGITAL),)
    output_ports = (PortSpec("symbols", S),)
    parameter_specs = (
        ParameterSpec("modulation", ParameterType.CHOICE, default="qpsk", choices=_FMT),
        ParameterSpec(
            "polarizations",
            ParameterType.INT,
            default=1,
            minimum=1,
            maximum=2,
            description="1: single polarization; 2: x and y tributaries",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        seq: DigitalSequence = inputs["bits"]
        fmt = self.parameters["modulation"]
        k = bits_per_symbol(fmt)

        def mapped(bits: np.ndarray) -> np.ndarray:
            return map_bits(bits if bits.size % k == 0 else np.tile(bits, k), fmt)

        sym = mapped(seq.bits)
        if self.parameters["polarizations"] == 2:
            sym = np.vstack([sym, mapped(seq.bits[::-1])])
        md = {meta.MODULATION: fmt}
        if meta.PATTERN in seq.metadata:
            md[meta.PATTERN] = seq.metadata[meta.PATTERN]
        return {"symbols": SymbolSequence(sym, seq.bit_rate / k, md)}


class PulseShaper(Component):
    """Symbols -> in-phase and quadrature drive voltages with (root-)raised-cosine pulses.

    v_I + i v_Q = amplitude * sum_k s_k p(t - k T_s); symbol k centred at sample k*sps.
    For dual-polarization symbols, ``polarization`` selects the tributary
    (one shaper per IQ modulator).
    """

    type_id = "optobuild.dsp.pulse_shaper"
    version = "1.1.0"
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
        ParameterSpec(
            "polarization",
            ParameterType.CHOICE,
            default="x",
            choices=("x", "y"),
            description="Tributary of dual-polarization symbols",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sym: SymbolSequence = inputs["symbols"]
        p = self.parameters
        sps = p["samples_per_symbol"]
        s = sym.symbols
        if s.ndim == 2:
            s = s[0] if p["polarization"] == "x" else s[1]
        grid = TimeGrid(sym.n_symbols * sps, 1.0 / (sym.symbol_rate * sps))
        y = p["amplitude"] * shape(s, sps, grid, p["rolloff"], p["pulse"])
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
    specification here; for gain and noise derived from the erbium inversion
    use ``components.lasers.ErbiumDopedFiberAmplifier``.
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

    Impairments: hybrid phase error eps (quadrature at 90 deg + eps), quadrature
    amplitude imbalance g (responsivity of the Q pair g R) and I/Q skew (Q current
    delayed by ``iq_skew``, after the photodiodes), physics.detection.
    For dual polarization use a PBS on the signal and on the LO and one receiver
    per polarization.
    """

    type_id = "optobuild.detector.coherent_receiver"
    version = "1.1.0"
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
        ParameterSpec(
            "hybrid_phase_error",
            ParameterType.FLOAT,
            default=0.0,
            unit="rad",
            display_unit="deg",
            minimum=-math.pi / 4,
            maximum=math.pi / 4,
            symbol="eps",
            description="Deviation of the I/Q phase difference from 90 deg",
        ),
        ParameterSpec(
            "quadrature_gain",
            ParameterType.FLOAT,
            default=1.0,
            unit="1",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="g",
            description="Q-to-I responsivity ratio (amplitude imbalance)",
        ),
        ParameterSpec(
            "iq_skew",
            ParameterType.FLOAT,
            default=0.0,
            unit="s",
            display_unit="ps",
            description="Delay of the Q current relative to I",
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
            phase_error=p["hybrid_phase_error"],
            quadrature_gain=p["quadrature_gain"],
        )
        if p["iq_skew"] != 0.0:
            q = delay_samples(q, p["iq_skew"], sig.grid)
        md = dict(sig.metadata)
        md[meta.CARRIER_WAVELENGTH] = sig.wavelength
        context.record("lo_power_w", lo.average_power())
        context.record("signal_power_w", sig.average_power())
        return {
            "i": ElectricalSignal(sig.grid, i, ElectricalQuantity.CURRENT, md),
            "q": ElectricalSignal(sig.grid, q, ElectricalQuantity.CURRENT, md),
        }


_FRONT_END_SPECS = (
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
        "iq_deskew",
        ParameterType.FLOAT,
        default=0.0,
        unit="s",
        display_unit="ps",
        description="Known I/Q skew to remove (Q advanced by this amount)",
    ),
    ParameterSpec(
        "iq_orthogonalization",
        ParameterType.BOOL,
        default=False,
        description="Gram-Schmidt orthogonalization of I and Q (hybrid imbalance)",
    ),
    ParameterSpec(
        "matched_filter",
        ParameterType.BOOL,
        default=True,
        description="RRC matched filter with the transmitter's roll-off",
    ),
)


def _front_end(
    name: str, params: Mapping[str, Any], si: ElectricalSignal, sq: ElectricalSignal
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Deskew, GSOP and CD compensation of one I/Q pair: (complex waveform, metadata, format)."""
    require_same_grid(si, sq, what=f"inputs of '{name}'")
    md = dict(si.metadata)
    meta.require_symbol_timing(md, name)
    fmt = md.get(meta.MODULATION)
    if fmt not in FORMATS:
        raise SignalTypeError(
            f"'{name}' needs the '{meta.MODULATION}' metadata.",
            hint="Generate symbols with the SymbolMapper.",
        )
    grid = si.grid
    i, q = si.samples, sq.samples
    if params["iq_deskew"] != 0.0:
        q = delay_samples(q, -params["iq_deskew"], grid)
    if params["iq_orthogonalization"]:
        i, q = gram_schmidt_orthogonalize(i, q)
    y = i + 1j * q
    if params["cd_compensation"] != 0.0:
        lam = md.get(meta.CARRIER_WAVELENGTH)
        if lam is None:
            raise SignalTypeError(
                "CD compensation needs the carrier wavelength metadata.",
                hint="Feed the DSP from a CoherentReceiver.",
            )
        h = cd_compensation_transfer(grid.angular_frequency(), params["cd_compensation"], lam)
        y = apply_transfer_function(y, h, grid)
    return y, md, fmt


def _warn_offset(context: RunContext, name: str, df: float, rs: float) -> None:
    if abs(df) > 0.1 * rs:
        context.warn(
            Diagnostic(
                Severity.WARNING,
                "dsp.frequency_offset_near_limit",
                f"Estimated offset {df:.3g} Hz is close to the 4th-power limit R_s/8.",
                hint="Tune the LO closer to the signal.",
                source=name,
            )
        )


class CoherentDSP(Component):
    """Receiver DSP: deskew, GSOP, CD compensation, RRC matched filter, timing, FOE, BPS."""

    type_id = "optobuild.dsp.coherent_dsp"
    version = "1.1.0"
    display_name = "Coherent DSP"
    category = ComponentCategory.DSP
    input_ports = (PortSpec("i", E), PortSpec("q", E))
    output_ports = (PortSpec("symbols", S),)
    parameter_specs = (
        *_FRONT_END_SPECS,
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
        p = self.parameters
        y, md, fmt = _front_end(self.name, p, inputs["i"], inputs["q"])
        rs, sps = meta.require_symbol_timing(md, self.name)
        grid = inputs["i"].grid
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
            _warn_offset(context, self.name, df, rs)
        context.record("timing_offset_s", timing)
        if p["phase_recovery"] == "bps":
            sym, _ = blind_phase_search(sym, fmt, p["bps_test_phases"], p["bps_half_window"])
        sym = normalize_power(sym)
        context.record("sampling_offset", offset)
        context.record("frequency_offset_hz", df)
        out_md = {k: md[k] for k in (meta.MODULATION, meta.PATTERN) if k in md}
        return {"symbols": SymbolSequence(sym, rs, out_md)}


class DualPolCoherentDSP(Component):
    """Dual-polarization receiver DSP with a 2x2 adaptive (CMA/RDE) MIMO equalizer.

    Per polarization: deskew, GSOP, CD compensation, RRC matched filter. Then:
    Oerder-Meyr timing on |x|^2 + |y|^2 (sps >= 4; at sps = 2 the fractionally
    spaced equalizer absorbs the timing phase), decimation to 2 samples/symbol,
    2x2 butterfly equalizer (analysis.coherent_dsp.mimo_equalize: rotation,
    PMD, residual ISI), 4th-power FOE over both outputs, BPS per polarization,
    normalization. The output polarization order/phase may be permuted; the
    analyzer resolves it against the reference.
    """

    type_id = "optobuild.dsp.dual_pol_coherent_dsp"
    version = "1.0.0"
    display_name = "Coherent DSP (dual polarization)"
    category = ComponentCategory.DSP
    input_ports = (
        PortSpec("xi", E, "x in-phase"),
        PortSpec("xq", E, "x quadrature"),
        PortSpec("yi", E, "y in-phase"),
        PortSpec("yq", E, "y quadrature"),
    )
    output_ports = (PortSpec("symbols", S),)
    parameter_specs = (
        *_FRONT_END_SPECS,
        ParameterSpec("mimo_taps", ParameterType.INT, default=15, minimum=1),
        ParameterSpec(
            "mimo_step",
            ParameterType.FLOAT,
            default=2.5e-4,
            unit="1",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="mu",
            description="Step size (unit-power input); larger values drift for 16/64-QAM",
        ),
        ParameterSpec("mimo_epochs", ParameterType.INT, default=6, minimum=2),
        ParameterSpec(
            "mimo_x_epochs",
            ParameterType.INT,
            default=2,
            minimum=1,
            description="Initial passes adapting only output x (singularity avoidance)",
        ),
        ParameterSpec(
            "mimo_algorithm",
            ParameterType.CHOICE,
            default="auto",
            choices=("auto", "cma", "cma_rde"),
            description="auto: CMA for (B/Q)PSK, CMA then RDE for QAM",
        ),
        ParameterSpec("frequency_offset_estimation", ParameterType.BOOL, default=True),
        ParameterSpec(
            "phase_recovery", ParameterType.CHOICE, default="bps", choices=("bps", "none")
        ),
        ParameterSpec("bps_test_phases", ParameterType.INT, default=32, minimum=4),
        ParameterSpec("bps_half_window", ParameterType.INT, default=16, minimum=0),
    )

    def validate(self) -> list[Diagnostic]:
        diags = super().validate()
        if self.parameters["mimo_x_epochs"] >= self.parameters["mimo_epochs"]:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "mimo.epochs",
                    "mimo_x_epochs must be smaller than mimo_epochs.",
                    hint="E.g. 2 x-only passes out of 4.",
                )
            )
        if self.parameters["mimo_taps"] % 2 == 0:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "mimo.even_taps",
                    f"mimo_taps must be odd, got {self.parameters['mimo_taps']}.",
                    hint="Use an odd number of taps (centre-tap initialization).",
                )
            )
        return diags

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        p = self.parameters
        wx, md, fmt = _front_end(self.name, p, inputs["xi"], inputs["xq"])
        wy, _, fmt_y = _front_end(self.name, p, inputs["yi"], inputs["yq"])
        require_same_grid(inputs["xi"], inputs["yi"], what=f"polarizations of '{self.name}'")
        if fmt_y != fmt:
            raise SignalTypeError(f"'{self.name}': x and y carry different modulation formats.")
        rs, sps = meta.require_symbol_timing(md, self.name)
        grid = inputs["xi"].grid
        if sps % 2:
            raise SamplingError(
                f"'{self.name}' needs an even number of samples per symbol, got {sps}.",
                hint="Use 2, 4, 6, ... samples per symbol.",
            )
        algo = p["mimo_algorithm"]
        if algo == "auto":
            algo = "cma" if fmt in ("bpsk", "qpsk") else "cma_rde"
        self._check_pattern(md, fmt, context)
        rolloff = float(md.get(meta.PULSE_ROLLOFF, 0.0))

        def equalize(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            """Matched filter, timing, 2 samples/symbol, MIMO: (symbols, taps, tau [samples])."""
            if p["matched_filter"]:
                wave = np.vstack([matched_filter(row, sps, grid, rolloff) for row in wave])
            tau = 0.0
            if sps >= 4:
                tau = estimate_timing_offset(np.sqrt(np.sum(np.abs(wave) ** 2, axis=0)), sps)
                wave = np.vstack([fractional_advance(row, tau) for row in wave])
            wave = wave[:, :: sps // 2]
            wave = wave / math.sqrt(float(np.mean(np.abs(wave) ** 2)))
            sym, taps = mimo_equalize(
                wave,
                fmt,
                p["mimo_taps"],
                p["mimo_step"],
                p["mimo_epochs"],
                algo == "cma_rde",
                p["mimo_x_epochs"],
            )
            return sym, taps, tau

        w = np.vstack([wx, wy])
        sym, taps, tau = equalize(w)
        df = 0.0
        if p["frequency_offset_estimation"]:
            # pass 1 (CMA/RDE are offset-invariant); pass 2 re-centres the matched filter
            # on the signal spectrum, as in the single-polarization DSP
            df = estimate_frequency_offset(sym, rs)
            sym, taps, tau = equalize(w * np.exp(-2j * np.pi * df * grid.time())[None, :])
            residual = estimate_frequency_offset(sym, rs)
            sym = np.vstack([remove_frequency_offset(row, residual, rs) for row in sym])
            df += residual
            _warn_offset(context, self.name, df, rs)
        if p["phase_recovery"] == "bps":
            sym = np.vstack(
                [
                    blind_phase_search(row, fmt, p["bps_test_phases"], p["bps_half_window"])[0]
                    for row in sym
                ]
            )
        sym = np.vstack([normalize_power(row) for row in sym])
        context.record("timing_offset_s", tau * grid.dt)
        context.record("frequency_offset_hz", df)
        context.record("mimo_algorithm", algo)
        context.record("equalizer_taps", taps)
        out_md = {k: md[k] for k in (meta.MODULATION, meta.PATTERN) if k in md}
        return {"symbols": SymbolSequence(sym, rs, out_md)}

    def _check_pattern(self, md: Mapping[str, Any], fmt: str, context: RunContext) -> None:
        """Warn when a PRBS recurrence spans < 3 symbols (non-i.i.d. higher-order statistics)."""
        pattern = str(md.get(meta.PATTERN, ""))
        if not pattern.startswith("PRBS") or not pattern[4:].isdigit():
            return
        order = int(pattern[4:])
        if order / bits_per_symbol(fmt) < 3:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "dsp.prbs_order_too_low",
                    f"{pattern} spans only {order / bits_per_symbol(fmt):.1f} {fmt} symbols; "
                    "its bit recurrence correlates the symbols and can mislead blind "
                    "(CMA/RDE) equalization.",
                    hint="Use a PRBS order >= 3 log2(M) (e.g. PRBS23 segment for 64-QAM).",
                    source=self.name,
                )
            )


def _ratio(signal: float, noise: float) -> float:
    """signal / noise, +inf for a noiseless (exactly reconstructed) signal."""
    return signal / noise if noise > 0.0 else math.inf


def _pol_metrics(received: np.ndarray, reference: np.ndarray, fmt: str) -> dict[str, Any]:
    """Alignment, least-squares gain, noise power and error counts of one polarization."""
    shift, rot, r = align_to_reference(received, reference, fmt)
    s = reference
    c = np.vdot(s, r) / np.vdot(s, s)
    bits_rx, bits_ref = demap_hard(r, fmt), demap_hard(s, fmt)
    return {
        "shift": shift,
        "rot": rot,
        "r": r,
        "signal": float(abs(c) ** 2 * np.mean(np.abs(s) ** 2)),
        "noise": float(np.mean(np.abs(r - c * s) ** 2)),
        "sym_err": int(np.count_nonzero(np.abs(decide(r, fmt) - s) > 1e-9)),
        "bit_err": int(np.count_nonzero(bits_rx != bits_ref)),
        "n_bits": int(bits_ref.size),
    }


class CoherentAnalyzer(Component):
    """EVM, SNR, symbol and bit errors against the transmitted symbols; constellation data.

    Alignment (circular shift) and the constellation's rotation ambiguity are
    resolved against the reference (simulation convention). SNR is estimated
    data-aided with a least-squares complex gain c = <r, s>/<s, s>:
    SNR = |c|^2 mean|s|^2 / mean|r - c s|^2 and EVM_rms = 1/sqrt(SNR)
    (unbiased by noise in the power normalization).

    Dual polarization (symbols of shape (2, n)): the output-to-tributary
    assignment (a possible x/y swap after blind equalization) is chosen by the
    larger total correlation; totals pool both polarizations (SNR = sum of
    signal powers / sum of noise powers) and per-polarization values are
    recorded with suffixes ``_x``/``_y`` (of the transmitted tributaries).
    """

    type_id = "optobuild.analyzer.coherent"
    version = "1.1.0"
    display_name = "Coherent analyzer (EVM/BER)"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("received", S, tap=True), PortSpec("reference", S, tap=True))

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        rx: SymbolSequence = inputs["received"]
        ref: SymbolSequence = inputs["reference"]
        fmt = ref.metadata.get(meta.MODULATION)
        if (
            fmt not in FORMATS
            or rx.n_symbols != ref.n_symbols
            or rx.symbols.shape != ref.symbols.shape
        ):
            raise SamplingError(
                f"Received {rx.symbols.shape} and reference {ref.symbols.shape} symbol "
                f"sequences must have equal shapes and a known modulation (got {fmt!r})."
            )
        if rx.symbols.ndim == 1:
            pols = [_pol_metrics(rx.symbols, ref.symbols, fmt)]
            swapped = False
        else:
            straight = [_pol_metrics(rx.symbols[p], ref.symbols[p], fmt) for p in range(2)]
            crossed = [_pol_metrics(rx.symbols[1 - p], ref.symbols[p], fmt) for p in range(2)]

            # normalized squared correlation |c|^2 P_s / mean|r|^2, bounded in [0, 1]
            def score(ms: list[dict[str, Any]]) -> float:
                return sum(m["signal"] / (m["signal"] + m["noise"]) for m in ms)

            swapped = score(crossed) > score(straight)
            pols = crossed if swapped else straight
        signal = sum(m["signal"] for m in pols)
        noise = sum(m["noise"] for m in pols)
        snr = _ratio(signal, noise)
        sym_err = sum(m["sym_err"] for m in pols)
        bit_err = sum(m["bit_err"] for m in pols)
        n_bits = sum(m["n_bits"] for m in pols)
        n_sym = rx.n_symbols * len(pols)
        lo, hi = clopper_pearson(bit_err, n_bits)
        context.record("n_symbols", n_sym)
        context.record("symbol_errors", sym_err)
        context.record("ser", sym_err / n_sym)
        context.record("n_bits", n_bits)
        context.record("bit_errors", bit_err)
        context.record("ber", bit_err / n_bits)
        context.record("ber_lower_95", lo)
        context.record("ber_upper_95", hi)
        context.record("snr_db", float(linear_to_db(snr)))
        context.record("evm_rms_percent", 100.0 / math.sqrt(snr))
        context.record("alignment_shift", pols[0]["shift"])
        context.record("rotation_deg", math.degrees(pols[0]["rot"] * SYMMETRY[fmt]))
        if len(pols) == 2:
            context.record("polarizations_swapped", swapped)
            for m, sfx in zip(pols, ("x", "y"), strict=True):
                snr_p = _ratio(m["signal"], m["noise"])
                context.record(f"snr_db_{sfx}", float(linear_to_db(snr_p)))
                context.record(f"ber_{sfx}", m["bit_err"] / m["n_bits"])
            context.record("constellation", np.vstack([m["r"] for m in pols]))
        else:
            context.record("constellation", pols[0]["r"])
        if bit_err < 10:
            context.warn(
                Diagnostic(
                    Severity.INFO,
                    "ber.low_error_count",
                    f"Only {bit_err} bit errors in {n_bits} bits (95 % interval "
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
    "DualPolCoherentDSP",
    "IQModulator",
    "PulseShaper",
    "SymbolMapper",
]
