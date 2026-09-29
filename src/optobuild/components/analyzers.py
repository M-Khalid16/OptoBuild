"""Analyzer components (sinks): optical power meter, optical spectrum analyzer,
eye-diagram analyzer, BER analyzer. Algorithms: optobuild.analysis.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optobuild.analysis.ber import count_errors, q_factor
from optobuild.analysis.decision import decide, max_variance_offset, sample_bits
from optobuild.analysis.eye import eye_diagram
from optobuild.analysis.power import measure_power
from optobuild.analysis.spectrum import optical_spectrum
from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.signals import DigitalSequence, ElectricalSignal, OpticalSignal, SignalKind
from optobuild.signals import metadata as meta


class OpticalPowerMeter(Component):
    """Records window-averaged and peak optical power."""

    type_id = "optobuild.analyzer.optical_power_meter"
    version = "1.0.0"
    display_name = "Optical power meter"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("in", SignalKind.OPTICAL, tap=True),)

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        m = measure_power(inputs["in"])
        context.record("average_power_w", m.average_w)
        context.record("average_power_dbm", m.average_dbm)
        context.record("peak_power_w", m.peak_w)
        return {}


class OpticalSpectrumAnalyzer(Component):
    """Records the optical PSD and the power per resolution bandwidth."""

    type_id = "optobuild.analyzer.optical_spectrum"
    version = "1.0.0"
    display_name = "Optical spectrum analyzer"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("in", SignalKind.OPTICAL, tap=True),)
    parameter_specs = (
        ParameterSpec(
            "resolution_bandwidth",
            ParameterType.FLOAT,
            default=12.5e9,
            unit="Hz",
            display_unit="GHz",
            minimum=0.0,
            symbol="RBW",
            description="Rectangular resolution bandwidth (0 = one FFT bin)",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: OpticalSignal = inputs["in"]
        spec = optical_spectrum(sig, self.parameters["resolution_bandwidth"])
        context.record("frequency_hz", spec.frequency_hz)
        context.record("wavelength_m", spec.wavelength_m)
        context.record("psd_w_per_hz", spec.psd_w_per_hz)
        context.record("power_per_rbw_w", spec.power_per_rbw_w)
        context.record("resolution_bandwidth_hz", spec.resolution_bandwidth_hz)
        context.record("total_power_w", spec.total_power())
        return {}


class EyeDiagramAnalyzer(Component):
    """Records eye traces centred on the max-variance sampling instant.

    Also records decision-directed level statistics at that instant (bits
    classified by the mean threshold, not by the transmitted data), which
    are biased when errors occur; use the BER analyzer for error counts.
    """

    type_id = "optobuild.analyzer.eye_diagram"
    version = "1.1.0"
    display_name = "Eye diagram analyzer"
    category = ComponentCategory.ANALYZER
    input_ports = (PortSpec("in", SignalKind.ELECTRICAL, tap=True),)
    parameter_specs = (
        ParameterSpec("symbols_per_trace", ParameterType.INT, default=2, minimum=1, maximum=8),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: ElectricalSignal = inputs["in"]
        bit_rate, sps = meta.require_timing(sig.metadata, self.name)
        n = self.parameters["symbols_per_trace"]
        best = max_variance_offset(sig.samples, sps)
        eye = eye_diagram(sig.samples, sps, 1.0 / bit_rate, n, offset=best - (n * sps) // 2)
        values = sample_bits(sig.samples, sps, best)
        dd_bits = decide(values, float(values.mean()))
        context.record("traces", eye.traces)
        context.record("time_s", eye.time_s)
        context.record("sampling_offset", best)
        context.record("unit", sig.unit)
        if 0 < dd_bits.sum() < dd_bits.size:
            mu1, mu0, s1, s0, q = q_factor(values, dd_bits)
            context.record("q_factor_decision_directed", q)
            context.record(
                "eye_opening", float(values[dd_bits == 1].min() - values[dd_bits == 0].max())
            )
        else:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "eye.single_level",
                    "All samples fall on one side of the threshold.",
                    source=self.name,
                )
            )
        return {}


class BERAnalyzer(Component):
    """Counts bit errors between the transmitted reference and the received bits."""

    type_id = "optobuild.analyzer.ber"
    version = "1.0.0"
    display_name = "BER analyzer"
    category = ComponentCategory.ANALYZER
    input_ports = (
        PortSpec("reference", SignalKind.DIGITAL, "transmitted bits", tap=True),
        PortSpec("received", SignalKind.DIGITAL, "decided bits", tap=True),
    )
    parameter_specs = (
        ParameterSpec(
            "align",
            ParameterType.BOOL,
            default=True,
            description="Search the circular bit shift that best aligns the sequences",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        ref: DigitalSequence = inputs["reference"]
        rx: DigitalSequence = inputs["received"]
        res = count_errors(ref.bits, rx.bits, align=self.parameters["align"])
        context.record("n_bits", res.n_bits)
        context.record("n_errors", res.n_errors)
        context.record("ber", res.ber)
        context.record("ber_lower_95", res.ber_lower_95)
        context.record("ber_upper_95", res.ber_upper_95)
        context.record("alignment_shift_bits", res.shift)
        if res.ber > 0.1:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "ber.alignment_unreliable",
                    f"BER {res.ber:.3g} is so high that sequence alignment is unreliable.",
                    source=self.name,
                )
            )
        if res.n_errors < 10:
            context.warn(
                Diagnostic(
                    Severity.INFO,
                    "ber.low_error_count",
                    f"Only {res.n_errors} errors in {res.n_bits} bits; the BER is statistically "
                    f"uncertain (95 % interval [{res.ber_lower_95:.2e}, {res.ber_upper_95:.2e}]).",
                    hint="Simulate more bits for a tighter estimate.",
                    source=self.name,
                )
            )
        return {}


__all__ = ["BERAnalyzer", "EyeDiagramAnalyzer", "OpticalPowerMeter", "OpticalSpectrumAnalyzer"]
