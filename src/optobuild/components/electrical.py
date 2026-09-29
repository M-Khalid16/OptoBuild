"""Electrical low-pass filter and decision circuit.

Equations: optobuild.numerics.filters, optobuild.analysis.decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from optobuild.analysis.decision import decide, max_variance_offset, sample_bits
from optobuild.components.base import Component, RunContext
from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.filters import FILTER_KINDS, lowpass_on_grid, noise_equivalent_bandwidth
from optobuild.signals import DigitalSequence, ElectricalSignal, SignalKind
from optobuild.signals import metadata as meta


class LowPassFilter(Component):
    """Electrical low-pass filter (Bessel, Butterworth, Gaussian or ideal) by -3 dB bandwidth."""

    type_id = "optobuild.electrical.lowpass_filter"
    version = "1.0.0"
    display_name = "Low-pass filter"
    category = ComponentCategory.ELECTRICAL
    input_ports = (PortSpec("in", SignalKind.ELECTRICAL),)
    output_ports = (PortSpec("out", SignalKind.ELECTRICAL),)
    parameter_specs = (
        ParameterSpec("kind", ParameterType.CHOICE, default="bessel", choices=FILTER_KINDS),
        ParameterSpec(
            "bandwidth",
            ParameterType.FLOAT,
            default=7.5e9,
            unit="Hz",
            display_unit="GHz",
            minimum=0.0,
            minimum_inclusive=False,
            symbol="B",
            description="-3 dB bandwidth",
        ),
        ParameterSpec(
            "order",
            ParameterType.INT,
            default=4,
            minimum=1,
            maximum=12,
            description="Order (Bessel/Butterworth only)",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: ElectricalSignal = inputs["in"]
        p = self.parameters
        if p["bandwidth"] >= sig.grid.nyquist_frequency:
            context.warn(
                Diagnostic(
                    Severity.WARNING,
                    "filter.bandwidth_above_nyquist",
                    f"Bandwidth {p['bandwidth']:g} Hz >= fs/2 = {sig.grid.nyquist_frequency:g} Hz; "
                    "the filter is truncated by the simulation band.",
                    hint="Increase samples per bit or reduce the filter bandwidth.",
                    source=self.name,
                )
            )
        h = lowpass_on_grid(sig.grid, p["kind"], p["bandwidth"], p["order"])
        y = apply_transfer_function(sig.samples, h, sig.grid)
        context.record("noise_equivalent_bandwidth_hz", noise_equivalent_bandwidth(sig.grid, h))
        return {"out": sig.replace(samples=np.real(y))}


class DecisionCircuit(Component):
    """Samples one value per bit and compares it with a threshold -> bit sequence.

    Timing: ``max_variance`` picks the sample offset with the largest spread
    of values (data-independent); ``fixed`` uses ``sampling_phase`` (fraction
    of the bit period). Threshold: ``mean`` uses the mean of the decision
    samples; ``fixed`` uses ``threshold`` in the input's unit (A or V).
    Bit timing comes from the signal's ``bit_rate``/``samples_per_bit`` metadata.
    """

    type_id = "optobuild.electrical.decision"
    version = "1.0.0"
    display_name = "Decision circuit"
    category = ComponentCategory.ELECTRICAL
    input_ports = (PortSpec("in", SignalKind.ELECTRICAL),)
    output_ports = (PortSpec("bits", SignalKind.DIGITAL),)
    parameter_specs = (
        ParameterSpec(
            "timing",
            ParameterType.CHOICE,
            default="max_variance",
            choices=("max_variance", "fixed"),
        ),
        ParameterSpec(
            "sampling_phase",
            ParameterType.FLOAT,
            default=0.5,
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
            description="Sampling instant as a fraction of the bit period (fixed)",
        ),
        ParameterSpec(
            "threshold_mode", ParameterType.CHOICE, default="mean", choices=("mean", "fixed")
        ),
        ParameterSpec(
            "threshold",
            ParameterType.FLOAT,
            default=0.0,
            unit="input",
            description="Decision threshold in the input's SI unit (fixed mode)",
        ),
    )

    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        sig: ElectricalSignal = inputs["in"]
        p = self.parameters
        bit_rate, sps = meta.require_timing(sig.metadata, self.name)
        if p["timing"] == "max_variance":
            offset = max_variance_offset(sig.samples, sps)
        else:
            offset = int(np.floor(p["sampling_phase"] * sps))
        values = sample_bits(sig.samples, sps, offset)
        threshold = float(values.mean()) if p["threshold_mode"] == "mean" else p["threshold"]
        bits = decide(values, threshold)
        context.record("sampling_offset", offset)
        context.record("threshold", threshold)
        context.record("decision_samples", values)
        md = {k: sig.metadata[k] for k in (meta.PATTERN,) if k in sig.metadata}
        return {"bits": DigitalSequence(bits, bit_rate, md)}


__all__ = ["DecisionCircuit", "LowPassFilter"]
