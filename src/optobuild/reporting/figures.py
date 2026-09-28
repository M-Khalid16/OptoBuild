"""Plot-ready arrays from simulation results (Qt-free, no physics).

Shared by the GUI (``gui.plotdata`` re-exports this module) and the report
generator (``reporting.html`` renders the curves as SVG).

These helpers only re-arrange and rescale values that the simulator already
computed (unit prefixes via ``core.units``). The one presentation choice is
the logarithmic display floor for spectra, which is reported in the label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optobuild.core.units import frequency_to_wavelength, from_si
from optobuild.signals import DigitalSequence, ElectricalSignal, OpticalSignal, SymbolSequence

_PREFIXES = [
    (1e-15, "f"),
    (1e-12, "p"),
    (1e-9, "n"),
    (1e-6, "µ"),
    (1e-3, "m"),
    (1.0, ""),
    (1e3, "k"),
    (1e6, "M"),
    (1e9, "G"),
    (1e12, "T"),
]
SPECTRUM_FLOOR_DBM = -100.0
"""Values below this (including exact zeros) are drawn at the floor."""


def engineering_scale(values: NDArray[np.floating] | float) -> tuple[float, str]:
    """(factor, prefix) so that max|values| / factor lies in [1, 1000)."""
    peak = float(np.nanmax(np.abs(values))) if np.size(values) else 0.0
    if peak == 0.0 or not math.isfinite(peak):
        return 1.0, ""
    factor, prefix = _PREFIXES[0]
    for f, p in _PREFIXES:
        if peak >= f:
            factor, prefix = f, p
    return factor, prefix


@dataclass(frozen=True)
class Curve:
    """One plot: x/y arrays (NaN separates line segments) and axis labels."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x_label: str
    y_label: str
    title: str = ""
    scatter: bool = False
    """Draw points instead of a line (constellations)."""


def eye_curve(results: dict[str, Any], quantity_unit: str = "A") -> Curve:
    """Eye-diagram traces from an eye analyzer's recorded results.

    Traces are concatenated with NaN separators so they can be drawn as one
    line item.
    """
    traces = np.asarray(results["traces"], dtype=float)
    t = np.asarray(results["time_s"], dtype=float)
    tf, tp = engineering_scale(t)
    yf, yp = engineering_scale(traces)
    n, m = traces.shape
    x = np.tile(np.append(t / tf, np.nan), n)
    y = np.hstack([traces / yf, np.full((n, 1), np.nan)]).ravel()
    return Curve(
        x, y, f"time [{tp}s]", f"signal [{yp}{quantity_unit}]", f"Eye diagram ({n} traces)"
    )


def spectrum_curve(results: dict[str, Any], axis: str = "wavelength") -> Curve:
    """Optical spectrum (power per RBW in dBm) vs wavelength [nm] or frequency [THz]."""
    p = np.asarray(results["power_per_rbw_w"], dtype=float)
    floor_w = 1e-3 * 10 ** (SPECTRUM_FLOOR_DBM / 10)
    dbm = np.asarray(from_si(np.maximum(p, floor_w), "dBm"), dtype=float)
    if axis == "wavelength":
        x = np.asarray(from_si(results["wavelength_m"], "nm"), dtype=float)
        x_label = "wavelength [nm]"
    else:
        x = np.asarray(from_si(results["frequency_hz"], "THz"), dtype=float)
        x_label = "frequency [THz]"
    rbw = float(from_si(results["resolution_bandwidth_hz"], "GHz"))
    return Curve(
        x,
        dbm,
        x_label,
        f"power / {rbw:.4g} GHz RBW [dBm] (floor {SPECTRUM_FLOOR_DBM:g} dBm)",
        "Optical spectrum",
    )


TRANSFER_FLOOR_DB = -100.0
"""Power transfer values below this (including exact zeros) are drawn at the floor."""


def transfer_ports(results: dict[str, Any]) -> list[str]:
    """Ports with a recorded device power transfer (``transfer_<port>`` results)."""
    if "transfer_frequency_hz" not in results:
        return []
    return [k[len("transfer_") :] for k in results if k.startswith("transfer_")
            and k != "transfer_frequency_hz"]  # fmt: skip


def transfer_curve(results: dict[str, Any], port: str) -> Curve:
    """Device power transfer |H|^2 [dB] of one output port vs wavelength [nm]."""
    t = np.asarray(results[f"transfer_{port}"], dtype=float)
    floor = 10 ** (TRANSFER_FLOOR_DB / 10)
    db = 10 * np.log10(np.maximum(t, floor))
    lam = np.asarray(frequency_to_wavelength(results["transfer_frequency_hz"]), dtype=float)
    return Curve(
        np.asarray(from_si(lam, "nm"), dtype=float),
        db,
        "wavelength [nm]",
        f"power transfer [dB] (floor {TRANSFER_FLOOR_DB:g} dB)",
        f"Device response: {port}",
    )


def waveform_curve(signal: Any) -> Curve:
    """Time-domain view of a signal: optical power, electrical samples or bits."""
    if isinstance(signal, OpticalSignal):
        t = signal.grid.time()
        y = signal.power()
        yf, yp = engineering_scale(y)
        tf, tp = engineering_scale(t if t.size else np.array([1.0]))
        return Curve(t / tf, y / yf, f"time [{tp}s]", f"optical power [{yp}W]", "Optical power")
    if isinstance(signal, ElectricalSignal):
        t = signal.grid.time()
        yf, yp = engineering_scale(signal.samples)
        tf, tp = engineering_scale(t)
        return Curve(
            t / tf,
            signal.samples / yf,
            f"time [{tp}s]",
            f"{signal.quantity.name.lower()} [{yp}{signal.unit}]",
            "Electrical signal",
        )
    if isinstance(signal, DigitalSequence):
        k = np.arange(signal.n_bits + 1, dtype=float)
        y = np.append(signal.bits, signal.bits[-1]).astype(float)
        return Curve(k, y, "bit index", "bit value", "Bit sequence (step)")
    if isinstance(signal, SymbolSequence):
        return constellation_curve(signal.symbols.ravel(), "Symbols")
    raise TypeError(f"No waveform view for {type(signal).__name__}.")


def constellation_curve(symbols: Any, title: str = "Constellation") -> Curve:
    """I/Q scatter of complex symbols (dimensionless, unit average energy after DSP)."""
    s = np.asarray(symbols, dtype=complex).ravel()
    return Curve(
        s.real.astype(float),
        s.imag.astype(float),
        "in-phase",
        "quadrature",
        f"{title} ({s.size} symbols)",
        scatter=True,
    )


def scalar_rows(result: Any) -> list[tuple[str, str, str]]:
    """(node, key, formatted value) for every scalar result, in execution order."""
    rows = []
    for node in result.order:
        for key, value in result.nodes[node].results.items():
            if isinstance(value, np.ndarray):
                continue
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            rows.append((node, key, text))
    return rows


__all__ = [
    "SPECTRUM_FLOOR_DBM",
    "TRANSFER_FLOOR_DB",
    "Curve",
    "constellation_curve",
    "engineering_scale",
    "eye_curve",
    "scalar_rows",
    "spectrum_curve",
    "transfer_curve",
    "transfer_ports",
    "waveform_curve",
]
