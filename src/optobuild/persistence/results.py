"""Simulation results in HDF5 (ADR-0006; requires the optional ``h5py``).

Layout of a results file (schema version 1)::

    /                      attrs: format="optobuild-results", schema_version,
                                  optobuild_version, seed, trial (-1 = none),
                                  order (JSON list), layout (JSON or ""),
                                  project (project JSON or ""), diagnostics (JSON)
    /nodes/n0000           attrs: name, key, cache_hit, elapsed_s, diagnostics (JSON)
        outputs/p0000      attrs: port, kind, metadata (JSON) + kind-specific attrs
            field | samples | bits      dataset (lossless dtype)
        results/r0000      dataset; attrs: key, type

Signals are stored with the attributes needed to rebuild them exactly:
optical (grid n_samples/dt/t0, center_frequency [Hz], field [sqrt(W)]),
electrical (grid, quantity, samples [V or A]), digital (bit_rate [bit/s], bits).
Groups are indexed (n0000, p0000, ...) with the real names as attributes, so
any component name is allowed. Nothing is pickled; loading never executes code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import optobuild
from optobuild.components.registry import ComponentRegistry
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import ProjectFormatError
from optobuild.engine.executor import SimulationResult
from optobuild.numerics.grid import TimeGrid
from optobuild.numerics.layout import SimulationLayout
from optobuild.persistence.project import Project, dumps_project, loads_project
from optobuild.signals import (
    DigitalSequence,
    ElectricalQuantity,
    ElectricalSignal,
    OpticalSignal,
    Signal,
    SignalKind,
)

RESULTS_FORMAT = "optobuild-results"
RESULTS_SCHEMA_VERSION = 1


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ProjectFormatError(
            "HDF5 result storage requires h5py.", hint="pip install 'optobuild[hdf5]'"
        ) from exc
    return h5py


def _json(value: Any, what: str) -> str:
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ProjectFormatError(f"Cannot store {what} as JSON: {exc}") from exc


def _diag_list(diags: Any) -> str:
    return _json(
        [
            {
                "severity": d.severity.value,
                "code": d.code,
                "message": d.message,
                "hint": d.hint,
                "source": d.source,
            }
            for d in diags
        ],
        "diagnostics",
    )


def _diag_from(text: str) -> tuple[Diagnostic, ...]:
    return tuple(
        Diagnostic(Severity(d["severity"]), d["code"], d["message"], d["hint"], d["source"])
        for d in json.loads(text)
    )


def _write_grid(group: Any, grid: TimeGrid) -> None:
    group.attrs["n_samples"] = grid.n_samples
    group.attrs["dt"] = grid.dt
    group.attrs["t0"] = grid.t0


def _read_grid(group: Any) -> TimeGrid:
    return TimeGrid(
        int(group.attrs["n_samples"]), float(group.attrs["dt"]), float(group.attrs["t0"])
    )


def _write_signal(group: Any, sig: Signal, compression: str | None) -> None:
    group.attrs["kind"] = sig.kind.value
    group.attrs["metadata"] = _json(dict(sig.metadata), "signal metadata")
    if isinstance(sig, OpticalSignal):
        _write_grid(group, sig.grid)
        group.attrs["center_frequency"] = sig.center_frequency
        group.create_dataset("field", data=sig.field, compression=compression)
    elif isinstance(sig, ElectricalSignal):
        _write_grid(group, sig.grid)
        group.attrs["quantity"] = sig.quantity.name
        group.create_dataset("samples", data=sig.samples, compression=compression)
    elif isinstance(sig, DigitalSequence):
        group.attrs["bit_rate"] = sig.bit_rate
        group.create_dataset("bits", data=sig.bits, compression=compression)
    else:
        raise ProjectFormatError(f"Cannot store signal of type {type(sig).__name__}.")


def _read_signal(group: Any) -> Signal:
    kind = SignalKind(group.attrs["kind"])
    md = json.loads(group.attrs["metadata"])
    if kind is SignalKind.OPTICAL:
        return OpticalSignal(
            _read_grid(group), group["field"][()], float(group.attrs["center_frequency"]), md
        )
    if kind is SignalKind.ELECTRICAL:
        return ElectricalSignal(
            _read_grid(group), group["samples"][()], ElectricalQuantity[group.attrs["quantity"]], md
        )
    if kind is SignalKind.DIGITAL:
        return DigitalSequence(group["bits"][()], float(group.attrs["bit_rate"]), md)
    raise ProjectFormatError(f"Unsupported stored signal kind {kind.value!r}.")


_SCALAR_TYPES = {bool: "bool", int: "int", float: "float", complex: "complex", str: "str"}


def _write_result(group: Any, name: str, key: str, value: Any, compression: str | None) -> None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        ds = group.create_dataset(
            name, data=value, compression=compression if value.size > 1 else None
        )
        ds.attrs["type"] = "ndarray"
    elif type(value) in _SCALAR_TYPES:
        ds = group.create_dataset(name, data=value)
        ds.attrs["type"] = _SCALAR_TYPES[type(value)]
    else:
        raise ProjectFormatError(
            f"Cannot store result '{key}' of type {type(value).__name__} in HDF5.",
            hint="Record numbers, strings or NumPy arrays.",
        )
    ds.attrs["key"] = key


def _read_result(ds: Any) -> Any:
    kind = ds.attrs["type"]
    value = ds[()]
    if kind == "ndarray":
        arr = np.asarray(value)
        arr.setflags(write=False)
        return arr
    if kind == "str":
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return {"bool": bool, "int": int, "float": float, "complex": complex}[kind](value)


def save_results(
    result: SimulationResult,
    path: str | Path,
    *,
    project: Project | None = None,
    include_signals: bool = True,
    compression: str | None = "gzip",
) -> Path:
    """Write ``result`` (and optionally the project that produced it) to an HDF5 file."""
    h5py = _h5py()
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = RESULTS_FORMAT
        f.attrs["schema_version"] = RESULTS_SCHEMA_VERSION
        f.attrs["optobuild_version"] = optobuild.__version__
        f.attrs["seed"] = str(result.seed)  # up to 128-bit; stored as text
        f.attrs["trial"] = -1 if result.trial is None else result.trial
        f.attrs["order"] = _json(list(result.order), "execution order")
        f.attrs["layout"] = (
            "" if result.layout is None else _json(result.layout.to_dict(), "layout")
        )
        f.attrs["project"] = "" if project is None else dumps_project(project)
        f.attrs["diagnostics"] = _diag_list(result.diagnostics)
        nodes = f.create_group("nodes")
        for i, name in enumerate(result.order):
            node = result.nodes[name]
            g = nodes.create_group(f"n{i:04d}")
            g.attrs["name"] = name
            g.attrs["key"] = node.key
            g.attrs["cache_hit"] = node.cache_hit
            g.attrs["elapsed_s"] = node.elapsed_s
            g.attrs["diagnostics"] = _diag_list(node.diagnostics)
            outs = g.create_group("outputs")
            if include_signals:
                for j, (port, sig) in enumerate(node.outputs.items()):
                    pg = outs.create_group(f"p{j:04d}")
                    pg.attrs["port"] = port
                    _write_signal(pg, sig, compression)
            res = g.create_group("results")
            for j, (key, value) in enumerate(node.results.items()):
                _write_result(res, f"r{j:04d}", key, value, compression)
    return path


@dataclass
class StoredResults:
    """Contents of a results file, addressable like :class:`SimulationResult`."""

    order: tuple[str, ...]
    seed: int
    trial: int | None
    layout: SimulationLayout | None
    signals: dict[tuple[str, str], Signal] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_diagnostics: dict[str, tuple[Diagnostic, ...]] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    project_json: str = ""
    optobuild_version: str = ""

    def signal(self, node: str, port: str) -> Signal:
        """Stored output signal of ``node.port``."""
        try:
            return self.signals[(node, port)]
        except KeyError:
            raise KeyError(f"No stored signal for {node}.{port}.") from None

    def result(self, node: str, key: str) -> Any:
        """Stored result ``key`` of ``node``."""
        try:
            return self.results[node][key]
        except KeyError:
            raise KeyError(f"No stored result '{key}' for node '{node}'.") from None

    def project(self, registry: ComponentRegistry | None = None) -> Project:
        """Rebuild the embedded project (raises if none was stored)."""
        if not self.project_json:
            raise ProjectFormatError("This results file does not embed its project.")
        return loads_project(self.project_json, registry)


def load_results(path: str | Path) -> StoredResults:
    """Read a file written by :func:`save_results`."""
    h5py = _h5py()
    with h5py.File(Path(path), "r") as f:
        if f.attrs.get("format") != RESULTS_FORMAT:
            raise ProjectFormatError(f"{path} is not an OptoBuild results file.")
        if int(f.attrs.get("schema_version", -1)) != RESULTS_SCHEMA_VERSION:
            raise ProjectFormatError(
                f"Unsupported results schema_version {f.attrs.get('schema_version')!r}.",
                hint="Upgrade OptoBuild.",
            )
        trial = int(f.attrs["trial"])
        layout_text = f.attrs["layout"]
        out = StoredResults(
            order=tuple(json.loads(f.attrs["order"])),
            seed=int(f.attrs["seed"]),
            trial=None if trial < 0 else trial,
            layout=SimulationLayout.from_dict(json.loads(layout_text)) if layout_text else None,
            diagnostics=_diag_from(f.attrs["diagnostics"]),
            project_json=str(f.attrs["project"]),
            optobuild_version=str(f.attrs["optobuild_version"]),
        )
        for gname in sorted(f["nodes"]):
            g = f["nodes"][gname]
            name = str(g.attrs["name"])
            out.node_diagnostics[name] = _diag_from(g.attrs["diagnostics"])
            for pname in sorted(g["outputs"]):
                pg = g["outputs"][pname]
                out.signals[(name, str(pg.attrs["port"]))] = _read_signal(pg)
            out.results[name] = {
                str(g["results"][r].attrs["key"]): _read_result(g["results"][r])
                for r in sorted(g["results"])
            }
    return out


def results_summary(stored: StoredResults | SimulationResult) -> Mapping[str, Any]:
    """Scalar results per node (for quick inspection)."""
    if isinstance(stored, SimulationResult):
        items = {n: dict(stored.nodes[n].results) for n in stored.order}
    else:
        items = stored.results
    return {
        n: {k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for n, r in items.items()
    }


__all__ = [
    "RESULTS_FORMAT",
    "RESULTS_SCHEMA_VERSION",
    "StoredResults",
    "load_results",
    "results_summary",
    "save_results",
]
