"""Project files (ADR-0006): versioned JSON (optionally YAML); never pickle.

Schema version 1::

    {
      "format": "optobuild-project",
      "schema_version": 1,
      "optobuild_version": "x.y.z",
      "metadata": {...},                      # free-form JSON object
      "simulation": {"seed": 0},
      "components": [
        {"name": "...", "type_id": "...", "version": "...",
         "parameters": {"p": value, ...},     # SI values
         "units": {"p": "V", ...}}            # SI unit of each numeric parameter
      ],
      "connections": [{"from": ["node", "port"], "to": ["node", "port"]}],
      "layout": {...}                         # GUI-only; ignored by the engine
    }

Loading is strict: unknown keys, unknown component types, unit mismatches
and unsupported schema versions raise :class:`ProjectFormatError`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optobuild
from optobuild.components.registry import ComponentRegistry, builtin_registry
from optobuild.components.spec import ParameterType
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import OptoBuildError, ProjectFormatError
from optobuild.core.rng import validate_seed
from optobuild.graph.model import SimulationGraph

FORMAT_NAME = "optobuild-project"
SCHEMA_VERSION = 1
_TOP_KEYS = {
    "format",
    "schema_version",
    "optobuild_version",
    "metadata",
    "simulation",
    "components",
    "connections",
    "layout",
}
_COMPONENT_KEYS = {"name", "type_id", "version", "parameters", "units"}


@dataclass
class Project:
    """A simulation graph plus the settings needed to reproduce a run."""

    graph: SimulationGraph
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    """Findings from loading (e.g. component version differences)."""


def project_to_dict(project: Project) -> dict[str, Any]:
    """Plain JSON-compatible representation of ``project``."""
    components = []
    for comp in project.graph.nodes:
        params = dict(comp.parameters)
        units = {
            s.name: s.unit
            for s in comp.parameter_specs
            if s.type in (ParameterType.FLOAT, ParameterType.INT)
        }
        components.append(
            {
                "name": comp.name,
                "type_id": comp.type_id,
                "version": comp.version,
                "parameters": params,
                "units": units,
            }
        )
    return {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "optobuild_version": optobuild.__version__,
        "metadata": dict(project.metadata),
        "simulation": {"seed": validate_seed(project.seed)},
        "components": components,
        "connections": [
            {"from": [c.source.node, c.source.port], "to": [c.target.node, c.target.port]}
            for c in project.graph.connections
        ],
        "layout": dict(project.layout),
    }


def _fail(message: str, hint: str | None = None) -> ProjectFormatError:
    return ProjectFormatError(message, hint=hint)


def _require(obj: Any, typ: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(obj, typ):
        raise _fail(f"{where} must be of type {getattr(typ, '__name__', typ)}, got {obj!r}.")
    return obj


def project_from_dict(
    data: Mapping[str, Any], registry: ComponentRegistry | None = None
) -> Project:
    """Rebuild a :class:`Project` from :func:`project_to_dict` output (strictly validated)."""
    registry = builtin_registry() if registry is None else registry
    _require(data, Mapping, "Project")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise _fail(f"Unknown top-level project keys: {sorted(unknown)}.")
    if data.get("format") != FORMAT_NAME:
        raise _fail(f"Not an OptoBuild project (format={data.get('format')!r}).")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise _fail(
            f"Unsupported project schema_version {version!r}; this build reads {SCHEMA_VERSION}.",
            hint="Upgrade OptoBuild or migrate the project file.",
        )
    diagnostics: list[Diagnostic] = []
    simulation = _require(data.get("simulation", {}), Mapping, "'simulation'")
    if set(simulation) - {"seed"}:
        raise _fail(f"Unknown simulation keys: {sorted(set(simulation) - {'seed'})}.")
    try:
        seed = validate_seed(simulation.get("seed", 0))
    except OptoBuildError as exc:
        raise _fail(f"Invalid seed: {exc.message}") from exc

    graph = SimulationGraph()
    for i, entry in enumerate(_require(data.get("components", []), list, "'components'")):
        where = f"components[{i}]"
        _require(entry, Mapping, where)
        if set(entry) - _COMPONENT_KEYS:
            raise _fail(f"{where} has unknown keys {sorted(set(entry) - _COMPONENT_KEYS)}.")
        name = _require(entry.get("name"), str, f"{where}.name")
        type_id = _require(entry.get("type_id"), str, f"{where}.type_id")
        cls = registry.get(type_id)
        params = dict(_require(entry.get("parameters", {}), Mapping, f"{where}.parameters"))
        units = _require(entry.get("units", {}), Mapping, f"{where}.units")
        for pname, unit in units.items():
            spec = next((s for s in cls.parameter_specs if s.name == pname), None)
            if spec is not None and spec.unit != unit:
                raise _fail(
                    f"{where} ('{name}'): parameter '{pname}' is stored in {unit!r} but "
                    f"{type_id} expects SI unit {spec.unit!r}.",
                    hint="Project files store SI values; convert the value and unit.",
                )
        saved_version = entry.get("version")
        if saved_version is not None and saved_version != cls.version:
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "project.component_version",
                    f"Saved with {type_id} version {saved_version}, loaded with {cls.version}; "
                    "numerical results may differ.",
                    source=name,
                )
            )
        try:
            graph.add(cls(name, params))
        except OptoBuildError as exc:
            raise _fail(f"{where} ('{name}'): {exc}") from exc

    for i, entry in enumerate(_require(data.get("connections", []), list, "'connections'")):
        where = f"connections[{i}]"
        _require(entry, Mapping, where)
        if set(entry) != {"from", "to"}:
            raise _fail(f"{where} must have exactly the keys 'from' and 'to'.")
        src, dst = entry["from"], entry["to"]
        for label, ep in (("from", src), ("to", dst)):
            if not (isinstance(ep, list) and len(ep) == 2 and all(isinstance(x, str) for x in ep)):
                raise _fail(f"{where}.{label} must be [node, port], got {ep!r}.")
        try:
            graph.connect(src[0], src[1], dst[0], dst[1])
        except OptoBuildError as exc:
            raise _fail(f"{where}: {exc}") from exc

    return Project(
        graph=graph,
        seed=seed,
        metadata=dict(_require(data.get("metadata", {}), Mapping, "'metadata'")),
        layout=dict(_require(data.get("layout", {}), Mapping, "'layout'")),
        diagnostics=diagnostics,
    )


def _is_yaml(path: Path) -> bool:
    return path.suffix.lower() in (".yaml", ".yml")


def dumps_project(project: Project) -> str:
    """Serialize to JSON text (NaN/Inf rejected)."""
    data = project_to_dict(project)
    try:
        return json.dumps(data, indent=2, allow_nan=False) + "\n"
    except ValueError as exc:
        raise _fail(f"Project contains non-finite numbers: {exc}") from exc


def loads_project(text: str, registry: ComponentRegistry | None = None) -> Project:
    """Parse JSON text produced by :func:`dumps_project`."""

    def _reject_constant(name: str) -> float:
        raise _fail(f"Non-finite number {name} is not allowed in project files.")

    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise _fail(f"Invalid JSON: {exc}") from exc
    return project_from_dict(data, registry)


def save_project(project: Project, path: str | Path) -> Path:
    """Write ``project`` to ``path`` (JSON, or YAML for .yaml/.yml)."""
    path = Path(path)
    if _is_yaml(path):
        yaml = _import_yaml()
        data = project_to_dict(project)
        _check_finite(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(dumps_project(project), encoding="utf-8")
    return path


def load_project(path: str | Path, registry: ComponentRegistry | None = None) -> Project:
    """Read a project written by :func:`save_project`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if _is_yaml(path):
        yaml = _import_yaml()
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise _fail(f"Invalid YAML in {path}: {exc}") from exc
        _check_finite(data)
        return project_from_dict(data, registry)
    return loads_project(text, registry)


def _check_finite(obj: Any) -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise _fail("Non-finite numbers are not allowed in project files.")
    if isinstance(obj, Mapping):
        for v in obj.values():
            _check_finite(v)
    elif isinstance(obj, list):
        for v in obj:
            _check_finite(v)


def _import_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise _fail("YAML support requires PyYAML.", hint="pip install 'optobuild[yaml]'") from exc
    return yaml


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "Project",
    "dumps_project",
    "load_project",
    "loads_project",
    "project_from_dict",
    "project_to_dict",
    "save_project",
]
