"""Third-party component plugins via Python entry points (ADR-0021).

A plugin is an installed Python package that declares, in its packaging
metadata, entry points in the group ``optobuild.components``::

    [project.entry-points."optobuild.components"]
    my_components = "my_package.optobuild_plugin:COMPONENTS"

The referenced object is a ``Component`` subclass or an iterable of them.
Loading is explicit (``load_plugins`` / ``plugin_registry``, the CLI's
``--plugins`` flag): plugins are ordinary Python code with full access to the
process, so only packages the user installed are ever imported, and project
files can never name code to load (they only reference registered
``type_id`` strings). A plugin that fails to import, exports something else,
or reuses an existing ``type_id`` is reported and skipped; it cannot replace a
built-in component.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from optobuild.components.base import Component
from optobuild.components.registry import ComponentRegistry, builtin_registry

ENTRY_POINT_GROUP = "optobuild.components"


@dataclass(frozen=True)
class PluginReport:
    """Outcome of loading one entry point."""

    name: str
    value: str
    type_ids: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def discover(group: str = ENTRY_POINT_GROUP) -> tuple[EntryPoint, ...]:
    """Entry points of installed packages in ``group``."""
    return tuple(entry_points(group=group))


def _classes(obj: object) -> list[type[Component]]:
    if isinstance(obj, type):
        items: Iterable[object] = [obj]
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        items = obj
    else:
        raise TypeError(f"exports {type(obj).__name__}, not Component classes")
    out = []
    for item in items:
        if not (isinstance(item, type) and issubclass(item, Component)):
            raise TypeError(f"{item!r} is not a Component subclass")
        out.append(item)
    if not out:
        raise TypeError("exports no components")
    return out


def load_plugins(
    registry: ComponentRegistry, points: Sequence[EntryPoint] | None = None
) -> list[PluginReport]:
    """Register the components of every entry point; never raises for a bad plugin."""
    reports = []
    for ep in discover() if points is None else points:
        try:
            classes = _classes(ep.load())
            taken = [c.type_id for c in classes if c.type_id in registry]
            if taken:
                raise ValueError(f"type_id already registered: {', '.join(taken)}")
            if len({c.type_id for c in classes}) != len(classes):
                raise ValueError("duplicate type_id within the plugin")
            for cls in classes:
                registry.register(cls)
            reports.append(PluginReport(ep.name, ep.value, tuple(c.type_id for c in classes)))
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop the program
            reports.append(PluginReport(ep.name, ep.value, error=f"{type(exc).__name__}: {exc}"))
    return reports


def plugin_registry(
    points: Sequence[EntryPoint] | None = None,
) -> tuple[ComponentRegistry, list[PluginReport]]:
    """Built-in registry extended with all installed plugins."""
    registry = builtin_registry()
    return registry, load_plugins(registry, points)


__all__ = ["ENTRY_POINT_GROUP", "PluginReport", "discover", "load_plugins", "plugin_registry"]
