"""Component registry: resolves stable ``type_id`` strings to component classes.

Registries are ordinary objects (no global mutable singleton). The built-in
library is obtained with :func:`builtin_registry`, which returns a fresh
registry each call; plugins (Phase 10) will add entry-point discovery.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from optobuild.components.base import Component
from optobuild.core.errors import ProjectFormatError


class ComponentRegistry:
    """Mapping ``type_id -> Component subclass``."""

    def __init__(self, classes: Iterable[type[Component]] = ()) -> None:
        self._classes: dict[str, type[Component]] = {}
        for cls in classes:
            self.register(cls)

    def register(self, cls: type[Component]) -> type[Component]:
        """Register a concrete component class (usable as a decorator)."""
        if not (isinstance(cls, type) and issubclass(cls, Component)):
            raise TypeError(f"{cls!r} is not a Component subclass.")
        if getattr(cls, "__abstractmethods__", None):
            raise TypeError(f"{cls.__name__} is abstract and cannot be registered.")
        existing = self._classes.get(cls.type_id)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"type_id {cls.type_id!r} is already registered to {existing.__qualname__}."
            )
        self._classes[cls.type_id] = cls
        return cls

    def get(self, type_id: str) -> type[Component]:
        """Class registered for ``type_id``; raises ``ProjectFormatError`` if unknown."""
        try:
            return self._classes[type_id]
        except KeyError:
            raise ProjectFormatError(
                f"Unknown component type {type_id!r}.",
                hint="Check the spelling or load the plugin that provides this component.",
            ) from None

    def create(
        self, type_id: str, name: str, parameters: Mapping[str, Any] | None = None
    ) -> Component:
        """Instantiate a registered component (parameters validated)."""
        return self.get(type_id)(name, parameters)

    def __contains__(self, type_id: object) -> bool:
        return type_id in self._classes

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._classes))

    def __len__(self) -> int:
        return len(self._classes)

    def describe(self, type_id: str) -> dict[str, Any]:
        """Plain-data description of a component class (for CLI/GUI/docs)."""
        cls = self.get(type_id)
        return {
            "type_id": cls.type_id,
            "version": cls.version,
            "display_name": cls.display_name,
            "category": cls.category.value,
            "inputs": [(p.name, p.kind.value) for p in cls.input_ports],
            "outputs": [(p.name, p.kind.value) for p in cls.output_ports],
            "parameters": [
                {
                    "name": s.name,
                    "type": s.type.value,
                    "default": s.default,
                    "unit": s.unit,
                    "display_unit": s.display_unit,
                    "description": s.description,
                }
                for s in cls.parameter_specs
            ],
            "doc": cls.documentation(),
        }


def builtin_registry() -> ComponentRegistry:
    """A new registry containing every built-in component."""
    from optobuild.components.library import BUILTIN_COMPONENTS

    return ComponentRegistry(BUILTIN_COMPONENTS)


__all__ = ["ComponentRegistry", "builtin_registry"]
