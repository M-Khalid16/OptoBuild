"""The list of built-in component classes (consumed by ``builtin_registry``)."""

from __future__ import annotations

from optobuild.components.base import Component
from optobuild.components.reference import REFERENCE_COMPONENTS

BUILTIN_COMPONENTS: tuple[type[Component], ...] = (*REFERENCE_COMPONENTS,)

__all__ = ["BUILTIN_COMPONENTS"]
