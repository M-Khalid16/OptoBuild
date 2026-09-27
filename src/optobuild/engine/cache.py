"""Result cache keyed by a content hash of each node's computation (ADR-0005).

A node's key hashes (type_id, version, name, canonical parameters, root seed,
layout, trial index, and the keys of the upstream outputs it consumes). Because components are
deterministic functions of exactly these inputs, an unchanged key implies an
unchanged result; changing a parameter changes the key of that node and of
everything downstream, which is the invalidation rule.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from optobuild.components.base import Component
from optobuild.numerics.layout import SimulationLayout


def node_key(
    component: Component,
    root_seed: int,
    inputs: Mapping[str, tuple[str, str]],
    *,
    layout: SimulationLayout | None = None,
    trial: int | None = None,
) -> str:
    """Hash identifying one node computation; ``inputs`` maps port -> (upstream key, port).

    The layout and trial index are part of every key: any component may read
    the layout, and the trial selects the random realization.
    """
    payload = {
        "type_id": component.type_id,
        "version": component.version,
        "name": component.name,
        "parameters": dict(component.parameters),
        "seed": root_seed,
        "inputs": {k: list(v) for k, v in sorted(inputs.items())},
        "layout": None if layout is None else layout.to_dict(),
        "trial": trial,
    }
    text = json.dumps(payload, sort_keys=True, default=repr, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResultCache:
    """Bounded LRU cache of node results."""

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1.")
        self.max_entries = max_entries
        self._data: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Cached value for ``key`` or ``None``."""
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Store a value, evicting the least recently used entries beyond the bound."""
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        """Remove every entry."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


__all__ = ["ResultCache", "node_key"]
