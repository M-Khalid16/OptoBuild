"""Deterministic random-number framework (ADR-0008).

Each component instance receives its own generator derived from the project
root seed and a *stable* hash of the instance name, so its random stream does
not depend on execution order or on other components in the graph::

    SeedSequence(entropy=root_seed, spawn_key=(stable_id(name), *extra))

``stable_id`` uses SHA-256 (Python's built-in ``hash`` is salted per process).
Monte Carlo trials extend the spawn key with the trial index.
"""

from __future__ import annotations

import hashlib

import numpy as np

from optobuild.core.errors import InvalidParameterError

MAX_SEED = 2**128 - 1


def validate_seed(seed: object) -> int:
    """Return ``seed`` as int if it is an integer in [0, 2**128); raise otherwise."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise InvalidParameterError(
            f"Seed must be an integer, got {seed!r}.", hint="Use a non-negative integer seed."
        )
    if not 0 <= int(seed) <= MAX_SEED:
        raise InvalidParameterError(
            f"Seed must be in [0, 2**128 - 1], got {seed}.", hint="Use a non-negative integer."
        )
    return int(seed)


def stable_id(name: str) -> int:
    """Process-independent 64-bit integer identifier of a string."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def seed_sequence(root_seed: int, name: str, *extra: int) -> np.random.SeedSequence:
    """SeedSequence for component ``name`` (and optional trial indices ``extra``)."""
    root = validate_seed(root_seed)
    key = (stable_id(name), *(int(e) for e in extra))
    return np.random.SeedSequence(entropy=root, spawn_key=key)


def component_generator(root_seed: int, name: str, *extra: int) -> np.random.Generator:
    """Independent PCG64 generator for component ``name`` under ``root_seed``."""
    return np.random.Generator(np.random.PCG64(seed_sequence(root_seed, name, *extra)))


__all__ = ["MAX_SEED", "component_generator", "seed_sequence", "stable_id", "validate_seed"]
