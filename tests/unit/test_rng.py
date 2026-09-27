from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from optobuild.core import rng
from optobuild.core.errors import InvalidParameterError


def test_stable_id_is_process_independent() -> None:
    """Python's hash() is salted per process; stable_id must not be."""
    code = "from optobuild.core.rng import stable_id; print(stable_id('noise'))"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": str(s)},
        ).stdout.strip()
        for s in (1, 2)
    }
    assert outputs == {str(rng.stable_id("noise"))}


def test_same_inputs_same_stream() -> None:
    a = rng.component_generator(7, "noise").normal(size=100)
    b = rng.component_generator(7, "noise").normal(size=100)
    np.testing.assert_array_equal(a, b)


def test_different_name_seed_or_trial_changes_stream() -> None:
    base = rng.component_generator(7, "noise").normal(size=100)
    for other in (
        rng.component_generator(8, "noise"),
        rng.component_generator(7, "noise2"),
        rng.component_generator(7, "noise", 1),
    ):
        assert not np.array_equal(base, other.normal(size=100))


def test_streams_are_statistically_independent() -> None:
    """Correlation of two 1e5-sample streams should be ~N(0, 1/sqrt(n)); 5 sigma bound."""
    n = 100_000
    a = rng.component_generator(0, "a").normal(size=n)
    b = rng.component_generator(0, "b").normal(size=n)
    assert abs(np.corrcoef(a, b)[0, 1]) < 5 / np.sqrt(n)


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True, 2**128])
def test_invalid_seeds(bad: object) -> None:
    with pytest.raises(InvalidParameterError):
        rng.validate_seed(bad)
