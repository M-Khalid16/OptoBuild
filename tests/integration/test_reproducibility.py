"""Reproducibility under fixed seeds (ADR-0008)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from optobuild.cli.demos import reference_project
from optobuild.components.reference import GaussianNoise, RampSource, Recorder
from optobuild.engine import FeedForwardExecutor
from optobuild.persistence import dumps_project, loads_project


def _noisy(seed: int, graph=None):  # type: ignore[no-untyped-def]
    g = reference_project().graph if graph is None else graph
    return FeedForwardExecutor().run(g, seed=seed).result("noisy", "samples")


def test_same_seed_identical_results() -> None:
    np.testing.assert_array_equal(_noisy(11), _noisy(11))


def test_different_seed_different_results() -> None:
    assert not np.array_equal(_noisy(11), _noisy(12))


def test_adding_unrelated_random_component_does_not_change_existing_streams() -> None:
    g = reference_project().graph
    before = _noisy(3, g)
    g.add(RampSource("other_src"))
    g.add(GaussianNoise("aaa_other_noise", {"sigma": 5.0}))  # runs earlier in any order
    g.add(Recorder("other_rec"))
    g.connect("other_src", "out", "aaa_other_noise", "in")
    g.connect("aaa_other_noise", "out", "other_rec", "in")
    np.testing.assert_array_equal(_noisy(3, g), before)


def test_results_identical_across_processes() -> None:
    """A fresh interpreter (different hash salt) reproduces the same samples bit-exactly."""
    text = dumps_project(reference_project(seed=2024))
    code = (
        "import sys\n"
        "from optobuild.persistence import loads_project\n"
        "from optobuild.engine import FeedForwardExecutor\n"
        "p = loads_project(sys.stdin.read())\n"
        "r = FeedForwardExecutor().run(p.graph, seed=p.seed)\n"
        "sys.stdout.write(r.result('noisy', 'samples').tobytes().hex())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "12345"},
    ).stdout
    local = FeedForwardExecutor().run(loads_project(text).graph, seed=2024)
    assert out == local.result("noisy", "samples").tobytes().hex()
