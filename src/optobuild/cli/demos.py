"""Built-in demonstration projects used by the CLI and examples."""

from __future__ import annotations

from optobuild.components.reference import Adder, Gain, GaussianNoise, RampSource, Recorder
from optobuild.graph.model import SimulationGraph
from optobuild.persistence.project import Project


def reference_project(seed: int = 1234) -> Project:
    """Framework reference pipeline (no optical physics).

    ::

        ramp (v_n = n) --> gain x2 --> adder.a
        dc   (v_n = 10) ------------> adder.b
        adder --> recorder "clean"                  (expected v_n = 2 n + 10 exactly)
        adder --> noise (sigma 0.1 V) --> recorder "noisy"
    """
    g = SimulationGraph()
    g.add(RampSource("ramp", {"n_samples": 16, "sample_rate": 1e9, "offset": 0.0, "step": 1.0}))
    g.add(RampSource("dc", {"n_samples": 16, "sample_rate": 1e9, "offset": 10.0, "step": 0.0}))
    g.add(Gain("gain", {"gain": 2.0}))
    g.add(Adder("adder"))
    g.add(GaussianNoise("noise", {"sigma": 0.1}))
    g.add(Recorder("clean"))
    g.add(Recorder("noisy"))
    g.connect("ramp", "out", "gain", "in")
    g.connect("gain", "out", "adder", "a")
    g.connect("dc", "out", "adder", "b")
    g.connect("adder", "out", "clean", "in")
    g.connect("adder", "out", "noise", "in")
    g.connect("noise", "out", "noisy", "in")
    return Project(graph=g, seed=seed, metadata={"title": "Framework reference pipeline"})


DEMOS = {"reference": reference_project}

__all__ = ["DEMOS", "reference_project"]
