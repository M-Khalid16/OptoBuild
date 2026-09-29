from __future__ import annotations

import numpy as np
import pytest

from optobuild.cli.demos import optical_link_project
from optobuild.components.modulators import NRZGenerator
from optobuild.components.sources import CWLaser, PRBSGenerator
from optobuild.core.errors import ComponentExecutionError, InvalidParameterError, SamplingError
from optobuild.engine import FeedForwardExecutor, ResultCache
from optobuild.numerics.layout import SimulationLayout
from optobuild.persistence import run_project
from optobuild.signals import DigitalSequence

LAYOUT = SimulationLayout(bit_rate=25e9, n_bits=127, samples_per_bit=8)


def test_derived_grid() -> None:
    assert LAYOUT.sample_rate == pytest.approx(200e9)
    assert LAYOUT.n_samples == 1016
    g = LAYOUT.grid()
    assert g.n_samples == 1016 and g.dt == pytest.approx(5e-12) and g.t0 == 0.0
    assert SimulationLayout.from_dict(LAYOUT.to_dict()) == LAYOUT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bit_rate": 0.0, "n_bits": 10, "samples_per_bit": 4},
        {"bit_rate": float("inf"), "n_bits": 10, "samples_per_bit": 4},
        {"bit_rate": 1e9, "n_bits": 0, "samples_per_bit": 4},
        {"bit_rate": 1e9, "n_bits": 10, "samples_per_bit": 1},
        {"bit_rate": 1e9, "n_bits": 10.0, "samples_per_bit": 4},
        {"bit_rate": True, "n_bits": 10, "samples_per_bit": 4},
    ],
)
def test_invalid_layouts(kwargs: dict) -> None:
    with pytest.raises(InvalidParameterError, match="Invalid simulation layout"):
        SimulationLayout(**kwargs)
    with pytest.raises(InvalidParameterError):
        SimulationLayout.from_dict({"bit_rate": 1e9})


def test_sources_follow_layout(run_component) -> None:  # type: ignore[no-untyped-def]
    laser = CWLaser("laser", {"timing_source": "layout"})
    sig = run_component(laser, layout=LAYOUT)[0]["out"]
    assert sig.grid.is_compatible(LAYOUT.grid())
    prbs = PRBSGenerator("prbs", {"order": 7, "timing_source": "layout", "bit_rate": 1.0})
    seq = run_component(prbs, layout=LAYOUT)[0]["out"]
    assert seq.n_bits == 127 and seq.bit_rate == 25e9  # own bit_rate is ignored
    nrz = NRZGenerator("nrz", {"timing_source": "layout", "samples_per_bit": 64})
    wf = run_component(nrz, {"bits": seq}, layout=LAYOUT)[0]["out"]
    assert wf.grid.is_compatible(LAYOUT.grid())
    assert wf.metadata["samples_per_bit"] == 8


def test_layout_mode_without_layout_is_explained(run_component) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SamplingError, match="has no layout"):
        run_component(CWLaser("laser", {"timing_source": "layout"}))


def test_nrz_rejects_sequence_inconsistent_with_layout(run_component) -> None:  # type: ignore[no-untyped-def]
    seq = DigitalSequence(np.ones(100, dtype=np.uint8), 25e9)
    nrz = NRZGenerator("nrz", {"timing_source": "layout"})
    with pytest.raises(SamplingError, match="layout specifies 127 bits"):
        run_component(nrz, {"bits": seq}, layout=LAYOUT)


def test_prbs_warns_when_layout_is_not_a_period(run_component) -> None:  # type: ignore[no-untyped-def]
    layout = SimulationLayout(10e9, 100, 4)
    _, ctx = run_component(PRBSGenerator("p", {"timing_source": "layout"}), layout=layout)
    assert [d.code for d in ctx.diagnostics] == ["sampling.pattern_periodicity"]


def test_laser_offset_checked_against_layout(run_component) -> None:  # type: ignore[no-untyped-def]
    # 150 GHz exceeds the laser's own (unused) 160 GS/s band but fits the layout at 200 GS/s
    laser = CWLaser("l", {"timing_source": "layout", "frequency_offset": 150e9})
    with pytest.raises(SamplingError, match="outside the layout"):
        run_component(laser, layout=SimulationLayout(25e9, 127, 4))  # fs/2 = 50 GHz
    big = SimulationLayout(25e9, 127, 16)  # fs/2 = 200 GHz, df = 25 GHz / 127
    assert run_component(laser, layout=big)[1].diagnostics == []  # 150 GHz = 762 df
    leaky = CWLaser("l2", {"timing_source": "layout", "frequency_offset": 150.1e9})
    _, ctx = run_component(leaky, layout=big)
    assert [d.code for d in ctx.diagnostics] == ["laser.offset_not_periodic"]
    ok = CWLaser("ok", {"timing_source": "layout", "frequency_offset": 25e9 / 127 * 10})
    assert run_component(ok, layout=LAYOUT)[1].diagnostics == []


def test_changing_layout_changes_the_whole_link() -> None:
    p = optical_link_project(prbs_order=7)
    a = run_project(p)
    p.layout = SimulationLayout(p.layout.bit_rate, p.layout.n_bits, 8)
    b = run_project(p)
    assert a.signal("laser", "out").grid.n_samples == 127 * 16
    assert b.signal("laser", "out").grid.n_samples == 127 * 8
    assert b.signal("filter", "out").metadata["samples_per_bit"] == 8


def test_layout_is_part_of_the_cache_key() -> None:
    p = optical_link_project(prbs_order=7)
    ex = FeedForwardExecutor(ResultCache())
    run_project(p, executor=ex)
    p.layout = SimulationLayout(p.layout.bit_rate, p.layout.n_bits, 8)
    res = run_project(p, executor=ex)
    assert not any(n.cache_hit for n in res.nodes.values())


def test_link_without_layout_fails_clearly() -> None:
    p = optical_link_project(prbs_order=7)
    with pytest.raises(ComponentExecutionError, match="no layout"):
        FeedForwardExecutor().run(p.graph, seed=0)
