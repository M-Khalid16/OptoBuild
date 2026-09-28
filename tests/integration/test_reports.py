"""HTML reports (Phase 10): content, escaping, self-containment and reproducibility."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import numpy as np

from optobuild.cli.demos import coherent_link_project, reference_project
from optobuild.optimization.optimizer import Objective, Variable, optimize
from optobuild.persistence import run_project
from optobuild.reporting.figures import Curve
from optobuild.reporting.html import optimization_report, run_report, sweep_report
from optobuild.reporting.svg import MAX_SCATTER_POINTS, nice_ticks, render_curve
from optobuild.sweeps.parameter_sweep import Axis, Probe, sweep


class _Checker(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link", "circle", "rect", "line", "polyline"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.tags: set[str] = set()

    def handle_starttag(self, tag, attrs):  # type: ignore[no-untyped-def]
        self.tags.add(tag)
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):  # type: ignore[no-untyped-def]
        self.tags.add(tag)

    def handle_endtag(self, tag):  # type: ignore[no-untyped-def]
        assert self.stack and self.stack[-1] == tag, f"unbalanced </{tag}>"
        self.stack.pop()


def _check(html: str) -> set[str]:
    c = _Checker()
    c.feed(html)
    assert not c.stack, f"unclosed tags {c.stack}"
    assert "script" not in c.tags and "iframe" not in c.tags
    # no external resources: the only URL is the SVG namespace
    assert set(re.findall(r"https?://[^\s\"'<]+", html)) <= {"http://www.w3.org/2000/svg"}
    return c.tags


def test_run_report_contents_and_escaping() -> None:
    project = reference_project()
    project.metadata["title"] = "<script>alert(1)</script> & test"
    result = run_project(project)
    html = run_report(project, result, timestamp=False)
    _check(html)
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; test" in html
    assert "<script>" not in html
    assert "gain = 2" in html and "clean" in html and "mean" in html
    assert '"version"' in html or "schema" in html  # embedded project file
    assert run_report(project, result, timestamp=False) == html  # reproducible


def test_coherent_report_has_constellation_figure() -> None:
    project = coherent_link_project(prbs_order=11)
    html = run_report(project, run_project(project), timestamp=False)
    tags = _check(html)
    assert "svg" in tags and "circle" in tags
    assert "analyzer: constellation" in html and "snr_db" in html


def test_sweep_and_optimization_reports() -> None:
    res = sweep(reference_project(), [Axis("gain", "gain", (1.0, 2.0, 3.0))],
                [Probe("clean", "mean")])  # fmt: skip
    html = sweep_report(res, timestamp=False)
    tags = _check(html)
    assert "polyline" in tags and "gain.gain" in html
    opt = optimize(reference_project(), [Variable("gain", "gain", -5.0, 5.0)],
                   Objective("clean", "rms", "min"))  # fmt: skip
    html = optimization_report(opt, timestamp=False)
    _check(html)
    assert "gain.gain" in html and "History" in html


def test_svg_ticks_decimation_and_empty_data() -> None:
    np.testing.assert_allclose(nice_ticks(0.0, 1.0), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    many = np.linspace(0, 1, 3 * MAX_SCATTER_POINTS)
    svg = render_curve(Curve(many, many, "x", "y", "t", scatter=True))
    assert svg.count("<circle") <= MAX_SCATTER_POINTS and "1 in 3 points shown" in svg
    empty = render_curve(Curve(np.array([np.nan]), np.array([np.nan]), "x", "y"))
    assert "no finite data" in empty
