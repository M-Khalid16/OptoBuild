"""Self-contained HTML reports of simulation runs, sweeps and optimizations (ADR-0021).

A report contains only what the simulator computed: project metadata, seed,
layout, every component with its parameters in display units, the
connections, all scalar results, all diagnostics, figures of the recorded
data (eye diagrams, spectra, constellations, device responses,
autocorrelations) as inline SVG, and the project file itself for provenance.
No external resources, scripts or network access; every text is escaped.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from html import escape
from typing import Any

import numpy as np

import optobuild
from optobuild.core.units import from_si
from optobuild.engine.executor import SimulationResult
from optobuild.persistence.project import Project, dumps_project
from optobuild.reporting.figures import (
    Curve,
    constellation_curve,
    eye_curve,
    scalar_rows,
    spectrum_curve,
    transfer_curve,
    transfer_ports,
)
from optobuild.reporting.svg import render_curve

_CSS = """
body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;color:#222}
h1{font-size:22px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #ccc}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 7px;
text-align:left;vertical-align:top}th{background:#f3f3f3}.warning{color:#b35c00}
.error{color:#c92a2a}figure{display:inline-block;margin:8px}pre{font-size:11px;
background:#f7f7f7;padding:8px;overflow:auto;max-height:400px}
"""


def _table(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def _format_value(value: Any, unit: str | None) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if unit:
        try:
            shown = float(from_si(value, unit))
            if math.isfinite(shown):
                return f"{shown:.6g} {unit}"
        except Exception:  # noqa: BLE001 - display only; fall back to SI
            pass
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _component_rows(project: Project) -> list[list[str]]:
    rows = []
    for comp in project.graph.nodes:
        params = []
        for spec in type(comp).parameter_specs:
            unit = spec.display_unit or (spec.unit if spec.unit not in (None, "1") else None)
            params.append(f"{spec.name} = {_format_value(comp.parameters[spec.name], unit)}")
        rows.append([comp.name, f"{comp.type_id} {comp.version}", "; ".join(params)])
    return rows


def _figures(result: SimulationResult) -> list[Curve]:
    curves = []
    for n in result.order:
        res = dict(result.nodes[n].results)
        if "traces" in res:
            c = eye_curve(res, str(res.get("unit", "A")))
            curves.append(_retitled(c, f"{n}: {c.title}"))
        if "power_per_rbw_w" in res:
            c = spectrum_curve(res)
            curves.append(_retitled(c, f"{n}: {c.title}"))
        if "constellation" in res:
            curves.append(constellation_curve(res["constellation"], f"{n}: constellation"))
        for port in transfer_ports(res):
            c = transfer_curve(res, port)
            curves.append(_retitled(c, f"{n}: {c.title}"))
        if "autocorrelation" in res and "delay_s" in res:
            tau = np.asarray(res["delay_s"]) * 1e12
            ac = np.asarray(res["autocorrelation"])
            label = f"{n}: intensity autocorrelation"
            curves.append(Curve(tau, ac, "delay [ps]", "autocorrelation (normalized)", label))
    return curves


def _retitled(curve: Curve, title: str) -> Curve:
    return Curve(curve.x, curve.y, curve.x_label, curve.y_label, title, curve.scatter)


def _page(title: str, body: str, generated: str | None) -> str:
    stamp = f" &middot; generated {escape(generated)}" if generated else ""
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1><p>OptoBuild {escape(optobuild.__version__)}{stamp}</p>"
        f"{body}</body></html>\n"
    )


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def run_report(
    project: Project,
    result: SimulationResult,
    *,
    title: str | None = None,
    timestamp: bool = True,
) -> str:
    """HTML report of one run (``timestamp=False`` gives byte-reproducible output)."""
    title = title or str(project.metadata.get("title", "Simulation report"))
    parts = ["<h2>Run</h2>"]
    run_rows = [["seed", str(result.seed)], ["trial", str(result.trial)]]
    if result.layout is not None:
        lay = result.layout
        run_rows += [
            ["bit rate", f"{lay.bit_rate:.6g} bit/s"],
            ["bits", str(lay.n_bits)],
            ["samples per bit", str(lay.samples_per_bit)],
        ]
    parts.append(_table(["quantity", "value"], run_rows))
    parts.append("<h2>Components</h2>")
    parts.append(_table(["name", "type", "parameters"], _component_rows(project)))
    parts.append("<h2>Connections</h2>")
    parts.append(_table(["from", "to"], [[str(c.source), str(c.target)]
                                          for c in project.graph.connections]))  # fmt: skip
    parts.append("<h2>Results</h2>")
    parts.append(_table(["component", "result", "value"], [list(r) for r in scalar_rows(result)]))
    diags = result.all_diagnostics()
    parts.append(f"<h2>Diagnostics ({len(diags)})</h2>")
    if diags:
        items = "".join(
            f'<li class="{escape(d.severity.name.lower())}">{escape(str(d))}</li>' for d in diags
        )
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append("<p>None.</p>")
    curves = _figures(result)
    if curves:
        parts.append("<h2>Figures</h2>")
        parts += [f"<figure>{render_curve(c)}</figure>" for c in curves]
    parts.append("<h2>Project file</h2><details><summary>JSON</summary><pre>")
    parts.append(escape(dumps_project(project)))
    parts.append("</pre></details>")
    return _page(title, "".join(parts), _now() if timestamp else None)


def sweep_report(sweep: Any, *, title: str = "Parameter sweep", timestamp: bool = True) -> str:
    """HTML report of a ``SweepResult``: table of trial means and, for one axis, a plot of
    every probe versus the axis."""
    parts = ["<h2>Axes</h2>"]
    parts.append(_table(["parameter", "values (SI)"],
                        [[a.label, ", ".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                                             for v in a.values)] for a in sweep.axes]))  # fmt: skip
    parts.append(
        f"<p>Trials per point: {len(sweep.trials)} (indices {escape(str(sweep.trials))}).</p>"
    )
    parts.append("<h2>Results (mean over trials)</h2><pre>")
    parts.append(escape(sweep.to_csv()))
    parts.append("</pre>")
    if len(sweep.axes) == 1 and all(isinstance(v, (int, float)) for v in sweep.axes[0].values):
        x = np.asarray(sweep.axes[0].values, dtype=float)
        for p in sweep.probes:
            c = Curve(x, sweep.mean(p.label), sweep.axes[0].label + " (SI)", p.label, p.label)
            parts.append(f"<figure>{render_curve(c)}</figure>")
    return _page(title, "".join(parts), _now() if timestamp else None)


def optimization_report(opt: Any, *, title: str = "Optimization", timestamp: bool = True) -> str:
    """HTML report of an ``OptimizationResult`` (optimum and evaluation history)."""
    parts = ["<h2>Optimum</h2>"]
    rows = [[k, f"{v:.9g}"] for k, v in opt.x.items()]
    rows.append([f"{opt.objective.label} ({opt.objective.sense})", f"{opt.value:.9g}"])
    rows.append(["evaluations", str(opt.n_evaluations)])
    rows.append(["converged", f"{opt.success} ({opt.message})"])
    parts.append(_table(["quantity", "value"], rows))
    parts.append("<h2>History</h2>")
    head = [v.label for v in opt.variables] + [opt.objective.label]
    parts.append(_table(head, [[f"{x:.9g}" for x in xs] + [f"{f:.9g}"] for xs, f in opt.history]))
    values = np.array([f for _, f in opt.history], dtype=float)
    c = Curve(np.arange(1, values.size + 1, dtype=float), values, "evaluation",
              opt.objective.label, "Objective per evaluation")  # fmt: skip
    parts.append(f"<figure>{render_curve(c)}</figure>")
    parts.append("<h2>Optimized project</h2><details><summary>JSON</summary><pre>")
    parts.append(escape(json.dumps(json.loads(dumps_project(opt.project)), indent=2)))
    parts.append("</pre></details>")
    return _page(title, "".join(parts), _now() if timestamp else None)


__all__ = ["optimization_report", "run_report", "sweep_report"]
