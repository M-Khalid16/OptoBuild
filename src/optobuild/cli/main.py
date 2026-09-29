"""Command-line interface.

Examples::

    optobuild components                 # list built-in components
    optobuild demo reference             # run a built-in demo project
    optobuild demo reference --save p.json
    optobuild run p.json --seed 7        # run a project file
    optobuild report demo:coherent_link -o report.html
    optobuild sweep demo:reference --axis gain.gain=1,2,3 --probe clean.mean
    optobuild sweep p.json --axis fiber.length=0:50e3:6 --probe ber.ber --trials 4 --workers 4
    optobuild optimize demo:reference --var gain.gain=-5:5 --minimize clean.rms
    optobuild --plugins components       # include installed plugin components
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

import optobuild
from optobuild.cli.demos import DEMOS
from optobuild.components.registry import ComponentRegistry, builtin_registry
from optobuild.core.errors import OptoBuildError
from optobuild.core.log import configure_logging
from optobuild.engine.executor import SimulationResult
from optobuild.persistence.project import Project, load_project, run_project, save_project


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return np.array2string(value, precision=6)
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, float):
        return f"{value:.6g}"
    return repr(value)


def print_result(result: SimulationResult, out: Any = None) -> None:
    """Human-readable summary of a run (to ``out``, default: current ``sys.stdout``)."""
    out = sys.stdout if out is None else out
    print(f"seed: {result.seed}", file=out)
    print("execution order: " + " -> ".join(result.order), file=out)
    for name in result.order:
        node = result.nodes[name]
        for key, value in node.results.items():
            print(f"  {name}.{key} = {_format_value(value)}", file=out)
    diags = result.all_diagnostics()
    if diags:
        print("diagnostics:", file=out)
        for d in diags:
            print(f"  {d}", file=out)


def _run_project(project: Project, seed: int | None) -> SimulationResult:
    for d in project.diagnostics:
        print(f"load: {d}")
    return run_project(project, seed=seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optobuild", description="OptoBuild simulator CLI")
    parser.add_argument("--version", action="version", version=optobuild.__version__)
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--plugins",
        action="store_true",
        help="load component plugins of installed packages (entry points)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("components", help="list built-in components")
    p_demo = sub.add_parser("demo", help="run a built-in demo project")
    p_demo.add_argument("name", choices=sorted(DEMOS))
    p_demo.add_argument("--seed", type=int, default=None)
    p_demo.add_argument("--save", metavar="PATH", help="also save the project file")
    p_run = sub.add_parser("run", help="run a project file (.json/.yaml)")
    p_run.add_argument("project")
    p_run.add_argument("--seed", type=int, default=None, help="override the project seed")
    p_fso = sub.add_parser(
        "fso-budget",
        help="free-space optical link budget, margin and outage",
        description="Quantities take units, e.g. --distance '2 km' --tx-power '10 dBm'.",
    )
    for flag, default in (
        ("--distance", "1 km"),
        ("--wavelength", "1550 nm"),
        ("--tx-power", "10 dBm"),
        ("--sensitivity", "-30 dBm"),
        ("--waist", "10 mm"),
        ("--divergence", "1 mrad"),
        ("--aperture", "8 cm"),
        ("--visibility", "0 km"),
        ("--rain", "0 mm/h"),
        ("--offset", "0 urad"),
        ("--jitter", "0 urad"),
        ("--tx-loss", "0 dB loss"),
        ("--rx-loss", "0 dB loss"),
    ):
        p_fso.add_argument(flag, default=default)
    p_fso.add_argument("--turbulence", default="none", choices=["none", "lognormal", "gamma_gamma"])
    p_fso.add_argument("--cn2", type=float, default=0.0, help="[m^-2/3]")
    p_fso.add_argument("--beam-wander", action="store_true")
    p_gui = sub.add_parser("gui", help="start the graphical editor (needs optobuild[gui])")
    p_gui.add_argument("project", nargs="?", help="project file or demo:<name>")
    p_ber = sub.add_parser("ber", help="accumulate BER over Monte Carlo noise trials")
    p_ber.add_argument("project", help="project file, or demo:<name> for a built-in demo")
    p_ber.add_argument("--trials", type=int, default=10)
    p_ber.add_argument("--node", default="ber", help="name of the BER analyzer node")
    p_ber.add_argument("--seed", type=int, default=None, help="override the project seed")
    p_rep = sub.add_parser("report", help="run a project and write a self-contained HTML report")
    p_rep.add_argument("project", help="project file, or demo:<name>")
    p_rep.add_argument("-o", "--output", required=True, metavar="PATH")
    p_rep.add_argument("--seed", type=int, default=None)
    p_rep.add_argument("--no-timestamp", action="store_true", help="reproducible output")
    p_sw = sub.add_parser(
        "sweep",
        help="sweep component parameters and collect scalar results",
        description="Axis values: comma list (units allowed, e.g. '10 km,20 km') or "
        "lo:hi:n for n linearly spaced SI values.",
    )
    p_sw.add_argument("project", help="project file, or demo:<name>")
    p_sw.add_argument("--axis", action="append", default=[], metavar="NODE.PARAM=VALUES")
    p_sw.add_argument("--probe", action="append", required=True, metavar="NODE.KEY")
    p_sw.add_argument("--trials", type=int, default=1)
    p_sw.add_argument("--workers", type=int, default=1)
    p_sw.add_argument("--seed", type=int, default=None)
    p_sw.add_argument("--csv", metavar="PATH")
    p_sw.add_argument("--html", metavar="PATH")
    p_opt = sub.add_parser("optimize", help="optimize component parameters for a result")
    p_opt.add_argument("project", help="project file, or demo:<name>")
    p_opt.add_argument("--var", action="append", required=True, metavar="NODE.PARAM=LO:HI[:START]")
    goal = p_opt.add_mutually_exclusive_group(required=True)
    goal.add_argument("--maximize", metavar="NODE.KEY")
    goal.add_argument("--minimize", metavar="NODE.KEY")
    p_opt.add_argument("--trials", type=int, default=1, help="trials averaged per evaluation")
    p_opt.add_argument(
        "--method", default="auto", choices=["auto", "bounded", "nelder-mead", "powell"]
    )
    p_opt.add_argument("--max-evaluations", type=int, default=200)
    p_opt.add_argument("--seed", type=int, default=None)
    p_opt.add_argument("--html", metavar="PATH")
    for p in (p_demo, p_run):
        p.add_argument(
            "--save-results",
            metavar="PATH",
            help="write signals and results to an HDF5 file (requires h5py)",
        )
    return parser


def _report(project: Project, args: argparse.Namespace) -> None:
    result = _run_project(project, args.seed)
    print_result(result)
    if args.save_results:
        from optobuild.persistence.results import save_results

        print(f"saved results to {save_results(result, args.save_results, project=project)}")


def _fso_budget(args: argparse.Namespace) -> int:
    import math

    from optobuild.analysis.link_budget import fso_link_budget
    from optobuild.core.units import parse_to_si
    from optobuild.physics.fso_channel import FSOChannelModel

    q = parse_to_si
    visibility = q(args.visibility, expect="length")
    model = FSOChannelModel(
        distance=q(args.distance, expect="length"),
        wavelength=q(args.wavelength, expect="length"),
        beam_waist=q(args.waist, expect="length"),
        divergence=q(args.divergence, expect="angle"),
        aperture_diameter=q(args.aperture, expect="length"),
        tx_efficiency=q(args.tx_loss, expect="ratio"),
        rx_efficiency=q(args.rx_loss, expect="ratio"),
        visibility=visibility if visibility > 0 else math.inf,
        rain_rate=q(args.rain, expect="rain_rate"),
        pointing_offset=q(args.offset, expect="angle"),
        pointing_jitter=q(args.jitter, expect="angle"),
        turbulence=args.turbulence,
        cn2=args.cn2,
        beam_wander=args.beam_wander,
    )
    budget = fso_link_budget(
        model, q(args.tx_power, expect="power"), q(args.sensitivity, expect="power")
    )
    print(budget.table())
    return 0


class _UsageError(Exception):
    """Invalid command-line input (exit code 2)."""


def _load(spec: str, registry: ComponentRegistry | None) -> Project:
    """A project file, or ``demo:<name>`` for a built-in demo."""
    if spec.startswith("demo:"):
        name = spec[5:]
        if name not in DEMOS:
            raise _UsageError(f"unknown demo {name!r}; choose from {sorted(DEMOS)}")
        return DEMOS[name]()
    return load_project(spec, registry)


def _number(text: str) -> float:
    """A float, or a quantity with a unit converted to SI (never evaluated as code)."""
    from optobuild.core.units import parse_to_si

    try:
        return float(text)
    except ValueError:
        return float(parse_to_si(text.strip()))


def _split_ref(text: str, what: str) -> tuple[str, str]:
    node, sep, rest = text.partition(".")
    if not sep or not node or not rest:
        raise _UsageError(f"{what} {text!r} must look like NODE.NAME")
    return node, rest


def _axis(text: str):  # type: ignore[no-untyped-def]
    from optobuild.sweeps.parameter_sweep import Axis

    ref, eq, values = text.partition("=")
    if not eq:
        raise _UsageError(f"axis {text!r} must look like NODE.PARAM=VALUES")
    node, param = _split_ref(ref, "axis")
    parts = values.split(":")
    if len(parts) == 3 and "," not in values:
        lo, hi, n = _number(parts[0]), _number(parts[1]), int(parts[2])
        vals = tuple(float(v) for v in np.linspace(lo, hi, n))
    else:
        vals = tuple(_number(v) for v in values.split(","))
    return Axis(node, param, vals)


def _variable(text: str):  # type: ignore[no-untyped-def]
    from optobuild.optimization.optimizer import Variable

    ref, eq, bounds = text.partition("=")
    parts = bounds.split(":")
    if not eq or len(parts) not in (2, 3):
        raise _UsageError(f"variable {text!r} must look like NODE.PARAM=LO:HI[:START]")
    node, param = _split_ref(ref, "variable")
    start = _number(parts[2]) if len(parts) == 3 else None
    return Variable(node, param, _number(parts[0]), _number(parts[1]), start)


def _write(path: str, text: str, what: str) -> None:
    from pathlib import Path

    Path(path).write_text(text, encoding="utf-8")
    print(f"wrote {what} to {path}")


def _sweep(args: argparse.Namespace, registry: ComponentRegistry | None) -> int:
    from optobuild.reporting.html import sweep_report
    from optobuild.sweeps.parameter_sweep import Probe, sweep

    project = _load(args.project, registry)
    axes = [_axis(a) for a in args.axis]
    probes = [Probe(*_split_ref(p, "probe")) for p in args.probe]
    res = sweep(
        project,
        axes,
        probes,
        trials=args.trials,
        seed=args.seed,
        workers=args.workers,
        registry=registry,
        on_point=lambda done, total: print(f"point {done}/{total}", file=sys.stderr),
    )
    print(res.to_csv(), end="")
    if args.csv:
        _write(args.csv, res.to_csv(), "CSV")
    if args.html:
        _write(args.html, sweep_report(res), "report")
    return 0


def _optimize(args: argparse.Namespace, registry: ComponentRegistry | None) -> int:
    from optobuild.optimization.optimizer import Objective, optimize
    from optobuild.reporting.html import optimization_report

    project = _load(args.project, registry)
    ref, sense = (args.maximize, "max") if args.maximize else (args.minimize, "min")
    node, key = _split_ref(ref, "objective")
    res = optimize(
        project,
        [_variable(v) for v in args.var],
        Objective(node, key, sense, tuple(range(args.trials))),
        method=args.method,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        registry=registry,
    )
    for label, value in res.x.items():
        print(f"{label} = {value:.9g}")
    print(f"{node}.{key} = {res.value:.9g} ({res.n_evaluations} evaluations, {res.message})")
    if args.html:
        _write(args.html, optimization_report(res), "report")
    return 0


def _html_report(args: argparse.Namespace, registry: ComponentRegistry | None) -> int:
    from optobuild.reporting.html import run_report

    project = _load(args.project, registry)
    result = _run_project(project, args.seed)
    _write(args.output, run_report(project, result, timestamp=not args.no_timestamp), "report")
    return 0


def _ber(args: argparse.Namespace, registry: ComponentRegistry | None = None) -> int:
    from optobuild.sweeps.monte_carlo import monte_carlo_ber

    project = _load(args.project, registry)

    def show(trial: int, n_err: int, n_bits: int) -> None:
        print(f"trial {trial:4d}: {n_err} errors / {n_bits} bits")

    mc = monte_carlo_ber(project, args.trials, ber_node=args.node, seed=args.seed, on_trial=show)
    print(
        f"total: {mc.n_errors} errors / {mc.n_bits} bits, BER = {mc.ber:.3e}, "
        f"95% CI [{mc.ber_lower_95:.2e}, {mc.ber_upper_95:.2e}]"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    registry: ComponentRegistry | None = None
    if args.plugins:
        from optobuild.plugins.discovery import plugin_registry

        registry, reports = plugin_registry()
        for r in reports:
            state = f"loaded {', '.join(r.type_ids)}" if r.ok else f"FAILED: {r.error}"
            print(f"plugin {r.name} ({r.value}): {state}", file=sys.stderr)
    try:
        if args.command == "components":
            reg = registry or builtin_registry()
            for type_id in reg:
                info = reg.describe(type_id)
                print(f"{type_id:45s} v{info['version']:8s} {info['display_name']}")
            return 0
        if args.command == "demo":
            project = DEMOS[args.name]()
            if args.save:
                print(f"saved project to {save_project(project, args.save)}")
            _report(project, args)
            return 0
        if args.command == "run":
            _report(load_project(args.project, registry), args)
            return 0
        if args.command == "ber":
            return _ber(args, registry)
        if args.command == "report":
            return _html_report(args, registry)
        if args.command == "sweep":
            return _sweep(args, registry)
        if args.command == "optimize":
            return _optimize(args, registry)
        if args.command == "fso-budget":
            return _fso_budget(args)
        if args.command == "gui":
            try:
                from optobuild.gui.qt.mainwindow import main as gui_main
            except ImportError as exc:
                print(
                    f"error: the GUI needs PySide6 and pyqtgraph ({exc}). "
                    "Install with: pip install 'optobuild[gui]'",
                    file=sys.stderr,
                )
                return 2
            return gui_main([args.project] if args.project else [])
    except (OptoBuildError, _UsageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
