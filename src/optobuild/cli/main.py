"""Command-line interface.

Examples::

    optobuild components                 # list built-in components
    optobuild demo reference             # run a built-in demo project
    optobuild demo reference --save p.json
    optobuild run p.json --seed 7        # run a project file
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

import optobuild
from optobuild.cli.demos import DEMOS
from optobuild.components.registry import builtin_registry
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
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("components", help="list built-in components")
    p_demo = sub.add_parser("demo", help="run a built-in demo project")
    p_demo.add_argument("name", choices=sorted(DEMOS))
    p_demo.add_argument("--seed", type=int, default=None)
    p_demo.add_argument("--save", metavar="PATH", help="also save the project file")
    p_run = sub.add_parser("run", help="run a project file (.json/.yaml)")
    p_run.add_argument("project")
    p_run.add_argument("--seed", type=int, default=None, help="override the project seed")
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.command == "components":
            registry = builtin_registry()
            for type_id in registry:
                info = registry.describe(type_id)
                print(f"{type_id:45s} v{info['version']:8s} {info['display_name']}")
            return 0
        if args.command == "demo":
            project = DEMOS[args.name]()
            if args.save:
                print(f"saved project to {save_project(project, args.save)}")
            _report(project, args)
            return 0
        if args.command == "run":
            _report(load_project(args.project), args)
            return 0
    except OptoBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
