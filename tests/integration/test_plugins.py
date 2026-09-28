"""Entry-point plugins (Phase 10, ADR-0021): loading, validation, isolation of failures."""

from __future__ import annotations

import textwrap
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest

from optobuild.cli.demos import reference_project
from optobuild.core.errors import ProjectFormatError
from optobuild.persistence import dumps_project, loads_project, run_project
from optobuild.plugins.discovery import ENTRY_POINT_GROUP, load_plugins, plugin_registry

PLUGIN = textwrap.dedent(
    '''
    from optobuild.components.base import Component
    from optobuild.components.spec import (
        ComponentCategory, ParameterSpec, ParameterType, PortSpec)
    from optobuild.signals import SignalKind


    class Offset(Component):
        """Adds a constant to an electrical signal."""

        type_id = "example.offset"
        version = "1.0.0"
        display_name = "Offset (plugin)"
        category = ComponentCategory.ELECTRICAL
        input_ports = (PortSpec("in", SignalKind.ELECTRICAL),)
        output_ports = (PortSpec("out", SignalKind.ELECTRICAL),)
        parameter_specs = (ParameterSpec("offset", ParameterType.FLOAT, default=1.0, unit="V"),)

        def run(self, inputs, context):
            sig = inputs["in"]
            return {"out": sig.replace(samples=sig.samples + self.parameters["offset"])}


    class Clash(Offset):
        type_id = "optobuild.reference.gain"


    COMPONENTS = (Offset,)
    CLASH = (Clash,)
    NOT_COMPONENTS = ("hello",)
    '''
)


@pytest.fixture
def plugin_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    (tmp_path / "sample_optobuild_plugin.py").write_text(PLUGIN, encoding="utf-8")
    (tmp_path / "broken_optobuild_plugin.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "sample_optobuild_plugin"


def _ep(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)


def test_plugin_components_register_and_run(plugin_module: str) -> None:
    registry, reports = plugin_registry([_ep("sample", f"{plugin_module}:COMPONENTS")])
    assert reports[0].ok and reports[0].type_ids == ("example.offset",)
    assert "example.offset" in registry
    project = reference_project()
    project.graph.add(registry.create("example.offset", "shift", {"offset": 5.0}))
    project.graph.connect("adder", "out", "shift", "in")
    from optobuild.components.reference import Recorder

    project.graph.add(Recorder("shifted"))
    project.graph.connect("shift", "out", "shifted", "in")
    assert run_project(project).result("shifted", "mean") == pytest.approx(30.0)
    # a project using a plugin component loads only with the plugin's registry
    text = dumps_project(project)
    with pytest.raises(ProjectFormatError, match="load the plugin"):
        loads_project(text)
    again = loads_project(text, registry)
    np.testing.assert_allclose(run_project(again).result("shifted", "mean"), 30.0)


def test_bad_plugins_are_reported_not_fatal(plugin_module: str) -> None:
    registry, reports = plugin_registry(
        [
            _ep("clash", f"{plugin_module}:CLASH"),
            _ep("junk", f"{plugin_module}:NOT_COMPONENTS"),
            _ep("broken", "broken_optobuild_plugin:COMPONENTS"),
            _ep("missing", "no_such_module_xyz:COMPONENTS"),
        ]
    )
    errors = {r.name: r.error for r in reports}
    assert all(not r.ok for r in reports)
    assert "already registered" in errors["clash"]
    assert "not a Component" in errors["junk"]
    assert "boom" in errors["broken"]
    assert "ModuleNotFoundError" in errors["missing"]
    # the built-in component survived the clash
    from optobuild.components.reference import Gain

    assert registry.get("optobuild.reference.gain") is Gain


def test_no_installed_plugins_is_fine() -> None:
    from optobuild.components.registry import builtin_registry

    registry = builtin_registry()
    n = len(registry)
    assert load_plugins(registry, []) == [] and len(registry) == n
