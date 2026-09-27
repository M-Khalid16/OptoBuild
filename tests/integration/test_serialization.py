from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from optobuild.cli.demos import reference_project
from optobuild.components.registry import ComponentRegistry
from optobuild.core.errors import ProjectFormatError
from optobuild.engine import FeedForwardExecutor
from optobuild.persistence import (
    dumps_project,
    load_project,
    loads_project,
    project_from_dict,
    project_to_dict,
    save_project,
)


def _structure(project) -> tuple:  # type: ignore[no-untyped-def]
    g = project.graph
    return (
        [(c.name, c.type_id, dict(c.parameters)) for c in g.nodes],
        [str(c) for c in g.connections],
        project.seed,
        project.metadata,
    )


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_round_trip_file(tmp_path: Path, suffix: str) -> None:
    original = reference_project(seed=99)
    path = save_project(original, tmp_path / f"p{suffix}")
    loaded = load_project(path)
    assert _structure(loaded) == _structure(original)
    assert loaded.diagnostics == []


def test_round_trip_gives_identical_results(tmp_path: Path) -> None:
    original = reference_project(seed=42)
    loaded = loads_project(dumps_project(original))
    a = FeedForwardExecutor().run(original.graph, seed=original.seed)
    b = FeedForwardExecutor().run(loaded.graph, seed=loaded.seed)
    for rec in ("clean", "noisy"):
        np.testing.assert_array_equal(a.result(rec, "samples"), b.result(rec, "samples"))


def test_json_is_stable_and_human_readable() -> None:
    text = dumps_project(reference_project())
    assert dumps_project(loads_project(text)) == text
    data = json.loads(text)
    assert data["format"] == "optobuild-project" and data["schema_version"] == 1
    gain = next(c for c in data["components"] if c["name"] == "gain")
    assert gain["parameters"] == {"gain": 2.0} and gain["units"] == {"gain": "1"}
    assert {"from": ["ramp", "out"], "to": ["gain", "in"]} in data["connections"]


def _mutated(**edits):  # type: ignore[no-untyped-def]
    data = project_to_dict(reference_project())
    for path, value in edits.items():
        target = data
        keys = path.split("__")
        for k in keys[:-1]:
            target = target[int(k)] if k.isdigit() else target[k]
        last = keys[-1]
        if value is KeyError:
            del target[last]
        else:
            target[int(last) if last.isdigit() else last] = value
    return data


@pytest.mark.parametrize(
    ("edits", "fragment"),
    [
        ({"format": "other"}, "Not an OptoBuild project"),
        ({"schema_version": 2}, "Unsupported project schema_version"),
        ({"extra": 1}, "Unknown top-level"),
        ({"simulation": {"seed": -1}}, "Invalid seed"),
        ({"simulation": {"seed": 1, "x": 2}}, "Unknown simulation keys"),
        ({"components__0__type_id": "optobuild.nope"}, "Unknown component type"),
        ({"components__0__parameters": {"n_samples": 1}}, "must be >= 2"),
        ({"components__0__parameters": {"bogus": 1}}, "has no parameter"),
        ({"components__0__units": {"sample_rate": "GHz"}}, "expects SI unit 'Hz'"),
        ({"components__0__color": "red"}, "unknown keys"),
        ({"components__1__name": "ramp"}, "already exists"),
        ({"connections__0__to": ["gain", "nope"]}, "no input port"),
        ({"connections__0__to": "gain.in"}, "must be [node, port]"),
        ({"connections__0__via": 1}, "exactly the keys"),
        ({"components": {}}, "must be of type list"),
    ],
)
def test_strict_loading(edits: dict, fragment: str) -> None:
    with pytest.raises(ProjectFormatError) as info:
        project_from_dict(_mutated(**edits))
    assert fragment in str(info.value)


def test_version_mismatch_is_a_warning_not_an_error() -> None:
    data = _mutated(components__0__version="0.9.0")
    p = project_from_dict(data)
    assert [d.code for d in p.diagnostics] == ["project.component_version"]


def test_non_finite_numbers_rejected(tmp_path: Path) -> None:
    text = dumps_project(reference_project()).replace('"gain": 2.0', '"gain": NaN')
    with pytest.raises(ProjectFormatError, match="Non-finite"):
        loads_project(text)
    with pytest.raises(ProjectFormatError, match="Invalid JSON"):
        loads_project("{not json")
    bad_yaml = tmp_path / "p.yaml"
    bad_yaml.write_text("format: [unclosed", encoding="utf-8")
    with pytest.raises(ProjectFormatError, match="Invalid YAML"):
        load_project(bad_yaml)


def test_custom_registry_is_used() -> None:
    with pytest.raises(ProjectFormatError, match="Unknown component type"):
        project_from_dict(project_to_dict(reference_project()), ComponentRegistry())


def test_yaml_cannot_instantiate_python_objects(tmp_path: Path) -> None:
    path = tmp_path / "evil.yaml"
    path.write_text("!!python/object/apply:os.system ['echo pwned']\n", encoding="utf-8")
    with pytest.raises(ProjectFormatError):
        load_project(path)


def test_to_dict_does_not_alias_project() -> None:
    p = reference_project()
    d = project_to_dict(p)
    d2 = copy.deepcopy(d)
    d["metadata"]["title"] = "changed"
    assert p.metadata["title"] != "changed"
    assert project_to_dict(p) == d2
