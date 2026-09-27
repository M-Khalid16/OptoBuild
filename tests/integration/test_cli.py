from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from optobuild.cli.main import main


def test_demo_runs_and_prints_exact_clean_mean(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "reference"]) == 0
    out = capsys.readouterr().out
    assert "execution order: ramp -> dc -> gain -> adder -> noise -> clean -> noisy" in out
    assert "clean.mean = 25" in out


def test_demo_save_then_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "ref.json"
    assert main(["demo", "reference", "--save", str(path), "--seed", "5"]) == 0
    first = capsys.readouterr().out.split("saved project to")[1]
    assert main(["run", str(path), "--seed", "5"]) == 0
    second = capsys.readouterr().out
    assert first.split("\n", 1)[1] == second  # identical results from the saved file


def test_components_listing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["components"]) == 0
    assert "optobuild.reference.gain" in capsys.readouterr().out


def test_errors_exit_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["run", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_console_script_and_module_entry_points() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "optobuild", "demo", "reference"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "clean.mean = 25" in res.stdout


def test_ber_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ber", "demo:reference_nope"]) == 2
    assert main(["ber", "demo:optical_link", "--trials", "2"]) == 0
    out = capsys.readouterr().out
    assert "trial    1:" in out and "total:" in out and "4094 bits" in out
