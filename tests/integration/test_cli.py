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


def test_sweep_optimize_and_report_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from optobuild.cli.main import main

    assert main(["sweep", "demo:reference", "--axis", "gain.gain=1,2", "--probe", "clean.mean",
                 "--csv", str(tmp_path / "s.csv")]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "gain.gain,trial,clean.mean" in out and "1.0,0,17.5" in out and "2.0,0,25.0" in out
    assert (tmp_path / "s.csv").read_text().startswith("gain.gain")
    assert (
        main(["sweep", "demo:reference", "--axis", "gain.gain=0:2:3", "--probe", "clean.mean"]) == 0
    )
    assert "2.0,0,25.0" in capsys.readouterr().out  # linspace 0, 1, 2
    assert main(["optimize", "demo:reference", "--var", "gain.gain=-5:5", "--minimize",
                 "clean.rms"]) == 0  # fmt: skip
    assert "gain.gain = -0.9677" in capsys.readouterr().out
    report = tmp_path / "r.html"
    assert main(["report", "demo:reference", "-o", str(report), "--no-timestamp"]) == 0
    assert report.read_text().startswith("<!DOCTYPE html>")
    assert main(["sweep", "demo:reference", "--axis", "gain.gain", "--probe", "clean.mean"]) == 2
    assert main(["report", "demo:nope", "-o", str(report)]) == 2
    assert main(["--plugins", "components"]) == 0
