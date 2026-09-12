import json
import subprocess
import sys
from pathlib import Path
import tomllib


def test_headless_self_test_exercises_export_project_and_cad(tmp_path):
    output = tmp_path / "check with spaces"
    result = subprocess.run(
        [sys.executable, "-m", "assembly_workbench", "--self-test", str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((output / "self-test.json").read_text(encoding="utf-8"))
    assert receipt["passed"] is True
    assert receipt["registration"]["backend"] == "small_gicp"
    assert receipt["registration"]["rmse_mm"] < 0.08
    assert receipt["cad_surface_distances_mm"] == [5.0, 2.0]
    assert receipt["project_restored"] is True
    assert receipt['engineering']['tools_passed']==9
    assert receipt['engineering']['project_replay_verified'] is True
    assert receipt['engineering']['known_answers']['contact_volume_mm3']==8.0
    assert (output / "demo.vaw").is_file()
    assert (output / "report" / "report.html").is_file()
    assert (output / "report" / "aligned.xyz").is_file()


def test_version_does_not_launch_gui():
    result = subprocess.run(
        [sys.executable, "-m", "assembly_workbench", "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    project = tomllib.loads((Path(__file__).resolve().parents[1]/'pyproject.toml').read_text())
    assert result.stdout.strip() == f"Assembly Workbench {project['project']['version']}"


def test_screenshot_requires_demo_and_reports_invalid_arguments():
    result = subprocess.run(
        [sys.executable, "-m", "assembly_workbench", "--self-test", "unused", "--demo"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
