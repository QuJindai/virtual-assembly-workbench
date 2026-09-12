from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from assembly_workbench.gui import MainWindow


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(
        sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
        reason="Qt/VTK GUI tests require an X display on Linux",
    ),
]


@pytest.fixture
def window(qtbot):
    widget = MainWindow()
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _wait_for_work(window: MainWindow, qtbot, timeout: int = 60_000) -> None:
    qtbot.waitUntil(lambda: not window.busy, timeout=timeout)


def _run_screenshot_process(root: Path, destination: Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable,
        "-c",
        (
            "from assembly_workbench.gui import launch; "
            f"raise SystemExit(launch(demo=True, screenshot={str(destination)!r}))"
        ),
    ]
    return subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_demo_populates_visible_source_and_reference_controls(window, qtbot):
    """Catches a demo load that updates data without updating the visible workflow."""
    window.load_demo()

    assert len(window.assets) == 2
    assert window.asset_list.count() == 2
    assert window.source_combo.count() == 2
    assert window.target_combo.count() == 2
    assert window.source_combo.currentData() == window.assets[0].id
    assert window.target_combo.currentData() == window.assets[1].id
    assert window.source_combo.currentData() != window.target_combo.currentData()
    assert window.asset_list.item(0).checkState() == Qt.CheckState.Checked
    assert window.asset_list.item(1).checkState() == Qt.CheckState.Checked


def test_parameter_controls_do_not_offer_values_rejected_by_engine(window):
    """Catches GUI parameter ranges that exceed the core's validated limits."""
    assert window.iteration_spin.maximum() == 500
    assert window.max_samples_spin.maximum() == 2_000_000


def test_registration_is_native_async_and_requires_explicit_apply(window, qtbot):
    """Catches reversed registration direction or an implicit transform mutation."""
    window.load_demo()
    source = window.assets[0]
    before = source.transform.copy()

    window.run_registration()
    assert window.busy is True
    assert window.register_button.isEnabled() is False
    _wait_for_work(window, qtbot)

    result = window.last_registration
    assert result is not None
    assert result.source_id == source.id
    assert result.target_id == window.assets[1].id
    assert np.allclose(source.transform, before)
    assert window.apply_button.isEnabled()

    window.apply_result()

    assert not np.allclose(source.transform, before)
    assert window.last_registration is None
    assert window.last_deviation is None
    assert window.apply_button.isEnabled() is False


def test_measurement_finishes_with_real_mm_statistics_and_is_invalidated(window, qtbot):
    """Catches fake measurement results and stale exportable analysis."""
    window.load_demo()

    window.run_measurement()
    assert window.busy is True
    _wait_for_work(window, qtbot)

    result = window.last_deviation
    assert result is not None
    assert result.source_id == window.source_combo.currentData()
    assert result.target_id == window.target_combo.currentData()
    assert result.statistics["sample_count"] > 0
    assert np.isfinite(result.statistics["rms_mm"])
    assert "mm" in window.deviation_summary.text()
    assert window.export_action.isEnabled()

    history = window.history[-1]["result"]
    assert history["statistics"]["sample_count"] == result.statistics["sample_count"]
    assert "distances_mm" not in history
    assert "indices" not in history

    window.target_combo.setCurrentIndex(0)

    assert window.last_deviation is None
    assert window.export_action.isEnabled() is False


def test_visibility_checkbox_controls_the_real_viewport_actor(window, qtbot):
    """Catches an asset visibility checkbox disconnected from the viewport."""
    window.load_demo()
    item = window.asset_list.item(0)
    asset_id = window.assets[0].id

    item.setCheckState(Qt.CheckState.Unchecked)
    qtbot.waitUntil(lambda: not window.viewport.is_asset_visible(asset_id))
    item.setCheckState(Qt.CheckState.Checked)

    assert window.viewport.is_asset_visible(asset_id)


def test_toolbar_saves_and_reopens_self_contained_project(window, qtbot, monkeypatch, tmp_path):
    """Catches toolbar project actions that do not round-trip the visible assets."""
    project = tmp_path / "toolbar-roundtrip.vaw"
    window.load_demo()
    original_ids = [asset.id for asset in window.assets]
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(project), "工作台项目 (*.vaw)"),
    )

    window.save_action.trigger()
    _wait_for_work(window, qtbot)
    assert project.is_file()

    window.load_demo()
    assert [asset.id for asset in window.assets] != original_ids
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(project), "工作台项目 (*.vaw)"),
    )
    window.open_action.trigger()
    _wait_for_work(window, qtbot)

    assert [asset.id for asset in window.assets] == original_ids
    assert window.source_combo.currentText() == "合成扫描件 · 待配准"
    assert window.target_combo.currentText() == "合成基准件 · 固定"


def test_toolbar_exports_real_measurement_files(window, qtbot, monkeypatch, tmp_path):
    """Catches report export that omits the measured data or uses stale geometry."""
    window.load_demo()
    window.run_registration()
    _wait_for_work(window, qtbot)
    window.apply_result()
    window.run_measurement()
    _wait_for_work(window, qtbot)
    assert window.history_table.rowCount() == 2
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )

    window.export_action.trigger()
    _wait_for_work(window, qtbot)

    expected = {"report.json", "deviations.csv", "aligned.xyz", "report.html"}
    assert {path.name for path in tmp_path.iterdir()} == expected
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["units"] == "mm"
    assert report["deviation"]["statistics"]["sample_count"] > 0
    assert len(report["registrations"]) == 1
    assert report["registrations"][0]["method"] == "gicp"


def test_launch_captures_painted_window_and_exits(tmp_path):
    """Catches screenshot mode that captures before paint or leaves the app running."""
    screenshot = tmp_path / "workbench.png"
    root = Path(__file__).resolve().parents[1]
    completed = _run_screenshot_process(root, screenshot)

    assert completed.returncode == 0, completed.stderr
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert screenshot.stat().st_size > 10_000


def test_launch_reports_screenshot_write_failure_with_nonzero_exit(tmp_path):
    """Catches CI screenshot failures being reported as successful runs."""
    root = Path(__file__).resolve().parents[1]
    unwritable_destination = tmp_path / "existing-directory"
    unwritable_destination.mkdir()
    failed = _run_screenshot_process(root, unwritable_destination)

    assert failed.returncode != 0
