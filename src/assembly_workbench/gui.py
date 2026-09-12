"""Chinese Qt desktop workflow for the Virtual Assembly Workbench."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRect, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .core import (
    Dataset,
    DeviationResult,
    RegistrationResult,
    apply_registration,
    make_demo,
    measure_deviation,
    register,
    rigid_transform,
)
from .io import export_report, load_dataset, load_project, save_project
from .emma import EmmaImport, is_emma_csv, load_emma
from .viewport import AssemblyViewport


class _Worker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any]) -> None:
        super().__init__()
        self._operation = operation

    @Slot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self._operation())
        except Exception as exc:  # GUI boundary: convert engine failures to a useful message
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    """Main application window and end-to-end assembly workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.assets: list[Dataset] = []
        self.last_registration: RegistrationResult | None = None
        self.last_deviation: DeviationResult | None = None
        self.busy = False
        self.history: list[dict[str, Any]] = []
        self.project_path: Path | None = None
        self.engineering_dialog = None
        self._worker: _Worker | None = None
        self._worker_thread: QThread | None = None
        self._success_handler: Callable[[Any], None] | None = None
        self._operation_failed = False
        self._pending_close = False

        self.setWindowTitle("虚拟装配工作台 · Assembly Workbench")
        self.resize(1480, 900)
        self.setMinimumSize(1100, 700)
        self._build_toolbar()
        self._build_central_ui()
        self._apply_theme()
        self.statusBar().showMessage("就绪 · 所有长度单位为 mm")
        self._refresh_controls()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        brand = QWidget(toolbar)
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(8, 3, 22, 3)
        brand_layout.setSpacing(0)
        title = QLabel("虚拟装配工作台")
        title.setObjectName("brandTitle")
        subtitle = QLabel("ASSEMBLY WORKBENCH")
        subtitle.setObjectName("brandSubtitle")
        brand_layout.addWidget(title)
        brand_layout.addWidget(subtitle)
        toolbar.addWidget(brand)

        self.demo_action = self._toolbar_action(
            toolbar, "演示", QStyle.StandardPixmap.SP_MediaPlay, self.load_demo
        )
        self.import_action = self._toolbar_action(
            toolbar, "导入", QStyle.StandardPixmap.SP_DialogOpenButton, self.import_asset
        )
        self.open_action = self._toolbar_action(
            toolbar, "打开项目", QStyle.StandardPixmap.SP_DirOpenIcon, self.open_project
        )
        self.save_action = self._toolbar_action(
            toolbar, "保存项目", QStyle.StandardPixmap.SP_DialogSaveButton, self.save_project
        )
        self.export_action = self._toolbar_action(
            toolbar, "导出报告", QStyle.StandardPixmap.SP_DialogApplyButton, self.export_report
        )
        self.engineering_action = self._toolbar_action(
            toolbar, "尺寸工程", QStyle.StandardPixmap.SP_FileDialogDetailedView, self.open_engineering
        )
        toolbar.addSeparator()
        fit_action = self._toolbar_action(
            toolbar, "适合视图", QStyle.StandardPixmap.SP_DesktopIcon, self._fit_view
        )
        fit_action.setToolTip("显示全部可见几何")
        toolbar.addWidget(QLabel(" 点大小 "))
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(1.0, 12.0)
        self.point_size_spin.setValue(3.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setSuffix(" px")
        self.point_size_spin.valueChanged.connect(self._set_point_size)
        toolbar.addWidget(self.point_size_spin)

    def _toolbar_action(
        self,
        toolbar: QToolBar,
        text: str,
        icon: QStyle.StandardPixmap,
        callback: Callable[[], None],
    ) -> QAction:
        action = QAction(self.style().standardIcon(icon), text, self)
        action.triggered.connect(callback)
        toolbar.addAction(action)
        return action

    def _build_central_ui(self) -> None:
        vertical = QSplitter(Qt.Orientation.Vertical)
        horizontal = QSplitter(Qt.Orientation.Horizontal)
        vertical.addWidget(horizontal)

        horizontal.addWidget(self._build_asset_panel())
        self.viewport = AssemblyViewport()
        horizontal.addWidget(self.viewport)
        horizontal.addWidget(self._build_analysis_panel())
        horizontal.setSizes([245, 900, 330])
        horizontal.setStretchFactor(1, 1)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setObjectName("resultHistory")
        self.history_table.setHorizontalHeaderLabels(["类型", "移动件", "参考件", "结果"])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        history_frame = QFrame()
        history_layout = QVBoxLayout(history_frame)
        history_layout.setContentsMargins(8, 5, 8, 8)
        history_label = QLabel("结果历史")
        history_label.setObjectName("sectionTitle")
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.history_table)
        vertical.addWidget(history_frame)
        vertical.setSizes([700, 175])
        vertical.setStretchFactor(0, 1)
        self.setCentralWidget(vertical)

    def _build_asset_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(210)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 14, 12, 12)
        heading = QLabel("装配资源")
        heading.setObjectName("sectionTitle")
        help_text = QLabel("勾选控制可见性；选择移动件与固定参考件")
        help_text.setObjectName("hint")
        help_text.setWordWrap(True)
        self.asset_list = QListWidget()
        self.asset_list.setObjectName("assetList")
        self.asset_list.itemChanged.connect(self._asset_visibility_changed)
        layout.addWidget(heading)
        layout.addWidget(help_text)
        layout.addWidget(self.asset_list, 1)
        self.remove_button = QPushButton("移除所选")
        self.remove_button.clicked.connect(self.remove_selected_asset)
        layout.addWidget(self.remove_button)
        return panel

    def _build_analysis_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(315)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 14, 12, 12)
        heading = QLabel("装配分析")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        selectors = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.setObjectName("sourceSelector")
        self.target_combo = QComboBox()
        self.target_combo.setObjectName("targetSelector")
        self.source_combo.currentIndexChanged.connect(self._selection_changed)
        self.target_combo.currentIndexChanged.connect(self._selection_changed)
        selectors.addRow("移动件 / Source", self.source_combo)
        selectors.addRow("参考件 / Target", self.target_combo)
        layout.addLayout(selectors)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_registration_tab(), "配准")
        self.tabs.addTab(self._build_deviation_tab(), "偏差")
        self.tabs.addTab(self._build_transform_tab(), "刚体调整")
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_registration_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItem("GICP（推荐）", "gicp")
        self.method_combo.addItem("ICP", "icp")
        self.voxel_spin = self._distance_spin(0.01, 1000.0, 2.0, " mm")
        self.max_distance_spin = self._distance_spin(0.01, 10000.0, 20.0, " mm")
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 64)
        self.thread_spin.setValue(4)
        self.iteration_spin = QSpinBox()
        self.iteration_spin.setRange(1, 500)
        self.iteration_spin.setValue(60)
        form.addRow("方法", self.method_combo)
        form.addRow("体素", self.voxel_spin)
        form.addRow("最大对应距离", self.max_distance_spin)
        form.addRow("线程", self.thread_spin)
        form.addRow("最大迭代", self.iteration_spin)
        layout.addLayout(form)
        self.register_button = QPushButton("运行配准")
        self.register_button.setObjectName("primaryButton")
        self.register_button.clicked.connect(self.run_registration)
        self.apply_button = QPushButton("应用计算变换")
        self.apply_button.clicked.connect(self.apply_result)
        self.registration_summary = QLabel("尚未计算。结果需明确应用后才会改变零件。")
        self.registration_summary.setObjectName("resultCard")
        self.registration_summary.setWordWrap(True)
        layout.addWidget(self.register_button)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.registration_summary)
        layout.addStretch()
        return tab

    def _build_deviation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.tolerance_spin = self._distance_spin(0.0001, 10000.0, 0.2, " mm")
        self.tolerance_spin.setDecimals(4)
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(1, 2_000_000)
        self.max_samples_spin.setValue(5000)
        form.addRow("距离阈值", self.tolerance_spin)
        form.addRow("最大采样数", self.max_samples_spin)
        layout.addLayout(form)
        self.measure_button = QPushButton("测量几何偏差")
        self.measure_button.setObjectName("primaryButton")
        self.measure_button.clicked.connect(self.run_measurement)
        self.deviation_summary = QLabel("尚未测量。结果显示无符号几何距离。")
        self.deviation_summary.setObjectName("resultCard")
        self.deviation_summary.setWordWrap(True)
        layout.addWidget(self.measure_button)
        layout.addWidget(self.deviation_summary)
        layout.addStretch()
        return tab

    def _build_transform_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        hint = QLabel("对移动件应用增量刚体变换；旋转顺序为 Rz · Ry · Rx。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.transform_spins: list[QDoubleSpinBox] = []
        for label, suffix in (
            ("X 平移", " mm"),
            ("Y 平移", " mm"),
            ("Z 平移", " mm"),
            ("X 旋转", "°"),
            ("Y 旋转", "°"),
            ("Z 旋转", "°"),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(4)
            spin.setSuffix(suffix)
            self.transform_spins.append(spin)
            form.addRow(label, spin)
        layout.addLayout(form)
        self.transform_button = QPushButton("应用增量变换")
        self.transform_button.setObjectName("primaryButton")
        self.transform_button.clicked.connect(self.apply_manual_transform)
        layout.addWidget(self.transform_button)
        layout.addStretch()
        return tab

    @staticmethod
    def _distance_spin(
        minimum: float, maximum: float, value: float, suffix: str
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #101923; color: #dbe7f1; font-size: 13px; }
            QToolBar#mainToolbar { background: #071a2e; border: 0; border-bottom: 1px solid #28445c; spacing: 5px; padding: 5px; }
            QToolBar QToolButton { padding: 7px 9px; border-radius: 4px; }
            QToolBar QToolButton:hover { background: #183a53; }
            QLabel#brandTitle { font-size: 18px; font-weight: 650; color: #f3f8fc; }
            QLabel#brandSubtitle { font-size: 9px; letter-spacing: 2px; color: #51d1e1; }
            QLabel#sectionTitle { font-size: 15px; font-weight: 650; color: #edf6fc; padding-bottom: 2px; }
            QLabel#hint { color: #8096a8; font-size: 12px; }
            QFrame#sidePanel { background: #111f2b; border: 1px solid #233848; }
            QListWidget, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0b151e; border: 1px solid #2b4253; border-radius: 4px;
                selection-background-color: #13617a; padding: 4px;
            }
            QListWidget::item { padding: 7px 3px; }
            QListWidget::item:selected { background: #155a72; }
            QTabWidget::pane { border: 1px solid #2b4253; top: -1px; }
            QTabBar::tab { background: #172734; padding: 8px 11px; border: 1px solid #2b4253; }
            QTabBar::tab:selected { color: #63d8e7; background: #0d1b26; }
            QPushButton { background: #203646; border: 1px solid #395568; border-radius: 4px; padding: 8px; }
            QPushButton:hover { background: #29485d; }
            QPushButton:disabled { color: #667682; background: #18242e; }
            QPushButton#primaryButton { background: #08728c; border-color: #22a5bf; font-weight: 600; }
            QPushButton#primaryButton:hover { background: #0988a6; }
            QLabel#resultCard { background: #0a151e; border-left: 3px solid #2fc3d7; padding: 10px; color: #b9ccd9; }
            QHeaderView::section { background: #172734; color: #a9bdca; padding: 5px; border: 0; border-right: 1px solid #2b4253; }
            QStatusBar { background: #071a2e; color: #8ea9ba; }
            QSplitter::handle { background: #263b4b; }
            """
        )

    def load_demo(self) -> None:
        if self.busy:
            return
        source, target, _known_transform = make_demo()
        self.project_path = None
        self.history = []
        self._set_assets([source, target])
        self.statusBar().showMessage("已载入确定性装配演示 · 选择配准或偏差测量", 6000)

    def import_asset(self) -> None:
        if self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入几何或eMMA测量数据",
            "",
            "支持的几何 (*.csv *.xyz *.pcd *.ply *.stl *.obj *.step *.stp *.iges *.igs);;所有文件 (*)",
        )
        if not path:
            return
        unit = "mm"
        if Path(path).suffix.lower() in {".csv", ".xyz", ".pcd", ".ply", ".stl", ".obj"}:
            label, accepted = QInputDialog.getItem(
                self, "输入单位", "文件中的长度单位", ["毫米 (mm)", "米 (m)"], 0, False
            )
            if not accepted:
                return
            unit = "m" if label == "米 (m)" else "mm"
        self._start_operation(
            "正在导入几何…",
            lambda: load_emma(path, unit=unit) if is_emma_csv(path) else load_dataset(path, unit=unit),
            self._finish_import,
        )

    def _finish_import(self, asset: Dataset | EmmaImport) -> None:
        imported = asset.assets if isinstance(asset, EmmaImport) else [asset]
        if len(self.assets) + len(imported) > 100:
            self._show_error("资源数超限", "导入后资源数将超过100，请另建项目或减少样本。")
            return
        self.assets.extend(imported)
        self.project_path = None
        self._invalidate_results()
        self._refresh_asset_widgets()
        if isinstance(asset, EmmaImport):
            stats = asset.summary
            summary = (f"eMMA：{stats['actual_assets']}组实测、{stats['nominal_assets']}组名义；"
                       f"{stats['missing_xyz_rows']}行无完整XYZ；"
                       f"{stats['inferred_actual_rows']}行依据样本/时间识别为实测；"
                       f"{stats['invalid_xyz_rows']}行无效坐标、{stats['unknown_role_rows']}行角色不明；"
                       f"{stats['conflicting_keys']}个冲突编号")
            self._append_history("导入", stats['source_file'], "", summary, stats)
            self.statusBar().showMessage(summary)
        else:
            self.statusBar().showMessage(f"已导入 {asset.name} · {len(asset.points):,} 点", 6000)

    def open_project(self) -> None:
        if self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "工作台项目 (*.vaw)")
        if path:
            self._start_operation(
                "正在打开项目…", lambda: load_project(path), lambda value: self._finish_open(path, value)
            )

    def _finish_open(self, path: str, value: tuple[list[Dataset], list[dict]]) -> None:
        assets, history = value
        self.project_path = Path(path)
        self.history = list(history)
        self._set_assets(assets)
        self._refresh_history()
        self.statusBar().showMessage(f"已打开项目：{Path(path).name}", 6000)

    def save_project(self) -> None:
        if self.busy or not self.assets:
            return
        initial = str(self.project_path or Path.cwd() / "assembly.vaw")
        path, _ = QFileDialog.getSaveFileName(self, "保存项目", initial, "工作台项目 (*.vaw)")
        if not path:
            return
        if not path.lower().endswith(".vaw"):
            path += ".vaw"
        self._start_operation(
            "正在保存项目…",
            lambda: save_project(path, self.assets, self.history),
            lambda _result: self._finish_save(path),
        )

    def _finish_save(self, path: str) -> None:
        self.project_path = Path(path)
        self.statusBar().showMessage(f"项目已保存：{Path(path).name}", 6000)

    def export_report(self) -> None:
        if self.busy or self.last_deviation is None:
            return
        source, target = self._selected_pair(show_errors=True)
        if source is None or target is None:
            return
        if not self._deviation_is_current(source, target):
            self._invalidate_results()
            self._show_error("报告无法导出", "几何或选择已改变，请重新测量偏差。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择报告输出文件夹")
        if not folder:
            return
        deviation = self.last_deviation
        registrations = [
            row.get("result", row)
            for row in self.history
            if row.get("type") == "registration"
        ]
        self._start_operation(
            "正在导出报告…",
            lambda: export_report(folder, source, target, deviation, registrations),
            lambda paths: self.statusBar().showMessage(
                f"报告已导出 · {len(paths)} 个文件 · {paths[0].parent}", 8000
            ),
        )

    def remove_selected_asset(self) -> None:
        if self.busy:
            return
        item = self.asset_list.currentItem()
        if item is None:
            return
        asset_id = item.data(Qt.ItemDataRole.UserRole)
        self.assets = [asset for asset in self.assets if asset.id != asset_id]
        self.project_path = None
        self._invalidate_results()
        self._refresh_asset_widgets()

    def run_registration(self) -> None:
        if self.busy:
            return
        source, target = self._selected_pair(show_errors=True)
        if source is None or target is None:
            return
        method = str(self.method_combo.currentData())
        voxel = self.voxel_spin.value()
        max_distance = self.max_distance_spin.value()
        threads = self.thread_spin.value()
        iterations = self.iteration_spin.value()
        self._invalidate_results(redraw=False)
        self.registration_summary.setText("正在计算配准；完成前不会改变移动件…")
        self._start_operation(
            "正在后台运行配准…",
            lambda: register(
                source,
                target,
                method=method,
                voxel_mm=voxel,
                max_distance_mm=max_distance,
                threads=threads,
                max_iterations=iterations,
            ),
            self._finish_registration,
        )

    def _finish_registration(self, result: RegistrationResult) -> None:
        self.last_registration = result
        quality = "已收敛" if result.converged else "未收敛"
        next_step = (
            "变换尚未应用，请检查后点击“应用计算变换”。"
            if result.converged
            else "有效对应点不足，结果不可应用；请调整体素或对应距离后重试。"
        )
        self.registration_summary.setText(
            f"{quality} · RMSE {result.rmse_mm:.4f} mm\n"
            f"匹配率 {result.fitness:.1%} · {result.iterations} 次迭代 · {result.elapsed_s:.3f} s\n"
            f"{next_step}"
        )
        source = self._asset_by_id(result.source_id)
        target = self._asset_by_id(result.target_id)
        self._append_history(
            "registration",
            source.name if source else result.source_id,
            target.name if target else result.target_id,
            f"{result.method.upper()} · RMSE {result.rmse_mm:.4f} mm · {quality}",
            result.to_dict(),
        )
        self.statusBar().showMessage(
            "配准计算完成 · 变换尚未应用"
            if result.converged
            else "配准未收敛 · 调整参数后重试",
            6000,
        )

    def apply_result(self) -> None:
        if self.busy or self.last_registration is None:
            return
        source = self._asset_by_id(self.last_registration.source_id)
        if source is None:
            self._show_error("无法应用配准", "移动件已被移除，请重新运行配准。")
            self._invalidate_results()
            return
        try:
            apply_registration(source, self.last_registration)
        except ValueError as exc:
            self._show_error("无法应用配准", f"结果已过期：{exc}\n请重新运行配准。")
            self._invalidate_results()
            return
        self.project_path = None
        self._invalidate_results()
        self._update_asset_labels()
        self._refresh_viewport(fit=False)
        self.registration_summary.setText("变换已应用到移动件。几何已改变，请重新计算分析。")
        self.statusBar().showMessage("配准变换已应用 · 旧分析结果已失效", 6000)

    def run_measurement(self) -> None:
        if self.busy:
            return
        source, target = self._selected_pair(show_errors=True)
        if source is None or target is None:
            return
        tolerance = self.tolerance_spin.value()
        max_samples = self.max_samples_spin.value()
        self.last_deviation = None
        self._refresh_controls()
        self.deviation_summary.setText("正在测量无符号几何距离…")
        self._start_operation(
            "正在后台测量偏差…",
            lambda: measure_deviation(
                source, target, tolerance_mm=tolerance, max_samples=max_samples
            ),
            self._finish_measurement,
        )

    def _finish_measurement(self, result: DeviationResult) -> None:
        self.last_deviation = result
        stats = result.statistics
        self.deviation_summary.setText(
            f"RMS {stats['rms_mm']:.4f} mm · P95 {stats['p95_mm']:.4f} mm\n"
            f"最大 {stats['max_mm']:.4f} mm · 平均 {stats['mean_mm']:.4f} mm\n"
            f"阈值内 {stats['within_fraction']:.1%} · "
            f"采样 {int(stats['sample_count']):,}/{int(stats['total_count']):,}\n"
            f"方法：{result.method} · {result.elapsed_s:.3f} s"
        )
        if result.method == 'feature_id':
            self.deviation_summary.setText(self.deviation_summary.text() +
                f"\n编号对应 {stats['matched_count']:,} · 实测无对应 {stats['unmatched_count']:,}"
                f"\n参考覆盖 {stats['reference_coverage']:.1%} · 参考无对应 {stats['unmatched_reference_count']:,}")
        source = self._asset_by_id(result.source_id)
        target = self._asset_by_id(result.target_id)
        history_result = result.to_dict()
        # Per-point values belong in the current result/report. Persist a
        # compact audit history so large measurements remain saveable.
        history_result.pop("indices", None)
        history_result.pop("distances_mm", None)
        history_result.pop("target_indices", None)
        self._append_history(
            "deviation",
            source.name if source else result.source_id,
            target.name if target else result.target_id,
            f"RMS {stats['rms_mm']:.4f} mm · P95 {stats['p95_mm']:.4f} mm · 阈值内 {stats['within_fraction']:.1%}",
            history_result,
        )
        self._refresh_viewport(fit=False)
        self.statusBar().showMessage("偏差测量完成 · 视图颜色单位为 mm", 6000)

    def apply_manual_transform(self) -> None:
        if self.busy:
            return
        source = self._selected_source(show_errors=True)
        if source is None:
            return
        values = [spin.value() for spin in self.transform_spins]
        delta = rigid_transform(*values)
        source.transform = delta @ np.asarray(source.transform, dtype=np.float64)
        for spin in self.transform_spins:
            spin.setValue(0.0)
        self.project_path = None
        self._invalidate_results()
        self._update_asset_labels()
        self._refresh_viewport(fit=False)
        self.statusBar().showMessage("增量刚体变换已应用 · 旧分析结果已失效", 6000)

    def _set_assets(self, assets: list[Dataset]) -> None:
        self.assets = list(assets)
        self.last_registration = None
        self.last_deviation = None
        self._refresh_asset_widgets()
        self._refresh_history()

    def _refresh_asset_widgets(self) -> None:
        self.asset_list.blockSignals(True)
        self.source_combo.blockSignals(True)
        self.target_combo.blockSignals(True)
        self.asset_list.clear()
        self.source_combo.clear()
        self.target_combo.clear()
        for asset in self.assets:
            item = QListWidgetItem(self._asset_label(asset))
            item.setData(Qt.ItemDataRole.UserRole, asset.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.asset_list.addItem(item)
            self.source_combo.addItem(asset.name, asset.id)
            self.target_combo.addItem(asset.name, asset.id)
        if len(self.assets) > 1:
            self.target_combo.setCurrentIndex(1)
        self.asset_list.blockSignals(False)
        self.source_combo.blockSignals(False)
        self.target_combo.blockSignals(False)
        self._refresh_viewport(fit=True)
        self._refresh_controls()

    @staticmethod
    def _asset_label(asset: Dataset) -> str:
        placement = (
            "原始位置"
            if np.allclose(asset.transform, np.eye(4), rtol=0.0, atol=1e-12)
            else "已应用变换"
        )
        return f"{asset.name}\n{asset.kind} · {len(asset.points):,} 点 · {placement}"

    def _update_asset_labels(self) -> None:
        for row in range(self.asset_list.count()):
            item = self.asset_list.item(row)
            asset = self._asset_by_id(item.data(Qt.ItemDataRole.UserRole))
            if asset is not None:
                item.setText(self._asset_label(asset))

    @Slot()
    def _selection_changed(self) -> None:
        self._invalidate_results()
        self._refresh_viewport(fit=False)

    @Slot(QListWidgetItem)
    def _asset_visibility_changed(self, item: QListWidgetItem) -> None:
        asset_id = item.data(Qt.ItemDataRole.UserRole)
        self.viewport.set_asset_visible(asset_id, item.checkState() == Qt.CheckState.Checked)

    def _refresh_viewport(self, fit: bool) -> None:
        self.viewport.set_assets(
            self.assets,
            self.source_combo.currentData(),
            self.target_combo.currentData(),
            self.last_deviation,
        )
        for row in range(self.asset_list.count()):
            item = self.asset_list.item(row)
            self.viewport.set_asset_visible(
                item.data(Qt.ItemDataRole.UserRole),
                item.checkState() == Qt.CheckState.Checked,
            )
        if fit:
            self.viewport.fit_view()

    def _fit_view(self) -> None:
        self.viewport.fit_view()

    def open_engineering(self) -> None:
        if self.busy:
            return
        from .engineering_gui import EngineeringDialog
        if self.engineering_dialog is None:
            self.engineering_dialog = EngineeringDialog(self)
        else:
            self.engineering_dialog.refresh_assets()
            self.engineering_dialog.refresh_history()
        self.engineering_dialog.show()
        self.engineering_dialog.raise_()

    def _set_point_size(self, value: float) -> None:
        self.viewport.set_point_size(value)

    def _invalidate_results(self, redraw: bool = True) -> None:
        self.last_registration = None
        self.last_deviation = None
        self.registration_summary.setText("选择或几何已改变，请重新运行配准。")
        self.deviation_summary.setText("选择或几何已改变，请重新测量偏差。")
        if redraw:
            self._refresh_viewport(fit=False)
        self._refresh_controls()

    def _deviation_is_current(self, source: Dataset, target: Dataset) -> bool:
        result = self.last_deviation
        return bool(
            result is not None
            and result.source_id == source.id
            and result.target_id == target.id
            and np.array_equal(result.source_transform, source.transform)
            and np.array_equal(result.target_transform, target.transform)
        )

    def _selected_source(self, show_errors: bool = False) -> Dataset | None:
        source = self._asset_by_id(self.source_combo.currentData())
        if source is None and show_errors:
            self._show_error("缺少移动件", "请先导入几何，并选择一个移动件。")
        return source

    def _selected_pair(
        self, show_errors: bool = False
    ) -> tuple[Dataset | None, Dataset | None]:
        source = self._selected_source(show_errors=show_errors)
        target = self._asset_by_id(self.target_combo.currentData())
        if target is None and show_errors:
            self._show_error("缺少参考件", "请选择一个固定参考件。")
        if source is not None and target is not None and source.id == target.id:
            if show_errors:
                self._show_error("选择无效", "移动件和参考件必须是不同的资源。")
            return None, None
        return source, target

    def _asset_by_id(self, asset_id: str | None) -> Dataset | None:
        return next((asset for asset in self.assets if asset.id == asset_id), None)

    def _append_history(
        self,
        result_type: str,
        source_name: str,
        target_name: str,
        summary: str,
        result: dict[str, Any],
    ) -> None:
        row = {
            "type": result_type,
            "source": source_name,
            "target": target_name,
            "summary": summary,
            "result": result,
        }
        self.history.append(row)
        self._append_history_row(row)

    def _refresh_history(self) -> None:
        self.history_table.setRowCount(0)
        for row in self.history:
            self._append_history_row(row)

    def _append_history_row(self, row: dict[str, Any]) -> None:
        index = self.history_table.rowCount()
        self.history_table.insertRow(index)
        labels = {
            "registration": "配准",
            "deviation": "偏差",
            "transform": "变换",
            "engineering": "尺寸工程",
            "controls_import": "公差导入",
        }
        values = (
            labels.get(str(row.get("type")), str(row.get("type", "结果"))),
            str(row.get("source", "")),
            str(row.get("target", "")),
            str(row.get("summary", "")),
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            self.history_table.setItem(index, column, item)
        self.history_table.scrollToBottom()

    def _refresh_controls(self) -> None:
        has_assets = bool(self.assets)
        pair_selected = (
            self.source_combo.currentData() is not None
            and self.target_combo.currentData() is not None
            and self.source_combo.currentData() != self.target_combo.currentData()
        )
        for control in (
            self.demo_action,
            self.engineering_action,
            self.import_action,
            self.open_action,
            self.save_action,
            self.remove_button,
            self.source_combo,
            self.target_combo,
            self.method_combo,
            self.voxel_spin,
            self.max_distance_spin,
            self.thread_spin,
            self.iteration_spin,
            self.tolerance_spin,
            self.max_samples_spin,
            self.transform_button,
            *self.transform_spins,
        ):
            control.setEnabled(not self.busy)
        self.save_action.setEnabled(not self.busy and has_assets)
        self.remove_button.setEnabled(not self.busy and has_assets)
        self.register_button.setEnabled(not self.busy and pair_selected)
        self.measure_button.setEnabled(not self.busy and pair_selected)
        self.apply_button.setEnabled(
            not self.busy
            and self.last_registration is not None
            and self.last_registration.converged
        )
        source_available = not self.busy and self.source_combo.currentData() is not None
        self.transform_button.setEnabled(source_available)
        self.export_action.setEnabled(not self.busy and self.last_deviation is not None)
        self.asset_list.setEnabled(not self.busy)
        if self.engineering_dialog is not None:
            self.engineering_dialog.refresh_busy()

    def _start_operation(
        self,
        status: str,
        operation: Callable[[], Any],
        success_handler: Callable[[Any], None],
    ) -> None:
        if self.busy:
            return
        self.busy = True
        self._operation_failed = False
        self._success_handler = success_handler
        self._worker_thread = QThread(self)
        self._worker = _Worker(operation)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(
            self._worker_succeeded, Qt.ConnectionType.QueuedConnection
        )
        self._worker.failed.connect(self._worker_failed, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread_finished)
        self.statusBar().showMessage(status)
        self._refresh_controls()
        self._worker_thread.start()

    @Slot(object)
    def _worker_succeeded(self, result: Any) -> None:
        handler = self._success_handler
        if handler is None:
            return
        try:
            handler(result)
        except Exception as exc:
            self._operation_failed = True
            self._show_error("操作结果无法处理", str(exc))

    @Slot(str)
    def _worker_failed(self, detail: str) -> None:
        self._operation_failed = True
        self._show_error("操作失败", f"{detail}\n\n请检查输入、参数和文件权限后重试。")

    @Slot()
    def _worker_thread_finished(self) -> None:
        thread = self._worker_thread
        self._worker = None
        self._worker_thread = None
        self._success_handler = None
        self.busy = False
        self._refresh_controls()
        if self._operation_failed:
            self.statusBar().showMessage("操作失败 · 请检查提示后重试", 8000)
        if thread is not None:
            thread.deleteLater()
        if self._pending_close:
            self._pending_close = False
            QTimer.singleShot(0, self.close)

    def _show_error(self, title: str, detail: str) -> None:
        QMessageBox.critical(self, title, detail)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self.busy:
            self._pending_close = True
            self.statusBar().showMessage("后台计算完成后将安全关闭…")
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)


def _exit_after_work(window,app,exit_code):
    """Keep the event loop alive until the native worker has actually finished."""
    if window.busy:
        window.hide()
        QTimer.singleShot(50,lambda:_exit_after_work(window,app,exit_code))
        return
    quit_on_close=app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    window.close()
    app.setQuitOnLastWindowClosed(quit_on_close)
    app.exit(exit_code)


def launch(demo: bool = False, screenshot: str | Path | None = None, engineering_demo: str | None = None) -> int:
    """Launch the workbench, optionally capture the painted real window."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("虚拟装配工作台")
    app.setOrganizationName("Assembly Workbench")
    window = MainWindow()
    window.show()
    if demo:
        window.load_demo()

    capture_widget=window
    capture_viewport=window.viewport
    if engineering_demo:
        window.open_engineering()
        capture_widget=window.engineering_dialog
        capture_viewport=capture_widget.viewport
        capture_widget.load_example(engineering_demo)
        capture_widget.run_current()

    if screenshot is not None:
        destination = Path(screenshot)
        viewport_destination = destination.with_name(f"{destination.stem}-viewport.png")
        diagnostics_destination = destination.with_suffix(".render.json")
        screenshot_finished = False
        capture_timeout = QTimer(window)
        capture_timeout.setSingleShot(True)

        def finish_screenshot(exit_code: int) -> None:
            nonlocal screenshot_finished
            if screenshot_finished:
                return
            screenshot_finished = True
            capture_timeout.stop()
            _exit_after_work(window,app,exit_code)

        def write_diagnostics(diagnostics: dict[str, object]) -> bool:
            try:
                diagnostics_destination.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_destination.write_text(
                    json.dumps(diagnostics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True
            except OSError as exc:
                print(f"无法保存渲染诊断：{exc}", file=sys.stderr)
                return False

        def capture() -> None:
            if screenshot_finished:return
            if not capture_viewport.is_initialized or window.busy:
                QTimer.singleShot(100,capture)
                return
            diagnostics: dict[str, object] = {
                "initialized": capture_viewport.is_initialized
            }
            exit_code = 1
            try:
                diagnostics = capture_viewport.render_diagnostics()
                destination.parent.mkdir(parents=True, exist_ok=True)
                framebuffer, diagnostics = capture_viewport.capture_framebuffer()
                if not framebuffer.save(str(viewport_destination), "PNG"):
                    raise OSError(f"无法保存原始视口：{viewport_destination}")
                diagnostics["raw_viewport_path"] = viewport_destination.name

                cyan_count = int(diagnostics["cyan_pixel_count"])
                amber_count = int(diagnostics["amber_pixel_count"])
                diagnostics["demo_color_validation_required"] = bool(demo or engineering_demo)
                diagnostics["demo_colors_present"] = cyan_count > 20 and amber_count > 20
                if (demo or engineering_demo) and not diagnostics["demo_colors_present"]:
                    raise RuntimeError(
                        "VTK帧缓冲未包含演示的青色移动件和琥珀色参考件"
                    )

                if engineering_demo:
                    if capture_widget.receipt is None:
                        raise RuntimeError('Engineering example did not produce a result')
                    diagnostics['engineering_tool']=engineering_demo
                    diagnostics['engineering_result']=capture_widget.receipt['result']
                composite = capture_widget.grab()
                top_left = capture_viewport.interactor.mapTo(
                    capture_widget, capture_viewport.interactor.rect().topLeft()
                )
                target = QRect(top_left, capture_viewport.interactor.size())
                painter = QPainter(composite)
                if not painter.isActive():
                    raise RuntimeError("无法创建截图合成画布")
                try:
                    painter.drawImage(target, framebuffer)
                finally:
                    painter.end()
                diagnostics["composite_viewport_rect"] = [
                    target.x(),
                    target.y(),
                    target.width(),
                    target.height(),
                ]
                if not composite.save(str(destination), "PNG"):
                    raise OSError(f"无法保存截图：{destination}")
                diagnostics["screenshot_path"] = destination.name
                exit_code = 0
            except Exception as exc:  # screenshot mode must report every capture failure
                diagnostics["capture_error"] = f"{type(exc).__name__}: {exc}"
                window.statusBar().showMessage(f"无法保存截图：{exc}")
                print(diagnostics["capture_error"], file=sys.stderr)
            finally:
                if not write_diagnostics(diagnostics):
                    exit_code = 1
                finish_screenshot(exit_code)

        def capture_timed_out() -> None:
            diagnostics: dict[str, object] = {
                "initialized": capture_viewport.is_initialized
            }
            try:
                diagnostics = capture_viewport.render_diagnostics()
            except Exception as exc:
                diagnostics["diagnostics_error"] = f"{type(exc).__name__}: {exc}"
            diagnostics["capture_error"] = "VTK viewport did not initialize or compute example within 30 seconds"
            print(diagnostics["capture_error"], file=sys.stderr)
            write_diagnostics(diagnostics)
            finish_screenshot(1)

        def schedule_capture() -> None:
            QTimer.singleShot(0, capture)

        capture_viewport.rendering_ready.connect(schedule_capture)
        capture_timeout.timeout.connect(capture_timed_out)
        capture_timeout.start(30_000)
        if capture_viewport.is_initialized:
            schedule_capture()

    if owns_app:
        return app.exec()
    return 0


__all__ = ["MainWindow", "launch"]
