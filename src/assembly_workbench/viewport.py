"""VTK viewport used by the desktop workbench."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
# VTK's abstract factories need their implementation modules registered before
# the first window, renderer, interactor or text actor is constructed.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingFreeType  # noqa: F401
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkTextActor,
    vtkWindowToImageFilter,
)

if TYPE_CHECKING:
    from .core import Dataset, DeviationResult


SOURCE_COLOUR = (0.15, 0.82, 0.92)
TARGET_COLOUR = (1.0, 0.66, 0.18)
NEUTRAL_COLOUR = (0.55, 0.62, 0.70)
MAX_DISPLAY_POINTS = 250_000


class AssemblyViewport(QWidget):
    """Interactive VTK scene for transformed assembly assets."""

    rendering_ready = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.interactor = QVTKRenderWindowInteractor(self)
        # A concrete OpenGL render window needs an exposed native Qt window.
        # In particular, initializing here crashes on Windows before the
        # top-level MainWindow has been shown. Suppress QVTK's paintEvent until
        # _initialize_after_show has established that native lifecycle.
        self.interactor.setUpdatesEnabled(False)
        layout.addWidget(self.interactor)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.025, 0.055, 0.09)
        self.renderer.SetBackground2(0.075, 0.12, 0.17)
        self.renderer.GradientBackgroundOn()
        self.interactor.GetRenderWindow().AddRenderer(self.renderer)

        self._actors: dict[str, vtkActor] = {}
        self._display_indices: dict[str, np.ndarray] = {}
        self._point_size = 3.0
        self._initialized = False
        self._initialize_timer = QTimer(self)
        self._initialize_timer.setSingleShot(True)
        self._initialize_timer.timeout.connect(self._initialize_after_show)
        self._empty_text = vtkTextActor()
        self._empty_text.SetInput(
            "Load an asset or demo to begin\n"
            "Drag: rotate  |  Wheel: zoom  |  Middle drag: pan"
        )
        self._empty_text.SetDisplayPosition(34, 40)
        text_property = self._empty_text.GetTextProperty()
        text_property.SetColor(0.62, 0.72, 0.82)
        text_property.SetFontSize(18)
        text_property.SetLineSpacing(1.35)
        self.renderer.AddActor2D(self._empty_text)

        self._scalar_bar = vtkScalarBarActor()
        self._scalar_bar.SetTitle("Deviation (mm)")
        self._scalar_bar.SetNumberOfLabels(5)
        self._scalar_bar.SetWidth(0.085)
        self._scalar_bar.SetHeight(0.48)
        self._scalar_bar.SetPosition(0.89, 0.08)
        self._scalar_bar.GetTitleTextProperty().SetColor(0.92, 0.95, 0.98)
        self._scalar_bar.GetLabelTextProperty().SetColor(0.92, 0.95, 0.98)
        self._scalar_bar.SetVisibility(False)
        self.renderer.AddActor2D(self._scalar_bar)

        self.interactor.GetRenderWindow().SetMultiSamples(4)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if not self._initialized and not self._initialize_timer.isActive():
            self._initialize_timer.start(0)

    def _initialize_after_show(self) -> None:
        if self._initialized:
            return
        if not self.isVisible():
            return

        top_level = self.window().windowHandle()
        if top_level is None or not top_level.isExposed():
            self._initialize_timer.start(16)
            return

        self.interactor.setUpdatesEnabled(True)
        self.interactor.Initialize()
        self.interactor.Start()
        self._initialized = True
        self.interactor.GetRenderWindow().Render()
        self.rendering_ready.emit()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def render_diagnostics(self) -> dict[str, object]:
        """Return concise native renderer state for screenshot verification."""
        render_window = self.interactor.GetRenderWindow()
        diagnostics: dict[str, object] = {
            "initialized": self._initialized,
            "widget_size": [self.interactor.width(), self.interactor.height()],
            "device_pixel_ratio": float(self.interactor.devicePixelRatioF()),
            "render_window_size": list(render_window.GetSize()),
            "render_window_class": render_window.GetClassName(),
            "renderer_class": self.renderer.GetClassName(),
            "actor_count": len(self._actors),
        }
        if self._initialized:
            try:
                diagnostics["open_gl_supported"] = bool(render_window.SupportsOpenGL())
                diagnostics["open_gl_support_message"] = str(
                    render_window.GetOpenGLSupportMessage()
                )
                capabilities = str(render_window.ReportCapabilities())
                for label, key in (
                    ("OpenGL vendor string", "open_gl_vendor"),
                    ("OpenGL renderer string", "open_gl_renderer"),
                    ("OpenGL version string", "open_gl_version"),
                ):
                    line = next(
                        (line for line in capabilities.splitlines() if label in line),
                        None,
                    )
                    if line:
                        diagnostics[key] = line.split(":", 1)[-1].strip()
            except RuntimeError as exc:
                diagnostics["open_gl_diagnostics_error"] = str(exc)
        return diagnostics

    def capture_framebuffer(self) -> tuple[QImage, dict[str, object]]:
        """Capture the real VTK back buffer as an owned top-left RGB image."""
        if not self._initialized:
            raise RuntimeError("VTK viewport has not initialized")

        render_window = self.interactor.GetRenderWindow()
        render_window.Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(render_window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()

        output = capture.GetOutput()
        width, height, _depth = output.GetDimensions()
        scalars = output.GetPointData().GetScalars()
        if width < 1 or height < 1 or scalars is None:
            raise RuntimeError("VTK framebuffer capture returned no pixels")
        components = scalars.GetNumberOfComponents()
        if components < 3:
            raise RuntimeError(
                f"VTK framebuffer returned {components} color components; expected RGB"
            )

        values = vtk_to_numpy(scalars)
        pixels = values.reshape(height, width, components)[..., :3]
        # VTK images start at the lower-left; QImage starts at the upper-left.
        pixels = np.ascontiguousarray(np.flipud(pixels), dtype=np.uint8)
        image = QImage(
            pixels.data,
            width,
            height,
            int(pixels.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        image.setDevicePixelRatio(float(self.interactor.devicePixelRatioF()))

        red = pixels[..., 0].astype(np.int16)
        green = pixels[..., 1].astype(np.int16)
        blue = pixels[..., 2].astype(np.int16)
        cyan = (
            (green >= 130)
            & (blue >= 150)
            & (green > red + 70)
            & (blue > red + 80)
        )
        amber = (
            (red >= 150)
            & (green >= 90)
            & (green < 230)
            & (blue < 120)
            & (red > green + 40)
            & (green > blue + 40)
        )
        packed = (
            (red.astype(np.uint32) << 16)
            | (green.astype(np.uint32) << 8)
            | blue.astype(np.uint32)
        )
        diagnostics = self.render_diagnostics()
        diagnostics.update(
            {
                "framebuffer_size": [width, height],
                "color_components": 3,
                "unique_color_count": int(np.unique(packed).size),
                "cyan_pixel_count": int(np.count_nonzero(cyan)),
                "amber_pixel_count": int(np.count_nonzero(amber)),
            }
        )
        return image, diagnostics

    def set_assets(
        self,
        assets: Iterable[Dataset],
        source_id: str | None = None,
        target_id: str | None = None,
        deviation: DeviationResult | None = None,
    ) -> None:
        """Rebuild the scene from current world geometry."""
        visibility = {
            asset_id: bool(actor.GetVisibility()) for asset_id, actor in self._actors.items()
        }
        for actor in self._actors.values():
            self.renderer.RemoveActor(actor)
        self._actors.clear()
        self._display_indices.clear()

        assets = list(assets)
        for asset in assets:
            actor, display_indices = self._make_actor(asset)
            actor.GetProperty().SetColor(
                SOURCE_COLOUR
                if asset.id == source_id
                else TARGET_COLOUR
                if asset.id == target_id
                else NEUTRAL_COLOUR
            )
            actor.SetVisibility(visibility.get(asset.id, True))
            self._actors[asset.id] = actor
            self._display_indices[asset.id] = display_indices
            self.renderer.AddActor(actor)

        self._scalar_bar.SetVisibility(False)
        if deviation is not None and deviation.source_id in self._actors:
            source = next((asset for asset in assets if asset.id == deviation.source_id), None)
            if source is not None:
                self._apply_deviation(self._actors[source.id], source, deviation)

        self._empty_text.SetVisibility(not self._actors)
        self.render()

    def _make_actor(self, asset: Dataset) -> tuple[vtkActor, np.ndarray]:
        points_array = np.ascontiguousarray(asset.world_points(), dtype=np.float64)
        display_indices = np.arange(len(points_array), dtype=np.int64)
        if asset.triangles is None and len(points_array) > MAX_DISPLAY_POINTS:
            display_indices = np.linspace(
                0, len(points_array) - 1, MAX_DISPLAY_POINTS, dtype=np.int64
            )
            points_array = np.ascontiguousarray(points_array[display_indices])
        points = vtkPoints()
        points.SetData(numpy_to_vtk(points_array, deep=True))
        polydata = vtkPolyData()
        polydata.SetPoints(points)

        triangles = getattr(asset, "triangles", None)
        if triangles is not None and len(triangles):
            triangle_array = np.ascontiguousarray(triangles, dtype=np.int64)
            cells = vtkCellArray()
            encoded = np.column_stack(
                (np.full(len(triangle_array), 3, dtype=np.int64), triangle_array)
            ).ravel()
            cells.SetCells(len(triangle_array), numpy_to_vtkIdTypeArray(encoded, deep=True))
            polydata.SetPolys(cells)
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(0.82)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(0.12, 0.17, 0.22)
            actor.GetProperty().SetLineWidth(0.5)
        else:
            glyph = vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()
            mapper = vtkDataSetMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRepresentationToPoints()
            actor.GetProperty().SetPointSize(self._point_size)
        return actor, display_indices

    def _apply_deviation(
        self, actor: vtkActor, source: Dataset, deviation: DeviationResult
    ) -> None:
        count = len(source.points)
        distances = np.full(count, np.nan, dtype=np.float64)
        indices = np.asarray(deviation.indices, dtype=np.int64)
        values = np.asarray(deviation.distances_mm, dtype=np.float64)
        valid = (indices >= 0) & (indices < count)
        distances[indices[valid]] = values[valid]

        polydata = actor.GetMapper().GetInput()
        # Point-cloud mappers receive vtkVertexGlyphFilter output; mesh mappers
        # hold the source vtkPolyData directly. Both expose point data.
        displayed = distances[self._display_indices[source.id]]
        polydata.GetPointData().SetScalars(numpy_to_vtk(displayed, deep=True))
        finite = values[np.isfinite(values)]
        upper = float(np.max(finite)) if finite.size else 1.0
        upper = max(upper, float(deviation.tolerance_mm), 1e-9)
        table = vtkLookupTable()
        table.SetNumberOfTableValues(256)
        table.SetHueRange(0.66, 0.0)
        table.SetSaturationRange(0.85, 0.95)
        table.SetValueRange(0.95, 0.95)
        table.SetNanColor(0.35, 0.40, 0.46, 1.0)
        table.SetRange(0.0, upper)
        table.Build()
        mapper = actor.GetMapper()
        mapper.SetLookupTable(table)
        mapper.SetScalarRange(0.0, upper)
        mapper.SetScalarModeToUsePointData()
        mapper.ScalarVisibilityOn()
        self._scalar_bar.SetLookupTable(table)
        self._scalar_bar.SetVisibility(True)

    def set_asset_visible(self, asset_id: str, visible: bool) -> None:
        actor = self._actors.get(asset_id)
        if actor is not None:
            actor.SetVisibility(visible)
            self.render()

    def is_asset_visible(self, asset_id: str) -> bool:
        actor = self._actors.get(asset_id)
        return bool(actor and actor.GetVisibility())

    def set_point_size(self, size: float) -> None:
        self._point_size = float(size)
        for actor in self._actors.values():
            actor.GetProperty().SetPointSize(self._point_size)
        self.render()

    def fit_view(self) -> None:
        if self._actors:
            self.renderer.ResetCamera()
            self.renderer.ResetCameraClippingRange()
        self.render()

    def render(self) -> None:
        if self._initialized and self.isVisible():
            self.interactor.GetRenderWindow().Render()


Viewport = AssemblyViewport
