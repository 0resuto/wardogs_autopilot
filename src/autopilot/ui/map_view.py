"""Custom QGraphicsView with smooth pan, zoom, fit, and on-canvas HUD toolbar."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .icons import icon
from .map_scene import MapGraphicsScene


class MapGraphicsView(QGraphicsView):
    """Interactive map viewport supporting smooth pan, zoom, fit, and HUD controls."""

    def __init__(self, scene: MapGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.map_scene = scene

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._route_edit_mode = True
        self._pan_active = False
        self._pan_start_pos = QPointF()
        self._has_fitted = False

    def set_route_mode(self, enabled: bool) -> None:
        self._route_edit_mode = enabled
        self.map_scene.set_edit_mode(enabled)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self._has_fitted and self.width() > 100 and self.height() > 100:
            self.fit_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._has_fitted = True
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in(1.2)
        elif delta < 0:
            self.zoom_out(1.2)
        event.accept()

    def mousePressEvent(self, event: Any) -> None:
        # Middle click pans always, left click pans when not in route edit mode
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and not self._route_edit_mode
        ):
            self._pan_active = True
            self._has_fitted = True
            self._pan_start_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # In route edit mode, left click on empty scene or line adds a waypoint
        if event.button() == Qt.MouseButton.LeftButton and self._route_edit_mode:
            item = self.itemAt(event.position().toPoint())
            if item is None or item == self.map_scene.route_path_item:
                scene_pos = self.mapToScene(event.position().toPoint())
                self.map_scene.add_waypoint(scene_pos.x(), scene_pos.y())
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._pan_active:
            delta = event.position() - self._pan_start_pos
            self._pan_start_pos = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._pan_active:
            self._pan_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.fit_view()
        event.accept()

    def zoom_in(self, factor: float = 1.25) -> None:
        self._has_fitted = True
        self.scale(factor, factor)

    def zoom_out(self, factor: float = 1.25) -> None:
        self._has_fitted = True
        self.scale(1.0 / factor, 1.0 / factor)

    def fit_view(self) -> None:
        rect = self.map_scene.sceneRect()
        if not rect.isEmpty() and self.width() > 10 and self.height() > 10:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._has_fitted = True

    def center_on_coords(self, x: float, y: float) -> None:
        self._has_fitted = True
        self.centerOn(x, y)


class InteractiveMapWidget(QWidget):
    """Encapsulates MapGraphicsView with a compact top HUD toolbar."""

    def __init__(
        self,
        map_size: int = 32768,
        thumb: int = 8,
        enable_route_editing: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.scene = MapGraphicsScene(map_size, thumb, self)
        self.view = MapGraphicsView(self.scene, self)
        self.view.set_route_mode(enable_route_editing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Docked HUD toolbar (styled as a compact floating glass pill)
        tb = QWidget(self)
        tb.setObjectName("MapHud")
        tb.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.toolbar_layout = QHBoxLayout(tb)
        self.toolbar_layout.setContentsMargins(6, 3, 6, 3)
        self.toolbar_layout.setSpacing(4)

        # Zoom is the mouse wheel and Fit happens at maximum zoom-out; only
        # the follow-centering toggle stays here.
        self.btn_center = QPushButton("Center", tb)
        self.btn_center.setIcon(icon("crosshair"))
        self.btn_center.setCheckable(True)
        self.btn_center.setToolTip("Follow the vehicle position on the map")
        self.btn_center.toggled.connect(self._on_follow_toggled)
        self.toolbar_layout.addWidget(self.btn_center)

        if enable_route_editing:
            self.btn_mode = QPushButton("Route Mode", tb)
            self.btn_mode.setIcon(icon("pencil"))
            self.btn_mode.setCheckable(True)
            self.btn_mode.setChecked(True)
            self.btn_mode.toggled.connect(self._toggle_mode)
            self.toolbar_layout.addWidget(self.btn_mode)

        layout.addWidget(self.view, stretch=1)
        layout.addWidget(tb, alignment=Qt.AlignmentFlag.AlignLeft)

        self._on_center_cb: Any = None

    def add_hud_widget(self, widget: QWidget) -> None:
        """Append a custom widget to the HUD toolbar."""
        self.toolbar_layout.addWidget(widget)

    def set_on_center(self, callback: Any) -> None:
        self._on_center_cb = callback

    def is_follow_centered(self) -> bool:
        """Whether the view should keep re-centering on the current vehicle position."""
        return self.btn_center.isChecked()

    def _on_follow_toggled(self, checked: bool) -> None:
        self.btn_center.setText("Following" if checked else "Center")
        if checked:
            self._on_center_click()

    def _on_center_click(self) -> None:
        if self._on_center_cb:
            self._on_center_cb()
        else:
            self.view.fit_view()

    def _toggle_mode(self, checked: bool) -> None:
        self.view.set_route_mode(checked)
        self.btn_mode.setIcon(icon("pencil" if checked else "move"))
        self.btn_mode.setText("Route Mode" if checked else "Pan Mode")
