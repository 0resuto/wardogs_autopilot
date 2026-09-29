"""Interactive QGraphicsScene with pyramid mipmap background, waypoints, and vehicle marker."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
)

from .imaging import to_qimage
from .map_renderer import crop_map_viewport
from .theme import BG_DEEP, PANEL_BG, RED, RGB_CANVAS, TEXT, YELLOW


class RouteEndpointItem(QGraphicsItem):
    """Start/end marker of the route: fixed screen size, no interaction.

    The normal (non-edit) map shows only these two markers plus the route line:
    a numbered circle on every recorded point (hundreds for a manual route)
    merged into an unreadable "caterpillar" at map zoom.
    """

    RADIUS = 7.0  # Screen pixels

    def __init__(self, label: str, color: str) -> None:
        super().__init__()
        self._label = label
        self._color = QColor(color)
        self.setZValue(9)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

    def boundingRect(self) -> QRectF:
        return QRectF(-10, -10, 20, 20)

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self._color.darker(140), 1.5))
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor(f"{BG_DEEP}"))
        painter.drawText(QRectF(-8, -8, 16, 16), Qt.AlignmentFlag.AlignCenter, self._label)


class WaypointItem(QGraphicsItem):
    """Interactive route waypoint with fixed screen-space size and drag-and-drop support."""

    RADIUS = 10.0  # Screen pixels

    def __init__(
        self,
        index: int,
        x: float,
        y: float,
        on_moved: Callable[[int, float, float], None] | None = None,
        on_delete: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        self.index = index
        self._on_moved = on_moved
        self._on_delete = on_delete

        self.setPos(x, y)
        self.setZValue(10)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )

    def boundingRect(self) -> QRectF:
        return QRectF(-14, -14, 28, 28)

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Waypoint circle
        pen = QPen(QColor("#7ce06a"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(f"{PANEL_BG}")))
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)

        # Waypoint index number
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor(f"{TEXT}"))
        painter.drawText(
            QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.index + 1)
        )

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            if self._on_delete:
                self._on_delete(self.index)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        delta = event.scenePos() - event.lastScenePos()
        new_pos = self.pos() + delta
        self.setPos(new_pos)
        if self._on_moved:
            self._on_moved(self.index, new_pos.x(), new_pos.y())
        event.accept()


class VehicleMarkerItem(QGraphicsItem):
    """Vehicle marker with position ring and heading arrow (fixed screen size)."""

    def __init__(self) -> None:
        super().__init__()
        self.setZValue(20)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.heading = 0.0

    def boundingRect(self) -> QRectF:
        return QRectF(-20, -20, 40, 40)

    def update_pose(self, x: float, y: float, heading_deg: float) -> None:
        self.setPos(x, y)
        self.heading = heading_deg
        self.update()

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Yellow center vehicle ring
        pen = QPen(QColor(f"{YELLOW}"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 221, 0, 120)))
        painter.drawEllipse(QPointF(0, 0), 6, 6)

        # Red heading pointer arrow
        painter.save()
        painter.rotate(self.heading)
        arrow = QPolygonF(
            [
                QPointF(0, -16),
                QPointF(-5, 4),
                QPointF(0, 0),
                QPointF(5, 4),
            ]
        )
        arrow_pen = QPen(QColor(f"{TEXT}"), 1)
        arrow_pen.setCosmetic(True)
        painter.setPen(arrow_pen)
        painter.setBrush(QBrush(QColor(f"{RED}")))
        painter.drawPolygon(arrow)
        painter.restore()


class MapGraphicsScene(QGraphicsScene):
    """Scene managing native map coordinates (0..32768), mipmaps, and route items."""

    def __init__(
        self,
        map_size: int = 32768,
        thumb: int = 8,
        parent: Any = None,
    ) -> None:
        super().__init__(0, 0, map_size, map_size, parent)
        self._map_size = map_size
        self._thumb = thumb
        self._pyr: dict[int, np.ndarray] | None = None
        self._map8: np.ndarray | None = None

        # Route lines
        self.route_path_item = QGraphicsPathItem()
        route_pen = QPen(QColor("#7ce06a"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        route_pen.setCosmetic(True)
        self.route_path_item.setPen(route_pen)
        self.route_path_item.setZValue(5)
        self.addItem(self.route_path_item)

        # Waypoint items
        self.waypoint_items: list[WaypointItem] = []
        self._endpoint_items: list[RouteEndpointItem] = []
        self.route_pts: list[list[float]] = []
        self._edit_mode = True  # bare scenes keep numbered waypoints (tests/tools)

        # Vehicle marker
        self.vehicle_marker = VehicleMarkerItem()
        self.vehicle_marker.setVisible(False)
        self.addItem(self.vehicle_marker)

        # Callbacks
        self.on_route_changed: Callable[[], None] | None = None

    def set_map(
        self,
        map8: np.ndarray | None,
        map_pyr: dict[int, np.ndarray] | None,
        map_size: int = 32768,
        thumb: int = 8,
    ) -> None:
        self._map8 = map8
        self._pyr = map_pyr
        self._map_size = map_size
        self._thumb = thumb
        self.setSceneRect(0, 0, map_size, map_size)
        self.update()

    def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
        """Render visible slice of map pyramid directly into painter."""
        pyr = self._pyr or ({self._map8.shape[0]: self._map8} if self._map8 is not None else None)
        if not pyr:
            painter.fillRect(rect, QColor(*RGB_CANVAS))
            return

        scale = max(painter.transform().m11(), 1e-5)
        x0 = max(0.0, rect.left())
        y0 = max(0.0, rect.top())
        w = max(1.0, rect.width())
        h = max(1.0, rect.height())

        ru = x0 / float(self._thumb)
        rv = y0 / float(self._thumb)
        rw = w / float(self._thumb)
        rh = h / float(self._thumb)

        s_thumb = scale * float(self._thumb)
        bgr = crop_map_viewport(
            s_thumb, ru, rv, rw, rh, pyr, self._map_size, self._thumb, RGB_CANVAS
        )
        qimg = to_qimage(bgr)
        if not qimg.isNull():
            painter.drawImage(QRectF(x0, y0, w, h), qimg)

    def set_route(self, pts: list[list[float]]) -> None:
        self.route_pts = pts
        self.refresh_waypoints()

    def set_edit_mode(self, enabled: bool) -> None:
        """Switch between the decluttered view (line + endpoints) and numbered
        waypoints (the route editor needs per-point handles)."""
        enabled = bool(enabled)
        if enabled == self._edit_mode:
            return
        self._edit_mode = enabled
        self.refresh_waypoints()

    def refresh_waypoints(self) -> None:
        for wp in self.waypoint_items:
            self.removeItem(wp)
        self.waypoint_items.clear()
        for ep in self._endpoint_items:
            self.removeItem(ep)
        self._endpoint_items.clear()

        if self._edit_mode:
            for i, (x, y) in enumerate(self.route_pts):
                item = WaypointItem(i, x, y, self._on_waypoint_moved, self._on_waypoint_deleted)
                self.addItem(item)
                self.waypoint_items.append(item)
        else:
            if self.route_pts:
                start = RouteEndpointItem("S", "#7ce06a")
                start.setPos(self.route_pts[0][0], self.route_pts[0][1])
                self.addItem(start)
                self._endpoint_items.append(start)
            if len(self.route_pts) > 1:
                end = RouteEndpointItem("E", f"{YELLOW}")
                end.setPos(self.route_pts[-1][0], self.route_pts[-1][1])
                self.addItem(end)
                self._endpoint_items.append(end)

        self._update_route_line()

    def add_waypoint(self, x: float, y: float) -> None:
        clamped_x = max(0.0, min(float(self._map_size), x))
        clamped_y = max(0.0, min(float(self._map_size), y))
        idx = len(self.route_pts)
        self.route_pts.append([clamped_x, clamped_y])
        item = WaypointItem(
            idx, clamped_x, clamped_y, self._on_waypoint_moved, self._on_waypoint_deleted
        )
        self.addItem(item)
        self.waypoint_items.append(item)
        self._update_route_line()
        if self.on_route_changed:
            self.on_route_changed()

    def _on_waypoint_moved(self, index: int, x: float, y: float) -> None:
        if 0 <= index < len(self.route_pts):
            self.route_pts[index] = [x, y]
            self._update_route_line()
            if self.on_route_changed:
                self.on_route_changed()

    def _on_waypoint_deleted(self, index: int) -> None:
        if 0 <= index < len(self.route_pts):
            del self.route_pts[index]
            self.refresh_waypoints()
            if self.on_route_changed:
                self.on_route_changed()

    def _update_route_line(self) -> None:
        path = QPainterPath()
        if len(self.route_pts) > 1:
            path.moveTo(self.route_pts[0][0], self.route_pts[0][1])
            for pt in self.route_pts[1:]:
                path.lineTo(pt[0], pt[1])
        self.route_path_item.setPath(path)

    def update_vehicle(self, x: float, y: float, heading_deg: float) -> None:
        self.vehicle_marker.setVisible(True)
        self.vehicle_marker.update_pose(x, y, heading_deg)

    def hide_vehicle(self) -> None:
        self.vehicle_marker.setVisible(False)
