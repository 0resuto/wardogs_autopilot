"""Fullscreen rectangular zone picker for the minimap ROI (PySide6)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPen, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from .imaging import to_qpixmap


class RoiSelector(QWidget):
    """Shows the target monitor screenshot fullscreen; user drags a selection box."""

    def __init__(
        self,
        parent: QWidget | None,
        monitor_bgr: np.ndarray,
        monitor_geom: dict[str, int],
        on_roi: Callable[[list[int]], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__(None)  # Top-level window to avoid parent event stealing
        self._parent_ref = parent
        self._geom = monitor_geom
        self._on_roi = on_roi
        self._on_cancel = on_cancel

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap = to_qpixmap(monitor_bgr)
        self._img_w = monitor_bgr.shape[1]
        self._img_h = monitor_bgr.shape[0]
        self._left = int(monitor_geom.get("left", 0))
        self._top = int(monitor_geom.get("top", 0))
        self._w = int(monitor_geom.get("width", self._img_w))
        self._h = int(monitor_geom.get("height", self._img_h))

        app = QApplication.instance()
        target_screen = None
        if app is not None and hasattr(app, "screens"):
            for s in app.screens():
                sg = s.geometry()
                if abs(sg.left() - self._left) < 50 and abs(sg.top() - self._top) < 50:
                    target_screen = s
                    break
        if target_screen is not None:
            self.setScreen(target_screen)
            self.setGeometry(target_screen.geometry())
        else:
            self.setGeometry(self._left, self._top, self._w, self._h)

        self._dragging = False
        self._start_pos: QPoint | None = None
        self._pending_roi: tuple[int, int, int, int] | None = None  # (x, y, w, h) in widget coords
        self._apply_rect: QRect | None = None
        self._cancel_rect: QRect | None = None

        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        try:
            self.grabKeyboard()
        except Exception:
            pass

    def closeEvent(self, event: Any) -> None:
        try:
            self.releaseKeyboard()
        except Exception:
            pass
        super().closeEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._pixmap.isNull():
            painter.drawPixmap(self.rect(), self._pixmap)

        if self._pending_roi and self._pending_roi[2] > 5 and self._pending_roi[3] > 5:
            rx, ry, rw, rh = self._pending_roi
            box = QRect(rx, ry, rw, rh)

            # Dim outside only (leaving inside box completely clear and untouched)
            dim_region = QRegion(self.rect()).subtracted(QRegion(box))
            painter.setClipRegion(dim_region)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
            painter.setClipping(False)

            # Bright green border (no fill)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor("#7ce06a"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(box)

            # Floating control badge below or above the rectangle
            bx = max(10, min(self.width() - 320, rx))
            by = ry + rh + 10
            if by + 45 > self.height():
                by = max(10, ry - 50)

            # Background pill for action controls
            badge_rect = QRect(bx, by, 300, 38)
            painter.setPen(QPen(QColor("#383838"), 1))
            painter.setBrush(QColor(25, 25, 25, 235))
            painter.drawRoundedRect(badge_rect, 6, 6)

            # Dimensions text
            dim_txt = f"{rw}×{rh} px"
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(bx + 12, by + 23, dim_txt)

            # Apply button [ ✓ Apply ]
            self._apply_rect = QRect(bx + 115, by + 6, 85, 26)
            painter.setBrush(QColor("#107c41"))
            painter.setPen(QPen(QColor("#107c41"), 1))
            painter.drawRoundedRect(self._apply_rect, 4, 4)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(self._apply_rect, Qt.AlignmentFlag.AlignCenter, "✓ Apply")

            # Cancel button [ ✕ Cancel ]
            self._cancel_rect = QRect(bx + 208, by + 6, 80, 26)
            painter.setBrush(QColor("#454545"))
            painter.setPen(QPen(QColor("#555555"), 1))
            painter.drawRoundedRect(self._cancel_rect, 4, 4)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
            painter.drawText(self._cancel_rect, Qt.AlignmentFlag.AlignCenter, "✕ Cancel")
        else:
            self._apply_rect = None
            self._cancel_rect = None
            # Top instructional banner
            painter.fillRect(QRect(0, 0, self.width(), 40), QColor(20, 20, 20, 220))
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                20,
                26,
                "Drag a box over the minimap. Press Enter, double-click, or click Apply. Press Esc to cancel.",
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            # Click on Apply button
            if self._apply_rect and self._apply_rect.contains(pos):
                self._confirm()
                event.accept()
                return

            # Click on Cancel button
            if self._cancel_rect and self._cancel_rect.contains(pos):
                self._cancel()
                event.accept()
                return

            self._dragging = True
            self._start_pos = pos
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._start_pos is not None:
            x0, y0 = self._start_pos.x(), self._start_pos.y()
            x1, y1 = event.pos().x(), event.pos().y()
            rx = min(x0, x1)
            ry = min(y0, y1)
            rw = abs(x1 - x0)
            rh = abs(y1 - y0)
            self._pending_roi = (rx, ry, rw, rh)
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            if self._start_pos is not None:
                x0, y0 = self._start_pos.x(), self._start_pos.y()
                x1, y1 = event.pos().x(), event.pos().y()
                rx = min(x0, x1)
                ry = min(y0, y1)
                rw = abs(x1 - x0)
                rh = abs(y1 - y0)
                if rw > 10 and rh > 10:
                    self._pending_roi = (rx, ry, rw, rh)
                self.update()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pending_roi:
            rx, ry, rw, rh = self._pending_roi
            if QRect(rx, ry, rw, rh).contains(event.pos()):
                self._confirm()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            event.accept()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._confirm()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _confirm(self) -> None:
        if self._pending_roi and self._pending_roi[2] > 10 and self._pending_roi[3] > 10:
            rx, ry, rw, rh = self._pending_roi
            scale_x = self._img_w / float(self.width()) if self.width() > 0 else 1.0
            scale_y = self._img_h / float(self.height()) if self.height() > 0 else 1.0

            phys_x = int(round(rx * scale_x))
            phys_y = int(round(ry * scale_y))
            phys_w = int(round(rw * scale_x))
            phys_h = int(round(rh * scale_y))

            abs_x = self._left + phys_x
            abs_y = self._top + phys_y
            roi = [int(abs_x), int(abs_y), int(phys_w), int(phys_h)]
            try:
                self.releaseKeyboard()
            except Exception:
                pass
            self.close()
            self._on_roi(roi)
        else:
            self._cancel()

    def _cancel(self) -> None:
        try:
            self.releaseKeyboard()
        except Exception:
            pass
        self.close()
        self._on_cancel()
