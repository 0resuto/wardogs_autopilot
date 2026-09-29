"""Live capture preview card of the Capture section (frame + mask + keypoints)."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..flow_layout import FlowLayout
from ..imaging import to_qpixmap
from ..theme import BLUE, BORDER, GREEN, PANEL_BG, RED, TEXT_DIM, YELLOW
from .common import RoiTabBase

MAX_KP_DRAW = 300

_PREVIEW_STYLE = f"background-color: {PANEL_BG}; border-radius: 4px; border: 1px solid {BORDER};"
_TITLE_STYLE = "font-weight: bold; font-size: 9pt;"
SPEED_OK = "#7ce06a"


class PreviewPanel(QWidget):
    """Titled image panel that follows the width it is given and never clips.

    The panel asks for a comfortable height for whatever width the wrapping
    layout hands it (`heightForWidth`), so a wide stacked panel is taller than a
    narrow side-by-side one, and the frame inside is scaled to fit.
    """

    PREFERRED_WIDTH = 184
    MIN_WIDTH = 110

    def __init__(
        self,
        title: str,
        aspect: float,
        min_height: int,
        max_height: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.aspect = aspect
        self.min_height = min_height
        self.max_height = max_height
        # A vertically `Fixed` widget would be capped at its own hint height by
        # Qt (QWidgetItem::maximumSize), so the panel must be allowed to grow.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(3)

        self.title_lbl = QLabel(title, self)
        self.title_lbl.setStyleSheet(f"color: {BLUE}; {_TITLE_STYLE}")
        box.addWidget(self.title_lbl)

        self.preview_lbl = QLabel(self)
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet(_PREVIEW_STYLE)
        # The image label must never demand width of its own.
        self.preview_lbl.setMinimumSize(0, 0)
        self.preview_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        box.addWidget(self.preview_lbl, stretch=1)

    def flow_height_for_width(self, width: int) -> int:
        """Height the wrapping layout must give this panel at `width`."""
        return max(self.min_height, min(self.max_height, int(max(0, width) * self.aspect)))

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        return self.flow_height_for_width(width)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        return True

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(self.PREFERRED_WIDTH, self.flow_height_for_width(self.PREFERRED_WIDTH))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(self.MIN_WIDTH, self.min_height)

    def set_title(self, text: str, color: str) -> None:
        self.title_lbl.setText(text)
        self.title_lbl.setStyleSheet(f"color: {color}; {_TITLE_STYLE}")


class RoiDiagnosticsMixin(RoiTabBase):
    """Live preview (frame + mask + keypoints merged) for the Capture section."""

    def _build_diagnostics_card(self, layout: QVBoxLayout) -> None:
        # Live capture diagnostic card
        card_prev = QGroupBox("Live Capture Diagnostic", self)
        self.diagnostics_card = card_prev
        prev_layout = QVBoxLayout(card_prev)
        prev_layout.setContentsMargins(10, 12, 10, 10)
        prev_layout.setSpacing(6)

        # The previews share one row while they fit the sidebar width and wrap
        # onto separate rows when the column gets too narrow.
        panels_row = FlowLayout(h_spacing=8, v_spacing=8)
        panels_row.setContentsMargins(0, 0, 0, 0)

        # Merged capture panel: frame + mask overlay + keypoints/inliers
        self.capture_panel = PreviewPanel("Capture", aspect=0.50, min_height=104, max_height=175)
        self.capture_title_lbl = self.capture_panel.title_lbl
        self.preview_lbl = self.capture_panel.preview_lbl
        panels_row.addWidget(self.capture_panel)

        self.speed_panel = PreviewPanel("Speed OCR", aspect=0.32, min_height=76, max_height=125)
        self.speed_title_lbl = self.speed_panel.title_lbl
        self.speed_preview_lbl = self.speed_panel.preview_lbl
        panels_row.addWidget(self.speed_panel)

        prev_layout.addLayout(panels_row)
        layout.addWidget(card_prev)

    def _panel_placeholder(self, lbl: QLabel, text: str) -> None:
        """Muted text instead of an empty black rectangle."""
        lbl.setPixmap(QPixmap())
        lbl.setText(text)
        lbl.setStyleSheet(
            "background-color: rgba(26,27,30,0.85); border-radius: 4px; "
            "border: 1px solid rgba(234,234,234,0.10); color: rgba(234,234,234,0.45);"
        )

    def _show_panel_placeholders(self, reason: str) -> None:
        self._panel_placeholder(self.preview_lbl, f"no frames\n({reason})")
        self._panel_placeholder(self.speed_preview_lbl, "—")

    def update_preview(self) -> None:
        """Poll latest captured frame and render diagnostic preview."""
        loc = self.get_loc()
        if loc is None:
            self._show_panel_placeholders("no capture thread")
            return
        try:
            mm_gray, mm_bgr, mask, latest = loc.snapshot_debug()
        except Exception:
            self._show_panel_placeholders("capture unavailable")
            return
        if mm_gray is None:
            self._show_panel_placeholders("waiting for frames")
            return

        frame = (
            mm_bgr.copy()
            if (mm_bgr is not None and mm_bgr.size)
            else cv2.cvtColor(mm_gray, cv2.COLOR_GRAY2BGR)
        )
        h, w = frame.shape[:2]
        if h < 4 or w < 4:
            return

        diag = (latest.get("diag") or {}) if latest else {}

        def set_panel(lbl: QLabel, img: np.ndarray) -> None:
            ih, iw = img.shape[:2]
            lw = max(lbl.width() - 6, 20)
            lh = max(lbl.height() - 6, 20)
            s = min(lw / float(iw), lh / float(ih))
            if s > 0.05:
                nw = max(1, int(round(iw * s)))
                nh = max(1, int(round(ih * s)))
                disp = cv2.resize(
                    img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_NEAREST
                )
            else:
                disp = img
            pix = to_qpixmap(disp)
            if not pix.isNull():
                lbl.setPixmap(pix)

        # Merged capture panel: frame + mask overlay + keypoints/inliers
        p1 = frame.copy()
        mask_pct: float | None = None
        if mask is not None and mask.size:
            m = np.asarray(mask, bool)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            overlay = p1.copy()
            overlay[m] = (0, 30, 220)
            cv2.addWeighted(overlay, 0.28, p1, 0.72, 0, p1)
            cnts, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(p1, cnts, -1, (0, 160, 255), 1)
            mask_pct = (m.sum() / float(m.size)) * 100.0
        kp_pts = diag.get("kp_pts") or []
        inlier_pts = diag.get("inlier_pts") or []
        kp_draw = (
            kp_pts[:: int(np.ceil(len(kp_pts) / float(MAX_KP_DRAW)))]
            if len(kp_pts) > MAX_KP_DRAW
            else kp_pts
        )
        for pt in kp_draw:
            cv2.circle(p1, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 255), -1)
        for pt in inlier_pts:
            cv2.circle(p1, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 255, 0), -1)
            cv2.circle(p1, (int(round(pt[0])), int(round(pt[1]))), 6, (0, 200, 0), 1)
        n_kp = len(kp_pts)
        n_inl = len(inlier_pts)
        col_hex = GREEN if n_inl >= 4 else (f"{BLUE}" if n_kp > 0 else f"{RED}")
        mask_txt = f" | mask {mask_pct:.1f}%" if mask_pct is not None else ""
        self.capture_panel.set_title(
            f"Capture ({w}×{h}) | {n_kp} pts, {n_inl} inl{mask_txt}", col_hex
        )
        set_panel(self.preview_lbl, p1)

        # Speedometer OCR
        speed_frame = latest.get("speed_frame") if latest else None
        speed_kmh = latest.get("speed_kmh") if latest else None
        speed_ok = bool(latest.get("speed_ok")) if latest else False
        if speed_frame is not None and speed_frame.size > 0:
            p4 = (
                cv2.cvtColor(speed_frame, cv2.COLOR_GRAY2BGR)
                if speed_frame.ndim == 2
                else speed_frame.copy()
            )
            speed_mask = latest.get("speed_mask") if latest else None
            if speed_mask is not None and speed_mask.size > 0:
                overlay = p4.copy()
                overlay[speed_mask > 0] = (0, 180, 0)
                cv2.addWeighted(overlay, 0.45, p4, 0.55, 0, p4)
            speed_boxes = (latest.get("speed_boxes") or ()) if latest else ()
            for box in speed_boxes:
                bx, by, bw, bh = box
                cv2.rectangle(p4, (bx, by), (bx + bw, by + bh), (0, 255, 255), 1)
            if speed_ok and speed_kmh is not None:
                self.speed_panel.set_title(f"Speed OCR: {speed_kmh} km/h", SPEED_OK)
            else:
                self.speed_panel.set_title("Speed OCR: —", YELLOW)
        else:
            if self.cfg.get("capture", {}).get("speed_roi"):
                self.speed_panel.set_title("Speed OCR: waiting", TEXT_DIM)
                self._panel_placeholder(self.speed_preview_lbl, "waiting for the speed ROI…")
            else:
                self.speed_panel.set_title("Speed OCR: disabled", TEXT_DIM)
                self._panel_placeholder(
                    self.speed_preview_lbl, "disabled\n(pick a speed zone to enable)"
                )
            return
        set_panel(self.speed_preview_lbl, p4)
