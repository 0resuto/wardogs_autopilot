"""Live capture diagnostics card for the Capture tab: previews and snapshots."""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ... import PROJECT_ROOT, crashlog
from ..debug_collage import save_debug_snapshot
from ..icons import icon
from ..imaging import to_qpixmap
from ..theme import BLUE, BORDER, GREEN, PANEL_BG, RED, TEXT_DIM, YELLOW
from .common import RoiTabBase

MAX_KP_DRAW = 300


class RoiDiagnosticsMixin(RoiTabBase):
    """Live preview (frame + mask + keypoints merged) and diagnostic snapshots."""

    def _build_diagnostics_card(self, layout: QVBoxLayout) -> None:
        # Live capture diagnostic card
        card_prev = QGroupBox("Live Capture Diagnostic", self)
        self.diagnostics_card = card_prev
        prev_layout = QVBoxLayout(card_prev)
        prev_layout.setContentsMargins(10, 12, 10, 10)
        prev_layout.setSpacing(6)

        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        self.save_status_lbl = QLabel("", card_prev)
        self.save_status_lbl.setStyleSheet(f"color: {GREEN}; font-size: 8pt; font-weight: 500;")
        hdr_row.addWidget(self.save_status_lbl, stretch=1)

        self.open_snap_btn = QPushButton("Open snapshot", card_prev)
        self.open_snap_btn.setIcon(icon("folder"))
        self.open_snap_btn.setToolTip("Open last saved diagnostic snapshot folder")
        self.open_snap_btn.setVisible(False)
        self.open_snap_btn.clicked.connect(self._open_last_snapshot)
        hdr_row.addWidget(self.open_snap_btn)

        self.save_snap_btn = QPushButton("Save frame", card_prev)
        self.save_snap_btn.setIcon(icon("camera"))
        self.save_snap_btn.setToolTip(
            "Save diagnostic snapshot: 3 preview frames, map crop, and state log to output/"
        )
        self.save_snap_btn.clicked.connect(self.save_debug_frame)
        hdr_row.addWidget(self.save_snap_btn)
        prev_layout.addLayout(hdr_row)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(8)

        # Merged capture panel: frame + mask overlay + keypoints/inliers
        p1_box = QVBoxLayout()
        p1_box.setSpacing(4)
        self.capture_title_lbl = QLabel("Capture", card_prev)
        self.capture_title_lbl.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 9pt;")
        self.preview_lbl = QLabel(card_prev)
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet(
            f"background-color: {PANEL_BG}; border-radius: 4px; border: 1px solid {BORDER};"
        )
        self.preview_lbl.setMinimumSize(200, 120)
        p1_box.addWidget(self.capture_title_lbl)
        p1_box.addWidget(self.preview_lbl, stretch=1)
        panels_row.addLayout(p1_box, stretch=3)

        p4_box = QVBoxLayout()
        p4_box.setSpacing(4)
        self.speed_title_lbl = QLabel("Speed OCR", card_prev)
        self.speed_title_lbl.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 9pt;")
        self.speed_preview_lbl = QLabel(card_prev)
        self.speed_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speed_preview_lbl.setStyleSheet(
            f"background-color: {PANEL_BG}; border-radius: 4px; border: 1px solid {BORDER};"
        )
        self.speed_preview_lbl.setMinimumSize(140, 120)
        p4_box.addWidget(self.speed_title_lbl)
        p4_box.addWidget(self.speed_preview_lbl, stretch=1)
        panels_row.addLayout(p4_box, stretch=1)

        prev_layout.addLayout(panels_row, stretch=1)
        layout.addWidget(card_prev, stretch=1)

    def detach_diagnostics(self) -> QWidget:
        """Release the live preview card for the always-visible right pane."""
        layout = self.layout()
        if layout is not None:
            layout.removeWidget(self.diagnostics_card)
        return self.diagnostics_card

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
        self.capture_title_lbl.setText(f"Capture ({w}×{h}) | {n_kp} pts, {n_inl} inl{mask_txt}")
        self.capture_title_lbl.setStyleSheet(
            f"color: {col_hex}; font-weight: bold; font-size: 9pt;"
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
                self.speed_title_lbl.setText(f"Speed OCR: {speed_kmh} km/h")
                self.speed_title_lbl.setStyleSheet(
                    "color: #7ce06a; font-weight: bold; font-size: 9pt;"
                )
            else:
                self.speed_title_lbl.setText("Speed OCR: —")
                self.speed_title_lbl.setStyleSheet(
                    f"color: {YELLOW}; font-weight: bold; font-size: 9pt;"
                )
        else:
            if self.cfg.get("capture", {}).get("speed_roi"):
                self.speed_title_lbl.setText("Speed OCR: waiting")
                self._panel_placeholder(self.speed_preview_lbl, "waiting for the speed ROI…")
            else:
                self.speed_title_lbl.setText("Speed OCR: disabled")
                self._panel_placeholder(
                    self.speed_preview_lbl, "disabled\n(pick a speed zone to enable)"
                )
            self.speed_title_lbl.setStyleSheet(
                f"color: {TEXT_DIM}; font-weight: bold; font-size: 9pt;"
            )
            return
        set_panel(self.speed_preview_lbl, p4)

    def save_debug_frame(self) -> None:
        """Capture live diagnostic snapshot asynchronously."""
        if getattr(self, "_snap_busy", False):
            return
        loc = self.get_loc()
        if loc is None:
            self.save_status_lbl.setText("Locator not active")
            self.save_status_lbl.setStyleSheet(f"color: {YELLOW}; font-size: 8pt;")
            return
        self._snap_busy = True
        self.save_snap_btn.setEnabled(False)
        self.save_snap_btn.setText("Saving...")

        def worker() -> None:
            try:
                mm_gray, mm_bgr, mask, latest = loc.snapshot_debug()
                if mm_gray is None:
                    self.sig_save_failed.emit("No frame captured yet")
                    return
                pose = latest.get("pose") if isinstance(latest, dict) else None
                raw_diag = latest.get("diag") if isinstance(latest, dict) else None
                diag = dict(raw_diag) if isinstance(raw_diag, dict) else {}
                roi = self.cfg.get("capture", {}).get("mmap_roi")
                map_name = self.cfg.get("map", {}).get("name", "zestafona")
                out_dir = os.path.join(PROJECT_ROOT, "output")
                _, snap_dir, _ = save_debug_snapshot(
                    out_dir,
                    mm_gray,
                    mm_bgr,
                    mask,
                    pose,
                    diag,
                    latest,
                    map_name=map_name,
                    roi=roi,
                )
                self.sig_save_done.emit(snap_dir)
            except Exception as exc:
                crashlog.log("save debug snapshot", exc)
                self.sig_save_failed.emit(str(exc))
            finally:
                self._snap_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def _on_save_done(self, snap_dir: str) -> None:
        self._last_snapshot_dir = snap_dir
        folder_name = os.path.basename(snap_dir)
        self.save_status_lbl.setText(f"✓ Saved to {folder_name}")
        self.save_status_lbl.setStyleSheet(f"color: {GREEN}; font-size: 8pt; font-weight: 500;")
        self.open_snap_btn.setVisible(True)
        self.save_snap_btn.setEnabled(True)
        self.save_snap_btn.setText("Save frame")

    def _on_save_failed(self, err_msg: str) -> None:
        self.save_status_lbl.setText(f"Save failed: {err_msg}")
        self.save_status_lbl.setStyleSheet(f"color: {RED}; font-size: 9pt;")
        self.save_snap_btn.setEnabled(True)
        self.save_snap_btn.setText("Save frame")

    def _open_last_snapshot(self) -> None:
        if self._last_snapshot_dir and os.path.isdir(self._last_snapshot_dir):
            try:
                os.startfile(self._last_snapshot_dir)
            except Exception as exc:
                crashlog.log("open snapshot directory", exc)
