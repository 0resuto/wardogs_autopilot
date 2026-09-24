"""Capture zone (ROI) configuration, map cache management, and live capture preview (PySide6)."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ... import PROJECT_ROOT, crashlog
from ...vision import locator
from ..debug_collage import save_debug_snapshot
from ..imaging import to_qpixmap
from ..roi_selector import RoiSelector

MAX_KP_DRAW = 300


class _StringVarCompat:
    """Compatibility shim for Tkinter StringVar in tests."""

    def __init__(self, value: str = "") -> None:
        self._val = str(value)

    def get(self) -> str:
        return self._val

    def set(self, val: str) -> None:
        self._val = str(val)


class RoiTab(QWidget):
    """Tab widget for selecting minimap capture zone, managing SIFT cache, and previewing frames."""

    # Thread-safe Qt signals
    sig_roi_captured = Signal(object, object)
    sig_roi_failed = Signal(str)
    sig_cache_progress = Signal(str)
    sig_cache_done = Signal(str)
    sig_cache_failed = Signal(str)
    sig_save_done = Signal(str)
    sig_save_failed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None,
        cfg: dict[str, Any],
        save_cfg_fn: Callable[[], None],
        screen_cap_supplier: Callable[[], Any],
        loc_thread_supplier: Callable[[], Any],
        on_map_rebuilt: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.save_cfg = save_cfg_fn
        self.get_cap = screen_cap_supplier
        self.get_loc = loc_thread_supplier
        self.on_map_rebuilt = on_map_rebuilt

        self.roi_vars: dict[str, _StringVarCompat] = {}
        self._roi_pick_busy = False
        self._cache_rebuild_busy = False
        self._snap_busy = False
        self._last_snapshot_dir: str | None = None
        self._selector: RoiSelector | None = None

        # Connect signals to GUI thread slots
        self.sig_roi_captured.connect(self._roi_pick_open)
        self.sig_roi_failed.connect(self._roi_pick_fail)
        self.sig_cache_progress.connect(self._on_cache_progress)
        self.sig_cache_done.connect(self._on_cache_done)
        self.sig_cache_failed.connect(self._on_cache_failed)
        self.sig_save_done.connect(self._on_save_done)
        self.sig_save_failed.connect(self._on_save_failed)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Card 1: Minimap Capture Area
        card_roi = QGroupBox("Minimap Screen Capture Area", self)
        roi_layout = QVBoxLayout(card_roi)
        roi_layout.setContentsMargins(12, 14, 12, 12)
        roi_layout.setSpacing(8)

        top_row = QHBoxLayout()
        guide_lbl = QLabel(
            "1. Open minimap in-game (M key)   2. Drag selection box   3. Enter to confirm, Esc to cancel",
            card_roi,
        )
        guide_lbl.setStyleSheet("color: #a0a0a0;")
        top_row.addWidget(guide_lbl, stretch=1)

        cap = self.get_cap()
        if cap and len(cap.monitors) > 2:
            mon_lbl = QLabel("Monitor:", card_roi)
            mon_lbl.setStyleSheet("color: #88c0d0; font-weight: bold;")
            top_row.addWidget(mon_lbl)

            self._mon_sel = QComboBox(card_roi)
            for i in range(1, len(cap.monitors)):
                m = cap.monitors[i]
                self._mon_sel.addItem(f"{i}: {m['width']}x{m['height']}")
            cur_mon = int(self.cfg.get("capture", {}).get("monitor", 1) or 1)
            sel_idx = max(0, min(cur_mon - 1, len(cap.monitors) - 2))
            self._mon_sel.setCurrentIndex(sel_idx)
            self._mon_sel.currentIndexChanged.connect(self._on_monitor_changed)
            top_row.addWidget(self._mon_sel)

        self.pick_btn = QPushButton("⛶ Pick zone on screen", card_roi)
        self.pick_btn.setObjectName("AccentButton")
        self.pick_btn.clicked.connect(self.pick_roi)
        top_row.addWidget(self.pick_btn)

        self.mask_btn = QPushButton("📁 Open mask folder", card_roi)
        self.mask_btn.setToolTip("Open folder containing minimap mask (mm_mask.png)")
        self.mask_btn.clicked.connect(self.open_mask_folder)
        top_row.addWidget(self.mask_btn)
        roi_layout.addLayout(top_row)

        coord_row = QHBoxLayout()
        coord_row.setSpacing(6)
        coord_title = QLabel("Coordinates (px):", card_roi)
        coord_title.setStyleSheet("color: #88c0d0; font-weight: bold;")
        coord_row.addWidget(coord_title)

        current_roi = self.cfg.get("capture", {}).get("mmap_roi", [0, 0, 0, 0])
        self.coord_inputs: dict[str, QLineEdit] = {}
        for i, name in enumerate(("x", "y", "w", "h")):
            lbl = QLabel(name.upper(), card_roi)
            lbl.setStyleSheet("color: #b0b0b0;")
            coord_row.addWidget(lbl)

            val = str(current_roi[i]) if i < len(current_roi) else "0"
            inp = QLineEdit(val, card_roi)
            inp.setFixedWidth(54)
            coord_row.addWidget(inp)
            self.coord_inputs[name] = inp
            compat_var = _StringVarCompat(val)
            self.roi_vars[name] = compat_var

        apply_btn = QPushButton("Apply", card_roi)
        apply_btn.setFixedWidth(64)
        apply_btn.clicked.connect(self.apply_roi)
        coord_row.addWidget(apply_btn)

        self.status_lbl = QLabel("", card_roi)
        self.status_lbl.setStyleSheet("color: #8ae234; font-weight: 500;")
        coord_row.addWidget(self.status_lbl, stretch=1)
        roi_layout.addLayout(coord_row)
        layout.addWidget(card_roi)

        # Card 2: Map Cache & SIFT Feature Index Management
        card_cache = QGroupBox("Map Cache & SIFT Feature Index", self)
        cache_layout = QVBoxLayout(card_cache)
        cache_layout.setContentsMargins(12, 14, 12, 12)
        cache_layout.setSpacing(6)

        row_map = QHBoxLayout()
        row_map.setSpacing(8)
        row_map.addWidget(QLabel("Active Map:", card_cache))

        all_maps = locator.get_store().available_maps() or ["zestafona", "bakurani", "ozeti"]
        self._cache_map_sel = QComboBox(card_cache)
        self._cache_map_sel.addItems(all_maps)
        cur_map = self.cfg.get("map", {}).get("name", "zestafona")
        if cur_map in all_maps:
            self._cache_map_sel.setCurrentText(cur_map)
        self._cache_map_sel.currentTextChanged.connect(lambda: self.cache_status_refresh())
        row_map.addWidget(self._cache_map_sel)

        self._cache_rebuild_btn = QPushButton("Rebuild Cache & SIFT Index", card_cache)
        self._cache_rebuild_btn.clicked.connect(self._cache_rebuild_click)
        row_map.addWidget(self._cache_rebuild_btn)
        row_map.addStretch()
        cache_layout.addLayout(row_map)

        self._cache_status_lbl = QLabel("", card_cache)
        self._cache_status_lbl.setStyleSheet("color: #88c0d0;")
        cache_layout.addWidget(self._cache_status_lbl)
        self.cache_status_refresh()
        layout.addWidget(card_cache)

        # Card 3: Live 3-Panel Capture Diagnostic
        card_prev = QGroupBox("Live Capture Diagnostic", self)
        prev_layout = QVBoxLayout(card_prev)
        prev_layout.setContentsMargins(10, 12, 10, 10)
        prev_layout.setSpacing(6)

        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        self.save_status_lbl = QLabel("", card_prev)
        self.save_status_lbl.setStyleSheet("color: #8ae234; font-size: 8pt; font-weight: 500;")
        hdr_row.addWidget(self.save_status_lbl, stretch=1)

        self.open_snap_btn = QPushButton("📁 Open snapshot", card_prev)
        self.open_snap_btn.setToolTip("Open last saved diagnostic snapshot folder")
        self.open_snap_btn.setVisible(False)
        self.open_snap_btn.clicked.connect(self._open_last_snapshot)
        hdr_row.addWidget(self.open_snap_btn)

        self.save_snap_btn = QPushButton("📷 Save frame", card_prev)
        self.save_snap_btn.setToolTip("Save diagnostic snapshot: 3 preview frames, map crop, and state log to output/")
        self.save_snap_btn.clicked.connect(self.save_debug_frame)
        hdr_row.addWidget(self.save_snap_btn)
        prev_layout.addLayout(hdr_row)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(8)

        # Panel 1: Raw Frame
        p1_box = QVBoxLayout()
        p1_box.setSpacing(4)
        self.raw_title_lbl = QLabel("Raw Capture", card_prev)
        self.raw_title_lbl.setStyleSheet("color: #88c0d0; font-weight: bold; font-size: 9pt;")
        self.raw_preview_lbl = QLabel(card_prev)
        self.raw_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_preview_lbl.setStyleSheet("background-color: #1a1a1a; border-radius: 4px; border: 1px solid #2a2a2a;")
        self.raw_preview_lbl.setMinimumSize(120, 120)
        p1_box.addWidget(self.raw_title_lbl)
        p1_box.addWidget(self.raw_preview_lbl, stretch=1)
        panels_row.addLayout(p1_box, stretch=1)

        # Panel 2: Mask Overlay
        p2_box = QVBoxLayout()
        p2_box.setSpacing(4)
        self.mask_title_lbl = QLabel("Mask Overlay", card_prev)
        self.mask_title_lbl.setStyleSheet("color: #88c0d0; font-weight: bold; font-size: 9pt;")
        self.mask_preview_lbl = QLabel(card_prev)
        self.mask_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_preview_lbl.setStyleSheet("background-color: #1a1a1a; border-radius: 4px; border: 1px solid #2a2a2a;")
        self.mask_preview_lbl.setMinimumSize(120, 120)
        p2_box.addWidget(self.mask_title_lbl)
        p2_box.addWidget(self.mask_preview_lbl, stretch=1)
        panels_row.addLayout(p2_box, stretch=1)

        # Panel 3: SIFT Keypoints & Inliers
        p3_box = QVBoxLayout()
        p3_box.setSpacing(4)
        self.sift_title_lbl = QLabel("SIFT Keypoints", card_prev)
        self.sift_title_lbl.setStyleSheet("color: #88c0d0; font-weight: bold; font-size: 9pt;")
        self.sift_preview_lbl = QLabel(card_prev)
        self.sift_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sift_preview_lbl.setStyleSheet("background-color: #1a1a1a; border-radius: 4px; border: 1px solid #2a2a2a;")
        self.sift_preview_lbl.setMinimumSize(120, 120)
        p3_box.addWidget(self.sift_title_lbl)
        p3_box.addWidget(self.sift_preview_lbl, stretch=1)
        panels_row.addLayout(p3_box, stretch=1)

        prev_layout.addLayout(panels_row, stretch=1)
        layout.addWidget(card_prev, stretch=1)

        # Backward compatibility alias
        self.preview_lbl = self.raw_preview_lbl

    def _on_monitor_changed(self, index: int) -> None:
        mon_idx = index + 1
        self.cfg.setdefault("capture", {})["monitor"] = mon_idx
        self.save_cfg()
        loc = self.get_loc()
        if loc and hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
            loc.cfg.setdefault("capture", {})["monitor"] = mon_idx
        cap = self.get_cap()
        if cap:
            cap.monitor_index = mon_idx

    def open_mask_folder(self) -> None:
        """Open the directory containing the minimap mask in system file explorer."""
        mask_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "masks"))
        os.makedirs(mask_dir, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(mask_dir)):
            try:
                os.startfile(mask_dir)
            except Exception as exc:
                self.status_lbl.setText(f"Failed to open mask folder: {exc}")
                self.status_lbl.setStyleSheet("color: #ff3b3b;")

    def pick_roi(self) -> None:
        """Capture screen asynchronously and launch interactive ROI selector."""
        if self._roi_pick_busy:
            return
        self._roi_pick_busy = True
        self.status_lbl.setText("Grabbing the screen...")
        self.status_lbl.setStyleSheet("color: #88c0d0;")

        cap = self.get_cap()
        if cap is None:
            self._roi_pick_fail("Screen capture unavailable")
            return

        cfg_mon = int(self.cfg.get("capture", {}).get("monitor", 0) or 0)
        if hasattr(self, "_mon_sel"):
            sel_idx = self._mon_sel.currentIndex() + 1
            if 0 < sel_idx < len(cap.monitors):
                mon = cap.monitors[sel_idx]
            else:
                mon = cap.monitors[1] if len(cap.monitors) > 1 else cap.monitors[0]
        elif 0 < cfg_mon < len(cap.monitors):
            mon = cap.monitors[cfg_mon]
        elif len(cap.monitors) > 1:
            mon = cap.monitors[1]
        else:
            mon = cap.monitors[0]

        def grab_and_open() -> None:
            try:
                raw = cap._sct.grab(mon)
                bgr = np.array(raw)[:, :, :3].copy()
                self.sig_roi_captured.emit(bgr, mon)
            except Exception as exc:
                self.sig_roi_failed.emit(str(exc))

        threading.Thread(target=grab_and_open, daemon=True).start()

    def _roi_pick_fail(self, exc: str) -> None:
        self._roi_pick_busy = False
        self.status_lbl.setText(f"Screen grab failed: {exc}")
        self.status_lbl.setStyleSheet("color: #ff3b3b;")

    def _roi_pick_open(self, bgr: np.ndarray, mon: Any) -> None:
        self._roi_pick_busy = False
        self._selector = RoiSelector(
            self.window(),
            bgr,
            mon,
            on_roi=self._roi_done,
            on_cancel=self._roi_cancelled,
        )
        self._selector.show()

    def _roi_done(self, roi: list[int]) -> None:
        self._selector = None
        x, y, w, h = roi
        for name, val in zip(("x", "y", "w", "h"), (x, y, w, h), strict=True):
            self.coord_inputs[name].setText(str(val))
            self.roi_vars[name].set(str(val))

        self._apply_roi_values(roi)
        self.status_lbl.setText(f"OK: [{x}, {y}, {w}, {h}] — saved to config.json")
        self.status_lbl.setStyleSheet("color: #8ae234;")

    def _roi_cancelled(self) -> None:
        self._selector = None
        self.status_lbl.setText("Selection cancelled")
        self.status_lbl.setStyleSheet("color: #8a8a8a;")

    def apply_roi(self) -> None:
        """Parse coordinate entries, validate bounds, and update configuration."""
        try:
            roi = [int(self.roi_vars[n].get() or self.coord_inputs[n].text()) for n in ("x", "y", "w", "h")]
        except ValueError:
            self.status_lbl.setText("Error: integers are required")
            self.status_lbl.setStyleSheet("color: #ff3b3b;")
            return

        x, y, w, h = roi
        if x < 0 or y < 0 or w < 16 or h < 16:
            self.status_lbl.setText("Error: x>=0, y>=0, w>=16, h>=16 required")
            self.status_lbl.setStyleSheet("color: #ff3b3b;")
            return

        for name, val in zip(("x", "y", "w", "h"), (x, y, w, h), strict=True):
            self.coord_inputs[name].setText(str(val))
            self.roi_vars[name].set(str(val))

        self._apply_roi_values(roi)
        self.status_lbl.setText(f"OK: {roi}")
        self.status_lbl.setStyleSheet("color: #8ae234;")

    def _apply_roi_values(self, roi: list[int]) -> None:
        self.cfg.setdefault("capture", {})["mmap_roi"] = roi
        self.save_cfg()

        loc = self.get_loc()
        if loc is not None:
            if hasattr(loc, "set_roi"):
                loc.set_roi(roi)
            elif hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                loc.cfg.setdefault("capture", {})["mmap_roi"] = roi
            if hasattr(loc, "app_cfg") and hasattr(loc.app_cfg, "capture"):
                loc.app_cfg.capture.mmap_roi = roi
            if hasattr(loc, "capture_cfg"):
                loc.capture_cfg.mmap_roi = roi

        cap = self.get_cap()
        if cap is not None and hasattr(cap, "set_roi"):
            cap.set_roi(roi)

        self.update_preview()

    def cache_status_refresh(self) -> None:
        name = self._cache_map_sel.currentText()
        txt, color = self._get_map_cache_status(name)
        self._cache_status_lbl.setText(txt)
        self._cache_status_lbl.setStyleSheet(f"color: {color};")

    def _get_map_cache_status(self, name: str) -> tuple[str, str]:
        if not name:
            return "No map selected", "#8a8a8a"
        data_dir = os.path.join(PROJECT_ROOT, "data", "maps")
        mu_path = os.path.join(data_dir, f"{name}_mu.npy")
        feat_path = os.path.join(data_dir, f"{name}_feat.npz")

        has_mu = os.path.exists(mu_path)
        has_feat = os.path.exists(feat_path)
        missing_previews = [
            sz
            for sz in locator.PREVIEW_SIZES
            if not os.path.exists(os.path.join(data_dir, f"{name}_preview_{sz}.npy"))
        ]

        png_path = os.path.join(data_dir, f"{name}_map.png")
        has_png = os.path.exists(png_path)

        if not has_png and not has_mu and not has_feat:
            return f"Not downloaded: run python tools/download_map.py {name}", "#ff7c7c"
        if not has_mu and not has_feat:
            return "Cache not built: mu.npy and SIFT index missing (press Rebuild)", "#ff7c7c"
        if not has_mu:
            return "Cache incomplete: mu.npy missing (press Rebuild)", "#ff7c7c"
        if not has_feat:
            return "Cache incomplete: SIFT feature index missing (press Rebuild)", "#ffaa00"
        if missing_previews:
            return f"Cache incomplete: missing previews {missing_previews} (press Rebuild)", "#ffaa00"

        try:
            with np.load(feat_path) as idx:
                sig = str(idx.get("gray_sig", [""])[0])
                n_tiles = int(idx.get("gw", 0)) * int(idx.get("gh", 0))
                return f"Ready: mu OK, SIFT index OK ({n_tiles} tiles, {sig}), mipmaps OK", "#8ae234"
        except Exception:
            return "Ready: mu OK, SIFT index OK, mipmaps OK", "#8ae234"

    def _cache_rebuild_click(self) -> None:
        name = self._cache_map_sel.currentText().strip()
        if not name or self._cache_rebuild_busy:
            return
        png_path = os.path.join(PROJECT_ROOT, "data", "maps", f"{name}_map.png")
        if not os.path.exists(png_path):
            QMessageBox.critical(
                self,
                "Map Missing",
                f"Map file '{name}_map.png' not found.\n\nPlease download it with:\n  python tools/download_map.py {name}",
            )
            return

        res = QMessageBox.question(
            self,
            "Map Cache",
            f'Rebuild map cache and SIFT feature index for "{name}"?\n\n'
            "This will regenerate mu.npy, preview mipmaps, and the SIFT descriptor index (~1-2 min).",
        )
        if res != QMessageBox.StandardButton.Yes:
            return

        self._cache_rebuild_busy = True
        self._cache_rebuild_btn.setEnabled(False)
        self._cache_status_lbl.setText("Starting rebuild...")
        self._cache_status_lbl.setStyleSheet("color: #ffaa00;")
        threading.Thread(target=self._cache_rebuild_worker, args=(name,), daemon=True).start()

    def _cache_rebuild_worker(self, name: str) -> None:
        try:
            locator.rebuild_map_cache(name, progress_cb=lambda msg: self.sig_cache_progress.emit(msg))
            self.sig_cache_done.emit(name)
        except Exception as exc:
            crashlog.log(f"rebuild map cache {name}", exc)
            self.sig_cache_failed.emit(str(exc))

    def _on_cache_progress(self, msg: str) -> None:
        self._cache_status_lbl.setText(msg)

    def _on_cache_done(self, name: str) -> None:
        self._cache_rebuild_busy = False
        self._cache_rebuild_btn.setEnabled(True)
        self.cache_status_refresh()
        if self.on_map_rebuilt is not None:
            self.on_map_rebuilt(name)
        QMessageBox.information(self, "Map Cache", f'Map cache and SIFT index successfully rebuilt for "{name}"!')

    def _on_cache_failed(self, err_msg: str) -> None:
        self._cache_rebuild_busy = False
        self._cache_rebuild_btn.setEnabled(True)
        self._cache_status_lbl.setText(f"Rebuild error: {err_msg}")
        self._cache_status_lbl.setStyleSheet("color: #ff3b3b;")
        QMessageBox.critical(self, "Map Cache", f"Rebuild failed: {err_msg}")

    def update_preview(self) -> None:
        """Poll latest captured frame and render diagnostic preview."""
        loc = self.get_loc()
        if loc is None:
            return
        try:
            mm_gray, mm_bgr, mask, latest = loc.snapshot_debug()
        except Exception:
            return
        if mm_gray is None:
            return

        frame = mm_bgr.copy() if (mm_bgr is not None and mm_bgr.size) else cv2.cvtColor(mm_gray, cv2.COLOR_GRAY2BGR)
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
                disp = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_NEAREST)
            else:
                disp = img
            pix = to_qpixmap(disp)
            if not pix.isNull():
                lbl.setPixmap(pix)

        # 1. Raw Frame
        p1 = frame.copy()
        self.raw_title_lbl.setText(f"Raw Capture ({w}×{h})")
        set_panel(self.raw_preview_lbl, p1)

        # 2. Mask Overlay
        p2 = frame.copy()
        if mask is not None and mask.size:
            m = np.asarray(mask, bool)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            overlay = p2.copy()
            overlay[m] = (0, 30, 220)
            cv2.addWeighted(overlay, 0.45, p2, 0.55, 0, p2)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p2, cnts, -1, (0, 160, 255), 1)
            pct = (m.sum() / float(m.size)) * 100.0
            self.mask_title_lbl.setText(f"Mask Overlay ({pct:.1f}%)")
        else:
            self.mask_title_lbl.setText("Mask Overlay (none)")
        set_panel(self.mask_preview_lbl, p2)

        # 3. Keypoints & Inliers
        p3 = frame.copy()
        kp_pts = diag.get("kp_pts") or []
        inlier_pts = diag.get("inlier_pts") or []
        kp_draw = kp_pts[:: int(np.ceil(len(kp_pts) / float(MAX_KP_DRAW)))] if len(kp_pts) > MAX_KP_DRAW else kp_pts
        for pt in kp_draw:
            cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 255), -1)
        for pt in inlier_pts:
            cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 255, 0), -1)
            cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 6, (0, 200, 0), 1)
        n_kp = len(kp_pts)
        n_inl = len(inlier_pts)
        col_hex = "#7ce06a" if n_inl >= 4 else ("#88c0d0" if n_kp > 0 else "#ff7c7c")
        self.sift_title_lbl.setText(f"SIFT Features ({n_kp} pts, {n_inl} inl)")
        self.sift_title_lbl.setStyleSheet(f"color: {col_hex}; font-weight: bold; font-size: 9pt;")
        set_panel(self.sift_preview_lbl, p3)

    def save_debug_frame(self) -> None:
        """Capture live diagnostic snapshot asynchronously."""
        if getattr(self, "_snap_busy", False):
            return
        loc = self.get_loc()
        if loc is None:
            self.save_status_lbl.setText("Locator not active")
            self.save_status_lbl.setStyleSheet("color: #ffaa00; font-size: 8pt;")
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
        self.save_status_lbl.setStyleSheet("color: #8ae234; font-size: 8pt; font-weight: 500;")
        self.open_snap_btn.setVisible(True)
        self.save_snap_btn.setEnabled(True)
        self.save_snap_btn.setText("📷 Save frame")

    def _on_save_failed(self, err_msg: str) -> None:
        self.save_status_lbl.setText(f"Save failed: {err_msg}")
        self.save_status_lbl.setStyleSheet("color: #ff3b3b; font-size: 8pt;")
        self.save_snap_btn.setEnabled(True)
        self.save_snap_btn.setText("📷 Save frame")

    def _open_last_snapshot(self) -> None:
        if self._last_snapshot_dir and os.path.isdir(self._last_snapshot_dir):
            try:
                os.startfile(self._last_snapshot_dir)
            except Exception as exc:
                crashlog.log("open snapshot directory", exc)

