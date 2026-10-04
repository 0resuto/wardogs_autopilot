"""Capture zone (ROI) configuration for the studio (PySide6).

The tab is assembled from mixins that keep each responsibility in its own
module: `RoiCacheMixin` (map cache card, asset download, rebuild) and
`RoiDiagnosticsMixin` (the live preview panels shown in the right pane). This
module keeps the ROI/monitor selection and the assembly. Everything that writes
to `output/` lives in the separate Logs section (`logs_tab.py`).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ... import PROJECT_ROOT
from ..flow_layout import FlowLayout
from ..icons import icon
from ..param_tips import tip_for
from ..roi_selector import RoiSelector
from ..theme import BLUE, GREEN, RED, TEXT_DIM, TEXT_MUTED
from .common import StringVarCompat, compact_label
from .roi_cache import RoiCacheMixin, map_cache_status
from .roi_diagnostics import RoiDiagnosticsMixin

__all__ = ["RoiTab", "map_cache_status"]

#: Maximum absolute player-center calibration offset accepted by the Capture tab.
CENTER_LIMIT_PX = 200.0


class RoiTab(RoiCacheMixin, RoiDiagnosticsMixin, QWidget):
    """Tab widget for selecting minimap capture zone, managing SIFT cache, and previewing frames."""

    # Thread-safe Qt signals
    sig_roi_captured = Signal(object, object)
    sig_roi_failed = Signal(str)
    sig_cache_progress = Signal(str)
    sig_cache_done = Signal(str)
    sig_cache_failed = Signal(str)
    sig_download_done = Signal(str)
    sig_download_failed = Signal(str)

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

        self.roi_vars: dict[str, StringVarCompat] = {}
        self._pick_target = "minimap"
        self._roi_pick_busy = False
        self._cache_rebuild_busy = False
        self._cache_download_busy = False
        self._selector: RoiSelector | None = None

        # Connect signals to GUI thread slots
        self.sig_roi_captured.connect(self._roi_pick_open)
        self.sig_roi_failed.connect(self._roi_pick_fail)
        self.sig_cache_progress.connect(self._on_cache_progress)
        self.sig_cache_done.connect(self._on_cache_done)
        self.sig_cache_failed.connect(self._on_cache_failed)
        self.sig_download_done.connect(self._on_download_done)
        self.sig_download_failed.connect(self._on_download_failed)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # The live preview is the primary feedback of the Capture section, so it
        # goes first; configuration cards follow below it.
        self._build_diagnostics_card(layout)
        self._build_roi_card(layout)
        self._build_center_card(layout)
        self._build_speed_card(layout)
        self._build_cache_card(layout)
        layout.addStretch()  # keep the sidebar content top-aligned

    def _build_roi_card(self, layout: QVBoxLayout) -> None:
        # Card 1: Minimap Capture Area
        card_roi = QGroupBox("Minimap Capture", self)
        roi_layout = QVBoxLayout(card_roi)
        roi_layout.setContentsMargins(12, 14, 12, 12)
        roi_layout.setSpacing(8)

        guide_lbl = QLabel(
            "1. Open minimap in-game (M key)\n2. Drag selection box\n"
            "3. Enter to confirm, Esc to cancel",
            card_roi,
        )
        guide_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        guide_lbl.setWordWrap(True)
        compact_label(guide_lbl)
        roi_layout.addWidget(guide_lbl)

        # The button row wraps instead of pinning the sidebar to its width.
        top_row = FlowLayout(h_spacing=6, v_spacing=6)
        cap = self.get_cap()
        if cap and len(cap.monitors) > 2:
            mon_lbl = QLabel("Monitor:", card_roi)
            mon_lbl.setStyleSheet(f"color: {BLUE}; font-weight: bold;")
            top_row.addWidget(mon_lbl)

            self._mon_sel = QComboBox(card_roi)
            self._mon_sel.setFixedWidth(104)
            self._mon_sel.setToolTip(tip_for("monitor"))
            for i in range(1, len(cap.monitors)):
                m = cap.monitors[i]
                self._mon_sel.addItem(f"{i}: {m['width']}x{m['height']}")
            cur_mon = int(self.cfg.get("capture", {}).get("monitor", 1) or 1)
            sel_idx = max(0, min(cur_mon - 1, len(cap.monitors) - 2))
            self._mon_sel.setCurrentIndex(sel_idx)
            self._mon_sel.currentIndexChanged.connect(self._on_monitor_changed)
            top_row.addWidget(self._mon_sel)

        self.pick_btn = QPushButton("Pick zone", card_roi)
        self.pick_btn.setObjectName("AccentButton")
        self.pick_btn.setIcon(icon("crop"))
        self.pick_btn.setToolTip("Pick the minimap capture zone on screen")
        self.pick_btn.clicked.connect(self.pick_roi)
        top_row.addWidget(self.pick_btn)

        self.mask_btn = QPushButton("", card_roi)
        self.mask_btn.setIcon(icon("folder"))
        self.mask_btn.setFixedWidth(34)
        self.mask_btn.setToolTip("Open folder containing minimap mask (mm_mask.png)")
        self.mask_btn.clicked.connect(self.open_mask_folder)
        top_row.addWidget(self.mask_btn)
        roi_layout.addLayout(top_row)

        coord_title = QLabel("Coordinates (px):", card_roi)
        coord_title.setStyleSheet(f"color: {BLUE}; font-weight: bold;")
        roi_layout.addWidget(coord_title)

        current_roi = self.cfg.get("capture", {}).get("mmap_roi", [0, 0, 0, 0])
        self.coord_inputs: dict[str, QLineEdit] = {}
        for pair in (("x", "y"), ("w", "h")):
            coord_row = QHBoxLayout()
            coord_row.setSpacing(6)
            for name in pair:
                i = ("x", "y", "w", "h").index(name)
                lbl = QLabel(name.upper(), card_roi)
                lbl.setStyleSheet(f"color: {TEXT_MUTED};")
                coord_row.addWidget(lbl)

                val = str(current_roi[i]) if i < len(current_roi) else "0"
                inp = QLineEdit(val, card_roi)
                inp.setFixedWidth(48)
                inp.setToolTip(tip_for(f"mmap_roi.{name}"))
                coord_row.addWidget(inp)
                self.coord_inputs[name] = inp
                compat_var = StringVarCompat(val)
                self.roi_vars[name] = compat_var
            coord_row.addStretch()
            roi_layout.addLayout(coord_row)

        apply_row = QHBoxLayout()
        apply_row.setSpacing(6)
        apply_btn = QPushButton("Apply", card_roi)
        apply_btn.setFixedWidth(64)
        apply_btn.clicked.connect(self.apply_roi)
        apply_row.addWidget(apply_btn)

        self.status_lbl = QLabel("", card_roi)
        self.status_lbl.setStyleSheet(f"color: {GREEN}; font-weight: 500;")
        self.status_lbl.setWordWrap(True)
        compact_label(self.status_lbl)
        apply_row.addWidget(self.status_lbl, stretch=1)
        roi_layout.addLayout(apply_row)
        layout.addWidget(card_roi)

    def _build_center_card(self, layout: QVBoxLayout) -> None:
        # The map position is read at the ROI midpoint; if the in-game player
        # arrow is not exactly there, the reported point circles the true one
        # while the minimap rotates. The fields shift the assumed center.
        card_center = QGroupBox("Player Center Calibration", self)
        center_layout = QVBoxLayout(card_center)
        center_layout.setContentsMargins(12, 14, 12, 12)
        center_layout.setSpacing(8)

        guide_lbl = QLabel(
            "If the map marker drifts in a circle while turning, the player arrow\n"
            "is off the ROI midpoint. Align the magenta crosshair in the preview\n"
            "with the arrow using DX/DY (minimap px).",
            card_center,
        )
        guide_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        guide_lbl.setWordWrap(True)
        compact_label(guide_lbl)
        center_layout.addWidget(guide_lbl)

        loc_cfg = self.cfg.setdefault("locator", {})
        center_row = QHBoxLayout()
        center_row.setSpacing(6)
        self.center_inputs: dict[str, QLineEdit] = {}
        self.center_vars: dict[str, StringVarCompat] = {}
        for name, key in (("DX", "center_dx"), ("DY", "center_dy")):
            lbl = QLabel(name, card_center)
            lbl.setStyleSheet(f"color: {TEXT_MUTED};")
            center_row.addWidget(lbl)

            val = str(loc_cfg.get(key, 0.0) or 0.0)
            inp = QLineEdit(val, card_center)
            inp.setFixedWidth(48)
            inp.setToolTip(tip_for(key))
            center_row.addWidget(inp)
            self.center_inputs[key] = inp
            self.center_vars[key] = StringVarCompat(val)
        center_row.addStretch()
        center_layout.addLayout(center_row)

        center_apply_row = QHBoxLayout()
        center_apply_row.setSpacing(6)
        center_apply_btn = QPushButton("Apply", card_center)
        center_apply_btn.setFixedWidth(64)
        center_apply_btn.clicked.connect(self.apply_center)
        center_apply_row.addWidget(center_apply_btn)

        self.center_status_lbl = QLabel("", card_center)
        self.center_status_lbl.setStyleSheet(f"color: {GREEN}; font-weight: 500;")
        self.center_status_lbl.setWordWrap(True)
        compact_label(self.center_status_lbl)
        center_apply_row.addWidget(self.center_status_lbl, stretch=1)
        center_layout.addLayout(center_apply_row)
        layout.addWidget(card_center)

    def apply_center(self) -> None:
        """Parse and validate the player-center calibration fields."""
        values: dict[str, float] = {}
        for key, name in (("center_dx", "DX"), ("center_dy", "DY")):
            s = self.center_inputs[key].text().strip() or self.center_vars[key].get().strip()
            try:
                v = float(s)
            except ValueError:
                self.center_status_lbl.setText(f"Error: {name} must be a number")
                self.center_status_lbl.setStyleSheet(f"color: {RED};")
                return
            if not -CENTER_LIMIT_PX <= v <= CENTER_LIMIT_PX:
                self.center_status_lbl.setText(
                    f"Error: {name} in [-{CENTER_LIMIT_PX:.0f} .. +{CENTER_LIMIT_PX:.0f}] px"
                )
                self.center_status_lbl.setStyleSheet(f"color: {RED};")
                return
            values[key] = v

        for key, v in values.items():
            self.center_inputs[key].setText(str(v))
            self.center_vars[key].set(str(v))

        self.cfg.setdefault("locator", {}).update(values)
        self.save_cfg()

        loc = self.get_loc()
        if loc is not None:
            if hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                loc.cfg.setdefault("locator", {}).update(values)
            if hasattr(loc, "app_cfg") and hasattr(loc.app_cfg, "locator"):
                for key, v in values.items():
                    setattr(loc.app_cfg.locator, key, v)

        self.center_status_lbl.setText("applied")
        self.center_status_lbl.setStyleSheet(f"color: {GREEN};")
        self.update_preview()

    def _build_speed_card(self, layout: QVBoxLayout) -> None:
        card_speed = QGroupBox("Speedometer (OCR)", self)
        speed_layout = QVBoxLayout(card_speed)
        speed_layout.setContentsMargins(12, 14, 12, 12)
        speed_layout.setSpacing(8)

        speed_guide = QLabel(
            "1. Select a box around the speed digits only\n(units and labels are ignored)",
            card_speed,
        )
        speed_guide.setStyleSheet(f"color: {TEXT_MUTED};")
        speed_guide.setWordWrap(True)
        compact_label(speed_guide)
        speed_layout.addWidget(speed_guide)

        speed_top = FlowLayout(h_spacing=8, v_spacing=6)

        self.pick_speed_btn = QPushButton("Pick speed zone", card_speed)
        self.pick_speed_btn.setIcon(icon("crop"))
        self.pick_speed_btn.clicked.connect(self.pick_speed_roi)
        speed_top.addWidget(self.pick_speed_btn)

        self.disable_speed_btn = QPushButton("Disable", card_speed)
        self.disable_speed_btn.clicked.connect(self.disable_speed_roi)
        speed_top.addWidget(self.disable_speed_btn)
        speed_layout.addLayout(speed_top)

        saved_speed = self.cfg.get("capture", {}).get("speed_roi")
        speed_coord_title = QLabel("Coordinates (px):", card_speed)
        speed_coord_title.setStyleSheet(f"color: {BLUE}; font-weight: bold;")
        speed_layout.addWidget(speed_coord_title)

        self.speed_inputs: dict[str, QLineEdit] = {}
        self.speed_vars: dict[str, StringVarCompat] = {}
        for pair in (("x", "y"), ("w", "h")):
            speed_row = QHBoxLayout()
            speed_row.setSpacing(6)
            for name in pair:
                i = ("x", "y", "w", "h").index(name)
                lbl = QLabel(name.upper(), card_speed)
                lbl.setStyleSheet(f"color: {TEXT_MUTED};")
                speed_row.addWidget(lbl)
                val = str(saved_speed[i]) if saved_speed and i < len(saved_speed) else "0"
                inp = QLineEdit(val, card_speed)
                inp.setFixedWidth(48)
                inp.setToolTip(tip_for(f"speed_roi.{name}"))
                speed_row.addWidget(inp)
                self.speed_inputs[name] = inp
                self.speed_vars[name] = StringVarCompat(val)
            speed_row.addStretch()
            speed_layout.addLayout(speed_row)

        speed_apply_row = QHBoxLayout()
        speed_apply_row.setSpacing(6)
        speed_apply_btn = QPushButton("Apply", card_speed)
        speed_apply_btn.setFixedWidth(64)
        speed_apply_btn.clicked.connect(self.apply_speed_roi)
        speed_apply_row.addWidget(speed_apply_btn)

        self.speed_status_lbl = QLabel(
            "enabled" if saved_speed else "disabled",
            card_speed,
        )
        self.speed_status_lbl.setStyleSheet(
            f"color: {GREEN};" if saved_speed else f"color: {TEXT_DIM};"
        )
        self.speed_status_lbl.setWordWrap(True)
        compact_label(self.speed_status_lbl)
        speed_apply_row.addWidget(self.speed_status_lbl, stretch=1)
        speed_layout.addLayout(speed_apply_row)
        layout.addWidget(card_speed)

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
                self.status_lbl.setStyleSheet(f"color: {RED};")

    def _pick_status_label(self) -> QLabel:
        return self.status_lbl if self._pick_target == "minimap" else self.speed_status_lbl

    def pick_roi(self) -> None:
        """Capture screen asynchronously and launch interactive ROI selector."""
        self._start_pick("minimap")

    def pick_speed_roi(self) -> None:
        """Same as pick_roi, but the picked box is stored as capture.speed_roi."""
        self._start_pick("speed")

    def _start_pick(self, target: str) -> None:
        if self._roi_pick_busy:
            return
        self._pick_target = target
        self._roi_pick_busy = True
        label = self._pick_status_label()
        label.setText("Grabbing the screen...")
        label.setStyleSheet(f"color: {BLUE};")

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
        label = self._pick_status_label()
        label.setText(f"Screen grab failed: {exc}")
        label.setStyleSheet(f"color: {RED};")

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
        if self._pick_target == "speed":
            for name, val in zip(("x", "y", "w", "h"), (x, y, w, h), strict=True):
                self.speed_inputs[name].setText(str(val))
                self.speed_vars[name].set(str(val))
            self._apply_speed_roi_values(roi)
            self.speed_status_lbl.setText(f"OK: [{x}, {y}, {w}, {h}] — saved to config.json")
            self.speed_status_lbl.setStyleSheet(f"color: {GREEN};")
            return

        for name, val in zip(("x", "y", "w", "h"), (x, y, w, h), strict=True):
            self.coord_inputs[name].setText(str(val))
            self.roi_vars[name].set(str(val))

        self._apply_roi_values(roi)
        self.status_lbl.setText(f"OK: [{x}, {y}, {w}, {h}] — saved to config.json")
        self.status_lbl.setStyleSheet(f"color: {GREEN};")

    def _roi_cancelled(self) -> None:
        self._selector = None
        label = self._pick_status_label()
        label.setText("Selection cancelled")
        label.setStyleSheet(f"color: {TEXT_DIM};")

    def apply_speed_roi(self) -> None:
        """Parse and validate the speedometer ROI entry fields."""
        try:
            roi = [
                int(self.speed_inputs[n].text().strip() or self.speed_vars[n].get().strip())
                for n in ("x", "y", "w", "h")
            ]
        except ValueError:
            self.speed_status_lbl.setText("Error: integers are required")
            self.speed_status_lbl.setStyleSheet(f"color: {RED};")
            return

        x, y, w, h = roi
        if x < 0 or y < 0 or w < 8 or h < 8:
            self.speed_status_lbl.setText("Error: x>=0, y>=0, w>=8, h>=8 required")
            self.speed_status_lbl.setStyleSheet(f"color: {RED};")
            return

        for name, val in zip(("x", "y", "w", "h"), (x, y, w, h), strict=True):
            self.speed_inputs[name].setText(str(val))
            self.speed_vars[name].set(str(val))

        self._apply_speed_roi_values(roi)
        self.speed_status_lbl.setText(f"OK: {roi}")
        self.speed_status_lbl.setStyleSheet(f"color: {GREEN};")

    def disable_speed_roi(self) -> None:
        """Turn the speedometer OCR off (capture.speed_roi = None)."""
        for name in ("x", "y", "w", "h"):
            self.speed_inputs[name].setText("0")
            self.speed_vars[name].set("0")
        self._apply_speed_roi_values(None)
        self.speed_status_lbl.setText("disabled")
        self.speed_status_lbl.setStyleSheet(f"color: {TEXT_DIM};")

    def _apply_speed_roi_values(self, roi: list[int] | None) -> None:
        self.cfg.setdefault("capture", {})["speed_roi"] = roi
        self.save_cfg()

        loc = self.get_loc()
        if loc is not None:
            if hasattr(loc, "set_speed_roi"):
                loc.set_speed_roi(roi)
            elif hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                loc.cfg.setdefault("capture", {})["speed_roi"] = roi
            if hasattr(loc, "app_cfg") and hasattr(loc.app_cfg, "capture"):
                loc.app_cfg.capture.speed_roi = roi

        self.update_preview()

    def apply_roi(self) -> None:
        """Parse coordinate entries, validate bounds, and update configuration."""
        try:
            roi = [
                int(self.coord_inputs[n].text().strip() or self.roi_vars[n].get().strip())
                for n in ("x", "y", "w", "h")
            ]
        except ValueError:
            self.status_lbl.setText("Error: integers are required")
            self.status_lbl.setStyleSheet(f"color: {RED};")
            return

        x, y, w, h = roi
        if x < 0 or y < 0 or w < 16 or h < 16:
            self.status_lbl.setText("Error: x>=0, y>=0, w>=16, h>=16 required")
            self.status_lbl.setStyleSheet(f"color: {RED};")
            return

        for name, val in zip(("x", "y", "w", "h"), (x, y, w, h), strict=True):
            self.coord_inputs[name].setText(str(val))
            self.roi_vars[name].set(str(val))

        self._apply_roi_values(roi)
        self.status_lbl.setText(f"OK: {roi}")
        self.status_lbl.setStyleSheet(f"color: {GREEN};")

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
