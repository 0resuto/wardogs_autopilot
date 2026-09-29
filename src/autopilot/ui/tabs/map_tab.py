"""Unified Map tab: map view, route planning, autopilot control, and tuning.

The tab is assembled from mixins that keep each responsibility in its own
module: `MapTuningMixin` (locator/vehicle tuning panel), `MapPresetsMixin`
(preset management), `MapRouteEditMixin` (route editing and status).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...common.config import AppConfig
from ...navigation.follow import FollowDriver
from ...navigation.manual_record import ManualDriveRecorder
from ...vision import locator
from ..map_view import InteractiveMapWidget
from ..presets import PresetManager
from ..theme import BLUE, BORDER, GREEN, PANEL_BG, RED, TEXT, TEXT_DIM, TEXT_MUTED, YELLOW
from .common import StringVarCompat, compact_label
from .map_presets import MapPresetsMixin
from .map_route_edit import MapRouteEditMixin
from .map_tuning import MapTuningMixin


class MapTab(MapTuningMixin, MapPresetsMixin, MapRouteEditMixin, QWidget):
    """Unified tab widget for interactive map exploration, route planning, autopilot driving, and locator tuning."""

    def __init__(
        self,
        parent: QWidget | None,
        cfg: dict[str, Any] | AppConfig,
        save_cfg_fn: Callable[[], None],
        loc_thread_supplier: Callable[[], Any],
        map_name_supplier: Callable[[], str] | None = None,
        map_store_supplier: Callable[[], Any] | None = None,
        on_pick_roi: Callable[[], None] | None = None,
        on_open_capture: Callable[[], None] | None = None,
        app_cfg: AppConfig | None = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(cfg, AppConfig):
            self.app_cfg = cfg
            self.cfg = cfg.to_dict()
        else:
            self.cfg = cfg
            self.app_cfg = app_cfg or AppConfig()

        self.save_cfg = save_cfg_fn
        self.get_loc = loc_thread_supplier
        self.get_map_name = map_name_supplier or (
            lambda: str(self.cfg.get("map", {}).get("name", "zestafona"))
        )
        self.get_store = map_store_supplier or locator.get_store
        self.on_pick_roi = on_pick_roi
        self.on_open_capture = on_open_capture
        self._map_loaded = False

        self.map_name = self.get_map_name()
        self._disp_th: float | None = None
        self._last_loc: dict[str, Any] | None = None

        self.preset_mgr = PresetManager()
        self.driver: FollowDriver | None = None
        self._manual_rec: ManualDriveRecorder | None = None
        self._edit_snapshot: list[list[float]] | None = None

        self.tune_vars: dict[str, StringVarCompat] = {}
        self.tune_inputs: dict[str, QLineEdit] = {}

        self._build_ui()

    @property
    def route_pts(self) -> list[list[float]]:
        return self.map_widget.scene.route_pts

    @route_pts.setter
    def route_pts(self, pts: list[list[float]]) -> None:
        self.map_widget.scene.set_route(pts)
        self.routes_refresh()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        self._build_toolbar(root_layout)
        self._build_tuning_panel(root_layout)
        self._build_map_area(root_layout)
        self._build_map_notice()
        self._build_bottom_bar(root_layout)
        self._update_map_notice()
        self.preset_reload()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if getattr(self, "_map_notice", None) is not None:
            self._place_map_notice()

    def _build_map_notice(self) -> None:
        """Empty-state card over the canvas (missing assets / loading)."""
        self._map_notice = QFrame(self.map_widget)
        self._map_notice.setObjectName("MapNotice")
        self._map_notice.setStyleSheet(
            f"QFrame#MapNotice {{ background-color: {PANEL_BG}; "
            f"border: 1px solid {BORDER}; border-radius: 12px; }}"
            f"QFrame#MapNotice QLabel {{ color: {TEXT}; font-size: 10pt; border: none; }}"
        )
        lay = QVBoxLayout(self._map_notice)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        self._map_notice_lbl = QLabel("", self._map_notice)
        self._map_notice_lbl.setWordWrap(True)
        self._map_notice_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._map_notice_lbl.setMaximumWidth(360)
        lay.addWidget(self._map_notice_lbl)
        self._map_notice_btn = QPushButton("Open Capture zone", self._map_notice)
        self._map_notice_btn.setObjectName("AccentButton")
        self._map_notice_btn.clicked.connect(self._open_capture_zone)
        lay.addWidget(self._map_notice_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._map_notice.hide()

    def _place_map_notice(self) -> None:
        notice = self._map_notice
        notice.adjustSize()
        area = self.map_widget.rect()
        notice.move(area.center() - notice.rect().center())

    def _open_capture_zone(self) -> None:
        if self.on_open_capture is not None:
            self.on_open_capture()

    def _show_map_notice(self, text: str, *, with_button: bool) -> None:
        self._map_notice_lbl.setText(text)
        self._map_notice_btn.setVisible(with_button and self.on_open_capture is not None)
        self._place_map_notice()
        self._map_notice.show()
        self._map_notice.raise_()

    def map_asset_state(self) -> str:
        """Asset state of the active map: 'missing', 'no_index' or 'ok'."""
        name = self.get_map_name()
        data_dir = getattr(self.get_store(), "data_maps_dir", None)
        if not data_dir:
            return "ok"

        def exists(fname: str) -> bool:
            return os.path.exists(os.path.join(str(data_dir), fname))

        if not any(exists(f"{name}_preview_{n}.npy") for n in locator.PREVIEW_SIZES):
            return "missing"
        if not exists(f"{name}_mu.npy"):
            return "missing"
        if not exists(f"{name}_feat.npz"):
            return "no_index"
        return "ok"

    def _update_map_notice(self) -> None:
        """Reflect the map asset/load state on the canvas overlay."""
        state = self.map_asset_state()
        if state == "missing":
            self._show_map_notice(
                f'Map assets for "{self.get_map_name()}" are not downloaded.\n'
                "Open Capture zone → Map Cache → Download.",
                with_button=True,
            )
        elif state == "no_index":
            self._show_map_notice(
                f'SIFT index for "{self.get_map_name()}" is missing '
                "(localization reports no_index).\n"
                "Use Capture zone → Map Cache → Download.",
                with_button=True,
            )
        elif not self._map_loaded:
            self._show_map_notice("Loading map…", with_button=False)
        else:
            self._map_notice.hide()

    def map_loading(self) -> None:
        """Called by the app when a map (re)load starts."""
        self._map_loaded = False
        self._update_map_notice()

    def map_loaded(self) -> None:
        """Called by the app once the map scene has been populated."""
        self._map_loaded = True
        self._update_map_notice()

    def _build_toolbar(self, root_layout: QVBoxLayout) -> None:
        # --- Top toolbar: Route presets & Tuning toggle ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        lbl_preset = QLabel("Preset:", self)
        lbl_preset.setStyleSheet("font-weight: bold;")
        top_bar.addWidget(lbl_preset)

        self.p_sel = QComboBox(self)
        self.p_sel.setFixedWidth(130)
        top_bar.addWidget(self.p_sel)

        self._preset_menu_btn = QPushButton("⋯", self)
        self._preset_menu_btn.setFixedWidth(30)
        self._preset_menu_btn.setToolTip("Preset actions: load, save as, new, delete, reload")
        preset_menu = QMenu(self._preset_menu_btn)
        act_load = preset_menu.addAction("Load preset")
        act_load.triggered.connect(lambda: self.preset_load_sel())
        act_save = preset_menu.addAction("Save As…")
        act_save.triggered.connect(self.preset_save)
        act_new = preset_menu.addAction("New route…")
        act_new.triggered.connect(self.route_new)
        act_del = preset_menu.addAction("Delete preset")
        act_del.triggered.connect(self.preset_delete)
        preset_menu.addSeparator()
        act_reload = preset_menu.addAction("Reload list")
        act_reload.triggered.connect(self.preset_reload)
        self._preset_menu_btn.setMenu(preset_menu)
        top_bar.addWidget(self._preset_menu_btn)

        btn_clear = QPushButton("🗑 Clear", self)
        btn_clear.clicked.connect(self.routes_clear)
        top_bar.addWidget(btn_clear)

        self._reverse_btn = QPushButton("⇄ Reverse", self)
        self._reverse_btn.setToolTip("Reverse the route direction (F8)")
        self._reverse_btn.clicked.connect(lambda: self.routes_invert())
        top_bar.addWidget(self._reverse_btn)

        self._edit_btn = QPushButton("✏ Edit", self)
        self._edit_btn.setToolTip("Edit the route points on the map")
        self._edit_btn.clicked.connect(lambda: self._set_edit_mode(True))
        top_bar.addWidget(self._edit_btn)

        self._apply_btn = QPushButton("✓ Apply", self)
        self._apply_btn.clicked.connect(self.route_edit_apply)
        self._apply_btn.hide()
        top_bar.addWidget(self._apply_btn)

        self._cancel_btn = QPushButton("✕ Cancel", self)
        self._cancel_btn.clicked.connect(self.route_edit_cancel)
        self._cancel_btn.hide()
        top_bar.addWidget(self._cancel_btn)

        self._route_locked: list[QWidget] = [
            self.p_sel,
            self._preset_menu_btn,
            btn_clear,
            self._reverse_btn,
        ]

        self.dbg_ck = QCheckBox("Nav log", self)
        self.dbg_ck.setChecked(bool(self.app_cfg.navigator.debug))
        top_bar.addWidget(self.dbg_ck)

        self._manual_rec_ck = QCheckBox("⏺ Record my driving", self)
        self._manual_rec_ck.setToolTip(
            "Temporary: log your own W/A/S/D/SPACE presses and poses to "
            "output/manual_dbg_*.jsonl while driving by hand"
        )
        self._manual_rec_ck.toggled.connect(self.toggle_manual_record)
        top_bar.addWidget(self._manual_rec_ck)

        top_bar.addStretch()

        self._tune_toggle_btn = QPushButton("⚙ Tuning ▾", self)
        self._tune_toggle_btn.clicked.connect(self.toggle_tuning_panel)
        top_bar.addWidget(self._tune_toggle_btn)

        root_layout.addLayout(top_bar)

    def _build_map_area(self, root_layout: QVBoxLayout) -> None:
        # --- Interactive Map Canvas / Scene ---
        # Route editing is off until the Edit button is pressed: by default the
        # left button pans and the route cannot be changed by an accidental click.
        self.map_widget = InteractiveMapWidget(enable_route_editing=False, parent=self)
        self.map_widget.scene.on_route_changed = self.routes_refresh
        self.map_widget.set_on_center(self._center_vehicle)
        root_layout.addWidget(self.map_widget, stretch=1)

    def _build_bottom_bar(self, root_layout: QVBoxLayout) -> None:
        # --- Bottom Command & Navigation Control Center ---
        bot_box = QVBoxLayout()
        bot_box.setContentsMargins(0, 0, 0, 0)
        bot_box.setSpacing(2)

        bot_bar = QHBoxLayout()
        bot_bar.setSpacing(8)

        self.follow_btn = QPushButton("▶ Start Follow (F6)", self)
        self.follow_btn.setObjectName("SuccessButton")
        self.follow_btn.clicked.connect(self.follow_toggle)
        bot_bar.addWidget(self.follow_btn)

        self.estop_btn = QPushButton("🛑 EMERGENCY STOP (F7)", self)
        self.estop_btn.setObjectName("EStopButton")
        self.estop_btn.clicked.connect(self.emergency_stop)
        bot_bar.addWidget(self.estop_btn)

        self.routes_status = QLabel("", self)
        self.routes_status.setStyleSheet(f"color: {BLUE}; font-weight: bold;")
        compact_label(self.routes_status)
        bot_bar.addWidget(self.routes_status, stretch=1)
        bot_box.addLayout(bot_bar)

        info_bar = QHBoxLayout()
        info_bar.setSpacing(8)

        self.map_status = QLabel("", self)
        self.map_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9pt;")
        compact_label(self.map_status)
        info_bar.addWidget(self.map_status, stretch=1)

        self._hint_lbl = QLabel("LMB: Pan | Wheel: Zoom | ✏ Edit to modify the route", self)
        self._hint_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        info_bar.addWidget(self._hint_lbl)
        bot_box.addLayout(info_bar)

        root_layout.addLayout(bot_box)

    def toggle_manual_record(self, enabled: bool) -> None:
        """Start/stop recording the user's own driving (temporary tuning aid)."""
        if not enabled:
            self.stop_manual_record()
            return
        loc = self.get_loc()
        if loc is None:
            self._manual_rec_ck.setChecked(False)
            return
        params = dict(
            source="manual",
            map=self.get_map_name(),
            vehicle=self.app_cfg.navigator.vehicle_profile,
            speed_cap_kmh=self.app_cfg.navigator.speed_cap_kmh,
        )
        self._manual_rec = ManualDriveRecorder(
            loc=loc, route=self.route_pts, params=params, out_dir="output"
        )
        self._manual_rec.start()

    def stop_manual_record(self) -> None:
        """Stop manual-driving recording if it is running."""
        rec = self._manual_rec
        if rec is not None:
            rec.stop()
            self._manual_rec = None
        if self._manual_rec_ck.isChecked():
            self._manual_rec_ck.setChecked(False)

    def update_loc(self, last_loc: dict[str, Any] | None) -> None:
        """Receive latest localization pose from main event loop."""
        self._last_loc = last_loc
        if last_loc is None:
            self.map_widget.scene.hide_vehicle()
            self.map_status.setText("")
            return

        pose = last_loc.get("pose")
        mp = last_loc.get("map_px_disp") or last_loc.get("map_px")
        if pose is None or mp is None:
            self.map_status.setText("")
            self.map_widget.scene.hide_vehicle()
            return

        heading = locator.heading_deg(pose)
        if self._disp_th is None:
            self._disp_th = heading
        else:
            dth = (heading - self._disp_th + 540.0) % 360.0 - 180.0
            self._disp_th = (self._disp_th + dth * 0.4) % 360.0
        heading = self._disp_th

        self.map_widget.scene.update_vehicle(mp[0], mp[1], heading)
        if self.map_widget.is_follow_centered():
            self.map_widget.view.center_on_coords(mp[0], mp[1])
        speed_txt = ""
        if last_loc.get("speed_ok") and last_loc.get("speed_kmh") is not None:
            speed_txt = f" | v={last_loc['speed_kmh']} km/h"
        self.map_status.setText(
            f"x={mp[0]:.0f} y={mp[1]:.0f} | {heading:.1f}° | "
            f"s={pose['s']:.2f} inl={pose.get('inl', 0)}{speed_txt}"
        )

        diag = last_loc.get("diag")
        if isinstance(diag, dict) and diag.get("reject"):
            self.dbg_text.setText(f"{diag.get('reject')}: {diag.get('detail', '')[:50]}")

    def _center_vehicle(self) -> None:
        loc = self._last_loc
        if loc is None:
            loc = getattr(self.get_loc(), "latest", None)
        mp = (loc.get("map_px_disp") or loc.get("map_px")) if loc else None
        if mp is not None:
            self.map_widget.view.center_on_coords(mp[0], mp[1])
        else:
            self.map_widget.view.fit_view()

    def _set_follow_state(self, running: bool) -> None:
        if running:
            self.follow_btn.setText("⏸ Pause (F6)")
            self.follow_btn.setObjectName("DangerButton")
        else:
            self.follow_btn.setText("▶ Start Follow (F6)")
            self.follow_btn.setObjectName("SuccessButton")
        self.follow_btn.style().unpolish(self.follow_btn)
        self.follow_btn.style().polish(self.follow_btn)
        self._set_route_controls_enabled()

    def follow_toggle(self, silent: bool = False) -> None:
        """Start or stop the background FollowDriver thread."""
        if self.driver is not None:
            self.driver.stop()
            self.driver = None
            self._set_follow_state(False)
            self.routes_status.setText("Autopilot stopped")
            self.routes_status.setStyleSheet(f"color: {BLUE};")
            return

        if self._edit_snapshot is not None:
            self.routes_status.setText("Finish editing (Apply or Cancel) before starting")
            self.routes_status.setStyleSheet(f"color: {YELLOW};")
            if not silent:
                QMessageBox.information(
                    self, "Routes", "Finish editing (Apply or Cancel) before starting"
                )
            return

        if len(self.route_pts) < 2:
            self.routes_status.setText("Route not set (at least 2 points required)")
            self.routes_status.setStyleSheet(f"color: {YELLOW};")
            if not silent:
                QMessageBox.warning(self, "Routes", "Route not set (at least 2 points required)")
            return

        self.stop_manual_record()
        loc_thread = self.get_loc()
        nav_cfg = self.app_cfg.navigator
        try:
            kb = self._make_kb(nav_cfg.port)
        except (OSError, RuntimeError) as exc:
            self.routes_status.setText(f"Key driver error: {exc}")
            self.routes_status.setStyleSheet(f"color: {RED};")
            if not silent:
                QMessageBox.critical(self, "Autopilot", f"Failed to create key driver:\n{exc}")
            return

        self.driver = FollowDriver(
            loc=loc_thread,
            pts=[(p[0], p[1]) for p in self.route_pts],
            nav_cfg=nav_cfg,
            kb=kb,
            px_per_m=self._map_px_per_m(),
            debug=self.dbg_ck.isChecked(),
        )
        self.driver.start()
        self._set_follow_state(True)
        self.routes_status.setText("Autopilot driving...")
        self.routes_status.setStyleSheet(f"color: {GREEN};")

    def _make_kb(self, port: str) -> Any:
        from ...hardware.arduino_keyboard import ArduinoKeyDriver

        try:
            return ArduinoKeyDriver(port)
        except Exception as exc:
            raise RuntimeError(f"Arduino ({port}) unavailable: {exc}") from exc

    def emergency_stop(self) -> None:
        """Emergency stop handler invoked via global hotkey or button."""
        if self.driver is not None:
            self.driver.stop()
            self.driver = None
        self._set_follow_state(False)
        self.routes_status.setText("EMERGENCY STOP (keys released)")
        self.routes_status.setStyleSheet(f"color: {RED};")

    def sync_driver_state(self) -> None:
        """Main-thread poll: finalize the UI when the driver finished the route itself."""
        d = self.driver
        if d is not None and (d.state == "finished" or not d.is_alive()):
            self.driver = None
            self._set_follow_state(False)
            msg = "Route finished — autopilot off" if d.state == "finished" else "Autopilot stopped"
            self.routes_status.setText(msg)
            self.routes_status.setStyleSheet(f"color: {BLUE};")
