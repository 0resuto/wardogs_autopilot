"""Route planning tab with interactive QGraphicsScene waypoint editing and autopilot controls (PySide6)."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...common.config import AppConfig
from ...navigation.follow import FollowDriver
from ..map_view import InteractiveMapWidget
from ..presets import PresetManager


class RoutesTab(QWidget):
    """Tab widget for interactive waypoint editing, preset management, and autopilot driving."""

    def __init__(
        self,
        parent: QWidget | None,
        app_cfg: AppConfig,
        loc_thread_supplier: Callable[[], Any],
        map_name_supplier: Callable[[], str],
        map_store_supplier: Callable[[], Any],
    ) -> None:
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.get_loc = loc_thread_supplier
        self.get_map_name = map_name_supplier
        self.get_store = map_store_supplier

        self.preset_mgr = PresetManager()
        self.driver: FollowDriver | None = None

        self._build_ui()

    @property
    def route_pts(self) -> list[list[float]]:
        return self.map_widget.scene.route_pts

    @route_pts.setter
    def route_pts(self, pts: list[list[float]]) -> None:
        self.map_widget.scene.set_route(pts)
        self.routes_refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top preset toolbar
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("Preset:", self))

        self.p_sel = QComboBox(self)
        self.p_sel.setFixedWidth(130)
        bar.addWidget(self.p_sel)

        btn_load = QPushButton("Load", self)
        btn_load.clicked.connect(self.preset_load_sel)
        bar.addWidget(btn_load)

        btn_del = QPushButton("Delete", self)
        btn_del.clicked.connect(self.preset_delete)
        bar.addWidget(btn_del)

        self.p_name = QLineEdit(self)
        self.p_name.setPlaceholderText("Preset name...")
        self.p_name.setFixedWidth(120)
        bar.addWidget(self.p_name)

        btn_save = QPushButton("Save As", self)
        btn_save.clicked.connect(self.preset_save)
        bar.addWidget(btn_save)

        btn_upd = QPushButton("Update", self)
        btn_upd.clicked.connect(self.preset_overwrite)
        bar.addWidget(btn_upd)

        bar.addStretch()

        btn_clear = QPushButton("🗑 Clear", self)
        btn_clear.clicked.connect(self.routes_clear)
        bar.addWidget(btn_clear)

        self.invert_ck = QCheckBox("Invert", self)
        self.invert_ck.toggled.connect(self.routes_invert)
        bar.addWidget(self.invert_ck)

        self.dbg_ck = QCheckBox("Nav log", self)
        self.dbg_ck.setChecked(bool(self.app_cfg.navigator.debug))
        bar.addWidget(self.dbg_ck)

        layout.addLayout(bar)

        # Interactive Map Canvas / Scene
        self.map_widget = InteractiveMapWidget(enable_route_editing=True, parent=self)
        self.map_widget.scene.on_route_changed = self.routes_refresh
        self.map_widget.set_on_center(self._center_vehicle)
        layout.addWidget(self.map_widget, stretch=1)

        # Bottom autopilot command center
        ctl = QHBoxLayout()
        ctl.setSpacing(8)

        self.follow_btn = QPushButton("▶ Start Follow (F6)", self)
        self.follow_btn.setObjectName("SuccessButton")
        self.follow_btn.clicked.connect(self.follow_toggle)
        ctl.addWidget(self.follow_btn)

        self.estop_btn = QPushButton("🛑 EMERGENCY STOP (F7)", self)
        self.estop_btn.setObjectName("EStopButton")
        self.estop_btn.clicked.connect(self.emergency_stop)
        ctl.addWidget(self.estop_btn)

        self.routes_status = QLabel("", self)
        self.routes_status.setStyleSheet("color: #88c0d0; font-weight: 500;")
        ctl.addWidget(self.routes_status, stretch=1)

        hint = QLabel("LMB: Add Point | Drag: Move | RMB: Delete", self)
        hint.setStyleSheet("color: #707070; font-size: 8pt;")
        ctl.addWidget(hint)

        layout.addLayout(ctl)
        self.preset_reload()

    def _center_vehicle(self) -> None:
        loc = self.get_loc()
        item = loc.latest if loc else None
        mp = (item.get("map_px_disp") or item.get("map_px")) if item else None
        if mp is not None:
            self.map_widget.view.center_on_coords(mp[0], mp[1])
        else:
            self.map_widget.view.fit_view()

    def preset_reload(self) -> None:
        """Reload list of presets for the active map."""
        self.preset_mgr = PresetManager(subdir=f"data/presets/{self.get_map_name()}")
        names = self.preset_mgr.list_presets()
        self.p_sel.clear()
        self.p_sel.addItems(names)
        if names:
            last = self.app_cfg.navigator.last_preset
            if last in names:
                self.p_sel.setCurrentText(last)
            else:
                self.p_sel.setCurrentIndex(0)

    def preset_save(self) -> None:
        name = self.p_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Presets", "Enter a preset name")
            return
        path = self.preset_mgr.preset_path(name)
        if os.path.exists(path):
            QMessageBox.information(self, "Presets", f'Preset "{name}" already exists. Use "Update" to overwrite.')
            return
        try:
            self.preset_mgr.save_preset(name, self.route_pts)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f"Failed to save preset: {exc}")
            return
        self.app_cfg.navigator.last_preset = name
        self.preset_reload()
        self.p_sel.setCurrentText(name)

    def preset_overwrite(self) -> None:
        name = self.p_name.text().strip() or self.p_sel.currentText().strip()
        if not name:
            QMessageBox.warning(self, "Presets", "Select or enter a preset name to overwrite")
            return
        try:
            self.preset_mgr.save_preset(name, self.route_pts)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f"Failed to update preset: {exc}")
            return
        self.app_cfg.navigator.last_preset = name
        self.preset_reload()
        self.p_sel.setCurrentText(name)

    def preset_load_sel(self) -> None:
        name = self.p_sel.currentText().strip()
        if not name:
            return
        try:
            pts = self.preset_mgr.load_preset(name)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f'Failed to read preset "{name}": {exc}')
            return
        self.route_pts = pts
        self.p_name.setText(name)
        self.app_cfg.navigator.last_preset = name
        self.routes_refresh()

    def preset_delete(self) -> None:
        name = self.p_sel.currentText().strip()
        if not name:
            return
        res = QMessageBox.question(self, "Presets", f'Delete preset "{name}"?')
        if res != QMessageBox.StandardButton.Yes:
            return
        self.preset_mgr.delete_preset(name)
        self.preset_reload()

    def routes_clear(self) -> None:
        self.route_pts = []
        self.routes_refresh()

    def routes_invert(self) -> None:
        if self.driver is not None:
            QMessageBox.information(self, "Routes", "Stop the autopilot before inverting the route")
            self.invert_ck.setChecked(False)
            return
        pts = list(reversed(self.route_pts))
        self.route_pts = pts
        self.routes_refresh()

    def routes_refresh(self) -> None:
        length_m = 0.0
        for i in range(1, len(self.route_pts)):
            p0, p1 = self.route_pts[i - 1], self.route_pts[i]
            d_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            length_m += d_px / 1.7
        speed_cap = getattr(self.app_cfg.navigator, "speed_cap_kmh", 36.0) or 36.0
        speed_mps = max(2.0, speed_cap / 3.6)
        est_sec = int(length_m / speed_mps)
        mins, secs = divmod(est_sec, 60)
        time_txt = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        self.routes_status.setText(
            f"Route: {len(self.route_pts)} pts | ~{int(length_m)} m ({time_txt} at {int(speed_cap)} km/h)"
        )

    def _set_follow_state(self, running: bool) -> None:
        if running:
            self.follow_btn.setText("⏸ Pause (F6)")
            self.follow_btn.setObjectName("DangerButton")
        else:
            self.follow_btn.setText("▶ Start Follow (F6)")
            self.follow_btn.setObjectName("SuccessButton")
        # Re-apply stylesheet to update button color dynamically
        self.follow_btn.style().unpolish(self.follow_btn)
        self.follow_btn.style().polish(self.follow_btn)

    def follow_toggle(self, silent: bool = False) -> None:
        """Start or stop the background FollowDriver thread."""
        if self.driver is not None:
            self.driver.stop()
            self.driver = None
            self._set_follow_state(False)
            self.routes_status.setText("Autopilot stopped")
            self.routes_status.setStyleSheet("color: #88c0d0;")
            return

        if len(self.route_pts) < 2:
            self.routes_status.setText("Route not set (at least 2 points required)")
            self.routes_status.setStyleSheet("color: #ffaa00;")
            if not silent:
                QMessageBox.warning(self, "Routes", "Route not set (at least 2 points required)")
            return

        loc_thread = self.get_loc()
        nav_cfg = self.app_cfg.navigator
        try:
            kb = self._make_kb(nav_cfg.port)
        except (OSError, RuntimeError) as exc:
            self.routes_status.setText(f"Key driver error: {exc}")
            self.routes_status.setStyleSheet("color: #ff7c7c;")
            if not silent:
                QMessageBox.critical(self, "Autopilot", f"Failed to create key driver:\n{exc}")
            return

        self.driver = FollowDriver(
            loc=loc_thread,
            pts=[(p[0], p[1]) for p in self.route_pts],
            nav_cfg=nav_cfg,
            kb=kb,
            debug=self.dbg_ck.isChecked(),
        )
        self.driver.start()
        self._set_follow_state(True)
        self.routes_status.setText("Autopilot driving...")
        self.routes_status.setStyleSheet("color: #8ae234;")

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
        self.routes_status.setStyleSheet("color: #ff3b3b;")

    def sync_driver_state(self) -> None:
        """Main-thread poll: finalize the UI when the driver finished the route itself."""
        d = self.driver
        if d is not None and (d.state == "finished" or not d.is_alive()):
            self.driver = None
            self._set_follow_state(False)
            msg = "Route finished — autopilot off" if d.state == "finished" else "Autopilot stopped"
            self.routes_status.setText(msg)
            self.routes_status.setStyleSheet("color: #88c0d0;")
