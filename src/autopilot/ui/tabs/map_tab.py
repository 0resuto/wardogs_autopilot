"""Unified Map tab combining interactive map view, route planning, and locator tuning (PySide6)."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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

from ...common.config import AppConfig, LocatorConfig
from ...common.log import get_logger
from ...navigation.follow import FollowDriver
from ...vision import locator
from ..map_view import InteractiveMapWidget
from ..presets import PresetManager

logger = get_logger("map_tab")


class _StringVarCompat:
    """Compatibility shim for tests checking Tkinter StringVar."""

    def __init__(self, value: str = "") -> None:
        self._val = str(value)

    def get(self) -> str:
        return self._val

    def set(self, val: str) -> None:
        self._val = str(val)


class MapTab(QWidget):
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

        self.map_name = self.get_map_name()
        self._disp_th: float | None = None
        self._last_loc: dict[str, Any] | None = None

        self.preset_mgr = PresetManager()
        self.driver: FollowDriver | None = None

        self.tune_vars: dict[str, _StringVarCompat] = {}
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

        # --- Top toolbar: Route presets & Tuning toggle ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        lbl_preset = QLabel("Preset:", self)
        lbl_preset.setStyleSheet("font-weight: bold;")
        top_bar.addWidget(lbl_preset)

        self.p_sel = QComboBox(self)
        self.p_sel.setFixedWidth(130)
        top_bar.addWidget(self.p_sel)

        btn_load = QPushButton("Load", self)
        btn_load.clicked.connect(self.preset_load_sel)
        top_bar.addWidget(btn_load)

        btn_del = QPushButton("Delete", self)
        btn_del.clicked.connect(self.preset_delete)
        top_bar.addWidget(btn_del)

        self.p_name = QLineEdit(self)
        self.p_name.setPlaceholderText("Preset name...")
        self.p_name.setFixedWidth(110)
        top_bar.addWidget(self.p_name)

        btn_save = QPushButton("Save As", self)
        btn_save.clicked.connect(self.preset_save)
        top_bar.addWidget(btn_save)

        btn_upd = QPushButton("Update", self)
        btn_upd.clicked.connect(self.preset_overwrite)
        top_bar.addWidget(btn_upd)

        btn_clear = QPushButton("🗑 Clear", self)
        btn_clear.clicked.connect(self.routes_clear)
        top_bar.addWidget(btn_clear)

        self.invert_ck = QCheckBox("Invert", self)
        self.invert_ck.toggled.connect(self.routes_invert)
        top_bar.addWidget(self.invert_ck)

        self.dbg_ck = QCheckBox("Nav log", self)
        self.dbg_ck.setChecked(bool(self.app_cfg.navigator.debug))
        top_bar.addWidget(self.dbg_ck)

        top_bar.addStretch()

        self._tune_toggle_btn = QPushButton("⚙ Tuning ▾", self)
        self._tune_toggle_btn.clicked.connect(self.toggle_tuning_panel)
        top_bar.addWidget(self._tune_toggle_btn)

        root_layout.addLayout(top_bar)

        # --- Collapsible Locator Tuning Panel (hidden by default) ---
        self._tune_container = QWidget(self)
        tune_vbox = QVBoxLayout(self._tune_container)
        tune_vbox.setContentsMargins(0, 2, 0, 2)
        tune_vbox.setSpacing(4)

        tune_row = QHBoxLayout()
        tune_row.setSpacing(6)

        # 1. Tracking
        grp_trk = QGroupBox("Tracking", self._tune_container)
        l_trk = QHBoxLayout(grp_trk)
        l_trk.setContentsMargins(6, 10, 6, 6)
        l_trk.setSpacing(4)
        self._add_tune_field(l_trk, grp_trk, "ratio", "ratio_local", 0.85, 38)
        self._add_tune_field(l_trk, grp_trk, "inl", "min_inl_local", 4, 26, is_int=True)
        self._add_tune_field(l_trk, grp_trk, "inl%", "min_inl_rate_local", 0.0, 34)
        self._add_tune_field(l_trk, grp_trk, "rad", "track_radius", 900, 38, is_int=True)
        tune_row.addWidget(grp_trk)

        # 2. Re-Acquisition
        grp_acq = QGroupBox("Re-Acquisition", self._tune_container)
        l_acq = QHBoxLayout(grp_acq)
        l_acq.setContentsMargins(6, 10, 6, 6)
        l_acq.setSpacing(4)
        self._add_tune_field(l_acq, grp_acq, "ratio", "ratio_global", 0.9, 38)
        self._add_tune_field(l_acq, grp_acq, "inl", "min_inl_global", 5, 26, is_int=True)
        self._add_tune_field(l_acq, grp_acq, "inl%", "min_inl_rate_global", 0.0, 34)
        tune_row.addWidget(grp_acq)

        # 3. Consensus
        grp_misc = QGroupBox("Consensus", self._tune_container)
        l_misc = QHBoxLayout(grp_misc)
        l_misc.setContentsMargins(6, 10, 6, 6)
        l_misc.setSpacing(4)
        self._add_tune_field(l_misc, grp_misc, "vote", "vote_need", 3, 24, is_int=True)
        self._add_tune_field(l_misc, grp_misc, "head°", "heading_gate_deg", 0, 30, is_int=True)
        self._add_tune_field(l_misc, grp_misc, "skip", "vote_inl_skip", 40, 30, is_int=True)
        self._add_tune_field(l_misc, grp_misc, "hold", "hold_frames", 5, 24, is_int=True)
        tune_row.addWidget(grp_misc)

        # Action buttons
        btn_box = QVBoxLayout()
        btn_box.setSpacing(4)
        top_btns = QHBoxLayout()
        apply_btn = QPushButton("Apply", self._tune_container)
        apply_btn.setObjectName("AccentButton")
        apply_btn.clicked.connect(self.apply_tune)
        top_btns.addWidget(apply_btn)

        reset_btn = QPushButton("Reset", self._tune_container)
        reset_btn.clicked.connect(self.reset_tune)
        top_btns.addWidget(reset_btn)
        btn_box.addLayout(top_btns)

        self.tune_status = QLabel("", self._tune_container)
        self.tune_status.setStyleSheet("color: #8ae234; font-weight: 500;")
        btn_box.addWidget(self.tune_status)
        tune_row.addLayout(btn_box)

        tune_vbox.addLayout(tune_row)

        # Diagnostic fail logs bar
        dbg_bar = QHBoxLayout()
        self._collect_ck = QCheckBox("Collect fail logs", self._tune_container)
        self._collect_ck.setChecked(
            bool(self.cfg.setdefault("debug", {}).get("collect_fail_logs", False))
        )
        self._collect_ck.toggled.connect(self.apply_collect_logs)
        dbg_bar.addWidget(self._collect_ck)

        self.dbg_text = QLabel("", self._tune_container)
        self.dbg_text.setStyleSheet(
            "background-color: #252526; color: #ffcf6a; padding: 2px 6px; border-radius: 4px;"
        )
        dbg_bar.addWidget(self.dbg_text, stretch=1)

        copy_btn = QPushButton("Copy", self._tune_container)
        copy_btn.clicked.connect(self.copy_debug)
        dbg_bar.addWidget(copy_btn)

        tune_vbox.addLayout(dbg_bar)

        self._tune_container.setVisible(False)
        root_layout.addWidget(self._tune_container)

        # --- Interactive Map Canvas / Scene ---
        self.map_widget = InteractiveMapWidget(enable_route_editing=True, parent=self)
        self.map_widget.scene.on_route_changed = self.routes_refresh
        self.map_widget.set_on_center(self._center_vehicle)
        root_layout.addWidget(self.map_widget, stretch=1)

        # --- Bottom Command & Navigation Control Center ---
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
        self.routes_status.setStyleSheet("color: #88c0d0; font-weight: bold;")
        bot_bar.addWidget(self.routes_status, stretch=1)

        self.map_status = QLabel("", self)
        self.map_status.setStyleSheet("color: #a0a0a0; font-size: 8pt;")
        bot_bar.addWidget(self.map_status)

        hint = QLabel("LMB: Add | Drag: Move | RMB: Del | Drag map: Pan | Wheel: Zoom", self)
        hint.setStyleSheet("color: #606060; font-size: 8pt;")
        bot_bar.addWidget(hint)

        root_layout.addLayout(bot_bar)
        self.preset_reload()

    def toggle_tuning_panel(self) -> None:
        """Toggle visibility of locator tuning controls."""
        visible = not self._tune_container.isVisible()
        self._tune_container.setVisible(visible)
        self._tune_toggle_btn.setText("⚙ Tuning ▴" if visible else "⚙ Tuning ▾")

    def _add_tune_field(
        self,
        layout: QHBoxLayout,
        parent: QWidget,
        lbl_text: str,
        var_name: str,
        default: Any,
        width: int,
        is_int: bool = False,
    ) -> None:
        lbl = QLabel(lbl_text, parent)
        lbl.setStyleSheet("color: #a0a0a0;")
        layout.addWidget(lbl)

        val = str(
            int(self._loc_tune_cur(var_name, default))
            if is_int
            else self._loc_tune_cur(var_name, default)
        )
        inp = QLineEdit(val, parent)
        inp.setFixedWidth(width)
        layout.addWidget(inp)
        self.tune_inputs[var_name] = inp
        self.tune_vars[var_name] = _StringVarCompat(val)

    def _loc_tune_cur(self, name: str, default: Any) -> Any:
        block = self.cfg.setdefault("locator", {})
        return block.get(name, default)

    def apply_tune(self) -> None:
        """Validate tuning inputs, update configuration, and notify LiveLocator."""
        ranges = {
            "ratio_local": (0.1, 1.0, float, "TRACK ratio in [0.1 .. 1.0]"),
            "min_inl_local": (1, 50, int, "TRACK min_inl in [1 .. 50]"),
            "min_inl_rate_local": (0.0, 1.0, float, "TRACK inl% in [0.0 .. 1.0]"),
            "track_radius": (50, 4000, int, "TRACK rad in [50 .. 4000] px"),
            "ratio_global": (0.1, 1.0, float, "RE-ACQ ratio in [0.1 .. 1.0]"),
            "min_inl_global": (1, 50, int, "RE-ACQ min_inl in [1 .. 50]"),
            "min_inl_rate_global": (0.0, 1.0, float, "RE-ACQ inl% in [0.0 .. 1.0]"),
            "vote_need": (1, 10, int, "vote in [1 .. 10]"),
            "heading_gate_deg": (0, 180, int, "head gate in [0 .. 180] deg"),
            "vote_inl_skip": (1, 200, int, "skip in [1 .. 200] inl"),
            "hold_frames": (0, 30, int, "hold in [0 .. 30] frames"),
        }
        parsed = {}
        for name, (lo, hi, typ, desc) in ranges.items():
            s = self.tune_inputs[name].text().strip()
            if not s:
                s = self.tune_vars[name].get().strip()
            try:
                v = typ(float(s))
            except ValueError:
                self.tune_status.setText(f"Invalid {desc}")
                self.tune_status.setStyleSheet("color: #ff7c7c;")
                return
            if not (lo <= v <= hi):
                self.tune_status.setText(f"Invalid {desc}")
                self.tune_status.setStyleSheet("color: #ff7c7c;")
                return
            parsed[name] = v

        for name, val in parsed.items():
            self.tune_inputs[name].setText(str(val))
            self.tune_vars[name].set(str(val))

        block = self.cfg.setdefault("locator", {})
        block.update(parsed)
        self.save_cfg()
        loc = self.get_loc()
        if loc is not None:
            if hasattr(loc, "apply_tune"):
                loc.apply_tune(block)
            elif hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                loc.cfg.setdefault("locator", {}).update(block)
        self.tune_status.setText("applied")
        self.tune_status.setStyleSheet("color: #8ae234;")

    def reset_tune(self) -> None:
        """Reset tuning parameters to schema defaults."""
        defaults = LocatorConfig().model_dump()
        for k, v in defaults.items():
            if k in self.tune_vars:
                self.tune_vars[k].set(str(v))
                if k in self.tune_inputs:
                    self.tune_inputs[k].setText(str(v))
        self.apply_tune()

    def apply_collect_logs(self) -> None:
        enabled = self._collect_ck.isChecked()
        self.cfg.setdefault("debug", {})["collect_fail_logs"] = enabled
        self.save_cfg()
        loc = self.get_loc()
        if loc is not None and hasattr(loc, "set_collect_fail_logs"):
            loc.set_collect_fail_logs(enabled)

    def copy_debug(self) -> None:
        QApplication.clipboard().setText(self.dbg_text.text())

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

    # --- Presets & Route Editing Logic ---
    def preset_reload(self) -> None:
        """Reload list of presets for the active map."""
        map_name = self.get_map_name()
        self.preset_mgr = PresetManager(subdir=f"data/presets/{map_name}")
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
            QMessageBox.information(
                self, "Presets", f'Preset "{name}" already exists. Use "Update" to overwrite.'
            )
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
