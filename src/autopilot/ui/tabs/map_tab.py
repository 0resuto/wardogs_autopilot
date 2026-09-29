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
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...common.config import AppConfig, LocatorConfig, NavigatorConfig
from ...common.log import get_logger
from ...navigation.follow import FollowDriver
from ...navigation.manual_record import ManualDriveRecorder
from ...vision import locator
from ..map_view import InteractiveMapWidget
from ..presets import PresetManager

logger = get_logger("map_tab")


class _StringVarCompat:
    """String holder mirroring the entry widgets for tests."""

    def __init__(self, value: str = "") -> None:
        self._val = str(value)

    def get(self) -> str:
        return self._val

    def set(self, val: str) -> None:
        self._val = str(val)


# Only used when the map catalog has no m_per_px entry for the active map.
_FALLBACK_PX_PER_M = 2.0


def _compact_label(label: QLabel) -> None:
    """Let dynamic status text clip instead of forcing the window to grow."""
    label.setMinimumWidth(0)
    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    label.setSizePolicy(policy)


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
        self._manual_rec: ManualDriveRecorder | None = None
        self._edit_snapshot: list[list[float]] | None = None

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

        btn_reload = QPushButton("⟳", self)
        btn_reload.setFixedWidth(28)
        btn_reload.setToolTip("Reload the preset list (e.g. after route_from_manual.py)")
        btn_reload.clicked.connect(self.preset_reload)
        top_bar.addWidget(btn_reload)

        btn_load = QPushButton("Load", self)
        btn_load.clicked.connect(lambda: self.preset_load_sel())
        top_bar.addWidget(btn_load)

        btn_del = QPushButton("Delete", self)
        btn_del.clicked.connect(self.preset_delete)
        top_bar.addWidget(btn_del)

        self._new_btn = QPushButton("New", self)
        self._new_btn.setToolTip("Create a new empty route preset, then draw it")
        self._new_btn.clicked.connect(self.route_new)
        top_bar.addWidget(self._new_btn)

        btn_save = QPushButton("Save As", self)
        btn_save.setToolTip("Save the current route under a new preset name")
        btn_save.clicked.connect(self.preset_save)
        top_bar.addWidget(btn_save)

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
            btn_reload,
            btn_load,
            btn_del,
            btn_save,
            btn_clear,
            self._reverse_btn,
            self._new_btn,
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
        self._add_tune_field(
            l_trk,
            grp_trk,
            "kps",
            "max_kp_frame",
            1200,
            40,
            is_int=True,
            tip="Keypoints kept per frame (response-ranked): lower = less latency,\n"
            "higher = more robust in low-texture areas",
        )
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
        _compact_label(self.tune_status)
        btn_box.addWidget(self.tune_status)
        tune_row.addLayout(btn_box)

        tune_vbox.addLayout(tune_row)

        tune_row2 = QHBoxLayout()
        tune_row2.setSpacing(6)

        grp_veh = QGroupBox("Vehicle", self._tune_container)
        l_veh = QHBoxLayout(grp_veh)
        l_veh.setContentsMargins(6, 10, 6, 6)
        l_veh.setSpacing(4)
        self._add_tune_field(
            l_veh,
            grp_veh,
            "gain",
            "yaw_gain",
            1.0,
            38,
            section="navigator",
            tip="Yaw-authority scale of the model (see tools/calibrate_vehicle.py)",
        )
        self._add_tune_field(
            l_veh,
            grp_veh,
            "lat g",
            "corner_lat_g",
            0.35,
            38,
            section="navigator",
            tip="Lateral grip budget for planning corner speeds: v = sqrt(lat_g*9.81*R).\n"
            "Lower = slower corners (if it slides wide), higher = faster (but the planner\n"
            "may outrun what the steering can hold).",
        )
        self._add_tune_field(
            l_veh,
            grp_veh,
            "brake g",
            "brake_g",
            0.45,
            38,
            section="navigator",
            tip="Braking deceleration budget for planning when to brake before a corner.\n"
            "Higher = brakes later/harder. Measured value from your runs: ~0.5g\n"
            "(python tools/calibrate_vehicle.py output/nav_dbg_*.jsonl).",
        )
        self._add_tune_field(
            l_veh,
            grp_veh,
            "min km/h",
            "corner_min_kmh",
            12.0,
            40,
            section="navigator",
            tip="Lower edge of the steady-corner hold window (km/h)",
        )
        self._add_tune_field(
            l_veh,
            grp_veh,
            "max km/h",
            "corner_max_kmh",
            22.0,
            40,
            section="navigator",
            tip="Upper edge of the steady-corner hold window: inside the window the\n"
            "driver neither accelerates nor brakes (no more brake/gas hunting)",
        )
        self._add_tune_field(
            l_veh,
            grp_veh,
            "ahead m",
            "plan_ahead_m",
            200.0,
            40,
            section="navigator",
            tip="Speed planning horizon along the route (meters)",
        )
        self._add_tune_field(
            l_veh,
            grp_veh,
            "cut m",
            "corner_cut_m",
            15.0,
            38,
            section="navigator",
            tip="Distance over which a sharp vertex is rounded by the planner",
        )
        tune_row2.addWidget(grp_veh)

        grp_corr = QGroupBox("Corridors", self._tune_container)
        l_corr = QHBoxLayout(grp_corr)
        l_corr.setContentsMargins(6, 10, 6, 6)
        l_corr.setSpacing(4)
        self._add_tune_field(
            l_corr,
            grp_corr,
            "inner m",
            "xte_m",
            4.0,
            38,
            section="navigator",
            tip="Inner corridor (normal driving). Deviations beyond it get firmer corrections.",
        )
        self._add_tune_field(
            l_corr,
            grp_corr,
            "outer m",
            "xte_outer_m",
            12.0,
            38,
            section="navigator",
            tip="Outer corridor (warning). Past it the driver slows down and steers hardest.",
        )
        self._add_tune_field(
            l_corr,
            grp_corr,
            "look s",
            "steer_look_s",
            1.6,
            34,
            section="navigator",
            tip="Steering lookahead in seconds of travel: the aim point ahead on the route.\n"
            "Lower = tighter line and more active steering; higher = smoother, cuts curves.",
        )
        self._add_tune_field(
            l_corr,
            grp_corr,
            "settle s",
            "settle_s",
            0.6,
            34,
            section="navigator",
            tip="Pause after a completed steering hold before the next one (s).\n"
            "Lower = more frequent corrections; higher = smoother but a dead wheel\n"
            "for that long after each correction. At speed the pause is also capped\n"
            "by distance (8 m), so it shortens automatically.",
        )
        self._add_tune_field(
            l_corr,
            grp_corr,
            "lead s",
            "steer_lead_s",
            0.25,
            34,
            section="navigator",
            tip="Release anticipation in seconds: how much heading change still arrives\n"
            "through the pose/key latency after the wheel is released. Raise it if the\n"
            "car systematically overshoots, lower it if it releases too early.",
        )
        self._add_tune_field(
            l_corr,
            grp_corr,
            "skip m",
            "skip_ahead_m",
            150.0,
            40,
            section="navigator",
            tip="Route re-acquisition window (m). Only when the car is outside the outer\n"
            "corridor: the active point may jump forward to the nearest route point\n"
            "within this route length (it never jumps backwards or to the final point).\n"
            "0 disables the re-acquisition.",
        )
        tune_row2.addWidget(grp_corr)

        tune_row2.addStretch()
        tune_vbox.addLayout(tune_row2)

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
        _compact_label(self.dbg_text)
        dbg_bar.addWidget(self.dbg_text, stretch=1)

        copy_btn = QPushButton("Copy", self._tune_container)
        copy_btn.clicked.connect(self.copy_debug)
        dbg_bar.addWidget(copy_btn)

        tune_vbox.addLayout(dbg_bar)

        self._tune_container.setVisible(False)
        root_layout.addWidget(self._tune_container)

        # --- Interactive Map Canvas / Scene ---
        # Route editing is off until the Edit button is pressed: by default the
        # left button pans and the route cannot be changed by an accidental click.
        self.map_widget = InteractiveMapWidget(enable_route_editing=False, parent=self)
        self.map_widget.scene.on_route_changed = self.routes_refresh
        self.map_widget.set_on_center(self._center_vehicle)
        root_layout.addWidget(self.map_widget, stretch=1)

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
        self.routes_status.setStyleSheet("color: #88c0d0; font-weight: bold;")
        _compact_label(self.routes_status)
        bot_bar.addWidget(self.routes_status, stretch=1)
        bot_box.addLayout(bot_bar)

        info_bar = QHBoxLayout()
        info_bar.setSpacing(8)

        self.map_status = QLabel("", self)
        self.map_status.setStyleSheet("color: #a0a0a0; font-size: 8pt;")
        _compact_label(self.map_status)
        info_bar.addWidget(self.map_status, stretch=1)

        self._hint_lbl = QLabel("LMB: Pan | Wheel: Zoom | ✏ Edit to modify the route", self)
        self._hint_lbl.setStyleSheet("color: #606060; font-size: 8pt;")
        info_bar.addWidget(self._hint_lbl)
        bot_box.addLayout(info_bar)

        root_layout.addLayout(bot_box)
        self.preset_reload()

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
        section: str = "locator",
        tip: str = "",
    ) -> None:
        lbl = QLabel(lbl_text, parent)
        lbl.setStyleSheet("color: #a0a0a0;")
        layout.addWidget(lbl)

        cur = self._nav_tune_cur if section == "navigator" else self._loc_tune_cur
        value = cur(var_name, default)
        val = str(int(value) if is_int else value)
        inp = QLineEdit(val, parent)
        inp.setFixedWidth(width)
        if tip:
            lbl.setToolTip(tip)
            inp.setToolTip(tip)
        layout.addWidget(inp)
        self.tune_inputs[var_name] = inp
        self.tune_vars[var_name] = _StringVarCompat(val)

    def _loc_tune_cur(self, name: str, default: Any) -> Any:
        block = self.cfg.setdefault("locator", {})
        return block.get(name, default)

    def _nav_tune_cur(self, name: str, default: Any) -> Any:
        block = self.cfg.setdefault("navigator", {})
        value = block.get(name, default)
        return default if value is None else value

    def apply_tune(self) -> None:
        """Validate tuning inputs, update configuration, and notify consumers."""
        ranges = {
            "ratio_local": ("locator", 0.1, 1.0, float, "TRACK ratio in [0.1 .. 1.0]"),
            "min_inl_local": ("locator", 1, 50, int, "TRACK min_inl in [1 .. 50]"),
            "min_inl_rate_local": ("locator", 0.0, 1.0, float, "TRACK inl% in [0.0 .. 1.0]"),
            "track_radius": ("locator", 50, 4000, int, "TRACK rad in [50 .. 4000] px"),
            "ratio_global": ("locator", 0.1, 1.0, float, "RE-ACQ ratio in [0.1 .. 1.0]"),
            "min_inl_global": ("locator", 1, 50, int, "RE-ACQ min_inl in [1 .. 50]"),
            "min_inl_rate_global": ("locator", 0.0, 1.0, float, "RE-ACQ inl% in [0.0 .. 1.0]"),
            "vote_need": ("locator", 1, 10, int, "vote in [1 .. 10]"),
            "heading_gate_deg": ("locator", 0, 180, int, "head gate in [0 .. 180] deg"),
            "vote_inl_skip": ("locator", 1, 200, int, "skip in [1 .. 200] inl"),
            "hold_frames": ("locator", 0, 30, int, "hold in [0 .. 30] frames"),
            "max_kp_frame": ("locator", 100, 6000, int, "TRACK max kp in [100 .. 6000]"),
            "yaw_gain": ("navigator", 0.05, 5.0, float, "VEH gain in [0.05 .. 5.0]"),
            "corner_lat_g": ("navigator", 0.05, 1.5, float, "VEH lat g in [0.05 .. 1.5]"),
            "brake_g": ("navigator", 0.05, 2.0, float, "VEH brake g in [0.05 .. 2.0]"),
            "corner_min_kmh": ("navigator", 0.0, 79.0, float, "VEH min km/h in [0 .. 79]"),
            "corner_max_kmh": ("navigator", 0.0, 79.0, float, "VEH max km/h in [0 .. 79]"),
            "plan_ahead_m": ("navigator", 20.0, 1000.0, float, "VEH ahead m in [20 .. 1000]"),
            "corner_cut_m": ("navigator", 4.0, 60.0, float, "VEH cut m in [4 .. 60]"),
            "xte_m": ("navigator", 1.0, 30.0, float, "COR inner m in [1 .. 30]"),
            "xte_outer_m": ("navigator", 4.0, 60.0, float, "COR outer m in [4 .. 60]"),
            "steer_look_s": ("navigator", 0.4, 4.0, float, "COR look s in [0.4 .. 4.0]"),
            "settle_s": ("navigator", 0.1, 2.0, float, "COR settle s in [0.1 .. 2.0]"),
            "steer_lead_s": ("navigator", 0.0, 1.0, float, "COR lead s in [0.0 .. 1.0]"),
            "skip_ahead_m": ("navigator", 0.0, 1000.0, float, "COR skip m in [0 .. 1000]"),
        }
        parsed: dict[str, dict[str, Any]] = {"locator": {}, "navigator": {}}
        for name, (section, lo, hi, typ, desc) in ranges.items():
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
            parsed[section][name] = v

        if parsed["navigator"]["xte_outer_m"] <= parsed["navigator"]["xte_m"]:
            self.tune_status.setText("Invalid COR outer m must be greater than inner m")
            self.tune_status.setStyleSheet("color: #ff7c7c;")
            return

        if parsed["navigator"]["corner_max_kmh"] <= parsed["navigator"]["corner_min_kmh"]:
            self.tune_status.setText("Invalid VEH max km/h must be greater than min km/h")
            self.tune_status.setStyleSheet("color: #ff7c7c;")
            return

        for values in parsed.values():
            for name, val in values.items():
                self.tune_inputs[name].setText(str(val))
                self.tune_vars[name].set(str(val))

        loc_block = self.cfg.setdefault("locator", {})
        loc_block.update(parsed["locator"])
        nav_block = self.cfg.setdefault("navigator", {})
        nav_block.update(parsed["navigator"])
        self.save_cfg()

        loc = self.get_loc()
        if loc is not None:
            if hasattr(loc, "apply_tune"):
                loc.apply_tune(loc_block)
            elif hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                loc.cfg.setdefault("locator", {}).update(loc_block)

        if parsed["navigator"]:
            for name, value in parsed["navigator"].items():
                setattr(self.app_cfg.navigator, name, value)
            if self.driver is not None and hasattr(self.driver, "apply_vehicle_tuning"):
                self.driver.apply_vehicle_tuning(self.app_cfg.navigator)

        self.tune_status.setText("applied")
        self.tune_status.setStyleSheet("color: #8ae234;")

    def reset_tune(self) -> None:
        """Reset tuning parameters to schema defaults."""
        defaults = LocatorConfig().model_dump()
        nav_defaults = NavigatorConfig().model_dump()
        if nav_defaults.get("yaw_gain") is None:
            nav_defaults["yaw_gain"] = 1.0
        defaults.update(nav_defaults)
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

    # --- Presets & Route Editing Logic ---
    def preset_reload(self) -> None:
        """Reload list of presets for the active map.

        The preset used in the last session is selected again, and its route is
        loaded when no route is set (fresh start), so the studio opens ready to
        drive.
        """
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
            if not self.route_pts:
                self.preset_load_sel(persist=False)

    def _ask_preset_name(self, title: str, ok_text: str, initial: str = "") -> str | None:
        """Ask for a preset name (field + OK/Cancel), or None when cancelled."""
        dialog = QInputDialog(self)
        dialog.setWindowTitle(title)
        dialog.setLabelText("Preset name:")
        dialog.setOkButtonText(ok_text)
        dialog.setCancelButtonText("Cancel")
        dialog.setTextValue(initial)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        name = dialog.textValue().strip()
        if not name:
            QMessageBox.warning(self, title, "Enter a preset name")
            return None
        return name

    def _save_route_to_preset(self, name: str) -> bool:
        """Write the current route points to the preset and select it."""
        try:
            self.preset_mgr.save_preset(name, self.route_pts)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f"Failed to save preset: {exc}")
            return False
        self.app_cfg.navigator.last_preset = name
        self.save_cfg()
        self.preset_reload()
        self.p_sel.setCurrentText(name)
        return True

    def preset_save(self) -> None:
        """Save the current route under a new name (Save As)."""
        current = self.p_sel.currentText().strip()
        name = self._ask_preset_name("Save route as", "Save", current)
        if name is None:
            return
        if name != current and os.path.exists(self.preset_mgr.preset_path(name)):
            res = QMessageBox.question(
                self, "Presets", f'Preset "{name}" already exists. Overwrite?'
            )
            if res != QMessageBox.StandardButton.Yes:
                return
        self._save_route_to_preset(name)

    def preset_load_sel(self, persist: bool = True) -> None:
        """Load the selected preset; `persist` remembers it for the next start."""
        name = self.p_sel.currentText().strip()
        if not name:
            return
        try:
            pts = self.preset_mgr.load_preset(name)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f'Failed to read preset "{name}": {exc}')
            return
        self.route_pts = pts
        self.app_cfg.navigator.last_preset = name
        if persist:
            self.save_cfg()
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

    def _set_edit_mode(self, editing: bool) -> None:
        """Toggle the route editor: points change only while Edit is active."""
        if editing:
            if self._edit_snapshot is None:
                self._edit_snapshot = [list(p) for p in self.route_pts]
            self.map_widget.view.set_route_mode(True)
        else:
            self._edit_snapshot = None
            self.map_widget.view.set_route_mode(False)
        self._set_route_controls_enabled()
        self._hint_lbl.setText(
            "LMB: Add point | Middle-drag: Pan | Wheel: Zoom | ✓ Apply / ✕ Cancel"
            if editing
            else "LMB: Pan | Wheel: Zoom | ✏ Edit to modify the route"
        )

    def route_edit_apply(self) -> None:
        """Commit the edited route, save it to the selected preset, leave the editor."""
        name = self.p_sel.currentText().strip()
        self._set_edit_mode(False)
        if name:
            self._save_route_to_preset(name)
        self.routes_refresh()

    def route_edit_cancel(self) -> None:
        """Restore the route as it was when Edit was pressed."""
        snapshot = self._edit_snapshot
        self._set_edit_mode(False)
        if snapshot is not None:
            self.route_pts = snapshot

    def route_new(self) -> None:
        """Create a new empty route preset after asking for a name."""
        name = self._ask_preset_name("New route", "Create")
        if name is None:
            return
        if os.path.exists(self.preset_mgr.preset_path(name)):
            QMessageBox.information(
                self,
                "New route",
                f'Preset "{name}" already exists. Load it or use "Save As" to overwrite.',
            )
            return
        self.route_pts = []
        if not self._save_route_to_preset(name):
            return
        self._set_edit_mode(True)

    def routes_clear(self) -> None:
        self.route_pts = []
        self.routes_refresh()

    def routes_invert(self, silent: bool = False) -> None:
        if self._edit_snapshot is not None:
            self.routes_status.setText("Finish editing (Apply or Cancel) before reversing")
            self.routes_status.setStyleSheet("color: #ffaa00;")
            return
        if self.driver is not None:
            if not silent:
                QMessageBox.information(
                    self, "Routes", "Stop the autopilot before inverting the route"
                )
            self.routes_status.setText("Stop the autopilot before inverting the route")
            self.routes_status.setStyleSheet("color: #ffaa00;")
            return
        pts = list(reversed(self.route_pts))
        self.route_pts = pts
        self.routes_refresh()

    def _map_px_per_m(self) -> float:
        """Known scale of the active map from the catalog (0 when unknown)."""
        store = self.get_store()
        getter = getattr(store, "px_per_m", None)
        if not callable(getter):
            return 0.0
        try:
            return max(0.0, float(getter(self.get_map_name())))
        except Exception:
            return 0.0

    def routes_refresh(self) -> None:
        px_per_m = self._map_px_per_m() or _FALLBACK_PX_PER_M
        length_m = 0.0
        for i in range(1, len(self.route_pts)):
            p0, p1 = self.route_pts[i - 1], self.route_pts[i]
            d_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            length_m += d_px / px_per_m
        speed_cap = getattr(self.app_cfg.navigator, "speed_cap_kmh", 36.0) or 36.0
        speed_mps = max(2.0, speed_cap / 3.6)
        est_sec = int(length_m / speed_mps)
        mins, secs = divmod(est_sec, 60)
        time_txt = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        self.routes_status.setText(
            f"Route: {len(self.route_pts)} pts | ~{int(length_m)} m ({time_txt} at {int(speed_cap)} km/h)"
        )

    def _set_route_controls_enabled(self) -> None:
        """Route controls are locked while editing or while the autopilot drives."""
        editing = self._edit_snapshot is not None
        locked = editing or self.driver is not None
        for widget in self._route_locked:
            widget.setEnabled(not locked)
        self._edit_btn.setVisible(not editing)
        self._edit_btn.setEnabled(not locked)
        self._apply_btn.setVisible(editing)
        self._cancel_btn.setVisible(editing)

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
            self.routes_status.setStyleSheet("color: #88c0d0;")
            return

        if self._edit_snapshot is not None:
            self.routes_status.setText("Finish editing (Apply or Cancel) before starting")
            self.routes_status.setStyleSheet("color: #ffaa00;")
            if not silent:
                QMessageBox.information(
                    self, "Routes", "Finish editing (Apply or Cancel) before starting"
                )
            return

        if len(self.route_pts) < 2:
            self.routes_status.setText("Route not set (at least 2 points required)")
            self.routes_status.setStyleSheet("color: #ffaa00;")
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
            self.routes_status.setStyleSheet("color: #ff7c7c;")
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
