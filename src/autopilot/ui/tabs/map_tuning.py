"""Collapsible locator/vehicle tuning panel for the Map tab (PySide6)."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...common.config import LocatorConfig, NavigatorConfig
from .common import MapTabBase, StringVarCompat, compact_label


class MapTuningMixin(MapTabBase):
    """Builds the tuning panel and applies/resets its values live."""

    def _build_tuning_panel(self, root_layout: QVBoxLayout) -> None:
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
        self._add_tune_field(
            l_misc,
            grp_misc,
            "brk",
            "early_inl",
            40,
            26,
            is_int=True,
            tip="Stop trying further scale-level candidates once a match reaches\n"
            "this many inliers (0 = scan every level; lower = snappier,\n"
            "higher = more thorough)",
        )
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
        compact_label(self.tune_status)
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
        compact_label(self.dbg_text)
        dbg_bar.addWidget(self.dbg_text, stretch=1)

        copy_btn = QPushButton("Copy", self._tune_container)
        copy_btn.clicked.connect(self.copy_debug)
        dbg_bar.addWidget(copy_btn)

        tune_vbox.addLayout(dbg_bar)

        self._tune_container.setVisible(False)
        root_layout.addWidget(self._tune_container)

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
        self.tune_vars[var_name] = StringVarCompat(val)

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
            "early_inl": ("locator", 0, 200, int, "brk in [0 .. 200] inl (0 = scan all)"),
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
