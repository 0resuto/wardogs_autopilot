"""Always-visible locator/vehicle tuning panel for the Map sidebar (PySide6).

The groups are stacked vertically and the fields sit in a two-column grid so
the narrow sidebar stays readable; the panel itself scrolls with the sidebar.
Log switches and the last-reject readout live in the Logs sidebar section
(`logs_tab.py`), not here.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...common.config import LocatorConfig, NavigatorConfig
from ..theme import GREEN, RED, TEXT_MUTED
from .common import MapTabBase, StringVarCompat, compact_label

# (label, config key, default, is_int, section, tooltip)
_TUNE_GROUPS: list[tuple[str, list[tuple[Any, ...]]]] = [
    (
        "Tracking",
        [
            ("ratio", "ratio_local", 0.85, False, "locator", ""),
            ("inl", "min_inl_local", 4, True, "locator", ""),
            ("inl%", "min_inl_rate_local", 0.0, False, "locator", ""),
            ("rad", "track_radius", 900, True, "locator", ""),
            (
                "kps",
                "max_kp_frame",
                1200,
                True,
                "locator",
                "Keypoints kept per frame (response-ranked): lower = less latency,\n"
                "higher = more robust in low-texture areas",
            ),
        ],
    ),
    (
        "Re-Acquisition",
        [
            ("ratio", "ratio_global", 0.9, False, "locator", ""),
            ("inl", "min_inl_global", 5, True, "locator", ""),
            ("inl%", "min_inl_rate_global", 0.0, False, "locator", ""),
        ],
    ),
    (
        "Consensus",
        [
            ("vote", "vote_need", 3, True, "locator", ""),
            ("head°", "heading_gate_deg", 0, True, "locator", ""),
            ("skip", "vote_inl_skip", 40, True, "locator", ""),
            (
                "brk",
                "early_inl",
                40,
                True,
                "locator",
                "Stop trying further scale-level candidates once a match reaches\n"
                "this many inliers (0 = scan every level; lower = snappier,\n"
                "higher = more thorough)",
            ),
            ("hold", "hold_frames", 5, True, "locator", ""),
        ],
    ),
    (
        "Vehicle",
        [
            (
                "gain",
                "yaw_gain",
                1.0,
                False,
                "navigator",
                "Yaw-authority scale of the model (see tools/calibrate_vehicle.py)",
            ),
            (
                "lat g",
                "corner_lat_g",
                0.35,
                False,
                "navigator",
                "Lateral grip budget for planning corner speeds: v = sqrt(lat_g*9.81*R).\n"
                "Lower = slower corners (if it slides wide), higher = faster (but the planner\n"
                "may outrun what the steering can hold).",
            ),
            (
                "brake g",
                "brake_g",
                0.45,
                False,
                "navigator",
                "Braking deceleration budget for planning when to brake before a corner.\n"
                "Higher = brakes later/harder. Measured value from your runs: ~0.5g\n"
                "(python tools/calibrate_vehicle.py output/nav_dbg_*.jsonl).",
            ),
            (
                "min km/h",
                "corner_min_kmh",
                12.0,
                False,
                "navigator",
                "Lower edge of the steady-corner hold window (km/h)",
            ),
            (
                "max km/h",
                "corner_max_kmh",
                22.0,
                False,
                "navigator",
                "Upper edge of the steady-corner hold window: inside the window the\n"
                "driver neither accelerates nor brakes (no more brake/gas hunting)",
            ),
            (
                "ahead m",
                "plan_ahead_m",
                200.0,
                False,
                "navigator",
                "Speed planning horizon along the route (meters)",
            ),
            (
                "cut m",
                "corner_cut_m",
                15.0,
                False,
                "navigator",
                "Distance over which a sharp vertex is rounded by the planner",
            ),
        ],
    ),
    (
        "Corridors",
        [
            (
                "inner m",
                "xte_m",
                4.0,
                False,
                "navigator",
                "Inner corridor (normal driving). Deviations beyond it get firmer corrections.",
            ),
            (
                "outer m",
                "xte_outer_m",
                12.0,
                False,
                "navigator",
                "Outer corridor (warning). Past it the driver slows down and steers hardest.",
            ),
            (
                "look s",
                "steer_look_s",
                1.6,
                False,
                "navigator",
                "Steering lookahead in seconds of travel: the aim point ahead on the route.\n"
                "Lower = tighter line and more active steering; higher = smoother, cuts curves.",
            ),
            (
                "settle s",
                "settle_s",
                0.6,
                False,
                "navigator",
                "Pause after a completed steering hold before the next one (s).\n"
                "Lower = more frequent corrections; higher = smoother but a dead wheel\n"
                "for that long after each correction. At speed the pause is also capped\n"
                "by distance (8 m), so it shortens automatically.",
            ),
            (
                "lead s",
                "steer_lead_s",
                0.25,
                False,
                "navigator",
                "Release anticipation in seconds: how much heading change still arrives\n"
                "through the pose/key latency after the wheel is released. Raise it if the\n"
                "car systematically overshoots, lower it if it releases too early.",
            ),
            (
                "skip m",
                "skip_ahead_m",
                150.0,
                False,
                "navigator",
                "Route re-acquisition window (m). Only when the car is outside the outer\n"
                "corridor: the active point may jump forward to the nearest route point\n"
                "within this route length (it never jumps backwards or to the final point).\n"
                "0 disables the re-acquisition.",
            ),
        ],
    ),
]

_FIELD_W = 48


class MapTuningMixin(MapTabBase):
    """Builds the tuning panel and applies/resets its values live."""

    def _build_tuning_panel(self, root_layout: QVBoxLayout) -> None:
        self._tune_container = QWidget(self)
        tune_vbox = QVBoxLayout(self._tune_container)
        tune_vbox.setContentsMargins(0, 2, 0, 2)
        tune_vbox.setSpacing(6)

        for title, fields in _TUNE_GROUPS:
            grp = QGroupBox(title, self._tune_container)
            grid = QGridLayout(grp)
            grid.setContentsMargins(8, 10, 8, 8)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            for i, (lbl_text, var, default, is_int, section, tip) in enumerate(fields):
                row, col = divmod(i, 2)
                self._add_tune_cell(
                    grid, grp, row, col, lbl_text, var, default, is_int, section, tip
                )
            tune_vbox.addWidget(grp)

        # Action footer: status left, Reset/Apply right (they apply to every group)
        footer = QHBoxLayout()
        footer.setSpacing(6)
        self.tune_status = QLabel("", self._tune_container)
        self.tune_status.setStyleSheet(f"color: {GREEN}; font-weight: 500;")
        compact_label(self.tune_status)
        footer.addWidget(self.tune_status, stretch=1)

        reset_btn = QPushButton("Reset", self._tune_container)
        reset_btn.clicked.connect(self.reset_tune)
        footer.addWidget(reset_btn)

        apply_btn = QPushButton("Apply", self._tune_container)
        apply_btn.setObjectName("AccentButton")
        apply_btn.clicked.connect(self.apply_tune)
        footer.addWidget(apply_btn)
        tune_vbox.addLayout(footer)

        root_layout.addWidget(self._tune_container)

    def _add_tune_cell(
        self,
        grid: QGridLayout,
        parent: QWidget,
        row: int,
        col: int,
        lbl_text: str,
        var_name: str,
        default: Any,
        is_int: bool,
        section: str,
        tip: str,
    ) -> None:
        cell = QHBoxLayout()
        cell.setSpacing(6)

        # Field first with its label to the right: both columns then align at
        # the left edge and a label can never read as the neighbouring field's
        # caption.
        cur = self._nav_tune_cur if section == "navigator" else self._loc_tune_cur
        value = cur(var_name, default)
        val = str(int(value) if is_int else value)
        inp = QLineEdit(val, parent)
        inp.setObjectName("TuneInput")
        inp.setFixedWidth(_FIELD_W)
        inp.setCursorPosition(0)  # narrow fields must show the leading digits
        if tip:
            inp.setToolTip(tip)
        cell.addWidget(inp)

        lbl = QLabel(lbl_text, parent)
        lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        if tip:
            lbl.setToolTip(tip)
        cell.addWidget(lbl)
        cell.addStretch(1)
        grid.addLayout(cell, row, col)
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
                self.tune_status.setStyleSheet(f"color: {RED};")
                return
            if not (lo <= v <= hi):
                self.tune_status.setText(f"Invalid {desc}")
                self.tune_status.setStyleSheet(f"color: {RED};")
                return
            parsed[section][name] = v

        if parsed["navigator"]["xte_outer_m"] <= parsed["navigator"]["xte_m"]:
            self.tune_status.setText("Invalid COR outer m must be greater than inner m")
            self.tune_status.setStyleSheet(f"color: {RED};")
            return

        if parsed["navigator"]["corner_max_kmh"] <= parsed["navigator"]["corner_min_kmh"]:
            self.tune_status.setText("Invalid VEH max km/h must be greater than min km/h")
            self.tune_status.setStyleSheet(f"color: {RED};")
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
        self.tune_status.setStyleSheet(f"color: {GREEN};")

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
