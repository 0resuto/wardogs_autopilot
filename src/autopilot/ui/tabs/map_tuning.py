"""Always-visible locator/vehicle tuning panel for the Map sidebar (PySide6).

The groups are stacked vertically and the fields sit in a two-column grid so
the narrow sidebar stays readable; the panel itself scrolls with the sidebar.
Log switches and the last-reject readout live in the Logs sidebar section
(`logs_tab.py`), not here.
"""

from __future__ import annotations

from typing import Any, cast

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...common.config import CaptureConfig, LocatorConfig, NavigatorConfig
from ..param_tips import tip_for
from ..theme import GREEN, RED, TEXT_MUTED
from .common import MapTabBase, StringVarCompat, compact_label

# (label, config key, default, is_int, section); hover text lives in
# ui/param_tips.py so the tuning panel and the bench share one source.
_TUNE_GROUPS: list[tuple[str, list[tuple[Any, ...]]]] = [
    (
        "Tracking",
        [
            ("ratio", "ratio_local", 0.85, False, "locator"),
            ("inl", "min_inl_local", 4, True, "locator"),
            ("inl%", "min_inl_rate_local", 0.0, False, "locator"),
            ("rad", "track_radius", 900, True, "locator"),
            ("kps", "max_kp_frame", 1200, True, "locator"),
            ("smooth", "smooth_alpha", 0.5, False, "locator"),
            ("reset px", "smooth_reset_px", 100, True, "locator"),
            ("ransac px", "ransac_px", 3.0, False, "locator"),
            ("xfeat cos", "xfeat_min_cos", 0.82, False, "locator"),
            ("xfeat kp", "xfeat_top_k", 2000, True, "locator"),
        ],
    ),
    (
        "Re-Acquisition",
        [
            ("ratio", "ratio_global", 0.9, False, "locator"),
            ("inl", "min_inl_global", 5, True, "locator"),
            ("inl%", "min_inl_rate_global", 0.0, False, "locator"),
        ],
    ),
    (
        "Consensus",
        [
            ("vote", "vote_need", 3, True, "locator"),
            ("head°", "heading_gate_deg", 0, True, "locator"),
            ("skip", "vote_inl_skip", 40, True, "locator"),
            ("brk", "early_inl", 40, True, "locator"),
            ("hold", "hold_frames", 5, True, "locator"),
        ],
    ),
    (
        "Vehicle",
        [
            ("gain", "yaw_gain", 1.0, False, "navigator"),
            ("lat g", "corner_lat_g", 0.35, False, "navigator"),
            ("brake g", "brake_g", 0.45, False, "navigator"),
            ("min km/h", "corner_min_kmh", 12.0, False, "navigator"),
            ("max km/h", "corner_max_kmh", 22.0, False, "navigator"),
            ("ahead m", "plan_ahead_m", 200.0, False, "navigator"),
            ("cut m", "corner_cut_m", 15.0, False, "navigator"),
        ],
    ),
    (
        "Centering",
        [
            ("dead m", "xte_m", 0.5, False, "navigator"),
            ("ref m", "xte_outer_m", 4.0, False, "navigator"),
            ("look s", "steer_look_s", 1.6, False, "navigator"),
            ("settle s", "settle_s", 0.6, False, "navigator"),
            ("lead s", "steer_lead_s", 0.25, False, "navigator"),
            ("skip m", "skip_ahead_m", 150.0, False, "navigator"),
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

        # Engine selector: the three localization variants apply live; the
        # tracker reconfigures the locator on its next frame.
        eng_row = QHBoxLayout()
        eng_row.setSpacing(6)
        eng_lbl = QLabel("engine", self._tune_container)
        eng_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        eng_row.addWidget(eng_lbl)
        self.engine_combo = QComboBox(self._tune_container)
        self.engine_combo.addItems(["sift", "orb", "xfeat", "hybrid"])
        current = str(self._loc_tune_cur("engine", "sift") or "sift")
        slot = self.engine_combo.findText(current)
        self.engine_combo.setCurrentIndex(slot if slot >= 0 else 0)
        self.engine_combo.setToolTip(tip_for("engine"))
        self.engine_combo.currentTextChanged.connect(self._engine_changed)
        eng_row.addWidget(self.engine_combo)

        # Capture cadence: applies live (the producer re-reads it every loop).
        # Hybrid benefits from 20-30 fps; SIFT drops the extra frames anyway.
        fps_lbl = QLabel("fps", self._tune_container)
        fps_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        eng_row.addWidget(fps_lbl)
        fps_val = str(self._capture_tune_cur("fps", 10))
        self.fps_input = QLineEdit(fps_val, self._tune_container)
        self.fps_input.setObjectName("TuneInput")
        self.fps_input.setFixedWidth(_FIELD_W)
        self.fps_input.setCursorPosition(0)
        self.fps_input.setToolTip(tip_for("fps"))
        self.fps_input.editingFinished.connect(self.apply_tune)
        eng_row.addWidget(self.fps_input)
        self.fps_var = StringVarCompat(fps_val)
        self.tune_inputs["fps"] = self.fps_input
        self.tune_vars["fps"] = self.fps_var
        eng_row.addStretch(1)
        tune_vbox.addLayout(eng_row)

        for title, fields in _TUNE_GROUPS:
            grp = QGroupBox(title, self._tune_container)
            grid = QGridLayout(grp)
            grid.setContentsMargins(8, 10, 8, 8)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            for i, (lbl_text, var, default, is_int, section) in enumerate(fields):
                row, col = divmod(i, 2)
                self._add_tune_cell(grid, grp, row, col, lbl_text, var, default, is_int, section)
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
        tip = tip_for(var_name)
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

    def _capture_tune_cur(self, name: str, default: Any) -> Any:
        block = self.cfg.setdefault("capture", {})
        value = block.get(name, default)
        return default if value is None else value

    def _nav_tune_cur(self, name: str, default: Any) -> Any:
        block = self.cfg.setdefault("navigator", {})
        value = block.get(name, default)
        return default if value is None else value

    def _engine_changed(self, kind: str) -> None:
        """Apply the engine choice live (the tracker picks it up next frame)."""
        kind = str(kind or "sift")
        self.cfg.setdefault("locator", {})["engine"] = kind
        self.app_cfg.locator.engine = cast(Any, kind)
        loc = self.get_loc()
        if loc is not None and hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
            loc.cfg.setdefault("locator", {})["engine"] = kind
        self.save_cfg()
        self.tune_status.setText(f"engine: {kind}")
        self.tune_status.setStyleSheet(f"color: {GREEN};")

    def apply_tune(self) -> None:
        """Validate tuning inputs, update configuration, and notify consumers."""
        ranges = {
            "fps": ("capture", 1, 60, int, "fps in [1 .. 60]"),
            "ratio_local": ("locator", 0.1, 1.0, float, "TRACK ratio in [0.1 .. 1.0]"),
            "min_inl_local": ("locator", 1, 50, int, "TRACK min_inl in [1 .. 50]"),
            "min_inl_rate_local": ("locator", 0.0, 1.0, float, "TRACK inl% in [0.0 .. 1.0]"),
            "ransac_px": ("locator", 0.5, 24.0, float, "TRACK ransac px in [0.5 .. 24]"),
            "xfeat_min_cos": ("locator", 0.5, 1.0, float, "XFEAT cos in [0.5 .. 1.0]"),
            "xfeat_top_k": ("locator", 100, 8000, int, "XFEAT kp in [100 .. 8000]"),
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
            "smooth_alpha": ("locator", 0.0, 1.0, float, "TRACK smooth in [0.0 .. 1.0]"),
            "smooth_reset_px": ("locator", 0.0, 5000.0, int, "TRACK reset px in [0 .. 5000]"),
            "yaw_gain": ("navigator", 0.05, 5.0, float, "VEH gain in [0.05 .. 5.0]"),
            "corner_lat_g": ("navigator", 0.05, 1.5, float, "VEH lat g in [0.05 .. 1.5]"),
            "brake_g": ("navigator", 0.05, 2.0, float, "VEH brake g in [0.05 .. 2.0]"),
            "corner_min_kmh": ("navigator", 0.0, 79.0, float, "VEH min km/h in [0 .. 79]"),
            "corner_max_kmh": ("navigator", 0.0, 79.0, float, "VEH max km/h in [0 .. 79]"),
            "plan_ahead_m": ("navigator", 20.0, 1000.0, float, "VEH ahead m in [20 .. 1000]"),
            "corner_cut_m": ("navigator", 4.0, 60.0, float, "VEH cut m in [4 .. 60]"),
            "xte_m": ("navigator", 0.0, 10.0, float, "CEN dead m in [0 .. 10]"),
            "xte_outer_m": ("navigator", 0.5, 20.0, float, "CEN ref m in [0.5 .. 20]"),
            "steer_look_s": ("navigator", 0.4, 4.0, float, "COR look s in [0.4 .. 4.0]"),
            "settle_s": ("navigator", 0.1, 2.0, float, "COR settle s in [0.1 .. 2.0]"),
            "steer_lead_s": ("navigator", 0.0, 1.0, float, "COR lead s in [0.0 .. 1.0]"),
            "skip_ahead_m": ("navigator", 0.0, 1000.0, float, "COR skip m in [0 .. 1000]"),
        }
        parsed: dict[str, dict[str, Any]] = {"locator": {}, "navigator": {}, "capture": {}}
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
            self.tune_status.setText("Invalid CEN ref m must be greater than dead m")
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
        cap_block = self.cfg.setdefault("capture", {})
        cap_block.update(parsed["capture"])
        self.save_cfg()

        loc = self.get_loc()
        if loc is not None:
            if hasattr(loc, "apply_tune"):
                loc.apply_tune(loc_block)
            elif hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                loc.cfg.setdefault("locator", {}).update(loc_block)

        if parsed["capture"]:
            for name, value in parsed["capture"].items():
                setattr(self.app_cfg.capture, name, value)
            if "fps" in parsed["capture"]:
                if loc is not None and hasattr(loc, "set_fps"):
                    loc.set_fps(int(parsed["capture"]["fps"]))
                elif loc is not None and hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                    loc.cfg.setdefault("capture", {})["fps"] = int(parsed["capture"]["fps"])

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
        defaults.update(CaptureConfig().model_dump())  # fps field lives in Capture
        for k, v in defaults.items():
            if k in self.tune_vars:
                self.tune_vars[k].set(str(v))
                if k in self.tune_inputs:
                    self.tune_inputs[k].setText(str(v))
        if hasattr(self, "engine_combo"):
            slot = self.engine_combo.findText(str(LocatorConfig().engine))
            if slot >= 0:
                self.engine_combo.setCurrentIndex(slot)
        self.apply_tune()
