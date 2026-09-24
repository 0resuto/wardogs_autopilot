"""Map diagnostics tab with interactive QGraphicsView, tuning controls, and frame capture."""

from __future__ import annotations

from collections.abc import Callable
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

from ...common.log import get_logger
from ...vision import locator
from ..map_view import InteractiveMapWidget

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
    """Tab widget providing interactive map navigation, SIFT live pose, and locator tuning."""

    def __init__(
        self,
        parent: QWidget | None,
        cfg: dict[str, Any],
        save_cfg_fn: Callable[[], None],
        loc_thread_supplier: Callable[[], Any],
        on_pick_roi: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.save_cfg = save_cfg_fn
        self.get_loc = loc_thread_supplier
        self.on_pick_roi = on_pick_roi

        self.map_name = "zestafona"
        self._disp_th: float | None = None
        self._last_loc: dict[str, Any] | None = None

        self.tune_vars: dict[str, _StringVarCompat] = {}
        self.tune_inputs: dict[str, QLineEdit] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top status / hint bar
        top = QHBoxLayout()
        top.setSpacing(8)

        self.map_status = QLabel("", self)
        self.map_status.setStyleSheet("color: #88c0d0; font-weight: bold;")
        top.addWidget(self.map_status, stretch=1)

        hint = QLabel("Drag: Pan | Wheel: Zoom | 2x Click: Fit", self)
        hint.setStyleSheet("color: #707070; font-size: 8pt;")
        top.addWidget(hint)
        layout.addLayout(top)

        # Locator tuning bar grouped in 3 clusters
        tune_row = QHBoxLayout()
        tune_row.setSpacing(6)

        # 1. Tracking
        grp_trk = QGroupBox("Tracking", self)
        l_trk = QHBoxLayout(grp_trk)
        l_trk.setContentsMargins(6, 10, 6, 6)
        l_trk.setSpacing(4)
        self._add_tune_field(l_trk, grp_trk, "ratio", "ratio_local", 0.85, 38)
        self._add_tune_field(l_trk, grp_trk, "inl", "min_inl_local", 3, 26, is_int=True)
        self._add_tune_field(l_trk, grp_trk, "inl%", "min_inl_rate_local", 0.0, 34)
        self._add_tune_field(l_trk, grp_trk, "rad", "track_radius", 900, 38, is_int=True)
        tune_row.addWidget(grp_trk)

        # 2. Re-Acquisition
        grp_acq = QGroupBox("Re-Acquisition", self)
        l_acq = QHBoxLayout(grp_acq)
        l_acq.setContentsMargins(6, 10, 6, 6)
        l_acq.setSpacing(4)
        self._add_tune_field(l_acq, grp_acq, "ratio", "ratio_global", 0.72, 38)
        self._add_tune_field(l_acq, grp_acq, "inl", "min_inl_global", 15, 26, is_int=True)
        self._add_tune_field(l_acq, grp_acq, "inl%", "min_inl_rate_global", 0.45, 34)
        tune_row.addWidget(grp_acq)

        # 3. Consensus & Gating
        grp_misc = QGroupBox("Consensus", self)
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
        apply_btn = QPushButton("Apply", self)
        apply_btn.setObjectName("AccentButton")
        apply_btn.clicked.connect(self.apply_tune)
        top_btns.addWidget(apply_btn)

        reset_btn = QPushButton("Reset", self)
        reset_btn.clicked.connect(self.reset_tune)
        top_btns.addWidget(reset_btn)
        btn_box.addLayout(top_btns)

        self.tune_status = QLabel("", self)
        self.tune_status.setStyleSheet("color: #8ae234; font-weight: 500;")
        btn_box.addWidget(self.tune_status)
        tune_row.addLayout(btn_box)

        layout.addLayout(tune_row)

        # Diagnostic logs bar
        dbg_bar = QHBoxLayout()
        self._collect_ck = QCheckBox("Collect fail logs", self)
        self._collect_ck.setChecked(bool(self.cfg.setdefault("debug", {}).get("collect_fail_logs", True)))
        self._collect_ck.toggled.connect(self.apply_collect_logs)
        dbg_bar.addWidget(self._collect_ck)

        self.dbg_text = QLabel("", self)
        self.dbg_text.setStyleSheet(
            "background-color: #252526; color: #ffcf6a; padding: 2px 6px; border-radius: 4px;"
        )
        dbg_bar.addWidget(self.dbg_text, stretch=1)

        copy_btn = QPushButton("Copy", self)
        copy_btn.clicked.connect(self.copy_debug)
        dbg_bar.addWidget(copy_btn)
        layout.addLayout(dbg_bar)

        # Interactive Map Canvas / View
        self.map_widget = InteractiveMapWidget(enable_route_editing=False, parent=self)
        self.map_widget.set_on_center(self._center_vehicle)

        layout.addWidget(self.map_widget, stretch=1)

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

        val = str(int(self._loc_tune_cur(var_name, default)) if is_int else self._loc_tune_cur(var_name, default))
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
            s = self.tune_vars[name].get().strip() or self.tune_inputs[name].text().strip()
            try:
                v = typ(float(s) if typ is float else int(s))
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
        from ...common.config import LocatorConfig

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
        self.map_status.setText(
            f"map px: x={mp[0]:.0f} y={mp[1]:.0f}   heading: {heading:.1f}°   "
            f"s={pose['s']:.3f} inl={pose.get('inl', 0)}"
        )

    def _center_vehicle(self) -> None:
        loc = self._last_loc
        mp = (loc.get("map_px_disp") or loc.get("map_px")) if loc else None
        if mp is not None:
            self.map_widget.view.center_on_coords(mp[0], mp[1])
        else:
            self.map_widget.view.fit_view()
