"""Route editing state and status line for the Map tab (PySide6)."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QMessageBox

from ...navigation.speed_profile import G, RouteSpeedPlanner
from ..theme import YELLOW
from .common import MapTabBase

# Only used when the map catalog has no m_per_px entry for the active map.
_FALLBACK_PX_PER_M = 2.0


class MapRouteEditMixin(MapTabBase):
    """Edit-mode state, route status, and the route controls lock."""

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
            "LMB: Add point | Middle-drag: Pan | Wheel: Zoom | Apply / Cancel"
            if editing
            else "LMB: Pan | Wheel: Zoom | Edit to modify the route"
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

    def routes_clear(self) -> None:
        self.route_pts = []
        self.routes_refresh()

    def routes_invert(self, silent: bool = False) -> None:
        if self._edit_snapshot is not None:
            self.routes_status.setText("Finish editing (Apply or Cancel) before reversing")
            self.routes_status.setStyleSheet(f"color: {YELLOW};")
            return
        if self.driver is not None:
            if not silent:
                QMessageBox.information(
                    self, "Routes", "Stop the autopilot before inverting the route"
                )
            self.routes_status.setText("Stop the autopilot before inverting the route")
            self.routes_status.setStyleSheet(f"color: {YELLOW};")
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
        est_sec, planned = self._estimate_route_seconds(length_m)
        mins, secs = divmod(est_sec, 60)
        time_txt = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        speed_cap = int(getattr(self.app_cfg.navigator, "speed_cap_kmh", 36.0) or 36.0)
        how = "planned" if planned else f"at max {speed_cap} km/h"
        self.routes_status.setText(
            f"Route: {len(self.route_pts)} pts | ~{int(length_m)} m ({time_txt}, {how})"
        )

    def _estimate_route_seconds(self, length_m: float) -> tuple[int, bool]:
        """(seconds, planned): planned=True when the corner planner timed it.

        A cap-only estimate assumes the whole route at top speed, which is
        optimistic whenever corners force braking; with the planner enabled the
        per-vertex speed limits are integrated instead. Falls back to the cap
        estimate when the planner is disabled or the map scale is unknown.
        """
        nav = self.app_cfg.navigator
        px_per_m = self._map_px_per_m()
        pts = self.route_pts
        speed_cap = float(getattr(nav, "speed_cap_kmh", 36.0) or 36.0)
        if getattr(nav, "speed_profile", False) and px_per_m > 0 and len(pts) >= 2:
            try:
                planner = RouteSpeedPlanner(
                    lat_accel_mps2=float(nav.corner_lat_g) * G,
                    brake_decel_mps2=float(nav.brake_g) * G,
                    min_speed_kmh=float(nav.corner_min_kmh),
                    lookahead_m=float(nav.plan_ahead_m),
                    cut_m=float(nav.corner_cut_m),
                )
                pts_t = [(float(p[0]), float(p[1])) for p in pts]
                total_s = 0.0
                for i in range(1, len(pts_t)):
                    p0, p1 = pts_t[i - 1], pts_t[i]
                    seg_m = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / px_per_m
                    if seg_m <= 0.0:
                        continue
                    v0 = planner.target_speed_kmh(p0, pts_t, i - 1, px_per_m)
                    v1 = planner.target_speed_kmh(p1, pts_t, i, px_per_m)
                    v_kmh = min(
                        v0 if v0 is not None else speed_cap,
                        v1 if v1 is not None else speed_cap,
                    )
                    total_s += seg_m / (max(v_kmh, 1.0) / 3.6)
                if total_s > 0.0:
                    return int(total_s), True
            except Exception:
                pass
        return int(length_m / max(2.0, speed_cap / 3.6)), False

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
