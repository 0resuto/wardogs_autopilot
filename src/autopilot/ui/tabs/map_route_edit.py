"""Route editing state and status line for the Map tab (PySide6)."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QMessageBox

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
