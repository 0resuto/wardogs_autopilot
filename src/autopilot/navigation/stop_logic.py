"""Final-waypoint stop logic for the route follower (extracted mixin).

Keeps the stop thresholds, the measured-pose/OCR freshness gates and the
finish hand-off out of the main control loop. The state lives on FollowDriver
(`_stop_s_t`, `_final_t0`), so this mixin only supplies the decisions.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from ..common.log import get_logger

logger = get_logger("follow")


class FinalStopMixin:
    """Full-stop state machine at the final waypoint."""

    path: Any
    speed_ctrl: Any
    planner: Any
    kb: Any
    speed: float
    state: str
    brake_d: float
    stop_speed_kmh: float
    stop_min_px_s: float
    stop_confirm_s: float
    stop_hold: float
    stop_timeout: float
    _speed_kmh: float
    _last_measured_t: float | None
    _last_speed_t: float | None
    _stop_s_t: float | None
    _final_t0: float | None
    _stop_ev: threading.Event

    def _final_brake_dist(self) -> float:
        """Distance to the final waypoint at which full-stop braking begins.

        Physical stopping distance from the plan's braking budget, with the
        legacy constant as fallback when no planner is configured.
        """
        mv = max(self.path.mv, 0.0)
        if self.planner is not None:
            distance = self.speed_ctrl.brake_distance_px(mv, self.planner.brake_decel)
            if math.isfinite(distance):
                return distance * 1.2 + 8.0
        return (mv**2 / (2.0 * self.brake_d) + 12.0) * 1.5

    def _stop_thr(self) -> float:
        """Full-stop speed threshold in px/s (km/h knob + configured noise floor)."""
        pm = self.speed_ctrl.px_per_m_now()
        if pm > 0:
            return max(self.stop_speed_kmh * pm / 3.6, self.stop_min_px_s)
        return self.stop_min_px_s

    def _pose_fresh(self, now: float) -> bool:
        """True when the last MEASURED pose is recent enough to trust the stop."""
        return (
            self._last_measured_t is not None
            and (now - self._last_measured_t) <= self.stop_confirm_s
        )

    def _speed_fresh(self, now: float) -> bool:
        """True when the last OCR speedometer reading is recent enough to trust."""
        return self._last_speed_t is not None and (now - self._last_speed_t) <= self.stop_confirm_s

    def _stop_state(self, now: float) -> str:
        """Final-waypoint stop state: 'stopped', 'braking' or 'timeout'.

        A fresh speedometer OCR reading is the ground truth (it is independent
        of localization, so a frozen pose cannot fake a stop). Without it the
        position-derived estimate is used, and only fresh measured poses may
        confirm it. 'timeout' is the emergency exit after stop_timeout seconds.
        """
        if self._speed_fresh(now):
            below = self._speed_kmh <= self.stop_speed_kmh
        elif self._pose_fresh(now):
            below = max(self.speed, self.path.mv) <= self._stop_thr()
        else:
            below = False

        if below:
            if self._stop_s_t is None:
                self._stop_s_t = now
            elif now - self._stop_s_t >= self.stop_hold:
                return "stopped"
        else:
            self._stop_s_t = None
        t0 = self._final_t0
        if t0 is not None and (now - t0) >= self.stop_timeout:
            return "timeout"
        return "braking"

    def _finish(self) -> None:
        """Full stop at the final waypoint: disable the autopilot (as with F7)."""
        logger.info("[nav] full stop at final waypoint — disabling autopilot")
        self.state = "finished"
        try:
            if self.kb is not None:
                self.kb.release_all()
        except OSError:
            pass
        self._stop_ev.set()
