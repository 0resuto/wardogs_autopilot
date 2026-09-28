"""Speed profiling, adaptive scale calibration, and braking logic."""

from __future__ import annotations


class SpeedController:
    """Manages driving speed, deceleration profiles, and braking decisions."""

    def __init__(
        self,
        speed_cap_kmh: float = 79.0,
        brake_d: float = 260.0,
        v_cruise: float = 45.0,
        v_min: float = 14.0,
        corner_deg: float = 22.0,
    ) -> None:
        self.speed_cap_kmh = float(speed_cap_kmh)
        self.brake_d = float(brake_d)
        self.v_cruise = float(v_cruise)
        self.v_min = float(v_min)
        self.corner_deg = float(corner_deg)

        self._vmax_px = 45.0
        self._px_per_m = 0.0
        self._braking = False

    @property
    def is_braking(self) -> bool:
        return self._braking

    def px_per_m_now(self) -> float:
        """Current pixel-to-meter scale conversion factor."""
        if self._px_per_m > 0:
            return self._px_per_m
        if self.speed_cap_kmh > 0 and self._vmax_px > 0:
            return self._vmax_px * 3.6 / self.speed_cap_kmh
        return 0.0

    def to_kmh(self, px_s: float) -> float:
        """Convert pixels/sec to km/h."""
        pm = self.px_per_m_now()
        return px_s * 3.6 / pm if pm > 0 else 0.0

    def to_meters(self, d_px: float) -> float:
        """Convert map pixels to meters."""
        pm = self.px_per_m_now()
        return d_px / pm if pm > 0 else 0.0

    def from_kmh(self, kmh: float) -> float:
        """Convert km/h to map pixels per second with the current scale."""
        pm = self.px_per_m_now()
        return kmh / 3.6 * pm if pm > 0 else 0.0

    def brake_distance_px(self, speed_px: float, decel_mps2: float) -> float:
        """Physical stopping distance in map pixels for a deceleration budget."""
        if decel_mps2 <= 0.0:
            return float("inf")
        pm = self.px_per_m_now()
        if pm <= 0:
            return float("inf")
        v_mps = max(0.0, speed_px) / pm
        return v_mps * v_mps / (2.0 * decel_mps2) * pm

    def update_scale(self, mv: float) -> None:
        """Refine speed cap and pixel-per-meter scale on the fly from pose motion."""
        if mv > self._vmax_px:
            self._vmax_px = mv
        self._px_per_m = self._vmax_px * 3.6 / self.speed_cap_kmh if self.speed_cap_kmh > 0 else 0.0

    def calc_target_speed(self, road_turn: float) -> float:
        """Target speed (px/s) from the curvature of the road ahead."""
        cruise = max(self._vmax_px, self.v_cruise)
        tgt_spd = cruise / (1.0 + (road_turn / self.corner_deg) ** 2)
        return max(tgt_spd, self.v_min)

    def decide_throttle_and_brake(
        self,
        mv: float,
        tgt_spd: float,
        steer: int,
        micro: bool,
        road_turn: float,
        turn_min: float,
        xte: float,
        xte_lim: float,
    ) -> tuple[bool, bool]:
        """Decide gas (W) and brake (SPACE) from the target speed.

        Braking follows the overspeed against the (already planned) target
        speed; the final waypoint stop is handled by the driver itself.
        Returns (gas_w, brake_space).
        """
        overspd = mv > tgt_spd * 1.25

        if self._braking and (mv <= tgt_spd or steer != 0):
            self._braking = False
        if not self._braking and overspd:
            self._braking = True

        if steer == 0 and self._braking:
            return False, True

        gas_w = True
        if steer != 0 and not micro:
            # Cut gas in sharp corners while holding the wheel inside the corridor
            if road_turn >= turn_min and abs(xte) <= xte_lim:
                gas_w = False
        elif overspd:
            gas_w = False

        return gas_w, False
