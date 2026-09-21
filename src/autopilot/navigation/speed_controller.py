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
        self._saw_motion = False
        self._runaway = False
        self._braking = False

    @property
    def saw_motion(self) -> bool:
        return self._saw_motion

    @property
    def runaway(self) -> bool:
        return self._runaway

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

    def update_scale(self, mv: float) -> None:
        """Refine speed cap and pixel-per-meter scale on the fly from pose motion."""
        if mv > self._vmax_px:
            self._vmax_px = mv
        self._px_per_m = self._vmax_px * 3.6 / self.speed_cap_kmh if self.speed_cap_kmh > 0 else 0.0
        if mv > 40.0:
            self._saw_motion = True

    def calc_target_speed(self, road_turn: float, xte: float, xte_lim: float) -> float:
        """Calculate target speed based on road curvature ahead."""
        cruise = max(self._vmax_px, self.v_cruise)
        tgt_spd = cruise / (1.0 + (road_turn / self.corner_deg) ** 2)
        tgt_spd = max(tgt_spd, self.v_min)

        # Alignment mode: outside the corridor, maintain cruise speed to turn the wheels
        if abs(xte) > xte_lim:
            tgt_spd = max(tgt_spd, cruise)
        return tgt_spd

    def decide_throttle_and_brake(
        self,
        mv: float,
        tgt_spd: float,
        dist: float,
        arrive_r: float,
        turn_angle: float,
        steer: int,
        micro: bool,
        road_turn: float,
        turn_min: float,
        err: float,
        xte: float,
        xte_lim: float,
    ) -> tuple[bool, bool]:
        """Decide gas (W) and brake (SPACE) pedal actions.

        Returns (gas_w, brake_space).
        """
        turn_scale = 1.0 + turn_angle / 90.0
        required_brake_d = turn_scale * (mv * mv / (2.0 * self.brake_d) + 12.0)

        # Runaway / heading loss tracking
        if self._saw_motion and mv < 40.0 and abs(err) > 60.0 and abs(xte) <= xte_lim:
            self._runaway = True
        elif abs(err) < 12.0 or abs(xte) > xte_lim:
            self._runaway = False

        # Braking hysteresis
        overspd = mv > tgt_spd * 1.25
        if self._braking and (mv <= tgt_spd or dist <= arrive_r + 10 or steer != 0):
            self._braking = False
        if not self._braking and overspd and dist > arrive_r + 10:
            self._braking = True

        braking = (
            not self._runaway
            and steer == 0
            and (self._braking or (dist < required_brake_d and mv > 12.0))
        )

        if braking:
            return False, True

        gas_w = True
        if steer != 0 and not micro:
            # Cut gas in sharp corner when holding wheel inside corridor to prevent skidding
            if road_turn >= turn_min and abs(xte) <= xte_lim:
                gas_w = False
        elif self._runaway or overspd:
            gas_w = False

        return gas_w, False
