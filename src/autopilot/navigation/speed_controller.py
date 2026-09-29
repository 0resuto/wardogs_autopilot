"""Speed profiling, scale handling, and braking logic.

The map's physical scale is data (`catalog.json` -> `m_per_px`, injected as
px_per_m) and is authoritative when known: no "max observed speed equals the
configured cap" guessing. The speedometer OCR ratio validates it and takes
over when it persistently disagrees (the OCR ratio is a direct measurement and
the catalog may describe an older map revision). The legacy speed-profile
estimate remains only for maps without catalog scale.
"""

from __future__ import annotations

import statistics
from collections import deque

from ..common.log import get_logger

logger = get_logger("speed_controller")


class SpeedController:
    """Manages driving speed, deceleration profiles, and braking decisions."""

    # rolling windows for the px/m scale estimate (one entry per measured pose)
    _SCALE_WINDOW = 200
    _SCALE_MIN_SAMPLES = 8
    _SCALE_MIN_KMH = 30.0
    _SCALE_RECONCILE = 0.15  # relative drift at which the measured scale wins

    def __init__(
        self,
        speed_cap_kmh: float = 79.0,
        brake_d: float = 260.0,
        v_cruise: float = 45.0,
        v_min: float = 14.0,
        corner_deg: float = 22.0,
        corner_min_kmh: float = 12.0,
        corner_max_kmh: float = 22.0,
        px_per_m: float = 0.0,
    ) -> None:
        self.speed_cap_kmh = float(speed_cap_kmh)
        self.brake_d = float(brake_d)
        self.v_cruise = float(v_cruise)
        self.v_min = float(v_min)
        self.corner_deg = float(corner_deg)
        self.corner_min_kmh = float(corner_min_kmh)
        self.corner_max_kmh = float(corner_max_kmh)

        self._vmax_px = 45.0
        self._px_per_m = max(0.0, float(px_per_m))
        self._scale_source = "map" if self._px_per_m > 0 else None
        self._braking = False
        # Scale anchoring: the OCR ratio is preferred; the speed-profile
        # fallback uses a rolling percentile, never a running max (a single
        # pose spike inflated the old max by ~60% and widened every corridor).
        self._ratio_hist: deque[float] = deque(maxlen=self._SCALE_WINDOW)
        self._mv_hist: deque[float] = deque(maxlen=self._SCALE_WINDOW)

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

    def update_scale(self, mv: float, ocr_kmh: float | None = None) -> None:
        """Refine or validate the pixel-per-meter scale from pose motion.

        A catalog scale (source 'map') is authoritative: the OCR ratio only
        validates it, and a persistent disagreement beyond `_SCALE_RECONCILE`
        is logged and adopted once - the OCR ratio is a direct physical
        measurement and the catalog may describe an older map revision.
        Without a catalog scale the OCR median is adopted when enough samples
        exist; the legacy "observed speed vs the configured cap" estimate is a
        last resort for maps without both.
        """
        if mv > 0.0:
            self._mv_hist.append(float(mv))
        if self._mv_hist:
            vals = sorted(self._mv_hist)
            p90 = vals[min(len(vals) - 1, int(0.9 * len(vals)))]
            self._vmax_px = max(45.0, p90)

        if ocr_kmh is not None and ocr_kmh >= self._SCALE_MIN_KMH and mv > 1.0:
            self._ratio_hist.append(float(mv) / (float(ocr_kmh) / 3.6))

        measured = None
        if len(self._ratio_hist) >= self._SCALE_MIN_SAMPLES:
            measured = float(statistics.median(self._ratio_hist))

        if self._scale_source == "map":
            self._reconcile_known_scale(measured)
            return
        if self._scale_source == "ocr":
            if measured is not None:
                self._px_per_m = measured
            return
        if measured is not None:
            self._px_per_m = measured
            self._scale_source = "ocr"
            return

        self._px_per_m = (
            self._vmax_px * 3.6 / self.speed_cap_kmh if self.speed_cap_kmh > 0 else 0.0
        )
        self._scale_source = "legacy" if self._px_per_m > 0 else None

    def _reconcile_known_scale(self, measured: float | None) -> None:
        """Adopt the measured scale once when it strongly disagrees."""
        if measured is None or self._px_per_m <= 0:
            return
        drift = abs(measured - self._px_per_m) / self._px_per_m
        if drift <= self._SCALE_RECONCILE:
            return
        logger.warning(
            "[speed] map scale mismatch: catalog %.3f px/m vs measured %.3f px/m "
            "(%.0f%% off) - adopting the measured value; check the map catalog",
            self._px_per_m,
            measured,
            drift * 100.0,
        )
        self._px_per_m = measured
        self._scale_source = "ocr"
        self._ratio_hist.clear()

    def calc_target_speed(self, road_turn: float) -> float:
        """Target speed (px/s) from the curvature of the road ahead.

        With a known scale the cruise baseline is the configured speed cap in
        px/s; without one the legacy observed-speed baseline is kept.
        """
        if self._scale_source in ("map", "ocr") and self._px_per_m > 0:
            cruise = self.from_kmh(self.speed_cap_kmh)
        else:
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
        hold_window: bool = True,
    ) -> tuple[bool, bool]:
        """Decide gas (W) and brake (SPACE) from the target speed.

        Braking follows the overspeed against the (already planned) target
        speed; the final waypoint stop is handled by the driver itself. The
        decision is speed-only: steering stays independent, so the driver may
        brake and steer at the same time (freezing the wheel while braking
        used to let a slide run away until the speed target was reached).

        Steady corners use a hold window [corner_min_kmh, corner_max_kmh]:
        when the plan is at/below its top, speeds inside the window coast (no
        gas, no brake), which removes the cyclic brake/accelerate hunting
        around the single threshold.
        Returns (gas_w, brake_space).
        """
        # The window is disabled when the driver is recovering from outside the
        # outer corridor (hold_window=False): then the low rejoin target must
        # actively brake, not coast. corner_min_kmh = 0 also disables it (no
        # lower edge -> the band would never ask for gas).
        if hold_window and self.corner_min_kmh > 0.0 and self.corner_max_kmh > self.corner_min_kmh:
            hi_px = self.from_kmh(self.corner_max_kmh)
            lo_px = self.from_kmh(self.corner_min_kmh)
            if hi_px > 0.0 and tgt_spd <= hi_px:
                if mv > hi_px:
                    self._braking = True
                    return False, True
                self._braking = False
                if mv < lo_px:
                    return True, False
                return False, False

        overspd = mv > tgt_spd * 1.25

        if self._braking and mv <= tgt_spd:
            self._braking = False
        if not self._braking and overspd:
            self._braking = True

        if self._braking:
            return False, True

        gas_w = True
        if steer != 0 and not micro:
            # Cut gas in sharp corners while holding the wheel inside the corridor
            if road_turn >= turn_min and abs(xte) <= xte_lim:
                gas_w = False
        elif overspd:
            gas_w = False

        return gas_w, False
