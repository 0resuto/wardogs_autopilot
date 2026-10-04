"""Data-driven vehicle dynamics model (reverse-engineered physics pack).

Uses only the high-confidence fields of data/vehicles/<name>.json: gearbox,
engine torque, wheel radius and the steering authority curves. The pack cannot
provide absolute scales for mass / wheelbase / assist effects, so those live in
a small calibration block that a single in-game run can pin:

    wheel rev/s = rpm / (gear * final_drive) / 60
    speed       = wheel rev/s * 2*pi*r
    drive force = torque(rpm) * gear * final_drive * efficiency / r
    yaw rate    = yaw_gain * v * tan(steer_angle_max * steer_limit(v)) / wheelbase

The top-speed identity is validated: v_max = max_rpm / (top_gear * final_drive)
/ 60 * 2*pi*r reproduces the in-game stats for every vehicle in the pack.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from .. import PROJECT_ROOT
from ..common.log import get_logger

logger = get_logger("vehicle_model")

VEHICLES_DIR = os.path.join(PROJECT_ROOT, "data", "vehicles")


class VehicleModel:
    """Per-vehicle dynamics: gearing, torque, steering authority."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.vehicle_id = str(data.get("vehicle_id", "?"))
        self.name = str(data.get("name", self.vehicle_id))
        self.max_speed_kmh = float(data.get("max_speed_kmh", 0.0))
        self.wheel_radius_m = float(data["wheel_radius_m"])
        self.gears_forward = [float(g) for g in data["gears_forward"]]
        self.gear_reverse = float(data.get("gear_reverse", 0.0))
        self.final_drive = float(data["final_drive"])
        self.idle_rpm = float(data.get("idle_rpm", 0.0))
        self.max_rpm = float(data["max_rpm"])
        self.torque_curve = [(float(r), float(t)) for r, t in data["torque_nm"]]
        self.steer_limit_curve = [(float(v), float(p)) for v, p in data["steer_limit_pct"]]
        self.steer_speed_curve = [(float(v), float(m)) for v, m in data["steer_speed_mult"]]
        self.tire_peak_force = float(data.get("tire_peak_force", 0.0))

        self.tire_slip_curve = [(float(s), float(f)) for s, f in data.get("tire_slip_curve", [])]
        self.brake_engagement = [(float(t), float(v)) for t, v in data.get("brake_engagement", [])]
        self.throttle_engagement = [
            (float(t), float(v)) for t, v in data.get("throttle_engagement", [])
        ]

        calibration = data.get("calibration", {})
        self.yaw_gain = float(calibration.get("yaw_gain", 1.0))
        self.wheelbase_m = float(calibration.get("wheelbase_m", 3.8))
        self.steer_angle_max_deg = float(calibration.get("steer_angle_max_deg", 35.0))
        self.drive_efficiency = float(calibration.get("drive_efficiency", 0.9))
        self.drive_accel_scale = float(calibration.get("drive_accel_scale", 7.5e-05))
        self.resist_a_mps2 = float(calibration.get("resist_a_mps2", 0.15))
        self.resist_b = float(calibration.get("resist_b", 0.0003))
        self.brake_decel_mps2 = float(calibration.get("brake_decel_mps2", 5.0))
        self.brake_yaw_gain = float(calibration.get("brake_yaw_gain", 1.8))
        self.drive_accel_curve = [
            (float(v), float(a)) for v, a in calibration.get("drive_accel_curve", [])
        ]
        self.coast_decel_curve = [
            (float(v), float(a)) for v, a in calibration.get("coast_decel_curve", [])
        ]

    @classmethod
    def load(cls, name: str = "ural", path: str | None = None) -> VehicleModel:
        """Load a vendored profile by name (data/vehicles/<name>.json)."""
        profile_path = path or os.path.join(VEHICLES_DIR, f"{name}.json")
        with open(profile_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data)

    @staticmethod
    def _interp(curve: list[tuple[float, float]], x: float) -> float:
        """Piecewise-linear interpolation with end clamping."""
        if not curve:
            return 0.0
        if x <= curve[0][0]:
            return curve[0][1]
        if x >= curve[-1][0]:
            return curve[-1][1]
        for (x0, y0), (x1, y1) in zip(curve, curve[1:], strict=False):
            if x0 <= x <= x1:
                if x1 == x0:
                    return y1
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return curve[-1][1]

    @property
    def top_gear(self) -> float:
        return min(self.gears_forward)

    def speed_kmh(self, rpm: float, gear: float) -> float:
        """Vehicle speed at an engine rpm in the given gear."""
        rev_per_s = rpm / (gear * self.final_drive) / 60.0
        return rev_per_s * 2.0 * math.pi * self.wheel_radius_m * 3.6

    def rpm_for_speed(self, speed_kmh: float, gear: float) -> float:
        """Engine rpm needed for a speed in the given gear."""
        rev_per_s = max(0.0, speed_kmh) / 3.6 / (2.0 * math.pi * self.wheel_radius_m)
        return rev_per_s * gear * self.final_drive * 60.0

    def top_speed_kmh(self) -> float:
        """Maximum speed in the top gear at max rpm."""
        return self.speed_kmh(self.max_rpm, self.top_gear)

    def torque_nm(self, rpm: float) -> float:
        """Engine torque at an rpm (clamped to the curve ends)."""
        return self._interp(self.torque_curve, rpm)

    def drive_force_n(self, speed_kmh: float, gear: float) -> float:
        """Wheel drive force at a speed in the given gear (N, model units)."""
        if speed_kmh <= 0.0:
            rpm = self.torque_curve[0][0] if self.torque_curve else self.idle_rpm
        else:
            rpm = self.rpm_for_speed(speed_kmh, gear)
        rpm = min(max(rpm, self.idle_rpm or rpm), self.max_rpm)
        return (
            self.torque_nm(rpm)
            * gear
            * self.final_drive
            * self.drive_efficiency
            / self.wheel_radius_m
        )

    def drive_force_n_best(self, speed_kmh: float) -> float:
        """Best wheel drive force across the forward gears (N, model units).

        The game's automatic gearbox keeps the engine in its band; a
        short-horizon predictor only needs the achievable envelope, not the
        exact gear. Gears whose rpm exceeds the engine limit are skipped.
        """
        best = 0.0
        for gear in self.gears_forward:
            if speed_kmh > 1.0:
                rpm = self.rpm_for_speed(speed_kmh, gear)
                if rpm > self.max_rpm * 1.05:
                    continue
            else:
                rpm = self.idle_rpm
            rpm = min(max(rpm, self.idle_rpm), self.max_rpm)
            force = (
                self.torque_nm(rpm)
                * gear
                * self.final_drive
                * self.drive_efficiency
                / self.wheel_radius_m
            )
            best = max(best, force)
        return best

    def drive_accel_kmh(self, speed_kmh: float) -> float | None:
        """Measured net acceleration under throttle at a speed (None if no table)."""
        if not self.drive_accel_curve:
            return None
        return self._interp(self.drive_accel_curve, speed_kmh)

    def coast_decel_kmh(self, speed_kmh: float) -> float | None:
        """Measured coast deceleration magnitude (None if no table)."""
        if not self.coast_decel_curve:
            return None
        return self._interp(self.coast_decel_curve, speed_kmh)

    def steer_limit(self, speed_kmh: float) -> float:
        """Share of the maximum steer angle available at a speed (0..1)."""
        return max(0.0, min(1.0, self._interp(self.steer_limit_curve, speed_kmh) / 100.0))

    def steer_speed_mult(self, speed_kmh: float) -> float:
        """Steering rate multiplier at a speed (>0)."""
        return max(1e-3, self._interp(self.steer_speed_curve, speed_kmh))

    def yaw_rate_max_deg_s(
        self,
        speed_kmh: float,
        lat_accel_mps2: float | None = None,
        include_gain: bool = True,
    ) -> float:
        """Maximum yaw rate (deg/s) from the model at the given speed.

        The steering geometry (bicycle model) sets the kinematic ceiling, but
        the tires cannot hold more than the lateral grip budget: above it the
        real yaw rate is `a_lat / v`. Pass the budget to get the capped value.
        `include_gain=False` returns the raw kinematic/grip ceiling without the
        calibrated yaw gain (the predictor applies the gain itself, once).
        """
        v = max(0.0, speed_kmh) / 3.6
        angle_rad = math.radians(self.steer_angle_max_deg) * self.steer_limit(speed_kmh)
        yaw = math.degrees(v * math.tan(angle_rad) / self.wheelbase_m)
        if lat_accel_mps2 is not None and lat_accel_mps2 > 0.0 and v > 1e-3:
            grip_limit = math.degrees(lat_accel_mps2 / v)
            yaw = min(yaw, grip_limit)
        # the calibrated gain scales the final authority (arcade handling and
        # assist effects shrink both the kinematic and the grip ceiling)
        return yaw * (self.yaw_gain if include_gain else 1.0)

    def corner_speed_kmh(self, radius_m: float, lat_accel_mps2: float) -> float:
        """Speed limit for a corner of the given radius (m/s^2 lateral budget)."""
        if radius_m <= 0.0 or lat_accel_mps2 <= 0.0:
            return 0.0
        return math.sqrt(lat_accel_mps2 * radius_m) * 3.6

    def brake_distance_m(self, speed_kmh: float, decel_mps2: float) -> float:
        """Stopping distance for a speed at a constant deceleration."""
        if decel_mps2 <= 0.0:
            return float("inf")
        v = max(0.0, speed_kmh) / 3.6
        return v * v / (2.0 * decel_mps2)
