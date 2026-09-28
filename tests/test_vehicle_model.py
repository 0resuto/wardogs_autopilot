"""Tests for the vendored Ural vehicle model built from the physics pack."""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.vehicle_model import VehicleModel  # noqa: E402


class TestUralProfile(unittest.TestCase):
    model: VehicleModel

    @classmethod
    def setUpClass(cls):
        cls.model = VehicleModel.load("ural")

    def test_profile_identity_and_radius(self):
        self.assertEqual(self.model.vehicle_id, "WHL_07")
        self.assertAlmostEqual(self.model.max_speed_kmh, 79.0, delta=0.1)
        self.assertAlmostEqual(self.model.wheel_radius_m, 0.65, delta=0.02)

    def test_top_speed_identity(self):
        self.assertAlmostEqual(self.model.top_speed_kmh(), 79.0, delta=0.5)

        first_gear_top = self.model.speed_kmh(self.model.max_rpm, max(self.model.gears_forward))
        self.assertLess(first_gear_top, 20.0)

    def test_torque_interpolation_and_clamping(self):
        self.assertAlmostEqual(self.model.torque_nm(600), 200.0, delta=1e-6)
        self.assertAlmostEqual(self.model.torque_nm(1200), 900.0, delta=1e-6)
        self.assertAlmostEqual(self.model.torque_nm(400), 200.0, delta=1e-6)
        self.assertAlmostEqual(self.model.torque_nm(9000), 650.0, delta=1e-6)

        between = self.model.torque_nm(1100)
        self.assertTrue(860.0 < between <= 900.0)

    def test_drive_force_decreases_with_speed_in_top_gear(self):
        low = self.model.drive_force_n(30.0, self.model.top_gear)
        high = self.model.drive_force_n(70.0, self.model.top_gear)
        self.assertGreater(low, high)
        self.assertGreater(high, 0.0)

    def test_low_gear_multiplies_wheel_force(self):
        first_gear = max(self.model.gears_forward)
        self.assertGreater(
            self.model.drive_force_n(10.0, first_gear),
            self.model.drive_force_n(10.0, self.model.top_gear),
        )

    def test_steer_limit_curve(self):
        self.assertAlmostEqual(self.model.steer_limit(2.1), 0.975, delta=1e-6)
        self.assertAlmostEqual(self.model.steer_limit(80.0), 0.5125, delta=1e-6)
        self.assertAlmostEqual(self.model.steer_limit(0.0), 0.975, delta=1e-6)

    def test_steer_speed_curve(self):
        self.assertAlmostEqual(self.model.steer_speed_mult(0.0), 1.425, delta=1e-6)
        self.assertAlmostEqual(self.model.steer_speed_mult(80.0), 0.5, delta=1e-6)

    def test_yaw_rate_grows_with_speed(self):
        slow = self.model.yaw_rate_max_deg_s(5.0)
        fast = self.model.yaw_rate_max_deg_s(80.0)

        self.assertGreater(slow, 0.0)
        self.assertGreater(fast, slow)
        self.assertAlmostEqual(self.model.yaw_rate_max_deg_s(0.0), 0.0, delta=1e-9)

    def test_yaw_rate_is_grip_limited(self):
        speed_kmh = 50.0
        v = speed_kmh / 3.6
        lat = 0.35 * 9.80665

        capped = self.model.yaw_rate_max_deg_s(speed_kmh, lat_accel_mps2=lat)
        uncapped = self.model.yaw_rate_max_deg_s(speed_kmh)

        self.assertAlmostEqual(capped, math.degrees(lat / v), delta=1e-6)
        self.assertLess(capped, uncapped)

    def test_low_speed_authority_stays_geometric(self):
        lat = 0.35 * 9.80665
        self.assertAlmostEqual(
            self.model.yaw_rate_max_deg_s(5.0, lat_accel_mps2=lat),
            self.model.yaw_rate_max_deg_s(5.0),
            delta=1e-6,
        )

    def test_corner_and_brake_helpers(self):
        self.assertAlmostEqual(
            self.model.corner_speed_kmh(50.0, 3.0),
            math.sqrt(150.0) * 3.6,
            delta=0.1,
        )
        self.assertAlmostEqual(
            self.model.brake_distance_m(50.0, 5.0),
            (50.0 / 3.6) ** 2 / 10.0,
            delta=0.05,
        )
        self.assertEqual(self.model.brake_distance_m(50.0, 0.0), float("inf"))


if __name__ == "__main__":
    unittest.main()
