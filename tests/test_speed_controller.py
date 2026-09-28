"""Tests for the speed controller after removing legacy braking heuristics."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.speed_controller import SpeedController  # noqa: E402


class TestBrakeDistance(unittest.TestCase):
    def test_physical_stopping_distance(self):
        ctrl = SpeedController()
        ctrl._px_per_m = 2.0

        # 20 px/s = 10 m/s; at 5 m/s^2 the stop takes 10 m = 20 px
        self.assertAlmostEqual(ctrl.brake_distance_px(20.0, 5.0), 20.0, delta=1e-6)

    def test_zero_decel_is_infinite(self):
        ctrl = SpeedController()
        ctrl._px_per_m = 2.0
        self.assertEqual(ctrl.brake_distance_px(20.0, 0.0), float("inf"))

    def test_unknown_scale_is_infinite(self):
        ctrl = SpeedController(speed_cap_kmh=0.0)
        ctrl._px_per_m = 0.0
        ctrl._vmax_px = 0.0
        self.assertEqual(ctrl.brake_distance_px(20.0, 5.0), float("inf"))


class TestTargetSpeed(unittest.TestCase):
    def test_straight_road_targets_cruise(self):
        ctrl = SpeedController()
        ctrl._vmax_px = 90.0
        self.assertAlmostEqual(ctrl.calc_target_speed(0.0), 90.0, delta=1e-6)

    def test_sharp_corner_reduces_target(self):
        ctrl = SpeedController(v_min=10.0)
        ctrl._vmax_px = 90.0
        target = ctrl.calc_target_speed(120.0)
        self.assertLess(target, 20.0)
        self.assertGreaterEqual(target, 10.0)


class TestThrottleBrake(unittest.TestCase):
    @staticmethod
    def _decide(
        ctrl: SpeedController, mv: float, tgt: float, steer: int = 0, road_turn: float = 0.0
    ):
        return ctrl.decide_throttle_and_brake(
            mv=mv,
            tgt_spd=tgt,
            steer=steer,
            micro=False,
            road_turn=road_turn,
            turn_min=10.0,
            xte=0.0,
            xte_lim=8.0,
        )

    def test_gas_on_below_target(self):
        gas, brake = self._decide(SpeedController(), 10.0, 20.0)
        self.assertTrue(gas)
        self.assertFalse(brake)

    def test_brakes_when_overspeeding(self):
        gas, brake = self._decide(SpeedController(), 40.0, 20.0)
        self.assertFalse(gas)
        self.assertTrue(brake)

    def test_corner_cuts_gas_inside_corridor(self):
        gas, brake = self._decide(SpeedController(), 10.0, 20.0, steer=1, road_turn=30.0)
        self.assertFalse(gas)
        self.assertFalse(brake)


if __name__ == "__main__":
    unittest.main()
