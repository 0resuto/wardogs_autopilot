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


class TestScaleEstimation(unittest.TestCase):
    def test_ocr_anchored_median_resists_spikes(self):
        ctrl = SpeedController()
        for _ in range(30):
            ctrl.update_scale(43.5, ocr_kmh=79.0)
        expect = 43.5 / (79.0 / 3.6)
        self.assertAlmostEqual(ctrl.px_per_m_now(), expect, delta=1e-3)

        ctrl.update_scale(70.0, ocr_kmh=79.0)  # pose spike

        self.assertAlmostEqual(ctrl.px_per_m_now(), expect, delta=1e-3)

    def test_without_ocr_a_single_spike_does_not_dominate(self):
        ctrl = SpeedController()
        for _ in range(20):
            ctrl.update_scale(43.5)
        ctrl.update_scale(70.0)

        scale = ctrl.px_per_m_now()

        self.assertLess(scale, 2.2)
        self.assertGreater(scale, 1.9)


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


class TestCornerHoldWindow(unittest.TestCase):
    @staticmethod
    def _decide(ctrl: SpeedController, mv: float, tgt: float):
        return ctrl.decide_throttle_and_brake(
            mv=mv,
            tgt_spd=tgt,
            steer=0,
            micro=False,
            road_turn=0.0,
            turn_min=10.0,
            xte=0.0,
            xte_lim=8.0,
        )

    def test_window_coasts_between_the_edges(self):
        ctrl = SpeedController(corner_min_kmh=12.0, corner_max_kmh=22.0)
        ctrl._px_per_m = 2.0
        lo = ctrl.from_kmh(12.0)
        hi = ctrl.from_kmh(22.0)

        self.assertEqual(self._decide(ctrl, (lo + hi) / 2.0, lo), (False, False))
        self.assertEqual(self._decide(ctrl, hi + 1.0, lo), (False, True))
        self.assertEqual(self._decide(ctrl, lo - 1.0, lo), (True, False))

    def test_window_does_not_apply_on_faster_plans(self):
        ctrl = SpeedController(corner_min_kmh=12.0, corner_max_kmh=22.0)
        ctrl._px_per_m = 2.0
        hi = ctrl.from_kmh(22.0)

        self.assertEqual(self._decide(ctrl, hi + 1.0, hi * 2.0), (True, False))

    def test_window_can_be_disabled_for_the_rejoin_brake(self):
        ctrl = SpeedController(corner_min_kmh=12.0, corner_max_kmh=22.0)
        ctrl._px_per_m = 2.0
        lo = ctrl.from_kmh(12.0)
        hi = ctrl.from_kmh(22.0)
        mid = (lo + hi) / 2.0

        self.assertEqual(self._decide(ctrl, mid, lo), (False, False))
        gas, brake = ctrl.decide_throttle_and_brake(
            mv=mid,
            tgt_spd=lo,
            steer=0,
            micro=False,
            road_turn=0.0,
            turn_min=10.0,
            xte=0.0,
            xte_lim=8.0,
            hold_window=False,
        )
        self.assertTrue(brake)
        self.assertFalse(gas)

    def test_zero_min_kmh_does_not_stall(self):
        ctrl = SpeedController(corner_min_kmh=0.0, corner_max_kmh=22.0)
        ctrl._px_per_m = 2.0
        hi = ctrl.from_kmh(22.0)

        gas, brake = self._decide(ctrl, 2.0, hi)

        self.assertTrue(gas)
        self.assertFalse(brake)


class TestKnownScale(unittest.TestCase):
    def test_injected_scale_drives_cruise_from_the_cap(self):
        ctrl = SpeedController(speed_cap_kmh=79.0, px_per_m=2.0)

        self.assertAlmostEqual(ctrl.px_per_m_now(), 2.0, delta=1e-9)
        self.assertAlmostEqual(ctrl.calc_target_speed(0.0), 79.0 / 3.6 * 2.0, delta=1e-6)

    def test_known_scale_survives_pose_spikes(self):
        ctrl = SpeedController(px_per_m=2.0)
        ctrl.update_scale(300.0)
        ctrl.update_scale(12.0)

        self.assertAlmostEqual(ctrl.px_per_m_now(), 2.0, delta=1e-9)
        self.assertAlmostEqual(ctrl.calc_target_speed(0.0), 79.0 / 3.6 * 2.0, delta=1e-6)

    def test_ocr_agreement_keeps_catalog_scale(self):
        ctrl = SpeedController(px_per_m=2.0)
        for _ in range(35):
            ctrl.update_scale(29.4, ocr_kmh=53.0)  # ~2.0 px/m

        self.assertAlmostEqual(ctrl.px_per_m_now(), 2.0, delta=1e-9)

    def test_ocr_mismatch_warns_and_adopts_measured_scale(self):
        ctrl = SpeedController(px_per_m=2.0)
        with self.assertLogs("speed_controller", level="WARNING") as captured:
            for _ in range(30):
                ctrl.update_scale(60.0, ocr_kmh=40.0)  # 5.4 px/m

        self.assertTrue(any("scale mismatch" in line for line in captured.output))
        self.assertAlmostEqual(ctrl.px_per_m_now(), 5.4, delta=0.05)

    def test_noisy_ocr_ratios_are_not_adopted(self):
        ctrl = SpeedController(px_per_m=2.0)
        for i in range(40):
            ocr = 40.0 if i % 2 == 0 else 80.0  # ratios 5.4 / 2.7, spread too wide
            ctrl.update_scale(60.0, ocr_kmh=ocr)

        self.assertAlmostEqual(ctrl.px_per_m_now(), 2.0, delta=1e-9)

    def test_implausible_ocr_scale_is_ignored(self):
        ctrl = SpeedController(px_per_m=2.0)
        with self.assertLogs("speed_controller", level="WARNING") as captured:
            for _ in range(30):
                ctrl.update_scale(1.4, ocr_kmh=100.0)  # 0.05 px/m

        self.assertTrue(any("implausible" in line for line in captured.output))
        self.assertAlmostEqual(ctrl.px_per_m_now(), 2.0, delta=1e-9)

    def test_no_catalog_adopts_solid_ocr_scale(self):
        ctrl = SpeedController(speed_cap_kmh=79.0)
        for _ in range(30):
            ctrl.update_scale(29.4, ocr_kmh=53.0)  # ~2.0 px/m

        self.assertAlmostEqual(ctrl.px_per_m_now(), 1.997, delta=0.01)
        self.assertAlmostEqual(ctrl.calc_target_speed(0.0), 79.0 / 3.6 * 1.997, delta=0.5)

    def test_legacy_estimate_without_catalog_scale(self):
        ctrl = SpeedController(speed_cap_kmh=79.0)
        for _ in range(20):
            ctrl.update_scale(40.0)

        self.assertAlmostEqual(ctrl.px_per_m_now(), 45.0 * 3.6 / 79.0, delta=1e-6)


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

    def test_braking_is_independent_of_steering(self):
        ctrl = SpeedController()
        gas, brake = self._decide(ctrl, 40.0, 20.0, steer=1, road_turn=30.0)
        self.assertFalse(gas)
        self.assertTrue(brake)
        self.assertTrue(ctrl.is_braking)

    def test_brake_releases_at_target_while_steering(self):
        ctrl = SpeedController()
        self._decide(ctrl, 40.0, 20.0)
        self.assertTrue(ctrl.is_braking)

        gas, brake = self._decide(ctrl, 19.0, 20.0, steer=-1)
        self.assertFalse(ctrl.is_braking)
        self.assertFalse(brake)
        self.assertTrue(gas)


if __name__ == "__main__":
    unittest.main()
