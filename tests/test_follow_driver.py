"""End-to-end regression for FollowDriver reaching the final waypoint.

Simulates a moving vehicle at 40 px/s (slower than the final-brake distance
threshold, the case that used to crash the driver thread) and asserts that the
drive ends in the "finished" state with a SPACE full stop and released keys.
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import NavigatorConfig  # noqa: E402
from autopilot.navigation.follow import FollowDriver  # noqa: E402


class _FakeLocator:
    def __init__(self):
        self.latest = None


class _FakeKeyboard:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def set_state(self, keys):
        self.events.append(("keys", {k: bool(v) for k, v in keys.items()}))

    def release_all(self):
        self.events.append(("release", {}))


class TestFinalWaypointStop(unittest.TestCase):
    def test_route_finishes_with_full_stop(self):
        loc = _FakeLocator()
        kb = _FakeKeyboard()
        nav = NavigatorConfig(
            arrive_r=25.0,
            brake_d=260.0,
            stop_hold=0.2,
            stop_timeout=3.0,
            poll=0.01,
        )
        driver = FollowDriver(
            loc=loc,
            pts=[(900.0, 0.0), (1000.0, 0.0)],
            nav_cfg=nav,
            kb=kb,
        )

        errors: list[threading.ExceptHookArgs] = []
        previous_hook = threading.excepthook
        threading.excepthook = errors.append
        try:
            driver.start()
            t0 = time.time()
            speed = 40.0
            while time.time() - t0 < 8.0 and driver.is_alive() and driver.state != "finished":
                now = time.time()
                x = min(1000.0, 890.0 + speed * (now - t0))
                loc.latest = dict(
                    ts=now,
                    pose=dict(th=90.0, s=1.0, inl=20),
                    map_px=(x, 0.0),
                    good=True,
                )
                time.sleep(0.005)
            time.sleep(0.2)
        finally:
            driver.stop()
            threading.excepthook = previous_hook

        self.assertEqual(errors, [])
        self.assertEqual(driver.state, "finished")
        self.assertFalse(driver.is_alive())
        self.assertTrue(
            any(event[1].get("SPACE") for event in kb.events if event[0] == "keys"),
            "final stop must press SPACE",
        )
        self.assertEqual(kb.events[-1][0], "release")

    def test_no_finish_during_localization_hold(self):
        loc = _FakeLocator()
        kb = _FakeKeyboard()
        nav = NavigatorConfig(
            arrive_r=25.0,
            brake_d=260.0,
            stop_hold=0.2,
            stop_confirm_s=0.3,
            stop_timeout=30.0,
            poll=0.01,
        )
        driver = FollowDriver(
            loc=loc,
            pts=[(900.0, 0.0), (1000.0, 0.0)],
            nav_cfg=nav,
            kb=kb,
        )

        def publish(x: float, good: bool) -> None:
            now = time.time()
            loc.latest = dict(
                ts=now,
                pose=dict(th=90.0, s=1.0, inl=20),
                map_px=(x, 0.0),
                good=good,
            )

        errors: list[threading.ExceptHookArgs] = []
        previous_hook = threading.excepthook
        threading.excepthook = errors.append
        try:
            driver.start()
            t0 = time.time()
            while time.time() - t0 < 8.0 and driver.is_alive() and driver.state != "final_stop":
                publish(min(1000.0, 890.0 + 40.0 * (time.time() - t0)), good=True)
                time.sleep(0.005)
            self.assertEqual(driver.state, "final_stop")

            hold_until = time.time() + 0.6
            while time.time() < hold_until:
                publish(1000.0, good=False)
                time.sleep(0.005)
            time.sleep(0.1)

            self.assertTrue(driver.is_alive())
            self.assertEqual(driver.state, "final_stop")

            recover_until = time.time() + 3.0
            while time.time() < recover_until and driver.is_alive() and driver.state != "finished":
                publish(1000.0, good=True)
                time.sleep(0.005)
            time.sleep(0.2)
        finally:
            driver.stop()
            threading.excepthook = previous_hook

        self.assertEqual(errors, [])
        self.assertEqual(driver.state, "finished")

    def test_no_finish_while_ocr_reports_motion(self):
        loc = _FakeLocator()
        kb = _FakeKeyboard()
        nav = NavigatorConfig(
            arrive_r=25.0,
            brake_d=260.0,
            stop_hold=0.2,
            stop_confirm_s=0.3,
            stop_timeout=30.0,
            poll=0.01,
        )
        driver = FollowDriver(
            loc=loc,
            pts=[(900.0, 0.0), (1000.0, 0.0)],
            nav_cfg=nav,
            kb=kb,
        )

        def publish(x: float, good: bool, kmh: int, ok: bool) -> None:
            now = time.time()
            loc.latest = dict(
                ts=now,
                pose=dict(th=90.0, s=1.0, inl=20),
                map_px=(x, 0.0),
                good=good,
                speed_kmh=kmh,
                speed_ok=ok,
            )

        errors: list[threading.ExceptHookArgs] = []
        previous_hook = threading.excepthook
        threading.excepthook = errors.append
        try:
            driver.start()
            t0 = time.time()
            while time.time() - t0 < 8.0 and driver.is_alive() and driver.state != "final_stop":
                publish(min(1000.0, 890.0 + 40.0 * (time.time() - t0)), True, 40, True)
                time.sleep(0.005)
            self.assertEqual(driver.state, "final_stop")

            hold_until = time.time() + 0.6
            while time.time() < hold_until:
                publish(1000.0, False, 30, True)
                time.sleep(0.005)
            time.sleep(0.1)
            self.assertTrue(driver.is_alive())
            self.assertEqual(driver.state, "final_stop")

            stop_until = time.time() + 3.0
            while time.time() < stop_until and driver.is_alive() and driver.state != "finished":
                publish(1000.0, False, 0, True)
                time.sleep(0.005)
            time.sleep(0.2)
        finally:
            driver.stop()
            threading.excepthook = previous_hook

        self.assertEqual(errors, [])
        self.assertEqual(driver.state, "finished")


class TestStaleMeasuredPose(unittest.TestCase):
    def test_frozen_hold_pose_releases_the_wheel(self):
        """The tracker republishes a frozen position with a fresh frame ts and
        good=False during a capture void; the wheel must not stay locked."""
        loc = _FakeLocator()
        kb = _FakeKeyboard()
        nav = NavigatorConfig(arrive_r=25.0, poll=0.01, settle_s=0.1)
        driver = FollowDriver(loc=loc, pts=[(0.0, 0.0), (100000.0, 0.0)], nav_cfg=nav, kb=kb)

        def publish(x: float, th: float, good: bool) -> None:
            now = time.time()
            loc.latest = dict(
                ts=now,
                pose=dict(th=th, s=1.0, inl=20),
                map_px=(x, 0.0),
                good=good,
            )

        try:
            driver.start()
            t0 = time.time()
            while time.time() - t0 < 2.0:
                publish(100.0 + 40.0 * (time.time() - t0), 90.0, True)
                time.sleep(0.005)
            # drive against the route -> the motion course gives a big error
            hold_start = time.time()
            while time.time() - hold_start < 1.2:
                publish(200.0 - 40.0 * (time.time() - hold_start), 210.0, True)
                time.sleep(0.005)
            steered = any(e[0] == "keys" and ("A" in e[1] or "D" in e[1]) for e in kb.events)
            self.assertTrue(steered)

            # freeze: fresh frame timestamps, frozen position, good=False
            kb.events.clear()
            freeze_start = time.time()
            while time.time() - freeze_start < 0.9:  # let the measured pose go stale
                publish(200.0, 210.0, False)
                time.sleep(0.005)

            kb.events.clear()
            late_start = time.time()
            while time.time() - late_start < 0.5:
                publish(200.0, 210.0, False)
                time.sleep(0.005)
        finally:
            driver.stop()

        late_steer = [e for e in kb.events if e[0] == "keys" and ("A" in e[1] or "D" in e[1])]
        self.assertEqual(late_steer, [])


class TestStopDecision(unittest.TestCase):
    @staticmethod
    def _driver(**nav_kwargs) -> FollowDriver:
        nav = NavigatorConfig(**nav_kwargs)
        return FollowDriver(
            loc=None,
            pts=[(0.0, 0.0), (100.0, 0.0)],
            nav_cfg=nav,
            kb=None,
        )

    def test_stop_threshold_uses_configured_floor(self):
        driver = self._driver(stop_min_px_s=3.5, stop_speed_kmh=2.0)
        self.assertAlmostEqual(driver._stop_thr(), 3.5, delta=1e-6)

        faster = self._driver(stop_min_px_s=0.5, stop_speed_kmh=79.0)
        self.assertGreater(faster._stop_thr(), 0.5)

    def test_stop_requires_fresh_measured_pose(self):
        driver = self._driver(stop_hold=0.2, stop_confirm_s=0.3)
        now = 100.0
        driver._final_t0 = now
        driver.speed = 0.0
        driver.path._mv = 0.0

        driver._last_measured_t = None
        self.assertEqual(driver._stop_state(now), "braking")

        driver._last_measured_t = now - 1.0
        self.assertEqual(driver._stop_state(now), "braking")
        self.assertIsNone(driver._stop_s_t)

        driver._last_measured_t = now
        self.assertEqual(driver._stop_state(now), "braking")
        driver._last_measured_t = now + 0.25
        self.assertEqual(driver._stop_state(now + 0.25), "stopped")

    def test_ocr_speed_overrides_position_estimate(self):
        driver = self._driver(stop_hold=0.2, stop_confirm_s=0.3)
        now = 100.0
        driver._final_t0 = now
        driver.speed = 50.0
        driver.path._mv = 50.0
        driver._last_measured_t = now
        driver._last_speed_t = now

        driver._speed_kmh = 30.0
        self.assertEqual(driver._stop_state(now), "braking")

        driver._speed_kmh = 1.0
        self.assertEqual(driver._stop_state(now), "braking")
        driver._last_speed_t = now + 0.25
        self.assertEqual(driver._stop_state(now + 0.25), "stopped")

    def test_final_brake_distance_uses_brake_budget(self):
        driver = self._driver()
        assert driver.planner is not None
        driver.path._mv = 40.0

        expected = driver.speed_ctrl.brake_distance_px(40.0, driver.planner.brake_decel) * 1.2 + 8.0
        self.assertAlmostEqual(driver._final_brake_dist(), expected, delta=1e-6)

    def test_final_brake_distance_falls_back_without_planner(self):
        driver = self._driver(speed_profile=False)
        driver.path._mv = 40.0

        expected = (40.0**2 / (2.0 * driver.brake_d) + 12.0) * 1.5
        self.assertAlmostEqual(driver._final_brake_dist(), expected, delta=1e-6)

    def test_stop_timeout_is_the_emergency_exit(self):
        driver = self._driver(stop_hold=0.2, stop_confirm_s=0.3, stop_timeout=1.0)
        driver._final_t0 = 0.0
        driver.speed = 100.0
        driver.path._mv = 100.0
        driver._last_measured_t = None

        self.assertEqual(driver._stop_state(0.5), "braking")
        self.assertEqual(driver._stop_state(1.5), "timeout")


class TestSpeedPlanning(unittest.TestCase):
    @staticmethod
    def _driver(**nav_kwargs) -> FollowDriver:
        nav = NavigatorConfig(**nav_kwargs)
        return FollowDriver(
            loc=None,
            pts=[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0)],
            nav_cfg=nav,
            kb=None,
        )

    def test_planner_limits_speed_on_sharp_corner(self):
        driver = self._driver()
        assert driver.planner is not None
        driver.path.idx = 1

        plan = driver._route_target_kmh((190.0, 0.0))

        assert plan is not None
        self.assertGreater(plan, driver.nav_cfg.corner_min_kmh)
        self.assertLess(plan, 60.0)

    def test_planner_can_be_disabled(self):
        driver = self._driver(speed_profile=False)

        self.assertIsNone(driver.planner)
        self.assertIsNone(driver._route_target_kmh((0.0, 0.0)))

    def test_planner_caps_cruise_instead_of_vertex_braking(self):
        driver = self._driver()
        driver.speed_ctrl._vmax_px = 90.0

        tgt = driver._speed_target(56.0, 55.0)

        self.assertAlmostEqual(tgt, driver.speed_ctrl.from_kmh(55.0), delta=1e-6)
        self.assertGreater(tgt, driver.speed_ctrl.v_min)

    def test_legacy_vertex_heuristic_without_planner(self):
        driver = self._driver(speed_profile=False)
        driver.speed_ctrl._vmax_px = 90.0

        tgt = driver._speed_target(56.0, None)

        self.assertEqual(tgt, driver.speed_ctrl.calc_target_speed(56.0))
        self.assertEqual(tgt, driver.speed_ctrl.v_min)


class TestKnownMapScale(unittest.TestCase):
    def test_driver_uses_injected_scale(self):
        driver = FollowDriver(
            loc=None,
            pts=[(0.0, 0.0), (100.0, 0.0)],
            nav_cfg=NavigatorConfig(),
            kb=None,
            px_per_m=2.0,
        )

        self.assertAlmostEqual(driver._px_per_m_now(), 2.0, delta=1e-9)
        self.assertAlmostEqual(driver._speed_target(0.0, None), 79.0 / 3.6 * 2.0, delta=1e-6)


class TestVehicleAuthority(unittest.TestCase):
    @staticmethod
    def _driver(**nav_kwargs) -> FollowDriver:
        nav = NavigatorConfig(**nav_kwargs)
        return FollowDriver(loc=None, pts=[(0.0, 0.0), (1.0, 0.0)], nav_cfg=nav, kb=None)

    def test_profile_is_loaded_and_authority_is_grip_limited(self):
        driver = self._driver()
        self.assertIsNotNone(driver.vehicle_model)
        assert driver.vehicle_model is not None
        self.assertEqual(driver.vehicle_model.vehicle_id, "WHL_07")

        slow = driver._yaw_rate_max(5.0)
        fast = driver._yaw_rate_max(80.0)
        assert slow is not None and fast is not None
        self.assertGreater(slow, fast)

    def test_authority_scales_with_lateral_budget(self):
        low_budget = self._driver(corner_lat_g=0.2)
        high_budget = self._driver(corner_lat_g=0.6)

        slow = low_budget._yaw_rate_max(50.0)
        fast = high_budget._yaw_rate_max(50.0)
        assert slow is not None and fast is not None
        self.assertAlmostEqual(fast / slow, 3.0, delta=1e-6)

    def test_profile_can_be_disabled(self):
        driver = self._driver(vehicle_profile="")
        self.assertIsNone(driver.vehicle_model)
        self.assertIsNone(driver._yaw_rate_max(50.0))

    def test_yaw_gain_override_scales_authority(self):
        base = self._driver()
        tuned = self._driver(yaw_gain=2.0)
        assert base.vehicle_model is not None and tuned.vehicle_model is not None
        self.assertAlmostEqual(tuned.vehicle_model.yaw_gain, 2.0, delta=1e-9)

        base_yaw = base._yaw_rate_max(5.0)
        tuned_yaw = tuned._yaw_rate_max(5.0)
        assert base_yaw is not None and tuned_yaw is not None
        self.assertAlmostEqual(tuned_yaw, base_yaw * 2.0, delta=1e-6)

    def test_apply_vehicle_tuning_updates_planner(self):
        from autopilot.navigation.speed_profile import G

        driver = self._driver()
        assert driver.planner is not None

        driver.apply_vehicle_tuning(
            NavigatorConfig(corner_lat_g=0.9, brake_g=0.8, corner_min_kmh=20.0)
        )

        self.assertAlmostEqual(driver.planner.lat_accel, 0.9 * G, delta=1e-6)
        self.assertAlmostEqual(driver.planner.brake_decel, 0.8 * G, delta=1e-6)
        self.assertAlmostEqual(driver.planner.min_speed_kmh, 20.0, delta=1e-6)

    def test_corner_window_is_wired_from_config(self):
        driver = self._driver()

        self.assertAlmostEqual(driver.speed_ctrl.corner_min_kmh, 12.0, delta=1e-9)
        self.assertAlmostEqual(driver.speed_ctrl.corner_max_kmh, 22.0, delta=1e-9)

        driver.apply_vehicle_tuning(NavigatorConfig(corner_min_kmh=15.0, corner_max_kmh=28.0))

        self.assertAlmostEqual(driver.speed_ctrl.corner_min_kmh, 15.0, delta=1e-9)
        self.assertAlmostEqual(driver.speed_ctrl.corner_max_kmh, 28.0, delta=1e-9)

    def test_steer_settle_is_wired_and_live_tunable(self):
        driver = self._driver()

        self.assertAlmostEqual(driver.steer_ctrl.settle_t, 0.6, delta=1e-9)

        driver.apply_vehicle_tuning(NavigatorConfig(settle_s=0.35))

        self.assertAlmostEqual(driver.steer_ctrl.settle_t, 0.35, delta=1e-9)

    def test_corridor_tuning_updates_the_path_live(self):
        driver = self._driver()

        driver.apply_vehicle_tuning(NavigatorConfig(xte_m=6.0, xte_outer_m=18.0, steer_look_s=1.1))

        self.assertAlmostEqual(driver.xte_m, 6.0, delta=1e-9)
        self.assertAlmostEqual(driver.xte_outer_m, 18.0, delta=1e-9)
        self.assertAlmostEqual(driver.steer_look_s, 1.1, delta=1e-9)
        self.assertAlmostEqual(driver.path.xte_m, 6.0, delta=1e-9)
        self.assertAlmostEqual(driver.path.xte_outer_m, 18.0, delta=1e-9)
        self.assertAlmostEqual(driver.path.steer_look_s, 1.1, delta=1e-9)

    def test_smooth_heading_rejects_unphysical_steps(self):
        driver = self._driver()
        driver._heading = 0.0
        driver._last_heading_t = 100.0

        heading = driver._smooth_heading(60.0, 100.05, 45.0)

        self.assertLess(abs(heading), 5.0)

    def test_smooth_heading_tracks_a_plausible_turn(self):
        driver = self._driver()
        driver._heading = 0.0
        driver._last_heading_t = 100.0
        heading = 0.0
        for i in range(10):
            heading = driver._smooth_heading(heading + 2.0, 100.05 + i * 0.05, 45.0)

        self.assertGreater(heading, 6.0)

    def test_speed_estimate_prefers_fresh_ocr(self):
        driver = self._driver()
        driver._speed_kmh = 42.0
        driver._last_speed_t = 100.0

        self.assertAlmostEqual(driver._speed_kmh_estimate(100.5), 42.0, delta=1e-6)


class TestRouteEntrySnap(unittest.TestCase):
    def test_engage_mid_route_targets_the_nearest_point(self):
        loc = _FakeLocator()
        kb = _FakeKeyboard()
        nav = NavigatorConfig(arrive_r=5.0, poll=0.01)
        pts = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)]
        driver = FollowDriver(loc=loc, pts=pts, nav_cfg=nav, kb=kb)

        try:
            driver.start()
            t0 = time.time()
            while time.time() - t0 < 2.0 and driver.path.idx != 1:
                now = time.time()
                loc.latest = dict(
                    ts=now,
                    pose=dict(th=90.0, s=1.0, inl=20),
                    map_px=(90.0, 0.0),
                    good=True,
                )
                time.sleep(0.005)
            idx = driver.path.idx
        finally:
            driver.stop()

        self.assertEqual(idx, 1)


class TestKeyboardCleanup(unittest.TestCase):
    def test_stop_closes_keyboard_driver(self):
        loc = _FakeLocator()
        kb = MagicMock()
        driver = FollowDriver(
            loc=loc,
            pts=[(0.0, 0.0), (1000.0, 0.0)],
            nav_cfg=NavigatorConfig(poll=0.01),
            kb=kb,
        )

        driver.start()
        time.sleep(0.05)
        driver.stop()

        kb.release_all.assert_called()
        kb.close.assert_called_once()
        self.assertFalse(driver.is_alive())

    def test_self_finish_closes_keyboard_driver(self):
        loc = _FakeLocator()
        kb = MagicMock()
        nav = NavigatorConfig(arrive_r=25.0, stop_hold=0.2, poll=0.01)
        driver = FollowDriver(loc=loc, pts=[(0.0, 0.0), (10.0, 0.0)], nav_cfg=nav, kb=kb)

        try:
            driver.start()
            t0 = time.time()
            while time.time() - t0 < 5.0 and driver.is_alive() and driver.state != "finished":
                now = time.time()
                loc.latest = dict(
                    ts=now,
                    pose=dict(th=90.0, s=1.0, inl=20),
                    map_px=(10.0, 0.0),
                    good=True,
                )
                time.sleep(0.005)
            time.sleep(0.3)

            self.assertEqual(driver.state, "finished")
            self.assertFalse(driver.is_alive())
            kb.close.assert_called_once()
        finally:
            driver.stop()


class TestMotionBeforeFinish(unittest.TestCase):
    def test_driver_does_not_finish_while_moving(self):
        loc = _FakeLocator()
        kb = _FakeKeyboard()
        nav = NavigatorConfig(arrive_r=25.0, brake_d=260.0, stop_hold=0.2, poll=0.01)
        driver = FollowDriver(
            loc=loc,
            pts=[(900.0, 0.0), (100000.0, 0.0)],
            nav_cfg=nav,
            kb=kb,
        )

        errors: list[threading.ExceptHookArgs] = []
        previous_hook = threading.excepthook
        threading.excepthook = errors.append
        try:
            driver.start()
            t0 = time.time()
            while time.time() - t0 < 1.0 and driver.is_alive():
                now = time.time()
                x = 900.0 + 40.0 * (now - t0)
                loc.latest = dict(
                    ts=now,
                    pose=dict(th=90.0, s=1.0, inl=20),
                    map_px=(x, 0.0),
                    good=True,
                )
                time.sleep(0.005)
            state_while_moving = driver.state
            alive_while_moving = driver.is_alive()
        finally:
            driver.stop()
            threading.excepthook = previous_hook

        self.assertEqual(errors, [])
        self.assertTrue(alive_while_moving)
        self.assertNotEqual(state_while_moving, "finished")


if __name__ == "__main__":
    unittest.main()
