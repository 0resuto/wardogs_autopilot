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
            self.assertNotEqual(driver.state, "finished")

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
            self.assertNotEqual(driver.state, "finished")

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
            # drive against the route -> the motion course gives a big error.
            # The pose heading stays constant: a synthetic 90 -> 210 flip is
            # (correctly) rejected by the pose heading-rate gate.
            hold_start = time.time()
            while time.time() - hold_start < 1.2:
                publish(200.0 - 40.0 * (time.time() - hold_start), 90.0, True)
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
        # explicit 1.0 baseline: the shipped profile carries a fitted gain;
        # the override scales the tracking-authority ceiling (include_gain),
        # while the driver's physical clamp stays gain-free by design
        base = self._driver(yaw_gain=1.0)
        tuned = self._driver(yaw_gain=2.0)
        assert base.vehicle_model is not None and tuned.vehicle_model is not None
        self.assertAlmostEqual(base.vehicle_model.yaw_gain, 1.0, delta=1e-9)
        self.assertAlmostEqual(tuned.vehicle_model.yaw_gain, 2.0, delta=1e-9)

        base_yaw = base.vehicle_model.yaw_rate_max_deg_s(5.0, include_gain=True)
        tuned_yaw = tuned.vehicle_model.yaw_rate_max_deg_s(5.0, include_gain=True)
        self.assertAlmostEqual(tuned_yaw, base_yaw * 2.0, delta=1e-6)
        base_cap = base._yaw_rate_max(5.0)
        tuned_cap = tuned._yaw_rate_max(5.0)
        assert base_cap is not None and tuned_cap is not None
        self.assertAlmostEqual(tuned_cap, base_cap, delta=1e-9)

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


class TestHeadingSource(unittest.TestCase):
    """The control heading must use the map pose when the motion course is stale.

    `mh` freezes below the 10 px / 0.5 s update threshold; continuing to steer
    by it ignored the accurate map heading and produced the low-speed
    oscillation seen in the 2026-10-01 test runs.
    """

    @staticmethod
    def _driver() -> FollowDriver:
        return FollowDriver(
            loc=None,
            pts=[(0.0, 0.0), (1000.0, 0.0)],
            nav_cfg=NavigatorConfig(),
            kb=None,
        )

    def test_fresh_motion_course_wins_at_speed(self):
        driver = self._driver()
        driver.path._mh = 77.0
        driver.path._mh_t = 100.0
        driver.path._mv = 30.0

        heading, source, age, mh_on = driver._heading_source(dict(th=70.0, inl=5), 100.2)

        self.assertEqual(source, "mh")
        self.assertTrue(mh_on)
        self.assertAlmostEqual(heading, 77.0, delta=1e-6)
        assert age is not None
        self.assertAlmostEqual(age, 0.2, delta=1e-6)

    def test_stale_motion_course_falls_back_to_the_map_heading(self):
        driver = self._driver()
        driver.path._mh = 77.0
        driver.path._mh_t = 100.0
        driver.path._mv = 30.0

        heading, source, age, mh_on = driver._heading_source(dict(th=4.4, inl=5), 101.5)

        self.assertEqual(source, "pose")
        self.assertFalse(mh_on)
        self.assertAlmostEqual(heading, 4.4, delta=1e-6)
        assert age is not None
        self.assertGreater(age, 1.0)

    def test_slow_motion_course_is_not_used(self):
        driver = self._driver()
        driver.path._mh = 77.0
        driver.path._mh_t = 100.0
        driver.path._mv = 2.0

        heading, source, _age, mh_on = driver._heading_source(dict(th=350.0, inl=5), 100.05)

        self.assertEqual(source, "pose")
        self.assertFalse(mh_on)
        self.assertAlmostEqual(heading, 350.0, delta=1e-6)

    def test_no_pose_heading_keeps_the_last_course(self):
        driver = self._driver()
        driver.path._mh = 77.0
        driver.path._mh_t = 100.0
        driver.path._mv = 2.0

        heading, source, _age, mh_on = driver._heading_source(dict(inl=0), 100.05)

        self.assertEqual(source, "none")
        self.assertFalse(mh_on)
        self.assertAlmostEqual(heading, 77.0, delta=1e-6)


class TestPoseQualityGates(unittest.TestCase):
    """A wrong re-acquisition after a pose void must not reach the controls."""

    @staticmethod
    def _driver() -> FollowDriver:
        return FollowDriver(
            loc=None,
            pts=[(0.0, 0.0), (1000.0, 0.0)],
            nav_cfg=NavigatorConfig(),
            kb=None,
        )

    def test_impossible_reacquisition_is_rejected(self):
        driver = self._driver()
        driver._th_hist.append((100.0, 55.4))
        driver._speed_kmh = 80.0
        driver._last_speed_t = 100.7

        # 31 deg in 0.79 s (~39 deg/s): the wrong-orientation lock of the
        # 2026-10-03 02:29 run must be treated as a ghost pose
        self.assertFalse(driver._heading_rate_ok(100.79, 24.2, 100.79))
        # a hard but physical turn (30 deg/s) stays accepted
        self.assertTrue(driver._heading_rate_ok(100.6, 37.4, 100.6))

    def test_low_speed_braking_turn_is_allowed(self):
        driver = self._driver()
        driver._th_hist.append((100.0, 0.0))
        driver._speed_kmh = 20.0
        driver._last_speed_t = 100.5
        driver.speed_ctrl._braking = True

        # 24 deg in 0.6 s = 40 deg/s: impossible at 80 km/h, physically
        # plausible at 20 km/h on SPACE (a_lat / v)
        self.assertTrue(driver._heading_rate_ok(100.6, 24.0, 100.6))

    def test_long_gap_history_is_still_rate_checked(self):
        """An old sample must not disable the check (the old `dt > 0.8: break`)."""
        driver = self._driver()
        driver._th_hist.append((100.0, 0.0))
        driver._speed_kmh = 80.0
        driver._last_speed_t = 100.9

        # 50 deg in 0.9 s (~56 deg/s): beyond the old 0.8 s window, where the
        # check used to be skipped and the flip was accepted
        self.assertFalse(driver._heading_rate_ok(100.9, 50.0, 100.9))
        # a slow drift over the same window stays physical
        self.assertTrue(driver._heading_rate_ok(100.9, 25.0, 100.9))

    def test_only_the_newest_eligible_sample_is_compared(self):
        """Ancient entries must not dominate the rate (3.6 s lockout bug)."""
        driver = self._driver()
        driver._th_hist.extend([(90.0, 0.0), (100.5, 0.0)])
        driver._speed_kmh = 80.0
        driver._last_speed_t = 100.66

        # the newest eligible entry (100.5) gives 20 deg / 0.16 s = 125 deg/s;
        # the ancient 90.0 entry would give 1.9 deg/s and must not be used
        self.assertFalse(driver._heading_rate_ok(100.66, 20.0, 100.66))

    def test_lockout_accepts_a_persistent_track_only_at_low_speed(self):
        """A real spin needs low speed: at 80 km/h a flip is a wrong lock.

        The 2026-10-04 02:52 run drove off route because the lockout accepted
        a flipped pose at speed; the driver must stay blind instead.
        """
        fast = self._driver()
        fast._th_hist.append((100.0, 0.0))
        fast._speed_kmh = 80.0
        fast._last_speed_t = 100.5

        self.assertFalse(fast._heading_acceptable(100.5, 90.0, 100.5))
        self.assertFalse(fast._heading_acceptable(100.6, 90.0, 100.6))
        self.assertFalse(fast._heading_acceptable(100.7, 90.0, 100.7))

        slow = self._driver()
        slow._th_hist.append((100.0, 0.0))
        slow._speed_kmh = 20.0
        slow._last_speed_t = 100.5

        self.assertFalse(slow._heading_acceptable(100.5, 90.0, 100.5))
        self.assertFalse(slow._heading_acceptable(100.6, 90.0, 100.6))
        self.assertTrue(slow._heading_acceptable(100.7, 90.0, 100.7))

    def test_blind_mode_after_a_pose_void(self):
        driver = self._driver()

        self.assertFalse(driver._pose_blind(100.0))
        driver._last_measured_t = 100.0
        self.assertFalse(driver._pose_blind(100.3))
        self.assertTrue(driver._pose_blind(100.5))


class TestStuckEscape(unittest.TestCase):
    """No route progress at a big error: speed floor, then a short reverse."""

    @staticmethod
    def _driver() -> FollowDriver:
        return FollowDriver(
            loc=None,
            pts=[(0.0, 0.0), (1000.0, 0.0)],
            nav_cfg=NavigatorConfig(),
            kb=None,
            px_per_m=2.0,
        )

    def test_no_progress_triggers_the_floor_and_a_reverse(self):
        driver = self._driver()
        # xte 10 m (off the 4 m inner corridor), err -70 deg, waypoint frozen
        self.assertFalse(driver._tick_stuck(100.0, (0.0, 0.0), -70.0, 20.0, 50.0, 1))
        self.assertFalse(driver._tick_stuck(102.0, (0.0, 0.0), -70.0, 20.0, 50.0, 1))
        stuck = driver._tick_stuck(103.1, (0.0, 0.0), -70.0, 20.0, 50.0, 1)

        self.assertTrue(stuck)
        self.assertIsNotNone(driver._stuck_since)

        # two more seconds without progress arm the reverse with opposite lock
        driver._tick_stuck(105.2, (0.0, 0.0), -70.0, 20.0, 50.0, 1)

        self.assertGreater(driver._escape_until, 105.2)
        self.assertEqual(driver._escape_steer, 1)  # err < 0 -> D while backing

    def test_progress_disarms_the_watchdog(self):
        driver = self._driver()
        driver._tick_stuck(100.0, (0.0, 0.0), -70.0, 20.0, 50.0, 1)
        # 15 m closer to the waypoint over the window: not stuck
        stuck = driver._tick_stuck(103.1, (0.0, 0.0), -70.0, 20.0, 20.0, 1)

        self.assertFalse(stuck)
        self.assertIsNone(driver._stuck_since)

    def test_creeping_sideways_counts_as_stuck(self):
        """Net displacement hid a sideways crawl (2026-10-03 23:53 run)."""
        driver = self._driver()
        driver._tick_stuck(100.0, (0.0, 0.0), -70.0, 20.0, 50.0, 1)

        # 10 m of movement, but the waypoint distance is unchanged
        stuck = driver._tick_stuck(103.1, (20.0, 0.0), -70.0, 20.0, 50.0, 1)

        self.assertTrue(stuck)

    def test_waypoint_advance_restarts_the_window(self):
        """dist to a NEW waypoint must not be compared with the old one.

        The 2026-10-04 00:15 run got a false stuck + reverse right after a
        waypoint switch: dist jumped from 28 to 146 px and the difference
        looked like 59 m of "no progress" against the previous target.
        """
        driver = self._driver()
        driver._tick_stuck(100.0, (0.0, 0.0), -70.0, 20.0, 28.0, 11)

        # the active waypoint advances mid-window: window restarts
        self.assertFalse(driver._tick_stuck(102.0, (0.0, 0.0), -70.0, 20.0, 146.0, 12))
        stuck = driver._tick_stuck(103.1, (0.0, 0.0), -70.0, 20.0, 146.0, 12)

        self.assertFalse(stuck)
        self.assertIsNone(driver._stuck_since)


class TestSpinRecoveryOverride(unittest.TestCase):
    """A spun/backwards car gets a turning speed, not the corridor crawl."""

    @staticmethod
    def _driver() -> FollowDriver:
        return FollowDriver(
            loc=None,
            pts=[(0.0, 0.0), (1000.0, 0.0)],
            nav_cfg=NavigatorConfig(),
            kb=None,
            px_per_m=2.0,
        )

    def test_spin_raises_the_target_to_a_turning_speed(self):
        driver = self._driver()
        crawl = driver.speed_ctrl.from_kmh(20.0)

        spin, tgt = driver._spin_recovery_override(crawl, 150.0)

        self.assertTrue(spin)
        self.assertAlmostEqual(driver.speed_ctrl.to_kmh(tgt), 35.0, delta=0.1)

    def test_normal_error_keeps_the_target(self):
        driver = self._driver()
        crawl = driver.speed_ctrl.from_kmh(20.0)

        spin, tgt = driver._spin_recovery_override(crawl, 40.0)

        self.assertFalse(spin)
        self.assertAlmostEqual(tgt, crawl, delta=1e-9)


class TestProgressiveRejoin(unittest.TestCase):
    """The speed cap while returning to the line after an excursion."""

    @staticmethod
    def _driver(**nav_kwargs) -> FollowDriver:
        nav = NavigatorConfig(**nav_kwargs)
        return FollowDriver(loc=None, pts=[(0.0, 0.0), (1.0, 0.0)], nav_cfg=nav, kb=None)

    def test_aligned_car_has_no_cap(self):
        driver = self._driver()

        self.assertIsNone(driver._recovery_speed_cap_kmh(0.5, 5.0, False))
        self.assertIsNone(driver._recovery_speed_cap_kmh(4.9, -9.0, False))

    def test_centered_cornering_does_not_cap(self):
        """Heading error alone is normal pure-pursuit aiming, not a recovery.

        Pure pursuit holds 15-25 deg of error through every bend; capping the
        speed for it braked through normal cornering (the "brakes for no
        reason" report from the 2026-10-01 22:19 run).
        """
        driver = self._driver()  # xte_m = 4.0

        self.assertIsNone(driver._recovery_speed_cap_kmh(1.0, 45.0, False))
        self.assertIsNone(driver._recovery_speed_cap_kmh(4.0, 25.0, False))

    def test_misaligned_cap_eases_down_with_the_error(self):
        driver = self._driver(xte_m=1.0, xte_outer_m=5.0)

        gentle = driver._recovery_speed_cap_kmh(3.0, 22.0, False)
        hard = driver._recovery_speed_cap_kmh(3.0, 40.0, False)
        severe = driver._recovery_speed_cap_kmh(3.0, 60.0, False)
        assert gentle is not None and hard is not None and severe is not None

        self.assertGreater(gentle, hard)
        self.assertGreater(hard, severe)
        self.assertAlmostEqual(severe, 38.0, delta=1e-6)
        self.assertLessEqual(gentle, driver.speed_cap_kmh)

    def test_cap_grows_continuously_from_the_alignment_gate(self):
        driver = self._driver(xte_m=1.0, xte_outer_m=5.0)

        at_gate = driver._recovery_speed_cap_kmh(3.0, 20.0 + 1e-6, False)
        above_gate = driver._recovery_speed_cap_kmh(3.0, 22.0, False)
        assert at_gate is not None and above_gate is not None

        self.assertAlmostEqual(at_gate, driver.speed_cap_kmh, delta=0.1)
        self.assertLess(above_gate, at_gate)

    def test_inside_cap_needs_a_sustained_misalignment(self):
        """A single-frame bearing spike must not brake the truck on a straight."""
        driver = self._driver(xte_m=2.0)

        # xte ~5.9 m (past 1.5x the 2 m inner edge), err 30 deg: debounce first
        self.assertIsNone(driver._rejoin_cap_debounced(12.0, 30.0, False, 100.0))
        self.assertIsNone(driver._rejoin_cap_debounced(12.0, 30.0, False, 100.2))
        self.assertIsNotNone(driver._rejoin_cap_debounced(12.0, 30.0, False, 100.5))

    def test_inside_cap_resets_on_a_single_spike(self):
        driver = self._driver(xte_m=2.0)

        self.assertIsNone(driver._rejoin_cap_debounced(12.0, 30.0, False, 100.0))
        # error collapses before the debounce: nothing to cap
        self.assertIsNone(driver._rejoin_cap_debounced(12.0, 5.0, False, 100.1))
        # and the next spike starts the debounce from scratch
        self.assertIsNone(driver._rejoin_cap_debounced(12.0, 30.0, False, 100.6))

    def test_outside_corridor_cap_is_immediate(self):
        driver = self._driver(xte_m=2.0, xte_outer_m=12.0)

        cap = driver._rejoin_cap_debounced(30.0, 5.0, True, 100.0)

        assert cap is not None
        self.assertLess(cap, driver.speed_cap_kmh)

    def test_off_corridor_cap_keeps_steering_authority(self):
        driver = self._driver(xte_m=1.0, xte_outer_m=5.0)

        shallow = driver._recovery_speed_cap_kmh(5.5, 0.0, True)
        deep = driver._recovery_speed_cap_kmh(12.0, 0.0, True)
        assert shallow is not None and deep is not None

        self.assertGreater(shallow, deep)
        self.assertGreaterEqual(deep, 20.0)
        self.assertLessEqual(deep, 25.0)

    def test_off_corridor_cap_has_no_cliff_at_the_edge(self):
        """A marginal excursion must not demand hard braking in the bend.

        The old step (cruise -> 30 km/h one metre past the edge) dropped the
        target from 79 to 29 at 85 km/h; with the wheel held that braked the
        truck into a spin (2026-10-03 02:19 run).
        """
        driver = self._driver(xte_m=1.0, xte_outer_m=5.0)

        just_out = driver._recovery_speed_cap_kmh(5.2, 0.0, True)
        half = driver._recovery_speed_cap_kmh(7.5, 0.0, True)
        assert just_out is not None and half is not None

        self.assertGreater(just_out, 70.0)
        self.assertGreater(half, 40.0)
        self.assertLess(half, just_out)

    def test_off_corridor_cap_respects_the_corner_minimum(self):
        driver = self._driver(corner_min_kmh=26.0)

        cap = driver._recovery_speed_cap_kmh(20.0, 0.0, True)

        assert cap is not None
        self.assertGreaterEqual(cap, 26.0)


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
