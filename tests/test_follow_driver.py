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
