"""Tests for the alpha-beta pose smoother and its tracker integration.

The smoother turns the noisy SIFT match stream (~1-2 m per-frame jitter on
the minimap maps) into a steady published pose: steady motion passes with no
lag, random noise is attenuated, confirmed relocations and measurement gaps
reset the state.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import AppConfig  # noqa: E402
from autopilot.vision import tracker as tracker_mod  # noqa: E402
from autopilot.vision.pose_filter import RESET_GAP_S, PoseSmoother  # noqa: E402
from autopilot.vision.tracker import LiveLocator  # noqa: E402


class TestPoseSmoother(unittest.TestCase):
    def test_disabled_alpha_is_raw_passthrough(self):
        sm = PoseSmoother(alpha=0.0)
        for i, x in enumerate((10.0, 11.5, 9.0, 500.0)):
            self.assertEqual(sm.update(x, 2.0 * i, 1.0 + 0.1 * i), (x, 2.0 * i))

    def test_first_measurement_is_trusted(self):
        sm = PoseSmoother()
        self.assertEqual(sm.update(1000.0, 2000.0, 0.0), (1000.0, 2000.0))

    def test_resting_noise_is_attenuated(self):
        sm = PoseSmoother(alpha=0.5)
        sm.update(1000.0, 1000.0, 0.0)
        noise = [2.0, -2.0] * 40
        out = [
            sm.update(1000.0 + n, 1000.0 - n, 0.1 * (i + 1))[0] - 1000.0
            for i, n in enumerate(noise)
        ]
        self.assertLess(np.std(out[10:]), 0.6 * np.std(noise))

    def test_constant_velocity_has_no_steady_state_lag(self):
        sm = PoseSmoother(alpha=0.5)
        t, x = 0.0, 0.0
        sm.update(x, 0.0, t)
        err = 0.0
        for _ in range(80):
            t += 0.1
            x += 5.0  # 50 px/s
            out_x, _ = sm.update(x, 0.0, t)
            err = x - out_x
        self.assertLess(abs(err), 0.5)

    def test_confirmed_relocation_jumps_instead_of_gliding(self):
        sm = PoseSmoother(alpha=0.5, reset_px=100.0)
        sm.update(0.0, 0.0, 0.0)
        for i in range(10):
            sm.update(0.0, 0.0, 0.1 * (i + 1))
        self.assertEqual(sm.update(500.0, 0.0, 1.1), (500.0, 0.0))

    def test_small_correction_is_blended(self):
        sm = PoseSmoother(alpha=0.5, reset_px=100.0)
        sm.update(0.0, 0.0, 0.0)
        sm.update(0.0, 0.0, 0.1)
        out_x, _ = sm.update(20.0, 0.0, 0.2)
        self.assertGreater(out_x, 0.0)
        self.assertLess(out_x, 20.0)

    def test_measurement_gap_resets(self):
        sm = PoseSmoother(alpha=0.5)
        sm.update(0.0, 0.0, 0.0)
        sm.update(0.0, 0.0, 0.1)
        self.assertEqual(sm.update(300.0, 0.0, 0.1 + RESET_GAP_S + 0.01), (300.0, 0.0))

    def test_reset_forgets_the_state(self):
        sm = PoseSmoother()
        sm.update(1000.0, 1000.0, 0.0)
        sm.update(1000.0, 1000.0, 0.1)
        sm.reset()
        self.assertEqual(sm.update(5.0, 7.0, 0.2), (5.0, 7.0))

    def test_configure_clamps_alpha(self):
        sm = PoseSmoother()
        sm.configure(2.0, -5.0)
        self.assertEqual(sm.alpha, 1.0)
        self.assertEqual(sm.reset_px, 0.0)


class _FakeLocator:
    """Stand-in for locator.global_pose: alternating +-2 px around x=1000."""

    def __init__(self) -> None:
        self.raw: list[float] = []

    def __call__(self, _mm, _mask, **_kwargs):
        x = 1000.0 + 2.0 * (-1) ** len(self.raw)
        self.raw.append(x)
        pose = dict(map_x=x, map_y=500.0, th=0.0, s=1.0, inl=50)
        return pose, dict(reject=None, detail="OK (fake)")


class TestCaptureFps(unittest.TestCase):
    def test_set_fps_updates_config_and_clamps(self):
        loc = LiveLocator(cfg=AppConfig(), mask=None)

        loc.set_fps(30)
        self.assertEqual(loc.cfg["capture"]["fps"], 30)
        self.assertEqual(loc.app_cfg.capture.fps, 30)
        self.assertEqual(loc.cap_cfg.fps, 30)

        loc.set_fps(999)
        self.assertEqual(loc.cap_cfg.fps, 60)
        loc.set_fps(0)
        self.assertEqual(loc.cap_cfg.fps, 1)


class TestTrackerSmoothingIntegration(unittest.TestCase):
    def test_published_pose_is_smoothed(self):
        fake = _FakeLocator()
        cfg = AppConfig()
        cfg.capture.fps = 100
        loc = LiveLocator(
            cfg=cfg,
            mask=np.zeros((64, 64), bool),
            frame_source=lambda: np.zeros((64, 64, 3), np.uint8),
        )

        with (
            patch.object(tracker_mod.locator, "load_global_map", lambda: None),
            patch.object(tracker_mod.locator, "global_pose", fake),
        ):
            loc.start()
            deadline = time.time() + 5.0
            while len(fake.raw) < 8 and time.time() < deadline:
                time.sleep(0.01)
            loc.stop()
            loc.join(timeout=2.0)

        self.assertGreaterEqual(len(fake.raw), 8, "tracker produced no poses")
        latest = loc.latest
        self.assertIsNotNone(latest)
        assert latest is not None
        pub_x = float(latest["map_px"][0])
        self.assertAlmostEqual(float(latest["pose"]["map_x"]), pub_x, delta=1e-9)
        # the filtered value must sit closer to the true centre than the raw
        # alternating match does (and differ from it: the filter is engaged)
        self.assertLess(abs(pub_x - 1000.0), abs(fake.raw[-1] - 1000.0))
        self.assertNotAlmostEqual(pub_x, fake.raw[-1], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
