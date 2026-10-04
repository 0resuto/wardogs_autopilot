"""Map-aligned ECC: per-frame absolute pose correction without drift."""

import math
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.vision.hybrid import HybridLocalizer, _inv3  # noqa: E402
from autopilot.vision.preprocessing import shadow_fill_norm  # noqa: E402


class FakeStore:
    def __init__(self, mu):
        self._mu = mu
        self._cfg = {
            "hybrid_map_ecc": True,
            "hybrid_map_min_cc": 0.1,
            "hybrid_reanchor_s": 1.0,
        }

    def loc_cfg(self):
        return self._cfg

    def map_name(self):
        return "test"

    def load_global_map(self):
        return self._mu

    def mini_scale(self):
        return 1.0


def render_frame(mu, x, y, th_deg, w, h, s=1.0):
    """Render the map into a frame at the given pose; returns (frame, M)."""
    th = math.radians(th_deg)
    a, b = s * math.cos(th), s * math.sin(th)
    cx, cy = w / 2.0, h / 2.0
    t = np.array([x, y], np.float64) - np.array([[a, -b], [b, a]], np.float64) @ np.array(
        [cx, cy], np.float64
    )
    M = np.array([[a, -b, t[0]], [b, a, t[1]]], np.float64)
    frame = cv2.warpAffine(
        mu,
        M[:2].astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return frame, M


class TestMapEcc(unittest.TestCase):
    def _setup(self, seed=7):
        rng = np.random.default_rng(seed)
        mu = rng.integers(0, 255, (2000, 2000), dtype=np.uint8)
        mu = cv2.GaussianBlur(mu, (5, 5), 1.5)
        store = FakeStore(mu)
        return HybridLocalizer(anchor=None, store=store), store, mu

    def test_prediction_offset_is_corrected(self):
        hybrid, store, mu = self._setup()
        w, h = 200, 160
        frame, _ = render_frame(mu, 1000.0, 1000.0, 30.0, w, h)
        _, M_pred = render_frame(mu, 1004.0, 1002.0, 31.0, w, h)
        pp = shadow_fill_norm(frame, None)

        ok, warp, cc, why = hybrid._track_map(pp, M_pred, (h, w))

        self.assertTrue(ok, why)
        pose = hybrid._pose_from_M(M_pred @ _inv3(warp), frame, store.loc_cfg())
        self.assertLess(math.hypot(pose["map_x"] - 1000.0, pose["map_y"] - 1000.0), 2.0)
        self.assertLess(abs((pose["th"] - 30.0 + 180.0) % 360.0 - 180.0), 1.5)

    def test_error_does_not_accumulate_over_a_trajectory(self):
        """Each frame is absolute: a wrong prediction is corrected, not kept."""
        hybrid, store, mu = self._setup(seed=11)
        w, h = 200, 160
        # start with a wrong prediction; every next prediction is the previous
        # corrected pose (the real flow: the truck moved only a few px since)
        _, M_cur = render_frame(mu, 996.0, 998.0, 21.0, w, h)
        for step in range(12):
            x = 1000.0 + 3.0 * step
            y = 1000.0 + 1.5 * step
            frame, _ = render_frame(mu, x, y, 20.0, w, h)
            pp = shadow_fill_norm(frame, None)
            ok, warp, _cc, why = hybrid._track_map(pp, M_cur, (h, w))
            self.assertTrue(ok, why)
            M_cur = M_cur @ _inv3(warp)
            pose = hybrid._pose_from_M(M_cur, frame, store.loc_cfg())
            err = math.hypot(pose["map_x"] - x, pose["map_y"] - y)
            self.assertLess(err, 2.5, "step %d drifted to %.1f px" % (step, err))

    def test_missing_map_patch_is_reported(self):
        hybrid, _store, _mu = self._setup()
        hybrid.store._mu = None

        ok, warp, cc, why = hybrid._track_map(np.zeros((10, 10), np.uint8), np.eye(3), (10, 10))

        self.assertFalse(ok)
        self.assertIsNone(warp)
        self.assertIn("map patch", why)


if __name__ == "__main__":
    unittest.main()
