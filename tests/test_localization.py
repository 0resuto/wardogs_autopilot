"""Unit test for SIFT minimap localization and cold start dark frame."""
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

from autopilot.vision import locator


class TestLocalization(unittest.TestCase):

    def setUp(self):
        locator.set_map("zestafona")

    def test_dark_frame_cold_start(self):
        """Global pose cold start must locate the dark debug frame with valid inliers."""
        frame_path = os.path.join(ROOT, "output", "debug_fail_20260920_183159.png")
        if not os.path.exists(frame_path):
            self.skipTest(f"Reference frame not found: {frame_path}")

        mm = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(mm, "Failed to read reference frame")
        assert mm is not None

        mask = locator.make_mask()
        if mask.shape[:2] != mm.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (mm.shape[1], mm.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0

        pose, diag = locator.global_pose(mm, mask, prev_xy=None, debug=True)

        self.assertIsNotNone(pose, f"Localization failed: {diag.get('reject')} - {diag.get('detail')}")
        assert pose is not None
        self.assertGreaterEqual(pose.get("inl", 0), 10, "Inlier count below expected threshold")

        # Known ground truth coordinates for this frame: (14897, 14342)
        exp_x, exp_y = 14897.2, 14342.2
        self.assertAlmostEqual(float(pose["map_x"]), exp_x, delta=3.0)
        self.assertAlmostEqual(float(pose["map_y"]), exp_y, delta=3.0)

    def test_synthetic_minimap_localization(self):
        """Synthetic minimap crop from global map must localize with high inliers."""
        mu = locator.load_global_map()
        cy, cx = 6752, 5472

        rh, rw = 160, 194

        win, _ = locator._crop_win(mu, cy, cx, rh, rw)
        mask = locator.make_mask()
        if mask.shape[:2] != win.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (win.shape[1], win.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0

        pose, diag = locator.global_pose(win, mask, prev_xy=None, debug=True)
        self.assertIsNotNone(pose, f"Synthetic localization failed: {diag.get('reject')}")
        assert pose is not None
        self.assertGreaterEqual(pose.get("inl", 0), 15)


if __name__ == "__main__":
    unittest.main()
