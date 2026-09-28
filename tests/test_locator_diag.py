"""Diagnostics tests for the locator index search (no map data needed).

A search cut off by the frame budget must report budget_timeout instead of
masking the real cause as index_no_match.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import AppConfig  # noqa: E402
from autopilot.vision import locator  # noqa: E402
from autopilot.vision import tracker as tracker_mod  # noqa: E402
from autopilot.vision.tracker import LiveLocator  # noqa: E402


class _FakeIndex:
    def __init__(self, pts: np.ndarray, desc: np.ndarray) -> None:
        self._entry = (0, pts, desc)

    def global_candidates(self, _cap: int):
        return [self._entry]

    def radius_candidates(self, _cx: float, _cy: float, _r: float):
        return [self._entry]


class TestBudgetDiagnostics(unittest.TestCase):
    def setUp(self):
        self.engine = locator.MapLocator()
        self.kp = [cv2.KeyPoint(float(i), float(i), 5.0) for i in range(6)]
        self.d1 = np.zeros((6, 128), np.float32)
        self.idx = _FakeIndex(
            np.zeros((3, 2), np.float32),
            np.zeros((3, 128), np.float32),
        )
        self.thr = dict(ratio=0.8, min_inl=4, min_inl_rate=0.0)

    def _run(self, **kwargs):
        return self.engine._pose_via_index(
            np.zeros((32, 32), np.uint8),
            None,
            self.idx,
            0.0,
            0.0,
            None,
            thr=self.thr,
            feats=(self.kp, self.d1),
            **kwargs,
        )

    def test_expired_budget_reports_budget_timeout(self):
        pose, diag = self._run(budget=0.5, t0=time.time() - 1.0)

        self.assertIsNone(pose)
        self.assertEqual(diag["reject"], "budget_timeout")
        self.assertIn("budget", diag["detail"])

    def test_no_budget_keeps_index_no_match(self):
        pose, diag = self._run()

        self.assertIsNone(pose)
        self.assertEqual(diag["reject"], "index_no_match")


class TestTrainSetThinning(unittest.TestCase):
    """BFMatcher aborts at 262144 train rows; oversized sets must be thinned."""

    def test_subsample_keeps_points_and_descriptors_aligned(self):
        pts = np.repeat(np.arange(20, dtype=np.float32)[:, None], 2, axis=1)
        desc = np.repeat(np.arange(20, dtype=np.float32)[:, None], 128, axis=1)

        pts2, desc2 = locator._subsample_train(pts, desc, cap=8)

        self.assertEqual(len(pts2), 8)
        np.testing.assert_array_equal(pts2[:, 0], desc2[:, 0])

    def test_pose_via_index_thins_oversized_candidates(self):
        engine = locator.MapLocator()
        kp = [cv2.KeyPoint(float(i), float(i), 5.0) for i in range(6)]
        d1 = np.zeros((6, 128), np.float32)
        idx = _FakeIndex(np.zeros((40, 2), np.float32), np.zeros((40, 128), np.float32))
        real = locator._subsample_train
        thinned: list[int] = []

        def _spy(pts, desc, cap=None):
            pts2, desc2 = real(pts, desc, cap)
            thinned.append(pts2.shape[0])
            return pts2, desc2

        with (
            patch.object(locator, "BF_MAX_TRAIN_DESC", 8),
            patch.object(locator, "_subsample_train", side_effect=_spy) as mock,
        ):
            pose, diag = engine._pose_via_index(
                np.zeros((32, 32), np.uint8),
                None,
                idx,
                0.0,
                0.0,
                None,
                thr=dict(ratio=0.8, min_inl=4, min_inl_rate=0.0),
                feats=(kp, d1),
            )

        self.assertIsNone(pose)
        self.assertIsNotNone(diag)
        mock.assert_called_once()
        self.assertEqual(mock.call_args[0][0].shape[0], 40)
        self.assertEqual(thinned, [8])


class TestTrackerFrameError(unittest.TestCase):
    """A per-frame locator error must not kill the tracking thread."""

    def test_locator_frame_error_degrades_to_bad_frame(self):
        loc = LiveLocator(cfg=AppConfig(), mask=None)
        mm = np.zeros((32, 32), np.uint8)

        with patch.object(tracker_mod.locator, "global_pose", side_effect=RuntimeError("boom")):
            pose, diag = loc._frame_pose(mm, None)

        self.assertIsNone(pose)
        self.assertEqual(diag["reject"], "locator_error")
        self.assertIn("boom", diag["detail"])


if __name__ == "__main__":
    unittest.main()
