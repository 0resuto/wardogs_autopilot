"""Diagnostics tests for the locator index search (no map data needed).

A search cut off by the frame budget must report budget_timeout instead of
masking the real cause as index_no_match.
"""

import os
import sys
import time
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.vision import locator  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
