"""Tests for the temporary manual-driving recorder."""

import json
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.manual_record import ManualDriveRecorder, build_route  # noqa: E402


class _FakeLocator:
    def __init__(self) -> None:
        self.latest = dict(
            ts=1.0,
            pose=dict(th=45.0, s=1.0, inl=10),
            map_px=(100.0, 200.0),
            good=True,
            speed_kmh=30,
            speed_ok=True,
        )


class TestManualDriveRecorder(unittest.TestCase):
    def test_records_header_and_rows_with_injected_keys(self):
        keys = {"W": True, "A": False, "S": False, "D": True, "SPACE": False}

        with tempfile.TemporaryDirectory() as tmp:
            rec = ManualDriveRecorder(
                _FakeLocator(),
                [(0.0, 0.0), (10.0, 0.0)],
                {"source": "manual"},
                out_dir=tmp,
                period=0.01,
                keys_source=lambda: keys,
            )
            rec.start()
            time.sleep(0.15)
            rec.stop()

            self.assertIsNotNone(rec.path)
            assert rec.path is not None
            with open(rec.path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]

        header, rows = lines[0], lines[1:]
        self.assertEqual(header["kind"], "manual-log")
        self.assertEqual(header["route"], [[0.0, 0.0], [10.0, 0.0]])
        self.assertEqual(header["params"]["source"], "manual")
        self.assertGreater(len(rows), 2)
        self.assertEqual(rows[0]["keys"], "WD")
        self.assertEqual(rows[0]["x"], 100.0)
        self.assertEqual(rows[0]["y"], 200.0)
        self.assertEqual(rows[0]["th"], 45.0)
        self.assertTrue(rows[0]["good"])
        self.assertEqual(rows[0]["speed_kmh"], 30)

    def test_stop_is_safe_without_start_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = ManualDriveRecorder(
                _FakeLocator(),
                [],
                {},
                out_dir=tmp,
                period=0.01,
                keys_source=lambda: {},
            )
            rec.stop()
            rec.start()
            time.sleep(0.05)
            rec.stop()
            rec.stop()

            self.assertFalse(rec.is_alive())
            assert rec.path is not None
            self.assertTrue(os.path.exists(rec.path))


class TestBuildRoute(unittest.TestCase):
    def test_resamples_smooths_and_drops_jumps(self):
        rows = [dict(x=float(i * 10), y=0.0, good=True) for i in range(11)]
        rows.append(dict(x=9999.0, y=9999.0, good=True))  # localization jump
        rows.append(dict(x=100.0, y=0.0, good=False))  # held pose

        pts = build_route(rows, step_px=25.0, smooth_win=1)

        self.assertAlmostEqual(pts[0][0], 0.0, delta=1e-6)
        self.assertAlmostEqual(pts[-1][0], 100.0, delta=1e-6)
        self.assertEqual(len(pts), 5)
        for a, b in zip(pts, pts[1:], strict=False):
            self.assertAlmostEqual(b[0] - a[0], 25.0, delta=0.5)

    def test_not_enough_poses_returns_raw_points(self):
        pts = build_route([dict(x=1.0, y=2.0, good=True)], step_px=10.0)

        self.assertEqual(pts, [[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
