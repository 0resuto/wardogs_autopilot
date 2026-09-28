"""Regression tests for PathTracker waypoint sequencing and segment math."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.path_tracker import PathTracker  # noqa: E402


class TestAdvanceWaypoint(unittest.TestCase):
    def test_does_not_jump_to_later_waypoint_on_uturn(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0), (0.0, 0.0)], arrive_r=25.0)
        tracker.idx = 1

        tx, ty, _dist, arrived = tracker.advance_waypoint((100.0, 0.0))

        self.assertEqual(tracker.idx, 1)
        self.assertEqual((tx, ty), (1000.0, 0.0))
        self.assertFalse(arrived)

    def test_advances_when_waypoint_passed_along_segment(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0), (0.0, 0.0)], arrive_r=25.0)
        tracker.idx = 1

        tx, ty, _dist, arrived = tracker.advance_waypoint((1020.0, 0.0))

        self.assertEqual(tracker.idx, 2)
        self.assertEqual((tx, ty), (0.0, 0.0))
        self.assertFalse(arrived)

    def test_does_not_skip_waypoint_when_far_off_route(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], arrive_r=25.0)
        tracker.idx = 1

        tx, ty, _dist, _arrived = tracker.advance_waypoint((1100.0, 400.0))

        self.assertEqual(tracker.idx, 1)
        self.assertEqual((tx, ty), (1000.0, 0.0))

    def test_consumes_tightly_spaced_waypoints(self):
        tracker = PathTracker([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)], arrive_r=25.0)

        _tx, _ty, _dist, arrived = tracker.advance_waypoint((0.0, 0.0))

        self.assertTrue(arrived)
        self.assertEqual(tracker.idx, 3)

    def test_arrives_at_final_waypoint(self):
        tracker = PathTracker([(0.0, 0.0), (100.0, 0.0)], arrive_r=25.0)

        _tx, _ty, _dist, arrived = tracker.advance_waypoint((0.0, 0.0))
        self.assertFalse(arrived)
        self.assertEqual(tracker.idx, 1)

        tx, ty, _dist, arrived = tracker.advance_waypoint((90.0, 0.0))
        self.assertTrue(arrived)
        self.assertEqual((tx, ty), (100.0, 0.0))
        self.assertEqual(tracker.idx, 2)


class TestCrossTrack(unittest.TestCase):
    def test_calc_xte_handles_completed_route(self):
        tracker = PathTracker([(0.0, 0.0), (100.0, 0.0)], arrive_r=25.0)
        tracker.idx = 2

        xte, xte_lim, bearing = tracker.calc_xte_and_bearing((110.0, 5.0), 2.0)

        self.assertIsInstance(bearing, float)
        self.assertGreater(xte_lim, 0.0)
        self.assertIsInstance(xte, float)

    def test_calc_xte_uses_active_segment(self):
        tracker = PathTracker([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], arrive_r=25.0)
        tracker.idx = 2

        xte, _lim, bearing = tracker.calc_xte_and_bearing((80.0, 30.0), 2.0)

        self.assertAlmostEqual(xte, -20.0, delta=1e-6)
        self.assertAlmostEqual(bearing, 90.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
