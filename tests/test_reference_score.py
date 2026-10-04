"""Unit tests for the reference-lap scoring helpers (tools/reference_score.py)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.reference_score import project_to_route, score, segment_times  # noqa: E402


class TestProjection(unittest.TestCase):
    def test_projection_arc_and_lateral(self):
        route = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]

        arc, lateral = project_to_route(route, 150.0, 10.0)

        self.assertAlmostEqual(arc, 150.0, delta=1e-6)
        self.assertAlmostEqual(abs(lateral), 10.0, delta=1e-6)

    def test_projection_clamps_to_the_segment_end(self):
        route = [(0.0, 0.0), (100.0, 0.0)]

        arc, _lateral = project_to_route(route, 150.0, 0.0)

        self.assertAlmostEqual(arc, 100.0, delta=1e-6)


class TestSegmentTimes(unittest.TestCase):
    def test_equal_segments(self):
        times = [0.0, 10.0, 25.0]
        arcs = [0.0, 100.0, 200.0]

        out = segment_times(times, arcs, 200.0, 100.0)

        self.assertEqual(len(out), 2)
        assert out[0] is not None and out[1] is not None
        self.assertAlmostEqual(out[0], 10.0, delta=1e-6)
        self.assertAlmostEqual(out[1], 15.0, delta=1e-6)

    def test_unreached_segment_is_none(self):
        out = segment_times([0.0], [0.0], 200.0, 100.0)

        self.assertEqual(out, [None, None])


class TestScoreGates(unittest.TestCase):
    def test_clean_run_passes(self):
        gates, median, worst = score([10.0, 12.0], [10.0, 12.0], 2.0, 2.0, 0.2, 0.1, 0)

        self.assertTrue(all(ok for _n, ok, _d in gates))
        self.assertAlmostEqual(median, 1.0, delta=1e-9)
        self.assertAlmostEqual(worst, 1.0, delta=1e-9)

    def test_slow_and_unsafe_run_fails(self):
        gates, _median, _worst = score([20.0, 24.0], [10.0, 12.0], 5.0, 2.0, 3.0, 0.5, 2)
        by_name = {name: ok for name, ok, _detail in gates}

        self.assertFalse(by_name["segment time median <= 1.10"])
        self.assertFalse(by_name["segment time worst <= 1.25"])
        self.assertFalse(by_name["zero safety events"])


if __name__ == "__main__":
    unittest.main()
