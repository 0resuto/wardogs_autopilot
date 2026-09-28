"""Tests for the route curvature/braking speed planner."""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.speed_profile import G, RouteSpeedPlanner  # noqa: E402

LAT_ACCEL = 0.35 * G
BRAKE_DECEL = 0.45 * G


def planner(**overrides) -> RouteSpeedPlanner:
    params = dict(
        lat_accel_mps2=LAT_ACCEL,
        brake_decel_mps2=BRAKE_DECEL,
        min_speed_kmh=12.0,
        lookahead_m=250.0,
        cut_m=15.0,
    )
    params.update(overrides)
    return RouteSpeedPlanner(**params)


class TestPlanner(unittest.TestCase):
    def test_straight_route_is_unconstrained(self):
        pts = [(0.0, 0.0), (400.0, 0.0), (800.0, 0.0)]
        self.assertIsNone(planner().target_speed_kmh((0.0, 0.0), pts, 0, 2.0))

    def test_sharp_sparse_corner_uses_cut_radius(self):
        pts = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0)]
        target = planner().target_speed_kmh((190.0, 0.0), pts, 1, 2.0)

        assert target is not None
        radius = 15.0 / (2.0 * math.sin(math.radians(45.0)))
        v_corner = math.sqrt(LAT_ACCEL * radius)
        expected = math.sqrt(v_corner * v_corner + 2.0 * BRAKE_DECEL * 5.0) * 3.6
        self.assertAlmostEqual(target, expected, delta=1.0)

    def test_dense_arc_uses_arc_radius(self):
        radius_m = 50.0
        px_per_m = 2.0
        step_deg = 5.0
        pts = []
        angle = 0.0
        while angle <= 90.0 + 1e-9:
            pts.append(
                (
                    radius_m * math.sin(math.radians(angle)) * px_per_m,
                    radius_m * math.cos(math.radians(angle)) * px_per_m,
                )
            )
            angle += step_deg
        mp = (2.0 * pts[0][0] - pts[1][0], 2.0 * pts[0][1] - pts[1][1])

        target = planner().target_speed_kmh(mp, pts, 0, px_per_m)

        assert target is not None
        self.assertTrue(45.0 < target < 65.0, target)

    def test_braking_planning_slows_with_distance(self):
        pts = [(-200.0, 0.0), (200.0, 0.0), (200.0, 400.0)]
        p = planner()
        far = p.target_speed_kmh((-200.0, 0.0), pts, 0, 2.0)
        near = p.target_speed_kmh((100.0, 0.0), pts, 1, 2.0)

        assert far is not None and near is not None
        self.assertGreater(far, near)

    def test_final_waypoint_plans_a_stop(self):
        pts = [(0.0, 0.0), (200.0, 0.0)]
        p = planner()
        far = p.target_speed_kmh((0.0, 0.0), pts, 0, 2.0)
        near = p.target_speed_kmh((160.0, 0.0), pts, 1, 2.0)
        crawling = p.target_speed_kmh((198.0, 0.0), pts, 1, 2.0)

        assert far is not None and near is not None and crawling is not None
        self.assertAlmostEqual(far, math.sqrt(2.0 * BRAKE_DECEL * 100.0) * 3.6, delta=2.0)
        self.assertLess(near, far)
        self.assertAlmostEqual(crawling, 12.0, delta=1e-6)

    def test_min_speed_clamps_hairpin(self):
        pts = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0)]
        target = planner(min_speed_kmh=35.0).target_speed_kmh((190.0, 0.0), pts, 1, 2.0)
        assert target is not None
        self.assertAlmostEqual(target, 35.0, delta=1e-6)

    def test_bad_scale_returns_none(self):
        pts = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0)]
        self.assertIsNone(planner().target_speed_kmh((0.0, 0.0), pts, 0, 0.0))


if __name__ == "__main__":
    unittest.main()
