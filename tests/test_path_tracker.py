"""Regression tests for PathTracker waypoint sequencing and segment math."""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.path_tracker import PathTracker  # noqa: E402
from autopilot.navigation.steering_controller import wrap180  # noqa: E402


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

    def test_advances_from_index_zero_when_start_is_already_behind(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], arrive_r=25.0)

        tx, ty, _dist, _arrived = tracker.advance_waypoint((50.0, 5.0))

        self.assertEqual(tracker.idx, 1)
        self.assertEqual((tx, ty), (1000.0, 0.0))

    def test_from_index_zero_keeps_start_when_car_is_before_it(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], arrive_r=25.0)

        tx, ty, _dist, _arrived = tracker.advance_waypoint((-50.0, 5.0))

        self.assertEqual(tracker.idx, 0)
        self.assertEqual((tx, ty), (0.0, 0.0))

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


class TestSnapToNearest(unittest.TestCase):
    def test_picks_nearest_entry_point_mid_route(self):
        tracker = PathTracker([(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)], arrive_r=25.0)

        idx = tracker.snap_to_nearest((190.0, 10.0))

        self.assertEqual(idx, 2)

    def test_skips_point_already_crossed(self):
        tracker = PathTracker([(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)], arrive_r=25.0)

        idx = tracker.snap_to_nearest((150.0, 5.0))

        self.assertEqual(idx, 2)

    def test_keeps_start_point_when_car_is_before_route(self):
        tracker = PathTracker([(0.0, 0.0), (100.0, 0.0)], arrive_r=25.0)

        idx = tracker.snap_to_nearest((-10.0, 0.0))

        self.assertEqual(idx, 0)

    def test_snap_advances_when_start_is_already_behind(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], arrive_r=25.0)

        idx = tracker.snap_to_nearest((50.0, 5.0))

        self.assertEqual(idx, 1)


class TestDenseRouteLookahead(unittest.TestCase):
    def test_aim_spans_dense_chords(self):
        pts = [(float(i * 10), 0.0) for i in range(21)]  # 10 px chords
        tracker = PathTracker(pts, arrive_r=25.0, xte_m=4.0)
        tracker.idx = 10

        _xte, _lim, bearing = tracker.calc_xte_and_bearing((95.0, 0.0), 2.0)
        self.assertAlmostEqual(bearing, 90.0, delta=1e-6)

        # 5 px lateral offset at the 16 px (8 m) lookahead floor (mv=0): the
        # pursuit part is ~atan(5/16) = 17.3 deg, not atan(5/15) = 18 deg with
        # a one-segment aim; the centering term then adds its share and the
        # total saturates the inner-corridor cap (20 deg).
        _xte, _lim, bearing = tracker.calc_xte_and_bearing((95.0, 5.0), 2.0)
        self.assertAlmostEqual(abs(wrap180(bearing - 90.0)), 20.0, delta=0.5)


class TestCorridorTiers(unittest.TestCase):
    @staticmethod
    def _corr(offset_m: float) -> float:
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)],
            arrive_r=25.0,
            xte_m=4.0,
            xte_outer_m=12.0,
        )
        tracker.idx = 1
        _xte, _lim, bearing = tracker.calc_xte_and_bearing((990.0, offset_m * 2.0), 2.0)
        return abs(wrap180(bearing - 90.0))

    def test_midcorridor_is_firmer_then_bounded_outside(self):
        inner = self._corr(2.0)
        mid = self._corr(8.0)
        outer = self._corr(20.0)

        # Firmer with the offset, bounded at every tier (the centering term
        # already saturates the inner cap, so the ratio bound is monotonic).
        self.assertGreater(mid, inner)
        self.assertGreater(outer, mid)
        self.assertLessEqual(outer, 45.0 + 1e-6)


class TestReacquire(unittest.TestCase):
    def test_jumps_forward_to_the_nearest_point_ahead(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0), (3000.0, 0.0)],
            arrive_r=25.0,
            xte_m=4.0,
            xte_outer_m=12.0,
            skip_ahead_m=2000.0,
        )

        moved = tracker.maybe_reacquire((1500.0, 60.0), 1.0, course_deg=90.0, now=100.0)

        self.assertTrue(moved)
        self.assertEqual(tracker.idx, 2)

    def test_rejects_a_candidate_facing_the_other_way(self):
        # out-and-back U: return leg runs west along y=100; the car drives east
        # near it. The return candidate is close (30 px) but faces 180 deg away,
        # the aligned outbound leg is outside the lateral gate -> no jump.
        route = [
            (0.0, 0.0),
            (1000.0, 0.0),
            (2000.0, 0.0),
            (2000.0, 100.0),
            (1000.0, 100.0),
            (300.0, 100.0),
        ]
        tracker = PathTracker(
            route, arrive_r=25.0, xte_m=4.0, xte_outer_m=12.0, skip_ahead_m=4000.0
        )

        moved = tracker.maybe_reacquire((1500.0, 130.0), 1.0, course_deg=90.0, now=100.0)

        self.assertFalse(moved)
        self.assertEqual(tracker.idx, 0)

    def test_respects_the_route_length_window(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0), (3000.0, 0.0)],
            arrive_r=25.0,
            xte_m=4.0,
            xte_outer_m=12.0,
            skip_ahead_m=100.0,
        )

        moved = tracker.maybe_reacquire((1500.0, 60.0), 1.0, course_deg=90.0, now=100.0)

        self.assertFalse(moved)
        self.assertEqual(tracker.idx, 0)

    def test_never_targets_the_final_point(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)],
            arrive_r=25.0,
            xte_m=4.0,
            xte_outer_m=12.0,
            skip_ahead_m=3000.0,
        )

        moved = tracker.maybe_reacquire((1900.0, 60.0), 1.0, course_deg=90.0, now=100.0)

        self.assertFalse(moved)

    def test_does_nothing_while_inside_the_corridor(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0), (3000.0, 0.0)],
            arrive_r=25.0,
            xte_m=4.0,
            xte_outer_m=12.0,
            skip_ahead_m=2000.0,
        )

        moved = tracker.maybe_reacquire((1500.0, 10.0), 1.0, course_deg=90.0, now=100.0)

        self.assertFalse(moved)
        self.assertEqual(tracker.idx, 0)

    def test_never_moves_backwards(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0), (3000.0, 0.0)],
            arrive_r=25.0,
            xte_m=4.0,
            xte_outer_m=12.0,
            skip_ahead_m=3000.0,
        )
        tracker.idx = 3

        moved = tracker.maybe_reacquire((100.0, 60.0), 1.0, course_deg=90.0, now=100.0)

        self.assertFalse(moved)
        self.assertEqual(tracker.idx, 3)

    def test_entry_snap_uses_the_course_gate(self):
        # driving west over an out-and-back: entry must land on the return leg
        route = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 20.0), (0.0, 20.0)]
        tracker = PathTracker(
            route, arrive_r=25.0, xte_m=4.0, xte_outer_m=12.0, reacquire_gate_deg=60.0
        )

        idx = tracker.snap_to_nearest((500.0, 10.0), course_deg=270.0)

        self.assertEqual(idx, 2)


class TestCorridorGainRamp(unittest.TestCase):
    @staticmethod
    def _corr(tracker: PathTracker, off_px: float) -> float:
        _xte, _lim, bearing = tracker.calc_xte_and_bearing((990.0, off_px), 1.0)
        return abs(wrap180(bearing - 90.0))

    def test_gain_ramps_without_a_step_at_the_inner_edge(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], xte_m=4.0, xte_outer_m=12.0
        )
        tracker.idx = 1
        tracker._mv = 20.0  # look = 32 px

        inside = self._corr(tracker, 3.8)
        edge = self._corr(tracker, 4.2)

        # The only jump left at the edge is the gain ramp (the centering term
        # itself is proportional to xte, hence the slightly larger bound).
        self.assertLess(abs(edge - inside), 3.0)

    def test_correction_sign_follows_the_offset_side(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], xte_m=4.0, xte_outer_m=12.0
        )
        tracker.idx = 1
        tracker._mv = 20.0

        xte_pos, _lim, bearing = tracker.calc_xte_and_bearing((990.0, 20.0), 1.0)
        corr_pos = wrap180(bearing - 90.0)
        xte_neg, _lim, bearing = tracker.calc_xte_and_bearing((990.0, -20.0), 1.0)
        corr_neg = wrap180(bearing - 90.0)

        self.assertLess(xte_pos, 0.0)
        self.assertLess(corr_pos, 0.0)
        self.assertGreater(xte_neg, 0.0)
        self.assertGreater(corr_neg, 0.0)

    def test_cross_track_gain_is_bounded_short_of_perpendicular(self):
        tracker = PathTracker(
            [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], xte_m=4.0, xte_outer_m=12.0
        )
        tracker.idx = 1
        tracker._mv = 20.0  # look = 32 px

        natural_6m = math.degrees(math.atan2(6.0, 32.0))

        self.assertGreater(self._corr(tracker, 6.0), natural_6m)
        self.assertLess(self._corr(tracker, 6.0), 30.0)
        self.assertAlmostEqual(self._corr(tracker, 20.0), 35.0, delta=0.5)

    def test_outward_drift_gets_an_earlier_correction(self):
        """The centering term predicts the lateral trend, not just the offset.

        At 2 px off with a 10 px/s outward rate the correction must already be
        stronger than the same offset with no drift - the drift used to run to
        the corridor edge before any wheel command appeared.
        """
        route = [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)]

        drifting = PathTracker(route, xte_m=4.0, xte_outer_m=12.0)
        drifting.idx = 1
        drifting._mv = 20.0
        drifting.calc_xte_and_bearing((990.0, 1.0), 1.0, now=100.0)
        _x, _l, bearing_drift = drifting.calc_xte_and_bearing((990.0, 2.0), 1.0, now=100.1)

        flat = PathTracker(route, xte_m=4.0, xte_outer_m=12.0)
        flat.idx = 1
        flat._mv = 20.0
        _x, _l, bearing_flat = flat.calc_xte_and_bearing((990.0, 2.0), 1.0)

        self.assertGreater(
            abs(wrap180(bearing_drift - 90.0)),
            abs(wrap180(bearing_flat - 90.0)),
        )

    def test_noisy_drift_does_not_oscillate_the_aim(self):
        """A 1 Hz lateral wobble must not swing the aim bearing.

        The 0.7 s trend horizon on the raw rate amplified the pose noise and
        the aim swung +-40 deg every 0.2-0.4 s (2026-10-04 15:31 run): the
        driver chased the swinging aim and the truck weaved.
        """
        route = [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)]
        tracker = PathTracker(route, xte_m=2.0, xte_outer_m=6.0)
        tracker.idx = 1
        tracker._mv = 20.0
        bearings = []
        for i in range(80):
            t = 100.0 + 0.04 * i
            y = 6.0 * math.sin(2.0 * math.pi * (t - 100.0)) + 0.5 * ((-1) ** i)
            _xte, _lim, bearing = tracker.calc_xte_and_bearing((990.0, y), 2.0, now=t)
            bearings.append(bearing)

        steps = [abs(wrap180(bearings[i] - bearings[i - 1])) for i in range(1, len(bearings))]
        self.assertLess(max(steps), 10.0)


class TestCrossTrack(unittest.TestCase):
    def test_bearing_is_continuous_across_the_corridor_limit(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], arrive_r=25.0, xte_m=4.0)
        tracker.idx = 1

        _xte, _lim, inside = tracker.calc_xte_and_bearing((500.0, 3.0), 1.0)
        _xte, _lim, outside = tracker.calc_xte_and_bearing((500.0, 5.0), 1.0)

        # The old perpendicular-foot mode swung the aim ~90 deg here and the
        # full-lock correction spun the truck; pure pursuit plus the tier gain
        # keeps the step at the corridor border small (a few degrees).
        self.assertLess(abs(wrap180(outside - inside)), 10.0)

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
        self.assertIsInstance(bearing, float)


class TestGhostPoseLimit(unittest.TestCase):
    """The ghost-pose speed limit must catch a teleporting wrong lock."""

    def test_limit_catches_an_implausible_reacquisition(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0)], arrive_r=25.0)
        tracker._mv = 44.0  # ~79 km/h

        # the 2026-10-03 23:37 wrong lock implied ~80 px/s at this speed
        self.assertLess(tracker.lost_limit(), 80.0)

    def test_slow_start_still_allows_a_plausible_move(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0)], arrive_r=25.0)
        tracker._mv = 0.0

        self.assertGreaterEqual(tracker.lost_limit(), 40.0)


class TestMotionCourseFreshness(unittest.TestCase):
    """`mh_t` must mark real course updates, not every incoming pose.

    A course frozen below the 10 px / 0.5 s update threshold used to look
    fresh forever; the driver then steered by it while the accurate map
    heading was ignored (the low-speed oscillation in the 2026-10-01 runs).
    """

    def test_below_threshold_does_not_refresh_the_course_age(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0)], arrive_r=25.0)

        tracker.push_pose(100.0, 0.0, 0.0)
        tracker.push_pose(100.5, 12.5, 0.0)  # 12.5 px in the window -> updated
        tracker.push_pose(101.0, 25.0, 0.0)  # updated again
        self.assertAlmostEqual(tracker.mh_t, 101.0, delta=1e-9)
        self.assertIsNotNone(tracker.mh)

        tracker.push_pose(101.5, 27.0, 0.0)  # 2 px: too slow to update the course
        tracker.push_pose(102.0, 29.0, 0.0)

        self.assertAlmostEqual(tracker.mh_t, 101.0, delta=1e-9)
        age = tracker.mh_age(102.0)
        assert age is not None
        self.assertAlmostEqual(age, 1.0, delta=1e-9)
        self.assertLess(tracker.mv, 10.0)

    def test_age_is_none_before_the_first_course_update(self):
        tracker = PathTracker([(0.0, 0.0), (1000.0, 0.0)], arrive_r=25.0)

        tracker.push_pose(100.0, 0.0, 0.0)
        tracker.push_pose(100.5, 2.0, 0.0)

        self.assertIsNone(tracker.mh)
        self.assertIsNone(tracker.mh_age(100.5))


if __name__ == "__main__":
    unittest.main()
