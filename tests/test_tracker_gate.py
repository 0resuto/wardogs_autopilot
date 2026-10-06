"""Unit tests for the tracker relocation vote gate helpers."""

import collections
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
from autopilot.vision.tracker import (  # noqa: E402
    LiveLocator,
    _course_from_hist,
    _needs_vote,
    _vote_decide,
)


class TestNeedsVote(unittest.TestCase):
    def test_clean_candidate_needs_no_vote(self):
        self.assertFalse(_needs_vote(None, after_gap=False, inl=5, inl_skip=40))

    def test_jumped_candidate_is_vote_gated(self):
        self.assertTrue(_needs_vote("jump 500 px", after_gap=False, inl=5, inl_skip=40))

    def test_strong_candidate_bypasses_the_vote(self):
        self.assertFalse(_needs_vote("jump 500 px", after_gap=False, inl=50, inl_skip=40))

    def test_gap_disables_the_strong_bypass(self):
        self.assertTrue(_needs_vote("gap 0.70 s", after_gap=True, inl=50, inl_skip=40))
        self.assertTrue(_needs_vote(None, after_gap=True, inl=50, inl_skip=40))


class TestVoteDecide(unittest.TestCase):
    def test_cluster_needs_the_required_agreement(self):
        buf: collections.deque = collections.deque(maxlen=5)

        self.assertIsNone(_vote_decide(buf, 3, 300, (100.0, 100.0), 45.0, 10.0))
        self.assertIsNone(_vote_decide(buf, 3, 300, (101.0, 100.0), 46.0, 10.0))
        cluster = _vote_decide(buf, 3, 300, (100.5, 100.0), 44.0, 10.0)

        self.assertIsNotNone(cluster)
        self.assertEqual(len(cluster), 3)

    def test_scattered_candidates_do_not_agree(self):
        buf: collections.deque = collections.deque(maxlen=5)
        _vote_decide(buf, 3, 300, (0.0, 0.0), 0.0, 10.0)
        _vote_decide(buf, 3, 300, (1000.0, 0.0), 0.0, 10.0)

        self.assertIsNone(_vote_decide(buf, 3, 300, (2000.0, 0.0), 0.0, 10.0))

    def test_flipped_heading_at_the_same_place_does_not_agree(self):
        """The known wrong-lock shape: same position, 180-degree flip.

        A position-only vote used to accept it after three frames.
        """
        buf: collections.deque = collections.deque(maxlen=5)
        _vote_decide(buf, 3, 300, (100.0, 100.0), 45.0, 10.0)
        _vote_decide(buf, 3, 300, (100.0, 100.0), 225.0, 10.0)

        self.assertIsNone(_vote_decide(buf, 3, 300, (100.0, 100.0), 45.0, 10.0))


class TestMotionCourse(unittest.TestCase):
    def test_not_moving_has_no_course(self):
        hist = collections.deque([(100.0, 0.0, 0.0), (100.5, 3.0, 0.0)])

        self.assertIsNone(_course_from_hist(hist, 100.5))

    def test_cardinal_directions(self):
        north = collections.deque([(100.0, 0.0, 0.0), (100.5, 0.0, -20.0)])
        east = collections.deque([(100.0, 0.0, 0.0), (100.5, 20.0, 0.0)])
        south = collections.deque([(100.0, 0.0, 0.0), (100.5, 0.0, 20.0)])
        c_north = _course_from_hist(north, 100.5)
        c_east = _course_from_hist(east, 100.5)
        c_south = _course_from_hist(south, 100.5)

        assert c_north is not None and c_east is not None and c_south is not None
        self.assertAlmostEqual(c_north, 0.0)
        self.assertAlmostEqual(c_east, 90.0)
        self.assertAlmostEqual(c_south, 180.0)

    def test_stale_samples_have_no_course(self):
        hist = collections.deque([(100.0, 0.0, 0.0), (100.5, 0.0, -20.0)])

        self.assertIsNone(_course_from_hist(hist, 103.0))


class TestMotionFlip(unittest.TestCase):
    def _loc(self) -> LiveLocator:
        return LiveLocator({}, np.zeros((4, 4), bool))

    def test_flip_at_speed_is_rejected(self):
        loc = self._loc()
        loc._motion_hist.extend([(100.0, 0.0, 0.0), (100.5, 0.0, -20.0)])
        item = {"speed_ok": True, "speed_kmh": 80.0}

        self.assertFalse(loc._motion_flip(100.5, 10.0, item))
        self.assertTrue(loc._motion_flip(100.5, 180.0, item))

    def test_stop_clears_the_latch(self):
        loc = self._loc()
        loc._course = 0.0
        loc._course_t = 100.0

        self.assertFalse(loc._motion_flip(101.0, 180.0, {"speed_ok": True, "speed_kmh": 5.0}))
        self.assertIsNone(loc._course)

    def test_latched_course_survives_a_blind_spell(self):
        """No new positions after the rejection, but the OCR says moving."""
        loc = self._loc()
        loc._motion_hist.extend([(100.0, 0.0, 0.0), (100.5, 0.0, -20.0)])
        item = {"speed_ok": True, "speed_kmh": 80.0}
        loc._motion_flip(100.5, 10.0, item)

        self.assertTrue(loc._motion_flip(103.0, 180.0, item))

    def test_stale_latch_expires_without_speed_evidence(self):
        loc = self._loc()
        loc._course = 0.0
        loc._course_t = 100.0

        self.assertFalse(loc._motion_flip(103.0, 180.0, {}))
        self.assertIsNone(loc._course)


class TestAnchorPrevTh(unittest.TestCase):
    """The anchor rotation prior is only valid while the truck moves."""

    def _loc(self) -> LiveLocator:
        return LiveLocator({}, np.zeros((4, 4), bool))

    def test_no_recent_accept_disables_the_prior(self):
        loc = self._loc()

        self.assertIsNone(loc._anchor_prev_th(100.0))

        loc._last_accepted = (0.0, 0.0)
        loc._last_accept_t = 90.0  # 10 s ago

        self.assertIsNone(loc._anchor_prev_th(100.0))

    def test_motion_course_enables_the_prior(self):
        loc = self._loc()
        loc._last_accepted = (0.0, 0.0)
        loc._last_accept_t = 100.0
        loc._prev_th = 188.0
        loc._motion_hist.extend([(100.0, 0.0, 0.0), (100.5, 0.0, -20.0)])

        self.assertEqual(loc._anchor_prev_th(100.5), 188.0)

    def test_ocr_speed_enables_the_prior_without_positions(self):
        loc = self._loc()
        loc._last_accepted = (0.0, 0.0)
        loc._last_accept_t = 100.0
        loc._prev_th = 188.0
        loc._speed_kmh = 80.0
        loc._speed_t = 100.0

        self.assertEqual(loc._anchor_prev_th(100.2), 188.0)

    def test_slow_without_motion_disables_the_prior(self):
        """At low speed a real spin may change the heading fast."""
        loc = self._loc()
        loc._last_accepted = (0.0, 0.0)
        loc._last_accept_t = 100.0
        loc._prev_th = 188.0
        loc._speed_kmh = 10.0
        loc._speed_t = 100.0

        self.assertIsNone(loc._anchor_prev_th(100.2))


class TestMapSwitchReset(unittest.TestCase):
    """A map switch must not keep seeding searches with the old map's pose."""

    def test_map_change_drops_the_pose_state(self):
        loc = LiveLocator({}, np.zeros((4, 4), bool))
        loc._prev_xy = (1.0, 2.0)
        loc._last_accepted = (1.0, 2.0)
        loc._vote_buf.append((1.0, 2.0, None))

        with patch.object(tracker_mod.locator, "map_name", lambda: "other-map"):
            loc._sync_map()

        self.assertIsNone(loc._prev_xy)
        self.assertIsNone(loc._last_accepted)
        self.assertEqual(len(loc._vote_buf), 0)

    def test_same_map_keeps_the_pose_state(self):
        loc = LiveLocator({}, np.zeros((4, 4), bool))
        loc._prev_xy = (1.0, 2.0)

        with patch.object(tracker_mod.locator, "map_name", lambda: loc._map):
            loc._sync_map()

        self.assertEqual(loc._prev_xy, (1.0, 2.0))


class TestVoteRejectResetsTrack(unittest.TestCase):
    """A rejected relocation candidate must not survive inside the engine."""

    def test_vote_reject_resets_the_engine_track(self):
        calls = {"n": 0}

        def fake_pose(_mm, _mask, **_kwargs):
            jumped = calls["n"] > 0
            calls["n"] += 1
            inl = 5 if jumped else 50
            x = 4000.0 if jumped else 0.0
            return (
                dict(map_x=x, map_y=0.0, th=0.0, s=1.0, inl=inl),
                dict(reject=None, detail="OK (fake)"),
            )

        cfg = AppConfig()
        cfg.capture.fps = 100
        loc = LiveLocator(
            cfg=cfg,
            mask=np.zeros((64, 64), bool),
            frame_source=lambda: np.zeros((64, 64, 3), np.uint8),
        )
        reset_calls = {"n": 0}

        def fake_reset():
            reset_calls["n"] += 1

        with (
            patch.object(tracker_mod.locator, "load_global_map", lambda: None),
            patch.object(tracker_mod.locator, "global_pose", fake_pose),
            patch.object(tracker_mod.locator, "reset_track", fake_reset),
        ):
            loc.start()
            deadline = time.time() + 5.0
            while reset_calls["n"] < 1 and time.time() < deadline:
                time.sleep(0.01)
            loc.stop()
            loc.join(timeout=2.0)

        self.assertGreaterEqual(reset_calls["n"], 1, "vote rejection did not reset the track")


if __name__ == "__main__":
    unittest.main()
