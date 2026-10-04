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
from autopilot.vision import index_search as index_search_mod  # noqa: E402
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
            patch.object(index_search_mod, "BF_MAX_TRAIN_DESC", 8),
            patch.object(index_search_mod, "_subsample_train", side_effect=_spy) as mock,
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


class _CountingBF:
    """BFMatcher proxy counting knnMatch calls (cv2 attrs are read-only)."""

    def __init__(self, real) -> None:
        self.real = real
        self.calls = 0

    def knnMatch(self, *args, **kwargs):
        self.calls += 1
        return self.real.knnMatch(*args, **kwargs)


class _MultiIndex:
    """Index stub with several (level, pts, desc) candidate sets."""

    def __init__(self, sets: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
        # like FeatureIndex: (level_index, pts, desc) with a levels array
        self.levels = np.asarray([lv for lv, _p, _d in sets], np.float32)
        self.sets = [(i, pts, desc) for i, (_lv, pts, desc) in enumerate(sets)]
        self.radius_calls = 0

    def total_features(self) -> int:
        return sum(len(desc) for _li, _pts, desc in self.sets)

    def radius_candidates(self, _cx: float, _cy: float, _r: float):
        self.radius_calls += 1
        return list(self.sets)

    def global_candidates(self, _cap: int):
        return list(self.sets)


class _CfgStore:
    """MapStore stub exposing just the locator config block."""

    def __init__(self, extra: dict) -> None:
        self._cfg = dict(extra)

    def loc_cfg(self):
        base = dict(
            local_radius=450.0,
            radius_growth=1.6,
            track_radius=900.0,
            ratio=0.8,
            min_inl=2,
            min_inl_rate=0.0,
            ratio_local=0.8,
            min_inl_local=2,
            min_inl_rate_local=0.0,
            ratio_global=0.8,
            min_inl_global=2,
            min_inl_rate_global=0.0,
        )
        base.update(self._cfg)
        return base


def _matchable_set(n_pts: int = 8):
    """(pts, desc) that matches the 8-descriptor query by a pure translation."""
    base = np.tile(np.arange(1, 9, dtype=np.float32)[:, None], (1, 128))
    desc = base + np.arange(n_pts, dtype=np.float32)[:, None] * 1e-3
    idx = np.arange(n_pts, dtype=np.float32) * 5.0
    pts = np.stack([idx + 10.0, idx + 3.0], axis=1)
    return pts, desc


class TestSearchOptimizations(unittest.TestCase):
    def setUp(self):
        self.engine = locator.MapLocator()
        self.kp = [cv2.KeyPoint(float(i * 5), float(i * 5), 5.0) for i in range(8)]
        self.d1 = np.tile(np.arange(1, 9, dtype=np.float32)[:, None], (1, 128))
        self.thr = dict(ratio=0.8, min_inl=2, min_inl_rate=0.0)

    def _run(self, idx, **kwargs):
        return self.engine._pose_via_index(
            np.zeros((32, 32), np.uint8),
            None,
            idx,
            0.0,
            0.0,
            450.0,
            thr=self.thr,
            feats=(self.kp, self.d1),
            **kwargs,
        )

    def test_radius_candidates_are_computed_once_per_step(self):
        idx = _MultiIndex([(1.0, *_matchable_set())])
        self.engine._index_find(
            np.zeros((32, 32), np.uint8),
            None,
            idx,
            100.0,
            100.0,
            min_inl=2,
            feats=(self.kp, self.d1),
        )

        self.assertEqual(idx.radius_calls, 1)

    def test_strong_first_match_stops_the_scan(self):
        idx = _MultiIndex([(1.0, *_matchable_set()), (0.8, *_matchable_set())])
        spy = _CountingBF(self.engine.bf)
        self.engine.bf = spy

        pose, diag = self._run(idx, early_inl=4)

        self.assertIsNotNone(pose)
        self.assertEqual(spy.calls, 1)

    def test_zero_early_exit_keeps_scanning_all_sets(self):
        idx = _MultiIndex([(1.0, *_matchable_set()), (0.8, *_matchable_set())])
        spy = _CountingBF(self.engine.bf)
        self.engine.bf = spy

        pose, _diag = self._run(idx, early_inl=0)

        self.assertIsNotNone(pose)
        self.assertEqual(spy.calls, 2)

    def test_early_break_uses_its_own_config_key(self):
        idx = _MultiIndex([(1.0, *_matchable_set()), (0.8, *_matchable_set())])
        spy = _CountingBF(self.engine.bf)
        self.engine.bf = spy
        self.engine.store = _CfgStore({"early_inl": 4, "vote_inl_skip": 999})

        pose = self.engine._index_find(
            np.zeros((32, 32), np.uint8),
            None,
            idx,
            100.0,
            100.0,
            min_inl=2,
            feats=(self.kp, self.d1),
        )[0]

        self.assertIsNotNone(pose)
        self.assertEqual(spy.calls, 1)

    def test_vote_skip_does_not_stop_the_scan(self):
        idx = _MultiIndex([(1.0, *_matchable_set()), (0.8, *_matchable_set())])
        spy = _CountingBF(self.engine.bf)
        self.engine.bf = spy
        self.engine.store = _CfgStore({"vote_inl_skip": 4})

        pose = self.engine._index_find(
            np.zeros((32, 32), np.uint8),
            None,
            idx,
            100.0,
            100.0,
            min_inl=2,
            feats=(self.kp, self.d1),
        )[0]

        self.assertIsNotNone(pose)
        self.assertEqual(spy.calls, 2)

    def test_previous_scale_prunes_far_levels(self):
        idx = _MultiIndex([(1.0, *_matchable_set()), (0.6, *_matchable_set())])
        spy = _CountingBF(self.engine.bf)
        self.engine.bf = spy

        pose, _diag = self._run(idx, prev_s=1.0, early_inl=0)

        self.assertIsNotNone(pose)
        self.assertEqual(spy.calls, 1)


class TestRansacThreshold(unittest.TestCase):
    """The reprojection threshold is config-driven and gates lazy matches.

    A tight threshold keeps the refined affine free of correspondences that
    are a few px off; the fallback threshold is the second chance for frames
    where the tight one leaves too few inliers.
    """

    def setUp(self):
        self.engine = locator.MapLocator()
        self.thr = dict(ratio=0.8, min_inl=2, min_inl_rate=0.0)

    def _run(self, idx, thr=None):
        kp = [cv2.KeyPoint(float(i * 5), float(i * 5), 5.0) for i in range(8)]
        d1 = np.tile(np.arange(1, 9, dtype=np.float32)[:, None], (1, 128))
        return self.engine._pose_via_index(
            np.zeros((32, 32), np.uint8),
            None,
            idx,
            0.0,
            0.0,
            450.0,
            thr=thr or self.thr,
            feats=(kp, d1),
        )

    def test_tight_threshold_rejects_lazy_matches(self):
        pts, desc = _matchable_set()
        lazy = pts.copy()
        lazy[5, 0] += 5.0
        idx = _MultiIndex([(1.0, lazy, desc)])

        self.engine.store = _CfgStore({"ransac_px": 2.0, "ransac_fallback_px": 0.0})
        tight, _ = self._run(idx)
        self.engine.store = _CfgStore({"ransac_px": 6.0, "ransac_fallback_px": 0.0})
        loose, _ = self._run(idx)

        assert tight is not None and loose is not None
        self.assertEqual(tight["inl"], 7)
        self.assertEqual(loose["inl"], 8)

    def test_fallback_recovers_frames_the_tight_threshold_drops(self):
        pts, desc = _matchable_set()
        lazy = pts.copy()
        offsets = {3: (5.0, 0.0), 4: (-5.0, 2.0), 5: (0.0, 5.0), 6: (4.0, -4.0), 7: (-4.0, -4.0)}
        for i, (dx, dy) in offsets.items():
            lazy[i, 0] += dx
            lazy[i, 1] += dy
        idx = _MultiIndex([(1.0, lazy, desc)])
        thr = dict(ratio=0.8, min_inl=6, min_inl_rate=0.0)

        self.engine.store = _CfgStore({"ransac_px": 2.0, "ransac_fallback_px": 0.0})
        pose, _ = self._run(idx, thr=thr)
        self.assertIsNone(pose)

        self.engine.store = _CfgStore({"ransac_px": 2.0, "ransac_fallback_px": 6.0})
        pose, _ = self._run(idx, thr=thr)
        self.assertIsNotNone(pose)


class TestTileGatherCache(unittest.TestCase):
    def _index(self):
        from autopilot.vision.featureindex import FeatureIndex

        class _Data(dict):
            @property
            def files(self):
                return list(self.keys())

        pts = np.array([[0.0, 0.0], [10.0, 0.0], [600.0, 0.0]], np.float32)
        desc = np.zeros((3, 128), np.float32)
        tile = np.array([0, 0, 1], np.int32)  # tx=1 in row 0
        data = _Data(
            ms=np.array(2.0),
            mu_h=np.array(1024),
            mu_w=np.array(1024),
            tile=np.array(512),
            levels=np.array([1.0], np.float32),
            norm=np.array(["raw"]),
            pts_lv0=pts,
            desc_lv0=desc,
            tile_lv0=tile,
        )
        return FeatureIndex("t", data)

    def test_same_tile_rect_reuses_the_gather(self):
        idx = self._index()
        li = idx._levels[0]

        first = idx.radius_candidates(5.0, 5.0, 100.0)
        cached = li["_tile_cache"]
        second = idx.radius_candidates(5.0, 5.0, 100.0)

        self.assertIsNotNone(cached)
        self.assertIs(li["_tile_cache"], cached)
        self.assertEqual(len(first[0][1]), len(second[0][1]))
        np.testing.assert_array_equal(first[0][1], second[0][1])

    def test_moving_center_updates_the_distance_filter(self):
        idx = self._index()

        near = idx.radius_candidates(0.0, 0.0, 20.0)
        farther = idx.radius_candidates(10.0, 0.0, 5.0)

        self.assertEqual(len(near[0][1]), 2)  # (0,0) and (10,0)
        self.assertEqual(len(farther[0][1]), 1)  # only (10,0)

    def test_far_tile_is_not_cached_into_the_query(self):
        idx = self._index()

        found = idx.radius_candidates(600.0, 0.0, 20.0)

        self.assertEqual(len(found[0][1]), 1)
        np.testing.assert_array_equal(found[0][1][0], [600.0, 0.0])


class TestTrackerFrameError(unittest.TestCase):
    """A per-frame locator error must not kill the tracking thread."""

    def test_locator_frame_error_degrades_to_bad_frame(self):
        loc = LiveLocator(cfg=AppConfig(), mask=None)
        mm = np.zeros((32, 32), np.uint8)

        with patch.object(tracker_mod.locator, "global_pose", side_effect=RuntimeError("boom")):
            pose, diag = loc._frame_pose(mm, None, time.time())

        self.assertIsNone(pose)
        self.assertEqual(diag["reject"], "locator_error")
        self.assertIn("boom", diag["detail"])


class TestXfeatRingDiagnostics(unittest.TestCase):
    """A rotation ring that found nothing must report why, not reject=None."""

    def _call(self, engine, **overrides):
        kwargs = dict(
            mm=np.zeros((32, 32), np.uint8),
            ui_mask=None,
            idx=object(),
            cx=None,
            cy=None,
            min_inl=2,
            budget=3.0,
            t0=time.time(),
            progress=None,
            prev_s=None,
            ms=1.0,
            max_kp=1000,
            diag={"reject": None, "detail": ""},
            debug=True,
        )
        kwargs.update(overrides)
        return engine._localize_xfeat(**kwargs)

    def test_all_attempts_failed_reports_the_last_attempt_diag(self):
        engine = locator.MapLocator()
        engine._last_th = None
        angles: list[float] = []

        def _fail(_mm, _ui, _idx, _cx, _cy, **kw):
            angles.append(kw["derotate_deg"])
            kw["diag"]["reject"] = "index_no_match"
            return None

        with patch.object(engine, "_localize_index", side_effect=_fail):
            pose, diag = self._call(engine)

        self.assertIsNone(pose)
        self.assertIsNone(engine._last_th)
        self.assertEqual(len(angles), 8)
        self.assertEqual(diag["reject"], "index_no_match")
        self.assertEqual(diag["derot"], 315.0)

    def test_expired_budget_reports_budget_timeout_and_skips_the_ring(self):
        engine = locator.MapLocator()
        engine._last_th = None

        with patch.object(engine, "_localize_index") as localize:
            pose, diag = self._call(engine, budget=0.5, t0=time.time() - 1.0)

        self.assertIsNone(pose)
        self.assertEqual(diag["reject"], "budget_timeout")
        self.assertIn("budget", diag["detail"])
        localize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
