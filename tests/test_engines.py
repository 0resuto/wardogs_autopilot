"""Tests for the shipped localization engine: the hybrid ECC tracker."""

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
from autopilot.vision.hybrid import MAX_ANCHOR_MISSES, HybridLocalizer  # noqa: E402
from autopilot.vision.tracker import LiveLocator  # noqa: E402


def _texture(seed: int = 0, size: int = 336) -> np.ndarray:
    """Blobby high-contrast texture suitable for SIFT detection."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 60, (size, size), np.uint8)
    for _ in range(70):
        x, y = (int(v) for v in rng.integers(4, size - 40, 2))
        w, h = (int(v) for v in rng.integers(10, 35, 2))
        img[y : y + h, x : x + w] = int(rng.integers(120, 256))
    for _ in range(40):
        c = (int(rng.integers(10, size - 10)), int(rng.integers(10, size - 10)))
        cv2.circle(img, c, int(rng.integers(3, 9)), int(rng.integers(80, 256)), -1)
    return img


class _FakeIndex:
    """Single (level, pts, desc) candidate set."""

    def __init__(self, pts: np.ndarray, desc: np.ndarray) -> None:
        self._entry = (0, pts, desc)

    def global_candidates(self, _cap: int):
        return [self._entry]

    def radius_candidates(self, _cx: float, _cy: float, _r: float):
        return [self._entry]


class _FakeStore:
    def __init__(self, cfg: dict | None = None, name: str = "test") -> None:
        self._cfg = dict(cfg or {})
        self._name = name

    def loc_cfg(self) -> dict:
        return self._cfg

    def map_name(self) -> str:
        return self._name

    def mini_scale(self) -> float:
        return 1.0


class _FakeAnchor:
    def __init__(self, pose: dict) -> None:
        self.pose = dict(pose)
        self.calls = 0

    def global_pose(self, _mm, _mask, **kwargs):
        self.calls += 1
        pose = dict(self.pose) if self.pose is not None else None
        if kwargs.get("debug"):
            return pose, dict(mode="index", reject=None, detail="OK (fake anchor)")
        return pose


class TestLocatorAnchor(unittest.TestCase):
    def test_map_locator_is_sift(self):
        engine = locator.MapLocator()

        self.assertIsInstance(engine.detector, cv2.SIFT)
        self.assertEqual(engine.kind, "sift")

    def test_sift_pose_round_trip_unchanged(self):
        img = _texture(seed=2)
        engine = locator.MapLocator()
        kp, desc = engine.detector.detectAndCompute(img, None)
        self.assertIsNotNone(desc)
        assert desc is not None
        src = np.array([k.pt for k in kp], np.float32)
        idx = _FakeIndex(src + np.array([60.0, -30.0], np.float32), np.asarray(desc, np.uint8))
        pose, diag = engine._pose_via_index(
            img,
            None,
            idx,
            0.0,
            0.0,
            450.0,
            thr=dict(ratio=0.8, min_inl=8, min_inl_rate=0.0),
            feats=(kp, np.asarray(desc, np.uint8)),
        )
        self.assertIsNotNone(pose, diag)
        assert pose is not None
        self.assertAlmostEqual(pose["t"][0], 60.0, delta=2.0)
        self.assertAlmostEqual(pose["t"][1], -30.0, delta=2.0)


class TestEngineFacade(unittest.TestCase):
    def test_set_engine_keeps_the_only_engine(self):
        locator.set_engine("hybrid")
        self.assertEqual(locator.engine(), "hybrid")
        self.assertIsInstance(locator._active_engine(), HybridLocalizer)

        locator.set_engine("garbage")
        self.assertEqual(locator.engine(), "hybrid")
        self.assertIsInstance(locator._active_engine(), HybridLocalizer)


class TestTrackerEngineSync(unittest.TestCase):
    """LiveLocator must apply locator.engine from the config and reset state."""

    def test_tracker_applies_engine_from_config(self):
        calls = {"n": 0}

        def fake_pose(_mm, _mask, **_kwargs):
            calls["n"] += 1
            pose = dict(map_x=100.0, map_y=200.0, th=0.0, s=1.0, inl=50)
            return pose, dict(reject=None, detail="OK (fake)")

        cfg = AppConfig()
        cfg.capture.fps = 100
        cfg.locator.engine = "hybrid"
        loc = LiveLocator(
            cfg=cfg,
            mask=np.zeros((64, 64), bool),
            frame_source=lambda: np.zeros((64, 64, 3), np.uint8),
        )
        try:
            with (
                patch.object(tracker_mod.locator, "load_global_map", lambda: None),
                patch.object(tracker_mod.locator, "global_pose", fake_pose),
            ):
                loc.start()
                deadline = time.time() + 5.0
                while calls["n"] < 3 and time.time() < deadline:
                    time.sleep(0.01)
                loc.stop()
                loc.join(timeout=2.0)

            self.assertGreaterEqual(calls["n"], 3, "tracker produced no poses")
            self.assertEqual(loc._engine, "hybrid")
            self.assertEqual(locator.engine(), "hybrid")
        finally:
            locator.set_engine("hybrid")


class TestHybridMath(unittest.TestCase):
    def test_pose_matrix_round_trip(self):
        store = _FakeStore()
        h = HybridLocalizer(anchor=None, store=store)
        mm = np.zeros((100, 200), np.uint8)
        pose = dict(s=1.0, th=30.0, map_x=500.0, map_y=700.0, inl=10, n_match=20)

        M = h._M_from_pose(pose, mm, store.loc_cfg())
        back = h._pose_from_M(M, mm, store.loc_cfg())

        self.assertAlmostEqual(back["s"], 1.0, delta=1e-9)
        self.assertAlmostEqual(back["th"], 30.0, delta=1e-9)
        self.assertAlmostEqual(back["map_x"], 500.0, delta=1e-9)
        self.assertAlmostEqual(back["map_y"], 700.0, delta=1e-9)

    def test_ecc_recovers_frame_to_frame_warp(self):
        prev = _texture(seed=3)
        W = cv2.getRotationMatrix2D((168.0, 168.0), 1.2, 1.0)
        W[0, 2] += 1.5
        W[1, 2] += 0.7
        curr = cv2.warpAffine(prev, W, (336, 336), flags=cv2.INTER_LINEAR)

        h = HybridLocalizer(anchor=None, store=_FakeStore())
        h._prev_pp = prev
        ok, warp, cc, why = h._track(curr)

        self.assertTrue(ok, why)
        assert warp is not None
        self.assertGreater(cc, 0.9)
        pts = np.array([[80, 80, 1], [168, 168, 1], [260, 250, 1]], np.float64).T
        err = np.max(np.linalg.norm((warp @ pts)[:2].T - (W @ pts)[:2].T, axis=1))
        self.assertLess(err, 1.0)


def _hybrid_step(h: HybridLocalizer, frame: np.ndarray, prev: tuple[float, float] | None = None):
    """Debug-mode hybrid step with the union return narrowed to (pose, diag)."""
    res = h.global_pose(frame, np.zeros(frame.shape, bool), prev_xy=prev, debug=True)
    assert isinstance(res, tuple)
    return res


class TestHybridFlow(unittest.TestCase):
    ANCHOR_POSE = dict(s=1.0, th=0.0, map_x=1000.0, map_y=2000.0, inl=30, n_match=40)

    def test_track_updates_position_without_lag(self):
        anchor = _FakeAnchor(self.ANCHOR_POSE)
        h = HybridLocalizer(anchor=anchor, store=_FakeStore({"hybrid_reanchor_s": 5.0}))
        prev = _texture(seed=4)
        dx, dy = 4.0, 2.0
        W = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
        curr = cv2.warpAffine(prev, W, (336, 336), flags=cv2.INTER_LINEAR)

        pose0, _ = _hybrid_step(h, prev)
        self.assertEqual(anchor.calls, 1)
        pose1, diag1 = _hybrid_step(h, curr, prev=(1000.0, 2000.0))

        assert pose0 is not None and pose1 is not None
        self.assertEqual(anchor.calls, 1, "no re-anchor inside the period")
        self.assertEqual(diag1.get("mode"), "hybrid")
        self.assertFalse(diag1.get("anchor"))
        # features move +dx/+dy, so the centre now shows the map point that
        # was dx/dy behind: the reported centre moves by -dx/-dy
        self.assertAlmostEqual(pose1["map_x"], 1000.0 - dx, delta=0.5)
        self.assertAlmostEqual(pose1["map_y"], 2000.0 - dy, delta=0.5)
        self.assertAlmostEqual(pose1["th"], 0.0, delta=0.5)

    def test_period_forces_anchor(self):
        anchor = _FakeAnchor(self.ANCHOR_POSE)
        h = HybridLocalizer(anchor=anchor, store=_FakeStore({"hybrid_reanchor_s": 1.0}))
        frame = _texture(seed=5)
        _hybrid_step(h, frame)
        h._anchor_t -= 5.0  # pretend the anchor is old
        _hybrid_step(h, frame, prev=(1000.0, 2000.0))
        self.assertEqual(anchor.calls, 2)

    def test_low_cc_triggers_anchor(self):
        anchor = _FakeAnchor(self.ANCHOR_POSE)
        h = HybridLocalizer(anchor=anchor, store=_FakeStore({"hybrid_reanchor_s": 10.0}))
        frame = _texture(seed=6)
        other = np.full_like(frame, 127)
        _hybrid_step(h, frame)
        pose, diag = _hybrid_step(h, other, prev=(1000.0, 2000.0))
        self.assertEqual(anchor.calls, 2)
        self.assertIsNotNone(pose)
        self.assertTrue(diag.get("anchor"))
        self.assertEqual(diag.get("engine"), "hybrid")

    def test_anchor_failure_clears_the_track(self):
        anchor = _FakeAnchor(dict(self.ANCHOR_POSE))
        anchor.pose = None  # type: ignore[assignment]
        h = HybridLocalizer(anchor=anchor, store=_FakeStore({"hybrid_reanchor_s": 10.0}))
        frame = _texture(seed=7)
        pose, diag = _hybrid_step(h, frame)
        self.assertIsNone(pose)
        self.assertIsNone(h._prev_pp)
        self.assertIsNone(h._M)
        self.assertTrue(diag.get("anchor"))
        self.assertEqual(diag.get("engine"), "hybrid")

    def test_anchor_miss_keeps_a_healthy_track(self):
        anchor = _FakeAnchor(dict(self.ANCHOR_POSE))
        h = HybridLocalizer(anchor=anchor, store=_FakeStore({"hybrid_reanchor_s": 10.0}))
        frame = _texture(seed=8)
        _hybrid_step(h, frame)
        anchor.pose = None  # type: ignore[assignment]
        h._anchor_t -= 20.0  # force the periodic anchor

        pose, diag = _hybrid_step(h, frame, prev=(1000.0, 2000.0))

        self.assertIsNotNone(pose)
        self.assertFalse(diag.get("anchor"))
        self.assertEqual(diag.get("mode"), "hybrid")
        self.assertIsNotNone(h._M)
        self.assertIsNotNone(h._prev_pp)

    def test_anchor_misses_are_bounded(self):
        anchor = _FakeAnchor(dict(self.ANCHOR_POSE))
        h = HybridLocalizer(anchor=anchor, store=_FakeStore({"hybrid_reanchor_s": 10.0}))
        frame = _texture(seed=9)
        _hybrid_step(h, frame)
        anchor.pose = None  # type: ignore[assignment]

        for miss in range(MAX_ANCHOR_MISSES):
            h._anchor_t -= 20.0
            pose, _diag = _hybrid_step(h, frame, prev=(1000.0, 2000.0))
            self.assertIsNotNone(pose, "the track must bridge miss %d" % (miss + 1))

        h._anchor_t -= 20.0
        pose, diag = _hybrid_step(h, frame, prev=(1000.0, 2000.0))
        self.assertIsNone(pose)
        self.assertIsNone(h._M)
        self.assertIsNone(h._prev_pp)

    def test_track_drift_past_the_cap_forces_an_anchor(self):
        anchor = _FakeAnchor(self.ANCHOR_POSE)
        h = HybridLocalizer(anchor=anchor, store=_FakeStore({"hybrid_reanchor_s": 10.0}))
        frame = _texture(seed=10)
        _hybrid_step(h, frame)
        h._anchor_th = 180.0  # the ECC heading (0) now contradicts the anchor

        pose, diag = _hybrid_step(h, frame, prev=(1000.0, 2000.0))

        self.assertEqual(anchor.calls, 2)
        self.assertTrue(diag.get("anchor"))
        self.assertIsNotNone(pose)


if __name__ == "__main__":
    unittest.main()
