"""Tests for the synthetic benchmark: scenarios, sweeps, tracker, runner, report.

No GPU, no ONNX models and no map/index files are needed: the scenario
geometry runs on a synthetic textured preview and the engine call is
monkeypatched, so the tracker rules, the metrics and the writers are checked in
isolation.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.bench import (  # noqa: E402
    ENGINE_PARAMS,
    SHARED_PARAMS,
    BenchTracker,
    ConfigOverlay,
    RunResult,
    Scenario,
    ScenarioSpec,
    available_engines,
    build_scenario,
    default_specs,
    expand_oat,
    param_registry,
    parse_sweep,
    run_case,
    spec_for,
    write_csv,
    write_jsonl,
    write_report,
)
from autopilot.bench.report import swept_keys  # noqa: E402
from autopilot.bench.runner import CSV_COLUMNS  # noqa: E402
from autopilot.bench.scenario import _build_frame, _pick_center  # noqa: E402
from autopilot.vision import locator  # noqa: E402
from autopilot.vision import xfeat as xfeat_mod  # noqa: E402

#: Native px per meter used by the synthetic maps in these tests.
PX_PER_M = 2.0

#: Textured stand-ins for a map (mu) and its preview level (native/2).
_MU = _TEXTURE = None
_PREVIEW = _PREVIEW_TEXTURE = None


def _texture(seed: int = 0, size: int = 4096) -> np.ndarray:
    """A blocky, high-contrast preview stand-in for a map mipmap level."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 70, (size, size), np.uint8)
    for _ in range(900):
        x, y = (int(v) for v in rng.integers(0, size - 60, 2))
        w, h = (int(v) for v in rng.integers(8, 60, 2))
        img[y : y + h, x : x + w] = int(rng.integers(110, 256))
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _mu() -> np.ndarray:
    global _MU, _TEXTURE
    if _MU is None:
        _MU = _TEXTURE = _texture(seed=7, size=2048)
    return _MU


def _preview() -> np.ndarray:
    global _PREVIEW, _PREVIEW_TEXTURE
    if _PREVIEW is None:
        _PREVIEW = _PREVIEW_TEXTURE = _texture(seed=7, size=1024)
    return _PREVIEW


def _tiny_scenario(spec: ScenarioSpec | None = None) -> Scenario:
    """A few frames of the 'straight' profile over a small synthetic map."""
    spec = spec or ScenarioSpec(name="straight", frames=4, fps=10.0, speed_mps=15.0)
    mu = _mu()
    return build_scenario(
        spec,
        mu,
        _preview(),
        ms=2.0,
        center_mu=(mu.shape[1] / 2.0, mu.shape[0] / 2.0),
        px_per_m=PX_PER_M,
    )


def _frozen_scenario(frames: int = 5, xy: tuple[float, float] = (1000.0, 2000.0)) -> Scenario:
    """A scenario whose ground truth does not move (so scripted poses can match)."""
    scen = _tiny_scenario(ScenarioSpec(name="straight", frames=frames))
    truth = [(xy[0], xy[1], 0.0)] * frames
    return Scenario(spec=scen.spec, frames=scen.frames, truth=truth, dt=scen.dt, mask=scen.mask)


class TestScenario(unittest.TestCase):
    def test_frames_and_truth_line_up(self):
        scen = _tiny_scenario()

        self.assertEqual(len(scen.frames), len(scen.truth))
        self.assertEqual(len(scen.frames), 4)
        for frame in scen.frames:
            self.assertEqual(frame.dtype, np.uint8)
            self.assertEqual(frame.shape[:2], (277, 336))
            self.assertTrue(np.isfinite(frame).all())

    def test_straight_leg_steps_by_speed_times_dt(self):
        spec = ScenarioSpec(name="straight", frames=6, fps=10.0, speed_mps=15.0)
        scen = _tiny_scenario(spec)

        step = spec.speed_mps * (1.0 / spec.fps) * PX_PER_M
        for k in range(len(scen.truth) - 1):
            dx = scen.truth[k + 1][0] - scen.truth[k][0]
            dy = scen.truth[k + 1][1] - scen.truth[k][1]
            self.assertAlmostEqual(float(np.hypot(dx, dy)), step, places=6)

    def test_heading_only_turns_after_the_straight_leg(self):
        spec = ScenarioSpec(name="turns", frames=40, fps=10.0, yaw_deg_s=12.0, straight_s=2.0)
        scen = _tiny_scenario(spec)

        headings = [t[2] for t in scen.truth]
        self.assertEqual(headings[:20], [headings[0]] * 20, "straight leg must not turn")
        self.assertGreater(headings[-1], headings[0], "the yaw leg must rotate")

    def test_mixed_s_alternates_the_yaw_sign(self):
        spec = ScenarioSpec(name="mixed_s", frames=80, fps=10.0, yaw_deg_s=30.0, straight_s=1.0)
        scen = _tiny_scenario(spec)

        headings = np.unwrap(np.radians([t[2] for t in scen.truth]))
        rate = np.diff(headings) * 10.0 * 180.0 / np.pi
        after = rate[10:]
        self.assertTrue((after > 0).any() and (after < 0).any(), "yaw sign must flip")

    def test_void_frames_are_flat(self):
        spec = ScenarioSpec(name="void", frames=6, void_frames=(2, 4))
        scen = _tiny_scenario(spec)

        for k in (2, 4):
            self.assertLess(float(scen.frames[k].std()), 1.0, "void frame is not flat")
        self.assertGreater(float(scen.frames[1].std()), 1.0)

    def test_same_seed_is_deterministic(self):
        spec = ScenarioSpec(name="turns", frames=8, seed=5)
        a = _tiny_scenario(spec)
        b = _tiny_scenario(spec)

        for fa, fb in zip(a.frames, b.frames, strict=True):
            np.testing.assert_array_equal(fa, fb)

    def test_different_seed_changes_the_noise_only(self):
        a = _tiny_scenario(ScenarioSpec(name="straight", frames=3, seed=1))
        b = _tiny_scenario(ScenarioSpec(name="straight", frames=3, seed=2))
        c = _tiny_scenario(ScenarioSpec(name="straight", frames=3, seed=1))

        self.assertFalse(np.array_equal(a.frames[0], b.frames[0]))
        np.testing.assert_array_equal(a.frames[0], c.frames[0])
        np.testing.assert_array_equal(a.truth, b.truth)

    def test_ui_mask_paints_the_overlay(self):
        mask = np.zeros((277, 336), bool)
        mask[:40, :40] = True
        clean_spec = ScenarioSpec(
            name="straight", frames=2, blur_sigma=0.0, noise_sigma=0.0, seed=7
        )
        kwargs: dict[str, Any] = {
            "ms": 2.0,
            "center_mu": (1024.0, 1024.0),
            "mask": mask,
            "px_per_m": PX_PER_M,
        }
        clean = build_scenario(clean_spec, _mu(), _preview(), **kwargs)

        scen = build_scenario(replace(clean_spec, ui_mask=True), _mu(), _preview(), **kwargs)

        self.assertIsNotNone(scen.mask)
        assert scen.mask is not None
        self.assertTrue(scen.mask.any())
        self.assertEqual(scen.mask.shape, (277, 336), "the mask is resized to the frame")
        fill = int(round(float(np.median(clean.frames[0]))))
        np.testing.assert_array_equal(scen.frames[0][:40, :40], np.full((40, 40), fill))
        self.assertNotEqual(
            int(clean.frames[0][20, 20]),
            fill,
            "the patch must differ from the fill so the overlay is visible",
        )

    def test_build_frame_convention_puts_the_center_on_the_truth(self):
        frame = _build_frame(_preview(), (800.0, 600.0), 30.0, 2.0)

        self.assertEqual(frame.shape, (277, 336))
        # preview = native/2, so a native center of (800, 600) is a preview
        # center of (400, 300) and the frame must be cut around it
        self.assertEqual(frame[277 // 2, 336 // 2].tolist(), _preview()[300, 400].tolist())

    def test_pick_center_finds_a_textured_window(self):
        mu = _texture(seed=11, size=4096)
        center = _pick_center(mu)

        assert center is not None
        self.assertEqual(len(center), 2)

    def test_default_specs_quick_and_full(self):
        quick = [s.name for s in default_specs(quick=True)]
        full = [s.name for s in default_specs(quick=False)]

        self.assertEqual(quick, ["straight", "turns", "void"])
        self.assertEqual(full, ["straight", "turns", "void", "mixed_s"])
        self.assertTrue(all(s.frames == 60 for s in default_specs(quick=True)))
        self.assertTrue(all(s.frames == 150 for s in default_specs(quick=False)))

    def test_spec_for_replaces_fields_and_rejects_unknown_names(self):
        spec = spec_for("turns", frames=17, fps=25.0)

        self.assertEqual(spec.name, "turns")
        self.assertEqual(spec.frames, 17)
        self.assertEqual(spec.fps, 25.0)
        self.assertEqual(spec_for("void").void_frames, (20, 35, 50))
        with self.assertRaises(ValueError):
            spec_for("nope")


class _FakeStore:
    """Minimal stand-in for MapStore.loc_cfg."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self._cfg = dict(cfg or {})
        self.calls = 0

    def loc_cfg(self) -> dict[str, Any]:
        self.calls += 1
        return dict(self._cfg)


class TestSweep(unittest.TestCase):
    def test_parse_sweep_reads_the_registry_types(self):
        key, values = parse_sweep("xfeat_min_cos=0.75,0.82,0.88")

        self.assertEqual(key, "xfeat_min_cos")
        self.assertEqual(values, [0.75, 0.82, 0.88])
        self.assertTrue(all(isinstance(v, float) for v in values))

        key, values = parse_sweep("max_kp_frame=600,1200")
        self.assertEqual(key, "max_kp_frame")
        self.assertEqual(values, [600, 1200])
        self.assertTrue(all(isinstance(v, int) for v in values))

    def test_parse_sweep_accepts_a_single_value(self):
        self.assertEqual(parse_sweep("vote_need=3"), ("vote_need", [3]))

    def test_unknown_key_lists_the_valid_keys(self):
        with self.assertRaises(ValueError) as ctx:
            parse_sweep("nonsense=1")

        message = str(ctx.exception)
        self.assertIn("nonsense", message)
        for key in ("ratio_local", "xfeat_min_cos", "hybrid_reanchor_s", "smooth_alpha"):
            self.assertIn(key, message)

    def test_malformed_sweep_is_rejected(self):
        for expr in ("ratio_local", "ratio_local=", "max_kp_frame=a,b"):
            with self.assertRaises(ValueError, msg=expr):
                parse_sweep(expr)

    def test_expand_oat_is_a_cartesian_product(self):
        grid = expand_oat([("xfeat_min_cos", [0.75, 0.82]), ("xfeat_top_k", [500, 1000, 2000])], {})

        self.assertEqual(len(grid), 6)
        self.assertIn({"xfeat_min_cos": 0.75, "xfeat_top_k": 500}, grid)
        self.assertIn({"xfeat_min_cos": 0.82, "xfeat_top_k": 2000}, grid)

    def test_expand_oat_carries_the_base_overrides(self):
        grid = expand_oat([("vote_need", [1, 3])], {"smooth_alpha": 0.35})

        self.assertEqual(len(grid), 2)
        for cfg in grid:
            self.assertEqual(cfg["smooth_alpha"], 0.35)
            self.assertIn(cfg["vote_need"], (1, 3))

    def test_expand_oat_without_sweeps_returns_the_base_once(self):
        self.assertEqual(expand_oat([], {"a": 1}), [{"a": 1}])
        self.assertEqual(expand_oat([]), [{}])

    def test_registry_covers_every_engine(self):
        for engine in ("sift", "orb", "xfeat", "hybrid"):
            registry = param_registry(engine)
            for spec in ENGINE_PARAMS[engine]:
                self.assertIn(spec.key, registry, engine)
                self.assertTrue(spec.values and spec.label)
        for spec in SHARED_PARAMS:
            self.assertIn(spec.key, param_registry("sift"))
            self.assertIn(spec.key, param_registry("xfeat"))


class TestConfigOverlay(unittest.TestCase):
    def test_overlay_merges_and_restores(self):
        store = _FakeStore({"ratio": 0.8, "min_inl": 4})

        with ConfigOverlay(store, {"ratio": 0.9}) as overlay:
            self.assertIs(overlay.store, store)
            self.assertEqual(store.loc_cfg(), {"ratio": 0.9, "min_inl": 4})

        self.assertEqual(store.loc_cfg(), {"ratio": 0.8, "min_inl": 4})
        self.assertNotIn("loc_cfg", vars(store), "the shadow must be removed")
        self.assertEqual(store.calls, 2)

    def test_overlay_restores_after_an_exception(self):
        store = _FakeStore({"ratio": 0.8})

        with self.assertRaises(RuntimeError), ConfigOverlay(store, {"ratio": 0.5}):
            self.assertEqual(store.loc_cfg()["ratio"], 0.5)
            raise RuntimeError("boom")

        self.assertEqual(store.loc_cfg(), {"ratio": 0.8})
        self.assertNotIn("loc_cfg", vars(store))

    def test_overlay_ignores_none_and_the_base_method_keeps_working(self):
        store = _FakeStore({"a": 1})

        with ConfigOverlay(store, {"a": None, "b": 2}):
            self.assertEqual(store.loc_cfg(), {"a": 1, "b": 2})

        self.assertEqual(store.loc_cfg(), {"a": 1})

    def test_nested_overlays_restore_in_order(self):
        store = _FakeStore({"a": 1})

        with ConfigOverlay(store, {"a": 2, "b": 3}):
            with ConfigOverlay(store, {"a": 9}):
                self.assertEqual(store.loc_cfg(), {"a": 9, "b": 3})
            self.assertEqual(store.loc_cfg(), {"a": 2, "b": 3}, "the outer shadow must survive")

        self.assertEqual(store.loc_cfg(), {"a": 1})
        self.assertNotIn("loc_cfg", vars(store))


def _pose(map_x: float, map_y: float, th: float = 0.0, inl: int = 50, **extra: Any):
    out = {
        "s": 1.0,
        "th": th,
        "t": (0.0, 0.0),
        "inl": inl,
        "n_match": inl * 2,
        "map_x": float(map_x),
        "map_y": float(map_y),
    }
    out.update(extra)
    return out


class _ScriptedEngine:
    """Patches locator.global_pose with a scripted pose list."""

    def __init__(self, script: list[Any], anchor_on: set[int] | None = None) -> None:
        self.script = list(script)
        self.anchor_on = anchor_on or set()
        self.calls: list[dict[str, Any]] = []
        self._patcher: Any = None

    def __enter__(self) -> _ScriptedEngine:
        def fake(mm, ui_mask, **kwargs):
            idx = len(self.calls)
            self.calls.append(kwargs)
            item = self.script[min(idx, len(self.script) - 1)]
            if isinstance(item, dict) and item.get("boom"):
                raise RuntimeError("engine exploded")
            if item is None:
                return None, {"reject": "no_match_local", "detail": "fake miss", "anchor": False}
            pose = _pose(**{k: v for k, v in item.items() if k not in ("anchor", "boom")})
            return pose, {
                "reject": None,
                "detail": "ok",
                "anchor": idx in self.anchor_on,
            }

        self._patcher = patch.object(locator, "global_pose", fake)
        self._patcher.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._patcher is not None:
            self._patcher.stop()

    def step_all(
        self,
        bench: BenchTracker,
        scenario: Scenario,
        truth: tuple[float, float, float] | None = None,
    ) -> list:
        return [
            bench.step(
                k,
                scenario.frames[k],
                scenario.mask,
                k * scenario.dt,
                truth if truth is not None else scenario.truth[k],
            )
            for k in range(len(scenario.frames))
        ]


def _tracker(engine: str = "sift", **cfg: Any) -> BenchTracker:
    base: dict[str, Any] = {"smooth_alpha": 0.0, "jump_gate_px": 3000, "vote_need": 3}
    base.update(cfg)
    return BenchTracker(engine, base, budget=1.0)


class TestBenchTracker(unittest.TestCase):
    def test_perfect_sequence_publishes_every_frame(self):
        bench = _tracker()
        scenario = _tiny_scenario()

        with _ScriptedEngine([_pose(1000.0, 2000.0)]) as engine:
            recs = engine.step_all(bench, scenario, (1000.0, 2000.0, 0.0))

        self.assertEqual(len(recs), 4)
        for rec in recs:
            self.assertTrue(rec.new_sample, f"frame {rec.idx}")
            self.assertEqual(rec.err_px, 0.0)
            self.assertEqual(rec.raw_err_px, 0.0)

    def test_raw_error_equals_published_error_without_smoothing(self):
        bench = _tracker(smooth_alpha=0.0)

        with _ScriptedEngine([_pose(1000.0, 2000.0)]) as engine:
            rec = engine.step_all(bench, _tiny_scenario(), (1004.0, 2000.0, 0.0))[0]

        self.assertAlmostEqual(rec.err_px, 4.0)
        self.assertEqual(rec.err_px, rec.raw_err_px)

    def test_vote_gate_rejects_a_lone_jump_and_accepts_the_third(self):
        bench = _tracker(jump_gate_px=300.0, vote_need=3, smooth_alpha=0.0)
        script = [
            _pose(1000.0, 2000.0, inl=10),
            _pose(9000.0, 9000.0, inl=10),
            _pose(9010.0, 9010.0, inl=10),
            _pose(9005.0, 9005.0, inl=10),
        ]

        with _ScriptedEngine(script) as engine:
            recs = engine.step_all(bench, _frozen_scenario(4))

        self.assertTrue(recs[0].new_sample)
        self.assertTrue(recs[1].vote_reject and not recs[1].new_sample)
        self.assertEqual(recs[1].reject, "vote_reject")
        self.assertTrue(recs[2].vote_reject, "two candidates cannot decide a 3-frame vote")
        self.assertTrue(recs[3].new_sample, "three agreeing candidates must publish")

    def test_strong_candidate_skips_the_vote(self):
        bench = _tracker(jump_gate_px=300.0, vote_inl_skip=40, smooth_alpha=0.0)
        script = [_pose(1000.0, 2000.0), _pose(9000.0, 9000.0, inl=41)]

        with _ScriptedEngine(script) as engine:
            recs = engine.step_all(bench, _frozen_scenario(2))

        self.assertTrue(recs[1].new_sample)
        self.assertFalse(recs[1].vote_reject)

    def test_heading_gate_triggers_a_vote(self):
        bench = _tracker(heading_gate_deg=20.0, vote_need=2, smooth_alpha=0.0)
        script = [_pose(1000.0, 2000.0, th=0.0), _pose(1001.0, 2000.0, th=90.0, inl=1)]

        with _ScriptedEngine(script) as engine:
            recs = engine.step_all(bench, _frozen_scenario(2))

        self.assertTrue(recs[1].vote_reject)
        self.assertIn("heading", recs[1].detail)

    def test_hold_reuses_the_last_pose_then_stops(self):
        bench = _tracker(hold_frames=2, smooth_alpha=0.0)
        script = [_pose(1000.0, 2000.0), None, None, None]

        with _ScriptedEngine(script) as engine:
            recs = engine.step_all(bench, _frozen_scenario(4))

        self.assertTrue(recs[0].new_sample)
        self.assertEqual([r.hold for r in recs[1:]], [True, True, False])
        self.assertEqual(recs[1].pose["map_x"], 1000.0)
        self.assertFalse(recs[3].new_sample)
        self.assertIsNone(recs[3].pose)
        self.assertEqual(recs[1].reject, "hold")

    def test_hold_disabled_publishes_nothing(self):
        bench = _tracker(hold_frames=0, smooth_alpha=0.0)
        script = [_pose(1000.0, 2000.0), None, None]

        with _ScriptedEngine(script) as engine:
            recs = engine.step_all(bench, _frozen_scenario(3))

        self.assertEqual([r.hold for r in recs], [False, False, False])
        self.assertIsNone(recs[2].pose)

    def test_smoothing_low_passes_measurement_noise(self):
        """A jittery matcher must reach the consumers quieter than it measures."""
        bench = _tracker(smooth_alpha=0.5)
        jitter = [0.0, 3.0, -3.0, 2.0, -2.0, 4.0, -4.0, 1.0, -1.0, 2.0]
        script = [_pose(1000.0 + j, 2000.0) for j in jitter]

        with _ScriptedEngine(script) as engine:
            recs = engine.step_all(bench, _frozen_scenario(len(jitter)))

        raw = [r.raw_err_px or 0.0 for r in recs]
        pub = [r.err_px or 0.0 for r in recs]
        self.assertLess(sum(pub) / len(pub), sum(raw) / len(raw))
        self.assertNotEqual(pub, raw, "alpha=0.5 must not publish the raw match")

    def test_smoothing_converges_on_a_constant_velocity(self):
        """The alpha-beta filter reaches zero lag on steady motion."""
        bench = _tracker(smooth_alpha=0.5)
        script = [_pose(1000.0 + 10.0 * k, 2000.0) for k in range(40)]

        with _ScriptedEngine(script) as engine:
            recs = engine.step_all(bench, _frozen_scenario(40))

        raw_x = 1000.0 + 10.0 * (len(script) - 1)
        self.assertLess(
            abs((recs[-1].pose or {})["map_x"] - raw_x), 1.0, "steady-state lag vanishes"
        )
        self.assertLess(recs[-1].err_px or 0.0, recs[-1].raw_err_px or 0.0)

    def test_engine_errors_propagate(self):
        bench = _tracker()

        with _ScriptedEngine([{"boom": True}]), self.assertRaises(RuntimeError):
            bench.step(0, np.zeros((8, 8), np.uint8), None, 0.0, (0.0, 0.0, 0.0))

    def test_reset_clears_the_tracker_and_the_engine_state(self):
        bench = _tracker(smooth_alpha=0.0)

        class _Eng:
            _last_th: float | None = 42.0

        engine = _Eng()
        with _ScriptedEngine([_pose(1000.0, 2000.0)]):
            bench.step(0, np.zeros((8, 8), np.uint8), None, 0.0, (0.0, 0.0, 0.0))
        with patch.object(locator, "_active_engine", lambda: engine):
            bench.reset()
        self.assertIsNone(engine._last_th, "the derotation heading must be dropped")

        self.assertIsNone(bench._prev_xy)
        self.assertIsNone(bench._last_accepted)
        self.assertEqual(len(bench._vote_buf), 0)
        with _ScriptedEngine([_pose(1000.0, 2000.0)]):
            rec = bench.step(0, np.zeros((8, 8), np.uint8), None, 0.0, (0.0, 0.0, 0.0))
        self.assertTrue(rec.new_sample)

    def test_reset_calls_the_engine_reset_when_present(self):
        bench = _tracker()
        seen: list[int] = []

        class _Eng:
            def reset(self) -> None:
                seen.append(1)

        with patch.object(locator, "_active_engine", lambda: _Eng()):
            bench.reset()

        self.assertEqual(seen, [1])

    def test_realtime_pacing_is_excluded_from_the_latency(self):
        bench = BenchTracker("hybrid", {"smooth_alpha": 0.0}, realtime=True)
        bench._wall0 = time.perf_counter() - 0.3  # the scenario is 0.5 s in

        with _ScriptedEngine([_pose(1000.0, 2000.0)]):
            start = time.perf_counter()
            rec = bench.step(0, np.zeros((8, 8), np.uint8), None, 0.5, (0.0, 0.0, 0.0))
            slept = time.perf_counter() - start

        self.assertGreater(slept, 0.1, "the frame must be paced to the wall clock")
        self.assertLess(rec.latency_ms, 100.0, "the sleep must not count as engine latency")

    def test_realtime_off_does_not_sleep(self):
        bench = BenchTracker("sift", {"smooth_alpha": 0.0}, realtime=False)

        with _ScriptedEngine([_pose(1000.0, 2000.0)]):
            start = time.perf_counter()
            bench.step(0, np.zeros((8, 8), np.uint8), None, 10.0, (0.0, 0.0, 0.0))
            elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.05)


class _FakeStoreFull:
    """MapStore stand-in with everything run_case touches."""

    #: No hold: a void frame must show up as a plain reject, not a held pose.
    CFG = {"vote_need": 3, "vote_frames": 5, "smooth_alpha": 0.0, "hold_frames": 0}

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self._cfg = dict(cfg or self.CFG)
        self.name = "fake"

    def loc_cfg(self) -> dict[str, Any]:
        return dict(self._cfg)

    def map_name(self) -> str:
        return self.name

    def m_per_px(self, _name: str | None = None) -> float:
        return 0.5

    def mini_scale(self) -> float:
        return 2.0

    def set_map(self, name: str) -> None:
        self.name = name


def _indexes(present: set[str]):
    """Patch reporting which feature-index kinds the fake map has."""
    return patch(
        "autopilot.bench.runner._index_ready",
        side_effect=lambda _map, kind: kind in present,
    )


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.store = _FakeStoreFull()
        for target, attr, value in (
            (locator, "get_store", lambda: self.store),
            (locator, "set_engine", lambda _kind: None),
        ):
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _scenario(self, frames: int = 5) -> Scenario:
        return _frozen_scenario(frames)

    def test_run_case_aggregates_the_metrics(self):
        script = [
            _pose(1000.0, 2000.0),
            _pose(1002.0, 2004.0),
            None,
            _pose(1000.0, 2000.0),
            _pose(1000.0, 2000.0),
        ]
        with _ScriptedEngine(script):
            result, records = run_case("sift", self._scenario(), {}, "fake", warmup=False)

        mean_err = math.hypot(2.0, 4.0) / 4.0
        self.assertEqual(result.frames, 5)
        self.assertEqual(len(records), 5)
        self.assertAlmostEqual(result.localized_pct, 80.0)
        self.assertAlmostEqual(result.pos_err_px_mean, mean_err, places=3)
        self.assertAlmostEqual(result.pos_err_px_max, math.hypot(2.0, 4.0), places=3)
        self.assertAlmostEqual(result.pos_err_px_median, 0.0)
        self.assertAlmostEqual(result.pos_err_m_mean, 0.5 * mean_err, places=4)
        self.assertGreater(result.latency_ms_mean, 0.0)
        self.assertGreater(result.achieved_fps, 0.0)
        self.assertEqual(result.reject_reasons.get("no_match_local"), 1)
        self.assertEqual(result.holds, 0)
        self.assertEqual(result.anchors, 0)
        self.assertEqual(result.fps, self._scenario().spec.fps)
        self.assertEqual(result.speed_mps, self._scenario().spec.speed_mps)
        self.assertEqual(result.m_per_px, 0.5)

    def test_anchors_are_counted(self):
        with _ScriptedEngine([_pose(1000.0, 2000.0, anchor=True)], anchor_on={0, 2}):
            result, _ = run_case("sift", self._scenario(), {}, "fake", warmup=False)

        self.assertEqual(result.anchors, 2)

    def test_two_identical_runs_agree(self):
        """Same config + same engine => identical decisions frame by frame."""

        def scripted():
            state = {"n": 0}

            def fake(mm, ui_mask, **kwargs):
                state["n"] += 1
                return _pose(1000.0 + state["n"], 2000.0), {"reject": None, "detail": "ok"}

            return fake

        def decisions(records):
            return [{k: v for k, v in r.as_json().items() if k != "latency_ms"} for r in records]

        with patch.object(locator, "global_pose", scripted()):
            a, ra = run_case("sift", self._scenario(), {}, "fake", warmup=False)
        with patch.object(locator, "global_pose", scripted()):
            b, rb = run_case("sift", self._scenario(), {}, "fake", warmup=False)

        self.assertEqual(decisions(ra), decisions(rb))
        self.assertEqual(a.pos_err_px_mean, b.pos_err_px_mean)
        self.assertEqual(a.localized_pct, b.localized_pct)

    def test_warmup_is_measured_and_excluded(self):
        with _ScriptedEngine([_pose(1000.0, 2000.0)]):
            result, records = run_case("sift", self._scenario(), {}, "fake", warmup=True)

        self.assertGreater(result.warmup_ms, 0.0)
        self.assertEqual(len(records), 5, "the warm-up frame must not be scored")

    def test_overrides_reach_the_engine_config_and_are_restored(self):
        with _ScriptedEngine([_pose(1000.0, 2000.0)]) as engine:
            result, _ = run_case(
                "sift", self._scenario(frames=2), {"ransac_px": 2.0}, "fake", warmup=False
            )

        self.assertEqual(result.config, {"ransac_px": 2.0})
        self.assertNotIn("ransac_px", self.store.loc_cfg())
        self.assertNotIn("loc_cfg", vars(self.store))
        self.assertEqual(len(engine.calls), 2)

    def test_engine_failure_propagates_out_of_run_case(self):
        with _ScriptedEngine([{"boom": True}]), self.assertRaises(RuntimeError):
            run_case("sift", self._scenario(frames=2), {}, "fake", warmup=False)

    def test_available_engines_needs_every_index_it_uses(self):
        with (
            _indexes({"sift", "orb", "xfeat"}),
            patch("autopilot.bench.runner.xfeat_skip_reason", return_value=None),
        ):
            self.assertEqual(available_engines("m"), ["sift", "orb", "xfeat", "hybrid"])
        with (
            _indexes({"sift"}),
            patch("autopilot.bench.runner.xfeat_skip_reason", return_value="no GPU"),
        ):
            self.assertEqual(available_engines("m"), ["sift", "hybrid"])
        with _indexes({"orb"}):
            self.assertEqual(available_engines("m"), ["orb"])

    def test_xfeat_skip_reason_reports_every_missing_piece(self):
        from autopilot.bench import runner as runner_mod

        with _indexes(set()), patch.object(runner_mod, "_gpu_provider", return_value=True):
            self.assertIn("index missing", str(runner_mod.xfeat_skip_reason("m")))

        with (
            _indexes({"xfeat"}),
            patch.object(xfeat_mod, "available", lambda: False),
            patch.object(runner_mod, "_gpu_provider", return_value=True),
        ):
            self.assertIn("backbone model missing", str(runner_mod.xfeat_skip_reason("m")))

        with (
            _indexes({"xfeat"}),
            patch.object(xfeat_mod, "available", lambda: True),
            patch.object(xfeat_mod, "MATCH_MODEL_FILE", "missing-xfeat-match.onnx"),
            patch.object(runner_mod, "_gpu_provider", return_value=True),
        ):
            self.assertIn("match model missing", str(runner_mod.xfeat_skip_reason("m")))

        with (
            _indexes({"xfeat"}),
            patch.object(xfeat_mod, "available", lambda: True),
            patch.object(xfeat_mod, "MATCH_MODEL_FILE", os.devnull),
            patch.object(runner_mod, "_gpu_provider", return_value=False),
        ):
            self.assertIn("no GPU", str(runner_mod.xfeat_skip_reason("m")))

        with (
            _indexes({"xfeat"}),
            patch.object(xfeat_mod, "available", lambda: True),
            patch.object(xfeat_mod, "MATCH_MODEL_FILE", os.devnull),
            patch.object(runner_mod, "_gpu_provider", return_value=True),
        ):
            self.assertIsNone(runner_mod.xfeat_skip_reason("m"))

    def test_available_engines_is_empty_without_any_index(self):
        with _indexes(set()):
            self.assertEqual(available_engines("m"), [])

    def test_csv_and_jsonl_round_trip(self):
        with _ScriptedEngine([_pose(1000.0, 2000.0), None]):
            result, records = run_case(
                "sift", self._scenario(frames=2), {"ratio_local": 0.9}, "fake", warmup=False
            )

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv([result], os.path.join(tmp, "nested", "results.csv"))
            jsonl_path = write_jsonl(records, os.path.join(tmp, "per_frame", "run.jsonl"))
            with open(csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            with open(jsonl_path, encoding="utf-8") as fh:
                objs = [json.loads(line) for line in fh if line.strip()]

        self.assertEqual(tuple(rows[0]), CSV_COLUMNS)
        self.assertEqual(len(rows), 2)
        cells = dict(zip(CSV_COLUMNS, rows[1], strict=True))
        self.assertEqual(json.loads(cells["config"]), {"ratio_local": 0.9})
        self.assertEqual(json.loads(cells["reject_reasons"]), {"no_match_local": 1})
        self.assertEqual(cells["engine"], "sift")
        self.assertEqual(cells["scenario"], "straight")
        self.assertEqual(len(objs), 2)
        self.assertEqual(objs[0]["idx"], 0)
        self.assertTrue(objs[0]["new_sample"])
        self.assertEqual(objs[0]["pose"]["map_x"], 1000.0)
        self.assertIsNone(objs[1]["pose"])
        self.assertFalse(objs[1]["new_sample"])

    def test_csv_header_is_stable_across_runs(self):
        with _ScriptedEngine([_pose(1000.0, 2000.0)]):
            a, _ = run_case("sift", self._scenario(frames=2), {}, "fake")
            b, _ = run_case("sift", self._scenario(frames=2), {"ransac_px": 6.0}, "fake")
        with tempfile.TemporaryDirectory() as tmp:
            p1 = write_csv([a], os.path.join(tmp, "a.csv"))
            p2 = write_csv([b], os.path.join(tmp, "b.csv"))
            with open(p1, encoding="utf-8") as f1, open(p2, encoding="utf-8") as f2:
                self.assertEqual(f1.readline(), f2.readline())

    def test_hybrid_is_paced_to_the_wall_clock(self):
        with _ScriptedEngine([_pose(1000.0, 2000.0)]):
            start = time.perf_counter()
            run_case("hybrid", self._scenario(frames=3), {}, "fake", warmup=False)
            elapsed = time.perf_counter() - start

        self.assertGreater(elapsed, 0.2, "hybrid runs at the scenario cadence")


def _result(engine: str = "sift", scenario: str = "straight", **kw: Any) -> RunResult:
    base: dict[str, Any] = {
        "engine": engine,
        "scenario": scenario,
        "config": {},
        "frames": 60,
        "localized_pct": 100.0,
        "pos_err_px_mean": 2.0,
        "pos_err_px_median": 1.5,
        "pos_err_px_p95": 4.0,
        "pos_err_px_max": 6.0,
        "pos_err_m_mean": 1.0,
        "heading_err_deg_mean": 0.5,
        "heading_err_deg_max": 1.2,
        "latency_ms_mean": 90.0,
        "latency_ms_p95": 120.0,
        "achieved_fps": 11.1,
        "vote_rejects": 0,
        "holds": 0,
        "anchors": 1,
        "warmup_ms": 12.0,
        "fps": 10.0,
        "speed_mps": 15.0,
        "m_per_px": 0.5,
    }
    base.update(kw)
    return RunResult(**base)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, results: list[RunResult]) -> str:
        return write_report(
            results,
            os.path.join(self.tmp.name, "report.md"),
            {"map": "zestafona", "mode": "quick", "timestamp": "2026-01-01 00:00:00"},
        )

    def _text(self, results: list[RunResult]) -> str:
        with open(self._write(results), encoding="utf-8") as fh:
            return fh.read()

    def test_report_has_a_row_per_run_and_a_header(self):
        results = [
            _result("sift", "straight"),
            _result("sift", "turns", localized_pct=50.0),
            _result("orb", "straight"),
        ]
        text = self._text(results)

        self.assertIn("# Synthetic localization benchmark", text)
        self.assertIn("`zestafona`", text)
        self.assertIn("## sift", text)
        self.assertIn("## orb", text)
        rows = [ln for ln in text.splitlines() if ln.startswith("| straight")]
        self.assertEqual(len(rows), 2, "one row per run that used the default config")
        self.assertEqual(len([ln for ln in text.splitlines() if ln.startswith("| turns")]), 1)

    def test_swept_keys_need_a_varied_and_always_present_key(self):
        results = [
            _result("sift", "straight", config={"ransac_px": 2.0}, reject_reasons={"x": 2}),
            _result("sift", "turns", config={"ransac_px": 3.0}, reject_reasons={"x": 1}),
            _result("sift", "straight", config={"ransac_px": 4.0, "smooth_alpha": 0.5}),
        ]
        text = self._text(results)

        self.assertEqual(swept_keys(results, "sift"), ["ransac_px"])
        self.assertEqual(swept_keys(results, "orb"), [])
        self.assertIn("### sift: ransac_px", text)
        self.assertNotIn("### sift: smooth_alpha", text)
        self.assertIn("| 2 |", text)
        self.assertIn("| 4 |", text)
        self.assertIn("top rejects: `x` 3", text)

    def test_empty_report_is_written(self):
        self.assertIn("_no runs were recorded_", self._text([]))

    def test_plots_are_skipped_when_matplotlib_is_missing(self):
        import builtins

        from autopilot.bench import report as report_mod

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("no matplotlib here")
            return real_import(name, *args, **kwargs)

        out = os.path.join(self.tmp.name, "plots")
        with patch.object(builtins, "__import__", fake_import):
            written = report_mod.plot_results([_result()], out)

        self.assertEqual(written, [])
        self.assertIn("plots skipped", report_mod._MISSING_MPL)
        self.assertFalse(os.path.exists(out))


class TestBenchPackageApi(unittest.TestCase):
    def test_public_names_are_importable(self):
        import autopilot.bench as bench

        for name in bench.__all__:
            self.assertTrue(hasattr(bench, name), name)

    def test_scenario_is_frozen_and_replaceable(self):
        spec = default_specs(quick=True)[0]

        self.assertTrue(replace(spec, frames=5).frames == 5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.frames = 5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
