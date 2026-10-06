"""BenchTracker: the live tracker rules replayed over synthetic frames.

`LiveLocator` cannot be reused for the bench: it starts a capture thread and a
screen grab. `BenchTracker` keeps only the decision part of that loop —

1. `locator.global_pose` (the engine itself is untouched),
2. the relocation vote gate,
3. the pose smoother,
4. the black/empty-frame hold,

— and emits a `FrameRecord` per frame so the runner can score it. The gate,
hold and smoothing rules are copied from `vision/tracker.py` on purpose: the
bench measures the shipped behavior, not a second implementation of it.

Engine exceptions are never swallowed: a broken run must fail loudly instead of
showing up as a localization loss.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..vision import locator
from ..vision.pose_filter import PoseSmoother
from ..vision.tracker import _ang_diff, _vote_decide

#: Defaults mirror LocatorConfig so a partial cfg dict still behaves.
_DEFAULTS: dict[str, Any] = {
    "vote_need": 3,
    "vote_frames": 5,
    "vote_radius_px": 300,
    "jump_gate_px": 3000,
    "heading_gate_deg": 0.0,
    "vote_inl_skip": 40,
    "hold_frames": 5,
    "smooth_alpha": 0.5,
    "smooth_reset_px": 100.0,
}


@dataclass
class FrameRecord:
    """Everything the bench learned from one frame.

    `pose` is the published pose (smoothed for a fresh match, the frozen last
    good pose for a hold, None when nothing was published). `new_sample` marks
    a freshly measured pose: only those count as localized, and `err_px` /
    `th_err_deg` describe them. `raw_err_px` is the raw match error and exists
    even for a vote-rejected candidate.
    """

    idx: int
    truth: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose: dict[str, Any] | None = None
    latency_ms: float = 0.0
    reject: str | None = None
    vote_reject: bool = False
    new_sample: bool = False
    hold: bool = False
    anchor: bool = False
    err_px: float | None = None
    raw_err_px: float | None = None
    th_err_deg: float | None = None
    detail: str = ""

    def as_json(self) -> dict[str, Any]:
        """Flat JSON-friendly view for the per-frame JSONL writer."""
        pose = self.pose
        return {
            "idx": self.idx,
            "truth": [round(v, 3) for v in self.truth],
            "pose": None
            if pose is None
            else {
                "map_x": round(float(pose["map_x"]), 3),
                "map_y": round(float(pose["map_y"]), 3),
                "th": round(float(pose["th"]), 3),
                "inl": int(pose.get("inl", 0) or 0),
            },
            "latency_ms": round(self.latency_ms, 3),
            "reject": self.reject,
            "vote_reject": self.vote_reject,
            "new_sample": self.new_sample,
            "hold": self.hold,
            "anchor": self.anchor,
            "err_px": None if self.err_px is None else round(self.err_px, 3),
            "raw_err_px": None if self.raw_err_px is None else round(self.raw_err_px, 3),
            "th_err_deg": None if self.th_err_deg is None else round(self.th_err_deg, 3),
            "detail": self.detail[:200],
        }


@dataclass
class _Gate:
    """Vote-gate, hold and smoothing knobs read from the locator config."""

    need: int = 3
    radius_px: float = 300.0
    jump_px: float = 3000.0
    heading_deg: float = 0.0
    inl_skip: int = 40
    hold_frames: int = 5
    smooth_alpha: float = 0.5
    smooth_reset_px: float = 100.0

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> _Gate:
        merged = {**_DEFAULTS, **dict(cfg or {})}
        return cls(
            need=int(merged["vote_need"]),
            radius_px=float(merged["vote_radius_px"]),
            jump_px=float(merged["jump_gate_px"]),
            heading_deg=float(merged["heading_gate_deg"]),
            inl_skip=int(merged["vote_inl_skip"]),
            hold_frames=int(merged["hold_frames"]),
            smooth_alpha=float(merged["smooth_alpha"]),
            smooth_reset_px=float(merged["smooth_reset_px"]),
        )


def _pos_err(pose: dict[str, Any] | None, truth: tuple[float, float, float]) -> float | None:
    """Distance between a pose's map position and the ground truth (px)."""
    if pose is None:
        return None
    return math.hypot(float(pose["map_x"]) - truth[0], float(pose["map_y"]) - truth[1])


def _th_err(pose: dict[str, Any] | None, truth: tuple[float, float, float]) -> float | None:
    """Smallest heading difference between a pose and the ground truth (deg)."""
    if pose is None:
        return None
    return _ang_diff(float(pose["th"]), truth[2])


class BenchTracker:
    """One run's tracker state: engine call + vote gate + smoother + hold."""

    def __init__(
        self,
        engine: str,
        cfg: dict[str, Any],
        *,
        realtime: bool = False,
        budget: float | None = 3.0,
        min_inl: int = 4,
    ) -> None:
        self.engine = str(engine)
        self.cfg = dict(cfg or {})
        self.gate = _Gate.from_cfg(cfg)
        self.realtime = bool(realtime)
        self.budget = budget
        self.min_inl = int(min_inl)
        self._vote_buf: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=max(1, int(self.cfg.get("vote_frames", _DEFAULTS["vote_frames"])))
        )
        self._smoother = PoseSmoother(
            alpha=self.gate.smooth_alpha, reset_px=self.gate.smooth_reset_px
        )
        self._prev_xy: tuple[float, float] | None = None
        self._prev_th: float = 0.0
        self._prev_s: float | None = None
        self._last_accepted: tuple[float, float] | None = None
        self._good_xy: tuple[float, float] | None = None
        self._good_pose: dict[str, Any] | None = None
        self._hold_left: int | None = None
        self._wall0: float | None = None

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Drop every piece of state, including the engine's own track.

        The hybrid keeps its ECC track between runs, so without this a later
        config would start from a stale pose.
        """
        reset = getattr(locator._active_engine(), "reset", None)
        if callable(reset):
            reset()
        self._prev_xy = None
        self._prev_th = 0.0
        self._prev_s = None
        self._last_accepted = None
        self._good_xy = None
        self._good_pose = None
        self._hold_left = None
        self._vote_buf.clear()
        self._smoother.reset()
        self._wall0 = None

    # ------------------------------------------------------------------- step

    def _pace(self, now: float) -> None:
        """Hold the wall clock to the scenario cadence (hybrid anchor timing).

        The hybrid re-anchor period is wall-clock based, so a hybrid bench run
        paces its frames; the other engines do not need to.
        """
        if not self.realtime:
            return
        if self._wall0 is None:
            self._wall0 = time.perf_counter()
        delay = (self._wall0 + float(now)) - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)

    def step(
        self,
        idx: int,
        frame: np.ndarray,
        ui_mask: np.ndarray | None,
        now: float,
        truth: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FrameRecord:
        """Localize one frame and apply the tracker rules.

        `now` is the scenario time in seconds (idx * dt); it feeds the smoother.
        `truth` is the ground truth (native x, y, heading) of this frame.
        """
        self._pace(now)
        t0 = time.perf_counter()
        pose, diag = locator.global_pose(
            frame,
            ui_mask,
            prev_xy=self._prev_xy,
            prev_th=self._prev_th,
            debug=True,
            budget=self.budget,
            prev_s=self._prev_s,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        rec = FrameRecord(
            idx=int(idx),
            truth=(float(truth[0]), float(truth[1]), float(truth[2])),
            latency_ms=latency_ms,
            reject=diag.get("reject"),
            anchor=bool(diag.get("anchor")),
            detail=str(diag.get("detail") or ""),
        )

        if pose is None:
            self._hold(rec)
            return rec

        raw = dict(pose)
        rec.raw_err_px = _pos_err(raw, rec.truth)
        if self._vote_gate_ok(raw, rec):
            self._commit(raw, now, rec)
        return rec

    # ------------------------------------------------------------------- rules

    def _vote_gate_ok(self, pose: dict[str, Any], rec: FrameRecord) -> bool:
        """Relocation vote gate; False means "do not publish this candidate".

        A pose that jumped far from the last accepted position (or flipped its
        heading past `heading_gate_deg`) must be confirmed by `vote_need`
        agreeing frames, unless it is strong enough (`inl >= vote_inl_skip`),
        which a wrong re-acquisition practically never is.
        """
        if self._last_accepted is None:
            self._vote_buf.clear()
            return True

        gate = self.gate
        x, y = float(pose["map_x"]), float(pose["map_y"])
        d_jump = math.hypot(x - self._last_accepted[0], y - self._last_accepted[1])
        d_th = _ang_diff(float(pose["th"]), self._prev_th)
        why = None
        if d_jump > gate.jump_px:
            why = "jump %.0f px" % d_jump
        elif gate.heading_deg > 0 and d_th > gate.heading_deg:
            why = "heading %.0f deg" % d_th
        if why is None or int(pose.get("inl", 0) or 0) >= gate.inl_skip:
            self._vote_buf.clear()
            return True

        vote = _vote_decide(self._vote_buf, gate.need, gate.radius_px, (x, y))
        if vote is not None:
            self._vote_buf.clear()
            return True
        rec.vote_reject = True
        rec.reject = "vote_reject"
        rec.detail = "%s needs %d agreeing frames (have %d)" % (
            why,
            gate.need,
            len(self._vote_buf),
        )
        return False

    def _commit(self, pose: dict[str, Any], now: float, rec: FrameRecord) -> None:
        """Accept a measured pose: the search center stays raw, publication smooths."""
        meas = (float(pose["map_x"]), float(pose["map_y"]))
        published = self._smoother.update(meas[0], meas[1], now)
        self._prev_xy = meas
        self._prev_th = float(pose["th"])
        self._prev_s = float(pose.get("s") or self._prev_s or 1.0)
        self._good_xy = published
        self._last_accepted = published
        self._good_pose = dict(pose, map_x=published[0], map_y=published[1])
        self._hold_left = None
        rec.new_sample = True
        rec.pose = self._good_pose
        rec.err_px = _pos_err(self._good_pose, rec.truth)
        rec.th_err_deg = _th_err(self._good_pose, rec.truth)

    def _hold(self, rec: FrameRecord) -> None:
        """Black/empty-frame hold: reuse the last good pose for `hold_frames`.

        A capture void is not a localization loss, so the last accepted pose
        keeps being published (with good=False semantics: it is not a new
        sample and is excluded from the error statistics).
        """
        if self.gate.hold_frames > 0 and self._good_xy is not None:
            if self._hold_left is None:
                self._hold_left = self.gate.hold_frames
            if self._hold_left > 0:
                self._hold_left -= 1
                rec.hold = True
                rec.reject = "hold"
                rec.detail = "holding last pose (%d left)" % self._hold_left
                rec.pose = self._good_pose
                rec.err_px = _pos_err(self._good_pose, rec.truth)
                rec.th_err_deg = _th_err(self._good_pose, rec.truth)
                return
        self._hold_left = None
