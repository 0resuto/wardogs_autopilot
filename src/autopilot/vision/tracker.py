"""Background live localization: region capture -> pose -> position on the map.

The capture (screen grab) runs in a producer thread at a fixed ~fps feeding a
single-slot queue (drop-old); the localization consumer pops the latest frame
and runs the pose search. Decoupling the two keeps the UI's "proc: N ms" from
growing when a coarse map search stalls: the producer keeps delivering fresh
frames at the capture rate no matter how long global_pose takes.
"""

import collections
import math
import queue
import threading
import time
from typing import Any

import numpy as np

from .. import crashlog
from ..common.config import AppConfig, CaptureConfig, LocatorConfig
from ..common.log import get_logger
from . import locator
from .capture_producer import _CaptureProducer
from .fail_dump import (
    FAIL_KEEP_FILES,
    FAIL_REASONS,
    FAIL_SAVE_PERIOD_S,
    fail_payload,
    prune_fail_files,
    save_fail_frame,
)
from .pose_filter import PoseSmoother

logger = get_logger("tracker")


def _ang_diff(a, b):
    """Smallest angular difference in degrees between two headings/angles."""
    return abs((a - b + 540.0) % 360.0 - 180.0)


def _vote_decide(buf, need, radius, pos):
    """Append pos to the vote buffer; return the agreeing cluster if at least
    `need` buffered jump candidates lie pairwise within `radius`, else None.

    Used by the relocation vote gate: a single lone candidate (a wrong
    re-acquisition) is kept for `need` frames and only published when enough
    consensus frames agree on the same place. The buffer is a maxlen deque, so
    stale candidates expire by themselves.
    """
    buf.append((float(pos[0]), float(pos[1])))
    pts = list(buf)
    for anchor in pts:
        cluster = [p for p in pts if math.hypot(p[0] - anchor[0], p[1] - anchor[1]) <= radius]
        if len(cluster) >= need and all(
            math.hypot(p[0] - q[0], p[1] - q[1]) <= radius
            for i, p in enumerate(cluster)
            for q in cluster[i + 1 :]
        ):
            return cluster
    return None


class LiveLocator(threading.Thread):
    """Background live localization: consumer of grabbed frames -> pose.

    The result of the last processed frame is always in self.latest; the
    processing delay of the last frame is in self.latest['elapsed'] (seconds).
    latest['good'] is True ONLY for a measured pose: the held pose of a capture
    void and all failed frames report good=False, so consumers must not derive
    motion or a stopped state from them.
    latest['speed_kmh']/latest['speed_ok'] carry the optional speedometer OCR
    (capture.speed_roi): speed_ok=False means the value is stale or unread.
    latest['map_px'] is the measured position low-passed by the alpha-beta
    smoother (locator.smooth_alpha; 0 disables) so matcher jitter does not
    reach the navigator; latest['map_px_disp'] additionally medians the last
    frames for the UI.
    """

    def __init__(self, cfg: AppConfig | dict, mask, frame_source=None) -> None:
        super().__init__(daemon=True)
        if isinstance(cfg, AppConfig):
            self.app_cfg = cfg
            self.cfg = cfg.to_dict()
        elif isinstance(cfg, dict):
            self.cfg = cfg
            try:
                self.app_cfg = AppConfig(**cfg)
            except Exception:
                self.app_cfg = AppConfig()
        else:
            self.app_cfg = AppConfig()
            self.cfg = self.app_cfg.to_dict()

        self.cap_cfg = self.app_cfg.capture
        self.loc_cfg = self.app_cfg.locator

        self.mask = np.asarray(mask, bool)
        self.frame_source = frame_source  # callable -> BGR ROI (simulator)
        self.error: str | None = None
        self.latest: dict[str, Any] | None = None
        self.mm_gray: np.ndarray | None = None  # last raw ROI frame for debugging
        self.mm_bgr: np.ndarray | None = None  # last raw ROI frame in color (debug only)
        self.attempt = 0
        self.phase = "loading map..."
        self._prev_xy: tuple[float, float] | None = None
        self._prev_th: float = 0.0
        self._prev_s: float | None = None  # last matched scale (level pruning)
        self._good_xy: tuple[float, float] | None = None  # last accurate (SIFT) position
        self._stop = threading.Event()
        self._poll = 0.03
        # guards the {mm_gray, mm_bgr, mask} trio: the producer thread swaps
        # them per frame, and a UI reader must never see a mismatched combo
        # (e.g. new frame + old mask — that was an IndexError in the collage)
        self._snap_lock = threading.Lock()
        self._vote_buf: collections.deque[tuple[Any, ...]] = collections.deque(
            maxlen=self._loc_vote_frames()
        )
        self._last_accepted: tuple[float, float] | None = None
        self._smoother = PoseSmoother(
            alpha=self.loc_cfg.smooth_alpha,
            reset_px=self.loc_cfg.smooth_reset_px,
        )
        self._disp_hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        self._good_pose: dict[str, Any] | None = (
            None  # last accepted pose dict (for black-frame hold)
        )
        self._hold_left: int | None = None  # frames of black-frame hold still left
        self._last_fail_save: float = 0.0  # time of the last fail-frame dump (throttle)
        self._counts: collections.Counter[str] = collections.Counter()  # reject reason tally
        self._last_frame_error: str | None = None  # dedup for per-frame error logging
        self._engine = locator.engine()  # synced to locator.engine config on first frame
        self.search_now: tuple[Any, ...] | str | None = (
            None  # sector being searched right now ('disc'/'global')
        )

    @property
    def locator_cfg(self) -> LocatorConfig:
        if (
            isinstance(self.cfg, dict)
            and "locator" in self.cfg
            and isinstance(self.cfg["locator"], dict)
        ):
            try:
                return LocatorConfig(**self.cfg["locator"])
            except Exception:
                pass
        return self.app_cfg.locator

    @property
    def capture_cfg(self) -> CaptureConfig:
        if (
            isinstance(self.cfg, dict)
            and "capture" in self.cfg
            and isinstance(self.cfg["capture"], dict)
        ):
            try:
                return CaptureConfig(**self.cfg["capture"])
            except Exception:
                pass
        return self.app_cfg.capture

    def stop(self) -> None:
        self._stop.set()

    def set_roi(self, roi: list[int] | tuple[int, ...]) -> None:
        """Update live capture ROI on the fly."""
        roi_list = [int(v) for v in roi]
        if isinstance(self.cfg, dict):
            self.cfg.setdefault("capture", {})["mmap_roi"] = roi_list
        if hasattr(self, "app_cfg") and hasattr(self.app_cfg, "capture"):
            self.app_cfg.capture.mmap_roi = roi_list
        if hasattr(self, "cap_cfg") and hasattr(self.cap_cfg, "mmap_roi"):
            self.cap_cfg.mmap_roi = roi_list
        if hasattr(self, "_prod") and self._prod is not None:
            self._prod.set_roi(roi_list)

    def set_fps(self, fps: int | float) -> None:
        """Update the live capture cadence (the producer re-reads it every loop)."""
        fps_i = max(1, min(60, int(fps)))
        if isinstance(self.cfg, dict):
            self.cfg.setdefault("capture", {})["fps"] = fps_i
        if hasattr(self, "app_cfg") and hasattr(self.app_cfg, "capture"):
            self.app_cfg.capture.fps = fps_i
        if hasattr(self, "cap_cfg") and hasattr(self.cap_cfg, "fps"):
            self.cap_cfg.fps = fps_i
        if hasattr(self, "_prod") and self._prod is not None:
            self._prod.capture_cfg.fps = fps_i
            if isinstance(self._prod.cfg, dict):
                self._prod.cfg.setdefault("capture", {})["fps"] = fps_i

    def set_speed_roi(self, roi: list[int] | tuple[int, ...] | None) -> None:
        """Update (or disable with None) the speedometer OCR ROI on the fly."""
        roi_list = None if roi is None else [int(v) for v in roi]
        if isinstance(self.cfg, dict):
            self.cfg.setdefault("capture", {})["speed_roi"] = roi_list
        if hasattr(self, "app_cfg") and hasattr(self.app_cfg, "capture"):
            self.app_cfg.capture.speed_roi = roi_list
        if hasattr(self, "cap_cfg") and hasattr(self.cap_cfg, "speed_roi"):
            self.cap_cfg.speed_roi = roi_list
        if hasattr(self, "_prod") and self._prod is not None:
            self._prod.set_speed_roi(roi_list)

    def set_collect_fail_logs(self, enabled: bool) -> None:
        """Live toggle of the fail-frame collector (Map tab checkbox)."""
        enabled = bool(enabled)
        self.app_cfg.debug.collect_fail_logs = enabled
        if isinstance(self.cfg, dict):
            self.cfg.setdefault("debug", {})["collect_fail_logs"] = enabled
        logger.info("[tracker] fail-frame collection %s", "enabled" if enabled else "disabled")

    def _reset_engine_state(self) -> None:
        """Drop the track/vote/smoothing state after an engine switch."""
        self._prev_xy = None
        self._prev_th = 0.0
        self._prev_s = None
        self._good_xy = None
        self._good_pose = None
        self._hold_left = None
        self._last_accepted = None
        self._vote_buf.clear()
        self._disp_hist.clear()
        self._smoother.reset()

    def _frame_pose(
        self, mm: np.ndarray, ui_mask: np.ndarray | None
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """One localization pass; a locator/cv2 failure degrades to a bad frame.

        Without this guard a single native matching error escaped run() and
        killed the whole locator thread: the preview froze on the last frame
        and tracking stopped for good.
        """
        try:
            res = locator.global_pose(
                mm,
                ui_mask,
                prev_xy=self._prev_xy,
                prev_th=self._prev_th,
                debug=True,
                budget=3.0,
                progress=self._search_progress,
                prev_s=self._prev_s,
            )
            self._last_frame_error = None
            return res
        except Exception as exc:  # noqa: BLE001
            tag = f"{type(exc).__name__}: {exc}"
            if tag != self._last_frame_error:
                self._last_frame_error = tag
                logger.error("[tracker] locator frame error: %s", tag)
                crashlog.log("locator frame error", exc)
            return None, dict(reject="locator_error", detail=tag[:300])

    def _search_progress(self, region):
        """Live callback from the locator: the sector searched at this instant.

        Called on the localization thread while a frame search runs; read by
        the UI every poll. Nothing heavier than a reference assignment.
        """
        self.search_now = region

    def snapshot_debug(self):
        """(mm_gray, mm_bgr, mask, latest) captured atomically for debug dumps.

        mm_gray and mask always come from the same captured frame; mixing them
        across frames (when the ROI changes shape) previously produced a
        boolean-index shape mismatch in the debug collage.
        """
        with self._snap_lock:
            return (self.mm_gray, self.mm_bgr, self.mask, self.latest)

    # --- config knobs for the relocation vote gate (locator block) ---
    def _loc_keys(self):
        lc = self.locator_cfg
        return (
            int(lc.vote_need),
            float(lc.vote_radius_px),
            float(lc.jump_gate_px),
            float(lc.heading_gate_deg),
            int(lc.vote_inl_skip),
        )

    def _loc_vote_frames(self):
        return int(self.locator_cfg.vote_frames)

    def _loc_hold_frames(self):
        """Black/empty-frame hold: frames to reuse the last good pose."""
        return int(self.locator_cfg.hold_frames)

    def _count_reject(self, diag) -> None:
        """Tally the reject reason and expose the top counts on the diag."""
        r = diag.get("reject")
        if r:
            self._counts[r] += 1
        diag["reject_tally"] = dict(self._counts.most_common(5))

    def run(self) -> None:
        try:
            locator.load_global_map()
            self.phase = "searching pose..."
            q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._prod = _CaptureProducer(self.cfg, self.mask, self.frame_source, self._stop, q)
            self._prod.start()
            while not self._stop.is_set():
                if self._prod is not None and self._prod.error:
                    self.error = self._prod.error
                    self.latest = dict(
                        ts=time.time(),
                        pose=None,
                        map_px=None,
                        good=False,
                        speed_kmh=None,
                        speed_ok=False,
                        speed_frame=None,
                        speed_mask=None,
                        speed_boxes=(),
                        elapsed=0.0,
                        diag=dict(reject="capture_error", detail=f"Screen capture: {self.error}"),
                    )
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    self._stop.wait(self._poll)
                    continue
                mm = item["gray"]
                with self._snap_lock:
                    self.mm_gray = mm
                    self.mm_bgr = item.get("bgr")
                    self.mask = item["mask"]
                roi = item["roi"]
                t0 = item["ts"]
                self.error = None
                # engine switch (Map -> Tracking combo): reconfigure the
                # locator facade and drop every prior-track state so the new
                # engine cold-starts instead of inheriting a stale pose
                try:
                    eng = str(self.locator_cfg.engine or "sift")
                except Exception:
                    eng = "sift"
                if eng != self._engine:
                    locator.set_engine(eng)
                    self._engine = eng
                    self._reset_engine_state()
                    self.phase = "engine: %s..." % eng
                # frame budget: coarse map search can stall for tens of
                # seconds (low-texture areas); with a budget the
                # localization returns within max 3 s and the thread does
                # not hang along with it (nor did steer and UI)
                pose, diag = self._frame_pose(mm, self.mask)
                self.search_now = None  # the frame's search is done
                self._count_reject(diag)
                slow = time.time() - t0
                if slow > 3.0 and self.attempt % 20 == 0:
                    logger.warning("[locator] frame took %.1f s (budget 3 s) — map search", slow)

                # autosave of a "failed" frame (black/empty or no pose)
                # for post-run analysis; enabled by the "collect fail logs"
                # checkbox in the app (config 'debug.collect_fail_logs')
                collect = bool(self.app_cfg.debug.collect_fail_logs)
                if collect and pose is None and diag.get("reject") in FAIL_REASONS:
                    now = time.time()
                    if now - self._last_fail_save >= FAIL_SAVE_PERIOD_S:
                        self._last_fail_save = now
                        self._save_fail_frame(diag, mm, item.get("bgr"), item.get("mask"))
                mp = None
                good = False
                cand = None  # pending vote approval: (px, py, heading, inl)
                if pose is not None:
                    px = pose["map_x"]
                    py = pose["map_y"]
                    good = True
                    cand = (px, py, pose["th"], int(pose.get("inl", 0)))
                elapsed = time.time() - t0
                diag["roi"] = list(roi)
                if cand is not None:
                    # relocation vote gate: a pose that jumped far away from
                    # the last accepted position (or flipped its heading beyond
                    # heading_gate_deg) must be confirmed by a few frames
                    # agreeing on the same place before it is published — UNLESS
                    # the pose is very strong (inl >= vote_inl_skip), which a
                    # wrong re-acquisition practically never is.
                    need, rad, gate, hgate, inl_skip = self._loc_keys()
                    if self._last_accepted is not None:
                        d_jump = math.hypot(
                            cand[0] - self._last_accepted[0], cand[1] - self._last_accepted[1]
                        )
                        d_th = _ang_diff(cand[2], self._prev_th)
                        why = None
                        if d_jump > gate:
                            why = "jump %.0f px" % d_jump
                        elif hgate > 0 and d_th > hgate:
                            why = "heading %.0f deg" % d_th
                        if why is not None and cand[3] < inl_skip:
                            vote = _vote_decide(self._vote_buf, need, rad, (cand[0], cand[1]))
                            if vote is None:
                                diag["reject"] = "vote_reject"
                                diag["detail"] = "%s needs %d agreeing frames (have %d)" % (
                                    why,
                                    need,
                                    len(self._vote_buf),
                                )
                                diag["vote"] = dict(
                                    count=len(self._vote_buf),
                                    need=need,
                                    dist=d_jump,
                                    th=d_th,
                                    accepted=False,
                                    inl=cand[3],
                                )
                                self.latest = dict(
                                    ts=t0,
                                    pose=None,
                                    map_px=None,
                                    good=False,
                                    speed_kmh=item.get("speed_kmh"),
                                    speed_ok=bool(item.get("speed_ok", False)),
                                    speed_frame=item.get("speed_frame"),
                                    speed_mask=item.get("speed_mask"),
                                    speed_boxes=item.get("speed_boxes", ()),
                                    elapsed=elapsed,
                                    diag=diag,
                                )
                                self.attempt += 1
                                continue
                            self._vote_buf.clear()
                            diag["vote"] = dict(
                                count=len(vote),
                                need=need,
                                dist=d_jump,
                                th=d_th,
                                accepted=True,
                                inl=cand[3],
                            )
                        elif why is not None:
                            self._vote_buf.clear()
                    else:
                        self._vote_buf.clear()
                    # approved: commit the state that _prev_xy feeds the next
                    # global_pose call with, and the position to publish. The
                    # search center keeps the RAW match; only the published
                    # position is low-passed so matcher jitter does not reach
                    # the marker or the steering.
                    if pose is not None:
                        meas = (cand[0], cand[1])
                        lc = self.locator_cfg
                        self._smoother.configure(float(lc.smooth_alpha), float(lc.smooth_reset_px))
                        mx, my = self._smoother.update(meas[0], meas[1], t0)
                        self._prev_xy = meas
                        self._prev_th = cand[2]
                        self._good_xy = (mx, my)
                        pose = dict(pose, map_x=mx, map_y=my)
                        self._good_pose = pose
                        self._prev_s = float(pose.get("s") or self._prev_s or 1.0)
                        mp = (mx, my)
                    else:
                        self._prev_xy = (cand[0], cand[1])  # bridge only
                        mp = (cand[0], cand[1])
                    self._last_accepted = mp
                    self._disp_hist.append(mp)
                disp = mp
                if len(self._disp_hist) >= 3:
                    xs = np.median([p[0] for p in self._disp_hist])
                    ys = np.median([p[1] for p in self._disp_hist])
                    disp = (float(xs), float(ys))
                # black/empty-frame hold: a flat capture (minimap briefly not
                # drawn) while a good pose exists is NOT a real localization
                # loss — reuse the last accepted pose for a few frames so the
                # marker and the navigator do not drop out on every capture void.
                if pose is None:
                    hf = self._loc_hold_frames()
                    if hf > 0 and self._good_xy is not None:
                        if self._hold_left is None:
                            self._hold_left = hf
                        if self._hold_left > 0:
                            self._hold_left -= 1
                            d2 = dict(diag)
                            d2["reject"] = "hold"
                            d2["detail"] = "holding last pose (%d left)" % self._hold_left
                            self.latest = dict(
                                ts=t0,
                                pose=self._good_pose,
                                map_px=self._good_xy,
                                map_px_disp=self._good_xy,
                                good=False,
                                speed_kmh=item.get("speed_kmh"),
                                speed_ok=bool(item.get("speed_ok", False)),
                                speed_frame=item.get("speed_frame"),
                                speed_mask=item.get("speed_mask"),
                                speed_boxes=item.get("speed_boxes", ()),
                                elapsed=elapsed,
                                diag=d2,
                            )
                            self.attempt += 1
                            continue
                    self._hold_left = None
                else:
                    self._hold_left = None
                self.latest = dict(
                    ts=t0,
                    pose=pose,
                    map_px=mp,
                    map_px_disp=disp,
                    good=good,
                    speed_kmh=item.get("speed_kmh"),
                    speed_ok=bool(item.get("speed_ok", False)),
                    speed_frame=item.get("speed_frame"),
                    speed_mask=item.get("speed_mask"),
                    speed_boxes=item.get("speed_boxes", ()),
                    elapsed=elapsed,
                    diag=diag,
                )
                self.attempt += 1
        except Exception as exc:  # noqa: BLE001
            crashlog.log("locator thread exited with an error", exc)
            self.error = str(exc)

    def _fail_payload(self, diag: dict, mm: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
        """Structured context of a failed frame (see fail_dump.fail_payload)."""
        return fail_payload(
            diag,
            mm,
            mask,
            locator_cfg=self.locator_cfg,
            prev_xy=self._prev_xy,
            attempt=self.attempt,
        )

    def _save_fail_frame(
        self,
        diag: dict,
        mm: np.ndarray,
        bgr: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        """Autosave a failed frame for post-run analysis (see fail_dump)."""
        save_fail_frame(
            diag,
            mm,
            bgr,
            mask,
            locator_cfg=self.locator_cfg,
            prev_xy=self._prev_xy,
            attempt=self.attempt,
        )

    @staticmethod
    def _prune_fail_files(out_dir: str, keep: int = FAIL_KEEP_FILES) -> None:
        """Keep only the newest `keep` debug_fail_* files (see fail_dump)."""
        prune_fail_files(out_dir, keep)
