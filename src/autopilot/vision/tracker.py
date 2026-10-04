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
    prune_fail_files,
    save_fail_frame,
)
from .pose_filter import PoseSmoother

logger = get_logger("tracker")


def _ang_diff(a, b):
    """Smallest angular difference in degrees between two headings/angles."""
    return abs((a - b + 540.0) % 360.0 - 180.0)


#: A pose gap (s) after which the next candidate must pass the vote gate even
#: when it looks strong: right after a void the matcher can re-acquire a
#: coherent but wrong place (2026-10-03 runs), and the strong-inlier bypass
#: would publish it immediately.
REACQUIRE_GAP_S = 0.4
#: Heading agreement the vote frames must show: a position-only vote passes a
#: 180-degree flip at the same place (the known wrong-lock shape).
VOTE_HEADING_TOL_DEG = 10.0
#: Motion-course heading gate: a pose whose heading contradicts the direction
#: of travel is a wrong lock (the 180-degree flip at the same place), not a
#: spin. This is the physical check the vote cannot make: a persistent flipped
#: track passes a 3-frame vote, the hybrid ECC keeps it alive and `prev_th`
#: feeds it back (2026-10-04 02:52 run: flip at 80 km/h, then 90 s of recovery
#: loops). The course comes from the accepted positions and is latched while
#: the truck is known to move, so the guard survives the blind spell that a
#: rejection itself causes.
MOTION_WINDOW_S = 0.6
MOTION_MIN_PX = 16.0
MOTION_TOL_DEG = 75.0
MOTION_MIN_KMH = 30.0  # OCR speed above which the truck is definitely moving
MOTION_HOLD_S = 2.0  # stale course lifetime without OCR speed evidence


def _course_from_hist(hist, now: float) -> float | None:
    """Direction of travel (deg, 0=north clockwise) or None when not moving."""
    if len(hist) < 2:
        return None
    base = next((s for s in hist if now - s[0] <= MOTION_WINDOW_S), None)
    tip = next((s for s in reversed(hist) if now - s[0] <= MOTION_WINDOW_S), None)
    if base is None or tip is None or tip is base:
        return None
    d = math.hypot(tip[1] - base[1], tip[2] - base[2])
    if d < MOTION_MIN_PX:
        return None
    return math.degrees(math.atan2(tip[1] - base[1], -(tip[2] - base[2]))) % 360.0


def _needs_vote(why: str | None, after_gap: bool, inl: int, inl_skip: int) -> bool:
    """Vote requirement for a relocation candidate.

    A jumped or flipped candidate is vote-gated unless it is very strong
    (`inl >= inl_skip`); after a pose gap the bypass is disabled, so the first
    candidates must be confirmed by agreeing frames.
    """
    if why is None and not after_gap:
        return False
    return after_gap or inl < inl_skip


def _vote_decide(buf, need, radius, pos, th=None, th_tol=0.0):
    """Append pos/heading to the vote buffer; return the agreeing cluster.

    At least `need` buffered candidates must lie pairwise within `radius` and,
    when headings are supplied with `th_tol > 0`, within `th_tol` degrees of
    each other: a position-only vote used to pass a 180-degree flip at the same
    place. The buffer is a maxlen deque, so stale candidates expire themselves.
    """
    buf.append((float(pos[0]), float(pos[1]), None if th is None else float(th)))
    pts = list(buf)

    def agrees(p, q) -> bool:
        if math.hypot(p[0] - q[0], p[1] - q[1]) > radius:
            return False
        if th_tol > 0.0 and p[2] is not None and q[2] is not None:
            return _ang_diff(p[2], q[2]) <= th_tol
        return True

    for anchor in pts:
        cluster = [p for p in pts if agrees(p, anchor)]
        if len(cluster) >= need and all(
            agrees(p, q) for i, p in enumerate(cluster) for q in cluster[i + 1 :]
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
    reach the navigator. It is the single source of truth: the UI marker
    draws this exact pose, with no extra filtering.
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
        self._last_accept_t: float = 0.0
        self._smoother = PoseSmoother(
            alpha=self.loc_cfg.smooth_alpha,
            reset_px=self.loc_cfg.smooth_reset_px,
        )
        self._motion_hist: collections.deque[tuple[float, float, float]] = collections.deque(
            maxlen=64
        )
        self._course: float | None = None  # latched direction of travel (deg)
        self._course_t: float = 0.0
        self._speed_kmh: float | None = None  # last fresh speedometer OCR
        self._speed_t: float = 0.0
        self._good_pose: dict[str, Any] | None = (
            None  # last accepted pose dict (for black-frame hold)
        )
        self._hold_left: int | None = None  # frames of black-frame hold still left
        self._last_fail_save: float = 0.0  # time of the last fail-frame dump (throttle)
        self._local_timeouts = 0  # consecutive local-search budget timeouts
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
        self._motion_hist.clear()
        self._course = None
        self._course_t = 0.0
        self._smoother.reset()

    def _motion_course(self, now: float) -> float | None:
        """Direction of travel from the accepted positions (None = not moving)."""
        return _course_from_hist(self._motion_hist, now)

    def _anchor_prev_th(self, now: float) -> float | None:
        """Last accepted heading for the anchor's rotation prior (None = off).

        The prior is only valid while the truck is known to move: at low speed
        a real spin changes the heading arbitrarily fast, and a gate would
        blind the tracker exactly when the driver's spin recovery needs the
        pose.
        """
        if self._last_accepted is None or (now - self._last_accept_t) > 2.0:
            return None
        if self._motion_course(now) is not None:
            return self._prev_th
        if (
            self._speed_kmh is not None
            and (now - self._speed_t) <= 1.0
            and self._speed_kmh > MOTION_MIN_KMH
        ):
            return self._prev_th
        return None

    def _motion_flip(self, t0: float, th: float, item: dict) -> bool:
        """True when a candidate heading contradicts the direction of travel.

        The course is refreshed from the accepted positions; while the OCR
        speed says the truck is moving fast the last course is latched, so the
        guard covers the blind spell a rejection causes (no new positions).
        A known stop clears the latch: a direction of travel only constrains
        the heading while there is travel.
        """
        course = self._motion_course(t0)
        if course is not None:
            self._course = course
            self._course_t = t0
        speed_kmh = None
        if bool(item.get("speed_ok", False)) and item.get("speed_kmh") is not None:
            speed_kmh = float(item["speed_kmh"])
        if speed_kmh is not None and speed_kmh <= MOTION_MIN_KMH:
            self._course = None
            return False
        if self._course is None:
            return False
        if course is None and speed_kmh is None and t0 - self._course_t > MOTION_HOLD_S:
            self._course = None
            return False
        return _ang_diff(th, self._course) > MOTION_TOL_DEG

    def _frame_pose(
        self, mm: np.ndarray, ui_mask: np.ndarray | None, t0: float
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
                prev_th=self._anchor_prev_th(t0),
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
                if bool(item.get("speed_ok", False)) and item.get("speed_kmh") is not None:
                    self._speed_kmh = float(item["speed_kmh"])
                    self._speed_t = t0
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
                pose, diag = self._frame_pose(mm, self.mask, t0)
                self.search_now = None  # the frame's search is done
                self._count_reject(diag)
                slow = time.time() - t0
                if slow > 3.0 and self.attempt % 20 == 0:
                    logger.warning("[locator] frame took %.1f s (budget 3 s) — map search", slow)

                # A repeated local timeout means the seed may be poisoned by
                # ECC drift: the next frame must recover globally instead of
                # growing the radius again (the fail dumps showed this exact
                # lockout pattern: budget_timeout with search_global=false).
                if diag.get("reject") == "budget_timeout":
                    self._local_timeouts += 1
                    if self._local_timeouts >= 2:
                        self._prev_xy = None
                        self._vote_buf.clear()
                        self._local_timeouts = 0
                else:
                    self._local_timeouts = 0

                # autosave of a "failed" frame (black/empty or no pose)
                # for post-run analysis; enabled by the "collect fail logs"
                # checkbox in the app (config 'debug.collect_fail_logs').
                # The vote rejections are dumped too (the vote runs later).
                collect = bool(self.app_cfg.debug.collect_fail_logs)
                self._maybe_save_fail(diag, mm, item, collect)
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
                    # motion-course gate (root fix for the 180-degree wrong
                    # lock): while the truck is known to move, a pose heading
                    # that contradicts the direction of travel is rejected and
                    # the hybrid track is reset, so the flipped anchor cannot
                    # feed itself back through `prev_th`/ECC.
                    if self._last_accepted is not None and self._motion_flip(t0, cand[2], item):
                        diag["reject"] = "motion_flip"
                        diag["detail"] = "heading %.0f vs course %.0f deg" % (
                            cand[2],
                            self._course if self._course is not None else -1.0,
                        )
                        locator.reset_track()
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
                        self._maybe_save_fail(diag, mm, item, collect)
                        self.attempt += 1
                        continue
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
                        after_gap = (t0 - self._last_accept_t) > REACQUIRE_GAP_S
                        why = None
                        if d_jump > gate:
                            why = "jump %.0f px" % d_jump
                        elif hgate > 0 and d_th > hgate:
                            why = "heading %.0f deg" % d_th
                        elif after_gap:
                            why = "gap %.2f s" % (t0 - self._last_accept_t)
                        if _needs_vote(why, after_gap, cand[3], inl_skip):
                            vote = _vote_decide(
                                self._vote_buf,
                                need,
                                rad,
                                (cand[0], cand[1]),
                                cand[2],
                                VOTE_HEADING_TOL_DEG,
                            )
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
                                # the vote rejection happens after the first
                                # dump point; without this the dominant reject
                                # mode was invisible in the fail dumps
                                self._maybe_save_fail(diag, mm, item, collect)
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
                    self._last_accept_t = t0
                    self._motion_hist.append((t0, mp[0], mp[1]))
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
                    good=good,
                    cc=(pose or {}).get("cc"),
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

    def _maybe_save_fail(self, diag: dict, mm: np.ndarray, item: dict, collect: bool) -> None:
        """Throttled fail-frame dump for any FAIL_REASONS reject (incl. vote)."""
        if not collect or diag.get("reject") not in FAIL_REASONS:
            return
        now = time.time()
        if now - self._last_fail_save < FAIL_SAVE_PERIOD_S:
            return
        self._last_fail_save = now
        self._save_fail_frame(diag, mm, item.get("bgr"), item.get("mask"))

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
