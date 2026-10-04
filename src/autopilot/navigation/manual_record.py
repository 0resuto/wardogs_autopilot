"""Temporary manual-driving recorder: the user's key presses + poses + route.

While the user drives by hand (autopilot off), W/A/S/D/SPACE are polled with
GetAsyncKeyState (focus-independent, no system hook) and sampled together with
the localizer pose into output/manual_dbg_*.jsonl (one row per poll). The route
and the run parameters are written to the kind="manual-log" header, so offline
analysis can compare the human line/speed with the driver's decisions.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..common.log import get_logger

logger = get_logger("manual_record")

_KEYS = ("W", "A", "S", "D", "SPACE")
_VK = {0x57: "W", 0x41: "A", 0x53: "S", 0x44: "D", 0x20: "SPACE"}


def build_route(
    rows: Sequence[dict[str, Any]],
    step_px: float = 12.0,
    smooth_win: int = 3,
    max_jump_px: float = 80.0,
) -> list[list[float]]:
    """Resampled route (map px) from recorded manual poses (teach and repeat).

    Keeps only measured poses, drops localization jumps, lightly smooths the
    driven line and resamples it at a uniform arc-length step, so the road the
    user actually drove (not the drawn vertices) can be saved as a preset.
    """
    pts: list[tuple[float, float]] = []
    for r in rows:
        if r.get("x") is None or not r.get("good"):
            continue
        p = (float(r["x"]), float(r["y"]))
        if pts and math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > max_jump_px:
            continue
        pts.append(p)
    if len(pts) < 3:
        return [[round(p[0], 1), round(p[1], 1)] for p in pts]

    if smooth_win and smooth_win > 1:
        half = smooth_win // 2
        smoothed: list[tuple[float, float]] = []
        for i in range(len(pts)):
            lo = max(0, i - half)
            hi = min(len(pts), i + half + 1)
            n = hi - lo
            smoothed.append(
                (
                    sum(pts[j][0] for j in range(lo, hi)) / n,
                    sum(pts[j][1] for j in range(lo, hi)) / n,
                )
            )
        pts = smoothed

    step = max(1.0, float(step_px))
    out: list[tuple[float, float]] = [pts[0]]
    dist = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        pos = 0.0
        while dist + (seg - pos) >= step:
            pos += step - dist
            k = pos / seg
            out.append((x0 + (x1 - x0) * k, y0 + (y1 - y0) * k))
            dist = 0.0
        dist += seg - pos
    tail = pts[-1]
    if math.hypot(tail[0] - out[-1][0], tail[1] - out[-1][1]) > 1e-9:
        out.append(tail)
    return [[round(x, 1), round(y, 1)] for x, y in out]


class KeyboardStateSource:
    """W/A/S/D/SPACE state via GetAsyncKeyState (no hook, focus-independent)."""

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32

    def snapshot(self) -> dict[str, bool]:
        return {name: bool(self._user32.GetAsyncKeyState(vk) & 0x8000) for vk, name in _VK.items()}


class ManualDriveRecorder(threading.Thread):
    """Samples key states + localizer poses into output/manual_dbg_*.jsonl."""

    def __init__(
        self,
        loc: Any,
        route: Sequence[Sequence[float]],
        params: dict[str, Any] | None = None,
        out_dir: str = "output",
        period: float = 0.05,
        keys_source: Callable[[], dict[str, bool]] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.loc = loc
        self.route = [(float(x), float(y)) for x, y in route]
        self.params = dict(params or {})
        self.out_dir = out_dir
        self.period = float(period)
        self._keys_source = keys_source
        self._stop_ev = threading.Event()
        self.path: str | None = None

    def run(self) -> None:
        keys_source = self._keys_source or KeyboardStateSource().snapshot

        self.path = os.path.join(
            self.out_dir, "manual_dbg_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl"
        )
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            fh = open(self.path, "w", encoding="utf-8")
        except OSError as exc:
            logger.warning("manual record: cannot open %s: %s", self.path, exc)
            return

        try:
            header = dict(
                kind="manual-log", ts=time.time(), route=list(self.route), params=dict(self.params)
            )
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            fh.flush()
            while not self._stop_ev.is_set():
                keys = keys_source()
                latest = getattr(self.loc, "latest", None) or {}
                pose = latest.get("pose") or {}
                mp = latest.get("map_px")
                th = pose.get("th")
                row = dict(
                    t=round(time.time(), 4),
                    keys="".join(k for k in _KEYS if keys.get(k)),
                    x=round(float(mp[0]), 1) if mp else None,
                    y=round(float(mp[1]), 1) if mp else None,
                    th=round(float(th), 2) if th is not None else None,
                    good=bool(latest.get("good", False)),
                    speed_kmh=latest.get("speed_kmh"),
                    speed_ok=bool(latest.get("speed_ok", False)),
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                self._stop_ev.wait(self.period)
        finally:
            fh.close()
            logger.info("manual record: saved %s", self.path)

    def stop(self) -> None:
        self._stop_ev.set()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout=1.5)
