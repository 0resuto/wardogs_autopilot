"""Flight recorder telemetry logger for navigation runs.

Writes JSONL logs of autopilot ticks to output/nav_dbg_*.jsonl with automatic
pruning of old runs (keeping at most 10 newest files).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..common.log import get_logger

logger = get_logger("nav_telemetry")


class NavTelemetryLogger:
    """Manages the lifecycle of a navigation JSONL telemetry log."""

    def __init__(self, dbg_target: bool | str = False) -> None:
        self.dbg = dbg_target
        self._dbg_f: Any = None
        self._dbg_name: str | None = None
        self._dbg_buf: list[dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        """True if debugging/logging is enabled."""
        return bool(self.dbg)

    def open(self, route_pts: list[tuple[float, float]], params: dict[str, Any]) -> None:
        """Open the JSONL log file and write metadata header."""
        if self._dbg_f is not None or not self.dbg:
            return

        if isinstance(self.dbg, str) and self.dbg:
            path = self.dbg
            dd = os.path.dirname(path)
            if dd:
                os.makedirs(dd, exist_ok=True)
        else:
            out_dir = "output"
            os.makedirs(out_dir, exist_ok=True)
            try:
                old_logs = sorted(
                    p for p in os.listdir(out_dir)
                    if p.startswith("nav_dbg_") and p.endswith(".jsonl")
                )
                while len(old_logs) >= 10:
                    try:
                        os.remove(os.path.join(out_dir, old_logs[0]))
                    except OSError:
                        pass
                    old_logs.pop(0)
            except Exception:
                pass
            path = os.path.join(
                out_dir, "nav_dbg_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl"
            )

        self._dbg_name = path
        try:
            self._dbg_f = open(path, "w", encoding="utf-8")
            meta = dict(
                kind="nav-log",
                ts=time.time(),
                route=list(route_pts),
                params=params,
            )
            self._dbg_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to open nav debug log %s: %s", path, exc)
            self._dbg_f = None

    def tick(
        self,
        row: dict[str, Any],
        route_pts: list[tuple[float, float]],
        params: dict[str, Any],
    ) -> None:
        """Record a single navigation tick into the buffer."""
        if not self.dbg:
            return
        if self._dbg_f is None:
            self.open(route_pts, params)
        self._dbg_buf.append(row)
        if len(self._dbg_buf) >= 64:
            self.flush()

    def flush(self) -> None:
        """Flush buffered telemetry rows to disk."""
        fh = self._dbg_f
        if fh is None:
            return
        for r in self._dbg_buf:
            try:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            except (ValueError, TypeError):
                pass
        self._dbg_buf = []
        try:
            fh.flush()
        except (ValueError, OSError):
            pass

    def close(self) -> None:
        """Flush remaining buffer and close the file handle."""
        self.flush()
        fh = self._dbg_f
        self._dbg_f = None
        if fh is None:
            return
        try:
            fh.close()
        except OSError:
            pass
