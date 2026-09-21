"""Top-level window state: size/position/maximize persistence.

Restores the last geometry from config, clamps it to the monitors that are
actually present, and saves it back (debounced) on move/resize/close.
"""

import ctypes
import re
import tkinter as tk
from ctypes import wintypes

_GEO_RE = re.compile(
    r"^(\d+)x(\d+)([+-]\d+|[+]-\d+)([+-]\d+|[+]-\d+)$")


def _offset(s: str) -> int:
    """"+3060" -> 3060, "-100" -> -100; old Tk could write "+-100"."""
    if s.startswith("+-"):
        s = "-" + s[2:]
    return int(s)


class _MonitorInfo(ctypes.Structure):
    """MONITORINFO (winuser.h): monitor and work areas."""
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def monitor_workareas():
    """Work areas of all monitors in virtual screen coordinates.

    (without taskbar) -> [(left, top, right, bottom)]. Also handles
    monitors positioned to the left of the primary one (negative coords).
    Empty list if enumeration is unavailable.
    """
    user32 = ctypes.windll.user32
    areas = []

    def _cb(hmon, hdc, lprc, data):
        mi = _MonitorInfo()
        mi.cbSize = ctypes.sizeof(_MonitorInfo)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            r = mi.rcWork
            areas.append((r.left, r.top, r.right, r.bottom))
        return True

    enum_cb = ctypes.WINFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT), ctypes.c_void_p)(_cb)
    try:
        if not user32.EnumDisplayMonitors(None, None, enum_cb, None):
            return []
    except Exception:  # noqa: BLE001
        return []
    return areas


def clamp_to_visible_monitor(x: int, y: int, w: int, h: int):
    """If a w×h window at (x,y) has drifted beyond all monitors, return a
    position centered on the work area of the nearest one.

    The check covers ALL monitors, not just the primary: a position on a
    secondary monitor (incl. right/left of the primary) is valid.
    """
    areas = monitor_workareas()
    if not areas:
        return x, y
    for ax0, ay0, ax1, ay1 in areas:
        ox0, oy0 = max(ax0, x), max(ay0, y)
        ox1, oy1 = min(ax1, x + w), min(ay1, y + h)
        if ox1 - ox0 >= 120 and oy1 - oy0 >= 40:
            return x, y   # enough is visible - keep as-is
    cx, cy = x + w / 2.0, y + h / 2.0
    nearest = min(
        areas,
        key=lambda a: ((a[0] + a[2]) / 2.0 - cx) ** 2
                      + ((a[1] + a[3]) / 2.0 - cy) ** 2)
    return (nearest[0] + nearest[2] - w) // 2, (nearest[1] + nearest[3] - h) // 2


class WindowState:
    """Save/restore the Tk window's size, position, and zoom state.

    root     — the Tk window being managed.
    cfg      — the config dict; ``cfg["window"]`` is read/updated.
    persist  — callable writing cfg to disk (e.g. App._save_cfg).

    The geometry write is debounced on "<Configure>" and flushed on demand via
    ``save_now()`` (used by the close handler).
    """

    def __init__(self, root: tk.Tk, cfg: dict, persist) -> None:
        self._root = root
        self._cfg = cfg
        self._persist = persist
        self._normal_geometry: str | None = None   # last "normal" (non-zoomed) geometry
        self._zoomed = False
        self._save_job: str | None = None

    def bind_configure(self) -> None:
        self._root.bind("<Configure>", self._on_configure)

    def restore(self, cfg: dict) -> str | None:
        """Restore the window's last size and position from config (window).

        The position is validated against ALL monitors (not just the primary):
        a window closed on a secondary monitor right/left of the main one
        should return there. If a monitor was unplugged, slide the window
        into the area of the nearest live one so it doesn't open "nowhere".
        Returns the geometry string to apply (or None for the default).
        """
        win = cfg.get("window") or {}
        g = win.get("geometry")
        if g:
            m = _GEO_RE.match(g)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
                x, y = _offset(m.group(3)), _offset(m.group(4))
                x, y = clamp_to_visible_monitor(x, y, w, h)
                self._normal_geometry = "%dx%d+%d+%d" % (w, h, x, y)
        self._zoomed = bool(win.get("zoomed"))
        if self._zoomed:
            # maximize after rendering, otherwise Windows may "forget" the
            # state: apply geometry/zoom to the already-created window
            self._root.after_idle(self._apply_zoom_state)
        return self._normal_geometry

    def _apply_zoom_state(self) -> None:
        try:
            if self._zoomed and self._root.state() != "zoomed":
                self._root.state("zoomed")
        except tk.TclError:  # noqa: BLE001
            self._zoomed = False

    def _on_configure(self, e=None) -> None:
        """Window moved/resized/maximized - remember and defer the write.

        The real layout fires dozens of <Configure> per second; we write to
        config with a debounce so the last state goes to disk once.
        Maximizing (zoomed) does not overwrite the "normal" geometry.
        """
        if e is not None and e.widget is not self._root:
            return
        try:
            state = self._root.state()
        except tk.TclError:
            return
        if state == "zoomed":
            self._zoomed = True
        elif state == "normal":
            self._zoomed = False
            geo = self._root.geometry()
            if geo:
                self._normal_geometry = geo
        if self._save_job is not None:
            self._root.after_cancel(self._save_job)
        self._save_job = self._root.after(600, self._flush)

    def _flush(self) -> None:
        """Flush the deferred geometry write to config."""
        self._save_job = None
        try:
            self.save_now()
        except Exception:  # noqa: BLE001
            pass

    def save_now(self) -> None:
        """Write the window size/position/maximize state to config (if changed)."""
        if self._save_job is not None:
            try:
                self._root.after_cancel(self._save_job)
            except Exception:  # noqa: BLE001
                pass
            self._save_job = None
        win = self._cfg.setdefault("window", {})
        old_geo, old_zoom = win.get("geometry"), win.get("zoomed")
        new_geo = self._normal_geometry or self._root.geometry() or None
        new_zoom = bool(self._zoomed)
        if old_geo == new_geo and bool(old_zoom) == new_zoom:
            return   # nothing changed - don't touch the disk
        if new_geo:
            win["geometry"] = new_geo
        win["zoomed"] = new_zoom
        try:
            self._persist()
        except OSError:  # noqa: BLE001
            pass
