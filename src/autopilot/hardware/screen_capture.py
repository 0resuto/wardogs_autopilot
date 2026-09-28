"""Screen capture (mss). Minimap region is given in absolute screen pixels."""

import ctypes

import mss
import numpy as np

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass


class ScreenCapture:
    def __init__(self, monitor: int = 0, roi=None) -> None:
        factory = getattr(mss, "MSS", None) or mss.mss
        self._sct = factory()
        self.monitor_index = monitor
        self.monitors = self._sct.monitors
        self.region: dict[str, int] | None = None
        self.set_roi(roi)

    def _region_for(self, roi) -> dict[str, int]:
        """Clamp an ROI box to the target monitor and return an mss region dict."""
        x, y, w, h = roi
        if 0 < self.monitor_index < len(self.monitors):
            mon = self.monitors[self.monitor_index]
        elif self.monitors:
            mon = self.monitors[0]
        else:
            mon = None

        if mon is None:
            return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}

        m_left = int(mon["left"])
        m_top = int(mon["top"])
        m_w = int(mon["width"])
        m_h = int(mon["height"])

        clamped_x = max(m_left, min(m_left + m_w - 16, int(x)))
        clamped_y = max(m_top, min(m_top + m_h - 16, int(y)))
        clamped_w = max(16, min(int(w), m_left + m_w - clamped_x))
        clamped_h = max(16, min(int(h), m_top + m_h - clamped_y))
        return {
            "left": clamped_x,
            "top": clamped_y,
            "width": clamped_w,
            "height": clamped_h,
        }

    def set_roi(self, roi) -> None:
        self.region = None if roi is None else self._region_for(roi)

    def grab(self) -> np.ndarray:
        """BGR (HxWx3). Without ROI - the whole target monitor."""
        if self.region is None:
            monitors = self.monitors or []
            if not monitors:
                raise OSError("screen capture: no monitors reported by mss")
            index = self.monitor_index if 0 <= self.monitor_index < len(monitors) else 0
            raw = self._sct.grab(monitors[index])
        else:
            raw = self._sct.grab(self.region)
        return np.array(raw)[:, :, :3].copy()

    def grab_region(self, roi) -> np.ndarray:
        """BGR (HxWx3) of an arbitrary ROI on the target monitor."""
        raw = self._sct.grab(self._region_for(roi))
        return np.array(raw)[:, :, :3].copy()

    def close(self) -> None:
        """Release underlying mss and GDI Device Context resources."""
        if hasattr(self, "_sct") and self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
