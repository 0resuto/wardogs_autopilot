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

    def set_roi(self, roi) -> None:
        if roi is None:
            self.region = None
            return
        x, y, w, h = roi
        self.region = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}

    def grab(self) -> np.ndarray:
        """BGR (HxWx3). Without ROI - the whole target monitor."""
        if self.region is None:
            mon = self.monitors[self.monitor_index]
            raw = self._sct.grab(mon)
        else:
            raw = self._sct.grab(self.region)
        return np.array(raw)[:, :, :3].copy()

    def close(self) -> None:
        """Release underlying mss and GDI Device Context resources."""
        if hasattr(self, "_sct") and self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
