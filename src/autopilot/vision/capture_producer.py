"""Screen-capture producer thread feeding the localization consumer."""

from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np

from .. import crashlog
from ..common.config import AppConfig, CaptureConfig
from ..hardware.screen_capture import ScreenCapture
from .hud_speed import SpeedSensor


class _CaptureProducer(threading.Thread):
    """Screen capture at a fixed rate into a single-slot drop-old queue.

    Each item: dict(ts=time.time(), gray=mm_grayscale, bgr=mm_color,
    roi=roi, mask=ui_bool, speed_kmh=int|None, speed_ok=bool) with the UI mask
    already resized to the frame. The color frame is kept for debug dumps
    only; the pipeline uses gray. Speed is read from the optional speedometer
    ROI (capture.speed_roi). Errors land in crash.log and self.error (the
    consumer surfaces it).
    """

    def __init__(
        self,
        cfg: CaptureConfig | AppConfig | dict,
        mask,
        frame_source,
        stop,
        out_q,
    ) -> None:
        super().__init__(daemon=True)
        if isinstance(cfg, CaptureConfig):
            self.capture_cfg = cfg
            self.cfg = {"capture": cfg.model_dump()}
        elif isinstance(cfg, AppConfig):
            self.capture_cfg = cfg.capture
            self.cfg = cfg.to_dict()
        elif isinstance(cfg, dict):
            self.cfg = cfg
            raw_cap = cfg.get("capture")
            cap_dict = raw_cap if isinstance(raw_cap, dict) else cfg
            try:
                self.capture_cfg = CaptureConfig(**cap_dict)
            except Exception:
                self.capture_cfg = CaptureConfig()
        else:
            self.capture_cfg = CaptureConfig()
            self.cfg = {"capture": self.capture_cfg.model_dump()}

        self.mask = np.asarray(mask, bool)
        self.frame_source = frame_source
        self._stop = stop
        self._queue = out_q
        self.error: str | None = None
        self.speed_sensor = SpeedSensor()

    def set_roi(self, roi: list[int] | tuple[int, ...]) -> None:
        roi_list = [int(v) for v in roi]
        if isinstance(self.cfg, dict):
            self.cfg.setdefault("capture", {})["mmap_roi"] = roi_list
        if hasattr(self, "capture_cfg"):
            self.capture_cfg.mmap_roi = roi_list

    def set_speed_roi(self, roi: list[int] | tuple[int, ...] | None) -> None:
        roi_list = None if roi is None else [int(v) for v in roi]
        if isinstance(self.cfg, dict):
            self.cfg.setdefault("capture", {})["speed_roi"] = roi_list
        if hasattr(self, "capture_cfg"):
            self.capture_cfg.speed_roi = roi_list

    def run(self) -> None:
        cap = None
        last_sign = None
        period = 1.0 / max(1.0, float(self.capture_cfg.fps))  # safe default for except
        try:
            while not self._stop.is_set():
                try:
                    # re-read every loop so the UI's fps field applies live
                    cap_cfg = self.cfg.get("capture") if isinstance(self.cfg, dict) else None
                    if cap_cfg and isinstance(cap_cfg, dict):
                        roi = cap_cfg.get("mmap_roi")
                        speed_roi = cap_cfg.get("speed_roi")
                        mon = int(cap_cfg.get("monitor", 0) or 0)
                        fps = float(
                            cap_cfg.get("fps", self.capture_cfg.fps) or self.capture_cfg.fps
                        )
                    else:
                        roi = self.capture_cfg.mmap_roi
                        speed_roi = self.capture_cfg.speed_roi
                        mon = self.capture_cfg.monitor
                        fps = float(self.capture_cfg.fps)
                    period = 1.0 / max(1.0, fps)
                    if not roi:
                        self._stop.wait(period)
                        continue
                    roi = tuple(int(v) for v in roi)
                    if self.frame_source is not None:
                        frame = self.frame_source()
                    else:
                        sign = (mon, roi)
                        if cap is None or sign != last_sign:
                            if cap is not None and hasattr(cap, "close"):
                                cap.close()
                            cap = ScreenCapture(mon, roi)
                            last_sign = sign
                        frame = cap.grab()
                    mm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = mm.shape
                    ui = self.mask
                    if ui.shape[:2] != (h, w):
                        ui = (
                            cv2.resize(ui.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            > 0
                        )
                    item = dict(
                        ts=time.time(),
                        gray=mm,
                        bgr=frame,
                        roi=roi,
                        mask=ui,
                        speed_kmh=None,
                        speed_ok=False,
                        speed_frame=None,
                        speed_mask=None,
                        speed_boxes=(),
                    )
                    if speed_roi and cap is not None and self.speed_sensor.available:
                        speed_frame = cap.grab_region(speed_roi)
                        kmh, ok = self.speed_sensor.update(speed_frame, item["ts"])
                        reading = self.speed_sensor.last_reading
                        item["speed_kmh"] = kmh
                        item["speed_ok"] = ok
                        if reading is not None:
                            item["speed_frame"] = reading.frame
                            item["speed_mask"] = reading.mask
                            item["speed_boxes"] = reading.boxes
                    try:
                        self._queue.get_nowait()  # drop the stale frame
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(item)
                    self.error = None
                except Exception as exc:  # noqa: BLE001
                    if cap is not None and hasattr(cap, "close"):
                        try:
                            cap.close()
                        except Exception:
                            pass
                    cap = None
                    last_sign = None
                    self.error = str(exc)
                    if not self._stop.is_set():
                        crashlog.log("capture producer error", exc)
                    self._stop.wait(max(period, 0.5))
                    continue
                self._stop.wait(period)
        finally:
            if cap is not None and hasattr(cap, "close"):
                cap.close()
