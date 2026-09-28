"""HUD speedometer digit recognition (OpenCV only, resolution independent).

The digit atlas is built once from data/hud/Barlow-Regular.ttf by
tools/build_hud_atlas.py into data/hud/barlow_digits.png. Frame processing:
bright-text binarization (background flattening + Otsu), connected-component
segmentation, per-glyph normalization into the canonical atlas cell and
normalized cross-correlation matching. Glyph size on screen does not matter:
every component is rescaled into the same cell, so the same atlas works on any
resolution / HUD scale.

A small temporal filter (SpeedFilter) suppresses single-frame misreads:
teleporting values must repeat before they are trusted.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .. import PROJECT_ROOT
from ..common.log import get_logger

logger = get_logger("hud_speed")

GLYPH_W = 24
GLYPH_H = 32
GLYPH_MARGIN = 2
DIGITS = "0123456789"
ATLAS_LABELS = DIGITS + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_DIGITS = 3

HUD_DIR = os.path.join(PROJECT_ROOT, "data", "hud")
ATLAS_PATH = os.path.join(HUD_DIR, "barlow_digits.png")
ATLAS_META_PATH = os.path.join(HUD_DIR, "barlow_atlas.json")

MIN_COMPONENT_H = 6
UPSCALE_TARGET_H = 64


def fit_glyph(gray: np.ndarray, out_w: int = GLYPH_W, out_h: int = GLYPH_H) -> np.ndarray:
    """Scale a cropped glyph into the canonical cell, aspect preserved, centered."""
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w), np.uint8)
    inner_w = max(1, out_w - 2 * GLYPH_MARGIN)
    inner_h = max(1, out_h - 2 * GLYPH_MARGIN)
    scale = min(inner_w / float(w), inner_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (nw, nh), interpolation=interpolation)
    cell = np.zeros((out_h, out_w), np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    cell[y0 : y0 + nh, x0 : x0 + nw] = resized
    return cell


class DigitAtlas:
    """Per-digit normalized glyph cells rendered from the game font."""

    def __init__(
        self,
        path: str = ATLAS_PATH,
        labels: str = ATLAS_LABELS,
        meta_path: str | None = ATLAS_META_PATH,
    ) -> None:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"digit atlas not found: {path} (run tools/build_hud_atlas.py)")
        if image.shape[0] != GLYPH_H or image.shape[1] < GLYPH_W * len(labels):
            raise ValueError(f"digit atlas has unexpected shape {image.shape}")
        self.labels = labels
        self.cells = [image[:, i * GLYPH_W : (i + 1) * GLYPH_W] for i in range(len(labels))]
        self.meta: dict[str, Any] = {}
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    self.meta = json.load(fh)
            except (OSError, ValueError):
                self.meta = {}

    def match(self, cell: np.ndarray, jitter: int = 2) -> tuple[str, float, float]:
        """Best label, its correlation score and the margin to the runner-up."""
        padded = cv2.copyMakeBorder(
            cell, jitter, jitter, jitter, jitter, cv2.BORDER_CONSTANT, value=0
        )
        best_label = ""
        best = -1.0
        second = -1.0
        for label, templ in zip(self.labels, self.cells, strict=True):
            result = cv2.matchTemplate(padded, templ, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best:
                second = best
                best = score
                best_label = label
            elif score > second:
                second = score
        return best_label, best, best - second


@dataclass(frozen=True)
class SpeedReading:
    """One frame of the speedometer ROI: recognized number or a rejection."""

    kmh: int | None
    score: float
    margin: float
    digits: int
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.kmh is not None


class SpeedRecognizer:
    """Reads an integer speed from a ROI around the speed digits.

    Extra glyphs inside the ROI (unit letters, HUD labels) are ignored: only
    the longest run of glyphs that confidently match a digit is returned.
    """

    def __init__(
        self,
        atlas: DigitAtlas | None = None,
        min_score: float = 0.60,
        min_margin: float = 0.04,
    ) -> None:
        self.atlas = atlas or DigitAtlas()
        self.min_score = min_score
        self.min_margin = min_margin

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Bright-text mask: flatten the background, then Otsu the residual."""
        sigma = max(2.0, min(gray.shape[:2]) / 8.0)
        background = cv2.GaussianBlur(gray, (0, 0), sigma)
        residual = cv2.subtract(gray, background)
        _, binary = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    @staticmethod
    def _components(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes: list[tuple[int, int, int, int]] = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if h < MIN_COMPONENT_H or area < 4:
                continue
            if h > 5 * w:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))
        return boxes

    @staticmethod
    def _digit_row(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Largest row of glyphs with matching vertical extents, left to right."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[0])
        rows: list[list[tuple[int, int, int, int]]] = []
        for box in boxes:
            placed = False
            for row in rows:
                y0 = min(b[1] for b in row)
                y1 = max(b[1] + b[3] for b in row)
                h = max(b[3] for b in [*row, box])
                overlap = min(y1, box[1] + box[3]) - max(y0, box[1])
                if overlap >= 0.6 * h:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])
        rows.sort(key=lambda r: (len(r), sum(b[2] for b in r)), reverse=True)
        return sorted(rows[0], key=lambda b: b[0])

    def read(self, frame: np.ndarray) -> SpeedReading:
        """Recognize the speed in one captured ROI frame (gray or BGR)."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return SpeedReading(None, 0.0, 0.0, 0, "empty frame")

        if min(gray.shape[:2]) < UPSCALE_TARGET_H:
            factor = max(2, int(np.ceil(UPSCALE_TARGET_H / max(1, min(gray.shape[:2])))))
            gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

        binary = self._binarize(gray)
        boxes = self._components(binary)
        row = self._digit_row(boxes)
        if not row:
            return SpeedReading(None, 0.0, 0.0, 0, "no glyphs")

        max_h = max(b[3] for b in row)
        row = [b for b in row if b[3] >= 0.55 * max_h]
        if not row:
            return SpeedReading(None, 0.0, 0.0, 0, "no glyphs")

        runs: list[list[tuple[str, float, float]]] = []
        current: list[tuple[str, float, float]] = []
        ignored = 0
        for x, y, w, h in row:
            crop = gray[max(0, y - 1) : y + h + 1, max(0, x - 1) : x + w + 1]
            label, score, margin = self.atlas.match(fit_glyph(crop))
            confident = score >= self.min_score and margin >= self.min_margin
            if confident and label in DIGITS:
                current.append((label, score, margin))
            else:
                if confident or current:
                    ignored += 1
                if current:
                    runs.append(current)
                    current = []
        if current:
            runs.append(current)

        if not runs:
            return SpeedReading(None, 0.0, 0.0, 0, f"no confident digits ({len(row)} glyphs)")
        run = max(runs, key=len)
        if len(run) > MAX_DIGITS:
            return SpeedReading(None, 0.0, 0.0, 0, f"digit run too long ({len(run)})")

        text = "".join(label for label, _score, _margin in run)
        score = min(s for _label, s, _margin in run)
        margin = min(m for _label, _score, m in run)
        detail = f"{text} (ignored {ignored})" if ignored else text
        return SpeedReading(int(text), score, margin, len(run), detail)


class SpeedFilter:
    """Temporal gate: reject single-frame jumps, report a 'not measured' flag."""

    def __init__(
        self,
        jump_kmh: float = 20.0,
        confirm_frames: int = 2,
        timeout_s: float = 1.0,
    ) -> None:
        self.jump_kmh = float(jump_kmh)
        self.confirm_frames = max(1, int(confirm_frames))
        self.timeout_s = float(timeout_s)
        self.value: int | None = None
        self._pending: int | None = None
        self._pending_n = 0
        self._last_ok_t: float | None = None

    def update(self, reading: SpeedReading, now: float) -> tuple[int | None, bool]:
        """Return (filtered value, measured flag) for the navigation layer."""
        if not reading.ok or reading.kmh is None:
            if self._last_ok_t is not None and now - self._last_ok_t <= self.timeout_s:
                return self.value, True
            return self.value, False

        value = reading.kmh
        if self.value is None or abs(value - self.value) <= self.jump_kmh:
            self.value = value
        elif value == self._pending:
            self._pending_n += 1
            if self._pending_n >= self.confirm_frames:
                self.value = value
                self._pending = None
                self._pending_n = 0
        else:
            self._pending = value
            self._pending_n = 1

        accepted = value == self.value
        if accepted:
            self._last_ok_t = now
        else:
            return self.value, False
        return self.value, True
