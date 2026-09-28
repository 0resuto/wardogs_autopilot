"""Tests for the HUD speedometer OCR: digit recognition and temporal filter.

Synthetic frames are composed from the committed digit atlas over a noisy
gradient background at several scales, so the tests exercise binarization,
segmentation, normalization and matching without any screen or font at runtime.
"""

import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.vision.hud_speed import (  # noqa: E402
    DigitAtlas,
    SpeedFilter,
    SpeedReading,
    SpeedRecognizer,
)


def _compose(atlas: DigitAtlas, text: str, scale: float, seed: int = 0) -> np.ndarray:
    """Bright atlas glyphs over a noisy gradient background (game-like ROI)."""
    glyphs = [
        cv2.resize(
            atlas.cells[atlas.labels.index(ch)],
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        for ch in text
    ]
    glyph_h = max(g.shape[0] for g in glyphs)
    gap = max(2, int(round(3 * scale)))
    width = sum(g.shape[1] for g in glyphs) + gap * (len(glyphs) + 1)
    height = glyph_h + 2 * gap
    canvas = np.full((height, width), 26, np.uint8)

    gradient = np.linspace(0, 55, height, dtype=np.float32)[:, None]
    canvas = np.clip(canvas.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)

    rng = np.random.default_rng(seed)
    noise = rng.integers(-7, 8, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    x = gap
    for glyph in glyphs:
        gh, gw = glyph.shape
        y = gap + (glyph_h - gh) // 2
        region = canvas[y : y + gh, x : x + gw]
        canvas[y : y + gh, x : x + gw] = np.maximum(region, glyph)
        x += gw + gap
    return cv2.GaussianBlur(canvas, (3, 3), 0.5)


class TestDigitRecognition(unittest.TestCase):
    atlas: DigitAtlas
    recognizer: SpeedRecognizer

    @classmethod
    def setUpClass(cls):
        cls.atlas = DigitAtlas()
        cls.recognizer = SpeedRecognizer(cls.atlas)

    def test_reads_values_across_scales(self):
        for text in ("0", "7", "44", "123"):
            for scale in (0.7, 1.0, 1.4, 2.0, 2.6):
                with self.subTest(text=text, scale=scale):
                    frame = _compose(self.atlas, text, scale, seed=hash((text, scale)) % 1000)
                    reading = self.recognizer.read(frame)
                    self.assertTrue(reading.ok, f"{text} @ {scale}: {reading.detail}")
                    self.assertEqual(reading.kmh, int(text))

    def test_ignores_unrelated_blob_above_the_row(self):
        frame = _compose(self.atlas, "42", 1.6, seed=7)
        blob = np.zeros((8, frame.shape[1]), np.uint8)
        blob[2:6, 2:6] = 255
        canvas = np.vstack([blob, frame])
        reading = self.recognizer.read(canvas)
        self.assertTrue(reading.ok, reading.detail)
        self.assertEqual(reading.kmh, 42)

    def test_ignores_letters_around_digits(self):
        frame = _compose(self.atlas, "44KMH", 1.4, seed=5)
        reading = self.recognizer.read(frame)
        self.assertTrue(reading.ok, reading.detail)
        self.assertEqual(reading.kmh, 44)

    def test_rejects_letter_only_row(self):
        frame = _compose(self.atlas, "SPD", 1.4, seed=6)
        reading = self.recognizer.read(frame)
        self.assertFalse(reading.ok)

    def test_rejects_blank_frame(self):
        frame = np.full((48, 120), 30, np.uint8)
        reading = self.recognizer.read(frame)
        self.assertFalse(reading.ok)
        self.assertIsNone(reading.kmh)

    def test_rejects_noise_only_frame(self):
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 255, (48, 120), dtype=np.uint8)
        reading = self.recognizer.read(frame)
        self.assertIsNone(reading.kmh)

    def test_rejects_too_many_glyphs(self):
        frame = _compose(self.atlas, "1234", 1.5, seed=11)
        reading = self.recognizer.read(frame)
        self.assertFalse(reading.ok)


class TestRealCapture(unittest.TestCase):
    def test_reads_speed_from_real_hud_crop(self):
        """Reference frame: real in-game screenshot (tests/data/speed_44.png)."""
        path = os.path.join(ROOT, "tests", "data", "speed_44.png")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image, f"fixture not found: {path}")
        assert image is not None

        digits_roi = image[63:96, 15:60]
        reading = SpeedRecognizer(DigitAtlas()).read(digits_roi)
        self.assertTrue(reading.ok, reading.detail)
        self.assertEqual(reading.kmh, 44)

    def test_reads_digits_from_wider_row_with_unit(self):
        """A ROI that also contains 'KM/H' must still read the digits."""
        path = os.path.join(ROOT, "tests", "data", "speed_44.png")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert image is not None

        reading = SpeedRecognizer(DigitAtlas()).read(image[63:96, 15:193])
        self.assertTrue(reading.ok, reading.detail)
        self.assertEqual(reading.kmh, 44)

    def test_rejects_unit_and_label_areas(self):
        """KM/H and the SPD label must not be mistaken for a speed."""
        path = os.path.join(ROOT, "tests", "data", "speed_44.png")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert image is not None

        recognizer = SpeedRecognizer(DigitAtlas())
        self.assertIsNone(recognizer.read(image[63:96, 90:193]).kmh)
        self.assertIsNone(recognizer.read(image[5:55, 15:185]).kmh)


class TestSpeedFilter(unittest.TestCase):
    @staticmethod
    def _reading(value: int | None) -> SpeedReading:
        if value is None:
            return SpeedReading(None, 0.0, 0.0, 0, "not read")
        return SpeedReading(value, 0.95, 0.4, len(str(value)), str(value))

    def test_accepts_initial_and_smooth_values(self):
        flt = SpeedFilter(jump_kmh=20.0, confirm_frames=2, timeout_s=1.0)
        value, ok = flt.update(self._reading(44), now=0.0)
        self.assertEqual((value, ok), (44, True))

        value, ok = flt.update(self._reading(51), now=0.1)
        self.assertEqual((value, ok), (51, True))

    def test_jump_requires_confirmation(self):
        flt = SpeedFilter(jump_kmh=20.0, confirm_frames=2, timeout_s=1.0)
        flt.update(self._reading(44), now=0.0)

        value, ok = flt.update(self._reading(160), now=0.1)
        self.assertEqual(value, 44)
        self.assertFalse(ok)

        value, ok = flt.update(self._reading(160), now=0.2)
        self.assertEqual((value, ok), (160, True))

    def test_unread_frames_expire_to_not_measured(self):
        flt = SpeedFilter(jump_kmh=20.0, confirm_frames=2, timeout_s=0.5)
        flt.update(self._reading(30), now=0.0)

        value, ok = flt.update(self._reading(None), now=0.3)
        self.assertEqual((value, ok), (30, True))

        value, ok = flt.update(self._reading(None), now=1.0)
        self.assertEqual(value, 30)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
