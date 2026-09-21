"""Unit tests for map viewer performance, crop geometry, tag preservation, and zoom."""

import json
import os
import sys
import time
import tkinter as tk
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.ui.app import App
from autopilot.ui.imaging import to_photo


def load_test_cfg():
    cfg_path = os.path.join(ROOT, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def wait_for_map(app, timeout=5.0):
    t0 = time.time()
    while app._map8 is None and time.time() - t0 < timeout:
        app.update()
        time.sleep(0.05)
    if app._map8 is None:
        app._load_map()


class TestMapViewer(unittest.TestCase):
    def test_photo_image_perf(self):
        """to_photo() must convert a 1000x800 image in under 35 ms."""
        root = tk.Tk()
        root.withdraw()
        try:
            test_img = np.random.randint(0, 255, (800, 1000, 3), dtype=np.uint8)
            _ = to_photo(test_img)  # Warmup

            iters = 10
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = to_photo(test_img)
            elapsed = (time.perf_counter() - t0) / iters * 1000.0
            self.assertLess(elapsed, 45.0, f"to_photo() took {elapsed:.2f} ms (expected <45 ms)")
        finally:
            root.destroy()

    def test_bg_crop_and_bounds(self):
        """_bg_crop() must render valid dimensions within timing budget across zoom scales."""
        cfg = load_test_cfg()
        app = App(cfg)
        app.withdraw()
        try:
            wait_for_map(app)
            scales_to_test = [0.15, 0.5, 1.0, 2.5, 8.0, 16.0]
            for s in scales_to_test:
                vis_wu = 800.0 / s
                vis_hu = 600.0 / s
                m = 0.25
                ru = 500.0 - m * vis_wu
                rv = 500.0 - m * vis_hu
                rw = vis_wu * (1 + 2 * m)
                rh = vis_hu * (1 + 2 * m)

                t0 = time.perf_counter()
                img = app._bg_crop(s, ru, rv, rw, rh)
                dt = (time.perf_counter() - t0) * 1000.0

                dw = max(1, int(round(rw * s)))
                dh = max(1, int(round(rh * s)))
                self.assertEqual(img.shape[:2], (dh, dw))
                self.assertLess(dt, 50.0, f"_bg_crop took {dt:.2f} ms at scale {s}")
        finally:
            app.destroy()

    def test_canvas_tag_preservation(self):
        """_bg_update() must not wipe overlay items ('marker', 'arrow', 'chunkbox')."""
        cfg = load_test_cfg()
        app = App(cfg)
        app.withdraw()
        try:
            wait_for_map(app)
            c = app._map_canvas
            c.winfo_width = lambda: 800  # type: ignore[method-assign]
            c.winfo_height = lambda: 600  # type: ignore[method-assign]

            c.create_line(10, 10, 50, 50, fill="red", tags="marker")
            c.create_oval(20, 20, 30, 30, fill="yellow", tags="arrow")
            c.create_rectangle(5, 5, 40, 40, outline="blue", tags="chunkbox")

            app._bg_update("map", 0.5, 10.0, 20.0)
            app._bg_update("map", 0.5, 15.0, 25.0)
            app._bg_update("map", 0.8, -100.0, -100.0)

            self.assertEqual(len(c.find_withtag("marker")), 1)
            self.assertEqual(len(c.find_withtag("arrow")), 1)
            self.assertEqual(len(c.find_withtag("chunkbox")), 1)
            self.assertEqual(len(c.find_withtag("bg")), 1)
            self.assertEqual(c.find_all()[0], c.find_withtag("bg")[0])
        finally:
            app.destroy()

    def test_rapid_zoom_accumulation(self):
        """Rapid mouse wheel events must smoothly accumulate scale without losing anchor."""
        cfg = load_test_cfg()
        app = App(cfg)
        app.withdraw()
        try:
            wait_for_map(app)
            c = app._map_canvas
            c.winfo_width = lambda: 800  # type: ignore[method-assign]
            c.winfo_height = lambda: 600  # type: ignore[method-assign]

            app._map_disp = (0.2, 100.0, 100.0)
            app._map_fit_scale = 0.2

            class FakeWheelEvent:
                def __init__(self, delta, x, y):
                    self.delta = delta
                    self.x = x
                    self.y = y

            app._map_zoom(FakeWheelEvent(120, 400, 300))
            assert app._map_pending is not None
            s1 = app._map_pending[0]
            app._map_zoom(FakeWheelEvent(120, 400, 300))
            assert app._map_pending is not None
            s2 = app._map_pending[0]
            app._map_zoom(FakeWheelEvent(120, 400, 300))
            assert app._map_pending is not None
            s3 = app._map_pending[0]

            self.assertGreater(s2, s1)
            self.assertGreater(s3, s2)
            expected = 0.2 * (1.2**3)
            self.assertAlmostEqual(s3, expected, places=4)

            app._map_flush_view()
            assert app._map_disp is not None
            self.assertEqual(app._map_disp[0], s3)
            self.assertIsNone(app._map_pending)
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
