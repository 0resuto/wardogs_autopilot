"""Unit tests for input validation, boundary checking, and preset path sanitization."""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.ui.app import App
from autopilot.vision import locator


def load_test_cfg():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


class TestCodeQuality(unittest.TestCase):
    def test_tune_range_validation(self):
        """_apply_tune() must reject out-of-range parameters and accept valid ones."""
        cfg = load_test_cfg()
        app = App(cfg)
        app.withdraw()
        try:
            # Test ratio > 1.0
            app._tune_vars["ratio_local"].set("1.5")
            app._apply_tune()
            self.assertIn("Invalid TRACK ratio", app._tune_status.cget("text"))

            # Test negative radius
            app._tune_vars["ratio_local"].set("0.85")
            app._tune_vars["track_radius"].set("-10")
            app._apply_tune()
            self.assertIn("Invalid TRACK rad", app._tune_status.cget("text"))

            # Test valid inputs
            app._tune_vars["track_radius"].set("900")
            app._apply_tune()
            self.assertIn("applied", app._tune_status.cget("text"))
        finally:
            app.destroy()

    def test_roi_bounds_validation(self):
        """_apply_roi() must reject negative or microscopic dimensions and accept valid ROIs."""
        cfg = load_test_cfg()
        app = App(cfg)
        app.withdraw()
        try:
            # Invalid tiny width
            app._roi_vars["w"].set("5")
            app._apply_roi()
            self.assertIn("Error:", app._roi_status.cget("text"))

            # Invalid negative x
            app._roi_vars["w"].set("300")
            app._roi_vars["x"].set("-20")
            app._apply_roi()
            self.assertIn("Error:", app._roi_status.cget("text"))

            # Valid ROI
            app._roi_vars["x"].set("45")
            app._roi_vars["y"].set("1009")
            app._roi_vars["w"].set("336")
            app._roi_vars["h"].set("277")
            app._apply_roi()
            self.assertIn("OK:", app._roi_status.cget("text"))
        finally:
            app.destroy()

    def test_preset_path_sanitization(self):
        """_preset_path() must sanitize filenames against directory traversal."""
        cfg = load_test_cfg()
        app = App(cfg)
        app.withdraw()
        try:
            p = app._preset_path("../../traversal_attack")
            expected_dir = os.path.normpath(app._preset_dir())
            actual_dir = os.path.dirname(os.path.normpath(p))
            self.assertEqual(actual_dir, expected_dir)
        finally:
            app.destroy()

    def test_locator_set_map(self):
        """locator.set_map() must cleanly switch active map without state corruption."""
        locator.set_map("zestafona")
        self.assertEqual(locator.map_name(), "zestafona")
        locator.set_map("bakurani")
        self.assertEqual(locator.map_name(), "bakurani")
        locator.set_map("zestafona")
        self.assertEqual(locator.map_name(), "zestafona")


if __name__ == "__main__":
    unittest.main()
