"""Input-validation tests driven through the real Qt widgets.

Tuning-panel and capture-zone values are typed into the actual QLineEdit
widgets and applied through the production handlers; config writes are stubbed
so the repository config.json is never touched. The App smoke test builds the
full main window with map loading and the locator thread stubbed out.
"""

import os
import sys
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import AppConfig  # noqa: E402
from autopilot.ui.presets import PresetManager  # noqa: E402
from autopilot.ui.tabs.map_tab import MapTab  # noqa: E402
from autopilot.ui.tabs.roi_tab import RoiTab  # noqa: E402
from autopilot.vision import locator  # noqa: E402

_QT_APP = QApplication.instance() or QApplication([])


class TestTuningValidation(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig()
        self.saved = 0
        self.tab = MapTab(
            None,
            self.cfg,
            save_cfg_fn=self._on_save,
            loc_thread_supplier=lambda: None,
            app_cfg=self.cfg,
        )

    def _on_save(self):
        self.saved += 1

    def test_typed_values_are_applied_after_validation(self):
        self.tab.tune_inputs["ratio_local"].setText("1.5")
        self.tab.apply_tune()
        self.assertIn("Invalid TRACK ratio", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["locator"]["ratio_local"], 0.85)
        self.assertEqual(self.saved, 0)

        self.tab.tune_inputs["ratio_local"].setText("0.85")
        self.tab.tune_inputs["track_radius"].setText("-10")
        self.tab.apply_tune()
        self.assertIn("Invalid TRACK rad", self.tab.tune_status.text())
        self.assertEqual(self.saved, 0)

        self.tab.tune_inputs["track_radius"].setText("900")
        self.tab.apply_tune()
        self.assertIn("applied", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["locator"]["track_radius"], 900)
        self.assertEqual(self.tab.tune_vars["track_radius"].get(), "900")
        self.assertEqual(self.saved, 1)

    def test_reset_restores_schema_defaults(self):
        self.tab.tune_inputs["ratio_local"].setText("0.5")
        self.tab.apply_tune()
        self.assertEqual(self.tab.cfg["locator"]["ratio_local"], 0.5)

        self.tab.reset_tune()
        self.assertEqual(self.tab.cfg["locator"]["ratio_local"], 0.85)
        self.assertEqual(self.tab.tune_inputs["ratio_local"].text(), "0.85")
        self.assertEqual(self.tab.tune_vars["ratio_local"].get(), "0.85")


class TestRoiValidation(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig().to_dict()
        self.saved = 0
        self.tab = RoiTab(
            None,
            self.cfg,
            save_cfg_fn=self._on_save,
            screen_cap_supplier=lambda: None,
            loc_thread_supplier=lambda: None,
        )

    def _on_save(self):
        self.saved += 1

    def test_typed_coordinates_are_validated_and_applied(self):
        self.tab.coord_inputs["w"].setText("5")
        self.tab.apply_roi()
        self.assertIn("Error:", self.tab.status_lbl.text())
        self.assertEqual(self.saved, 0)

        self.tab.coord_inputs["w"].setText("300")
        self.tab.coord_inputs["x"].setText("-20")
        self.tab.apply_roi()
        self.assertIn("Error:", self.tab.status_lbl.text())
        self.assertEqual(self.saved, 0)

        for name, value in (("x", "45"), ("y", "1009"), ("w", "336"), ("h", "277")):
            self.tab.coord_inputs[name].setText(value)
        self.tab.apply_roi()
        self.assertIn("OK:", self.tab.status_lbl.text())
        self.assertEqual(self.cfg["capture"]["mmap_roi"], [45, 1009, 336, 277])
        self.assertEqual(self.tab.roi_vars["w"].get(), "336")
        self.assertEqual(self.saved, 1)


class TestPresetSanitization(unittest.TestCase):
    def test_traversal_is_neutralized(self):
        manager = PresetManager()
        path = manager.preset_path("../../etc/passwd")
        self.assertEqual(os.path.dirname(path), str(manager.presets_dir))
        self.assertEqual(os.path.basename(path), "passwd.json")


class TestLocatorMapSwitch(unittest.TestCase):
    def test_set_map_switches_and_invalidates_caches(self):
        locator.set_map("zestafona")
        self.assertEqual(locator.map_name(), "zestafona")

        locator.set_map("bakurani")
        self.assertEqual(locator.map_name(), "bakurani")
        self.assertIsNone(locator.get_store()._g["mu"])

        locator.set_map("zestafona")
        self.assertEqual(locator.map_name(), "zestafona")


class TestAppSmoke(unittest.TestCase):
    def test_app_builds_with_unified_map_tab(self):
        from autopilot.ui.app import App
        from autopilot.vision.tracker import LiveLocator

        with (
            patch.object(LiveLocator, "start", lambda _self: None),
            patch.object(App, "_load_map_worker", lambda _self, _name: None),
        ):
            app = App(AppConfig())
            try:
                self.assertEqual(app.nb.count(), 2)
                self.assertIs(app.routes_tab, app.map_tab)
                self.assertIsNotNone(app._loc_thread)
                self.assertIsNotNone(app._hotkeys)
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
