"""Input-validation tests driven through the real Qt widgets.

Tuning-panel and capture-zone values are typed into the actual QLineEdit
widgets and applied through the production handlers; config writes are stubbed
so the repository config.json is never touched. The App smoke test builds the
full main window with map loading and the locator thread stubbed out.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import AppConfig, CaptureConfig  # noqa: E402
from autopilot.ui import theme  # noqa: E402
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
        self.tab.tune_inputs["corner_lat_g"].setText("0.9")
        self.tab.apply_tune()
        self.assertEqual(self.tab.cfg["locator"]["ratio_local"], 0.5)
        self.assertEqual(self.tab.cfg["navigator"]["corner_lat_g"], 0.9)

        self.tab.reset_tune()
        self.assertEqual(self.tab.cfg["locator"]["ratio_local"], 0.85)
        self.assertEqual(self.tab.tune_inputs["ratio_local"].text(), "0.85")
        self.assertEqual(self.tab.tune_vars["ratio_local"].get(), "0.85")
        self.assertEqual(self.tab.cfg["navigator"]["corner_lat_g"], 0.35)
        self.assertEqual(self.tab.cfg["navigator"]["yaw_gain"], 1.0)

    def test_vehicle_tuning_applies_live_to_the_driver(self):
        class _FakeDriver:
            def __init__(self) -> None:
                self.applied: list[object] = []

            def apply_vehicle_tuning(self, cfg) -> None:
                self.applied.append(cfg)

        driver: Any = _FakeDriver()
        self.tab.driver = driver

        self.tab.tune_inputs["yaw_gain"].setText("0.7")
        self.tab.tune_inputs["corner_lat_g"].setText("0.5")
        self.tab.tune_inputs["corner_min_kmh"].setText("20")
        self.tab.apply_tune()

        self.assertIn("applied", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["navigator"]["yaw_gain"], 0.7)
        self.assertEqual(self.tab.cfg["navigator"]["corner_lat_g"], 0.5)
        self.assertAlmostEqual(self.tab.app_cfg.navigator.corner_lat_g, 0.5, delta=1e-9)
        self.assertEqual(len(driver.applied), 1)

    def test_vehicle_tuning_rejects_out_of_range(self):
        self.tab.tune_inputs["corner_lat_g"].setText("5.0")
        self.tab.apply_tune()

        self.assertIn("Invalid VEH lat g", self.tab.tune_status.text())
        self.assertEqual(self.saved, 0)

    def test_corridor_values_apply_and_validate(self):
        self.tab.tune_inputs["xte_m"].setText("6")
        self.tab.tune_inputs["xte_outer_m"].setText("15")
        self.tab.tune_inputs["steer_look_s"].setText("1.2")
        self.tab.tune_inputs["settle_s"].setText("0.5")
        self.tab.apply_tune()

        self.assertIn("applied", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["navigator"]["xte_m"], 6)
        self.assertEqual(self.tab.cfg["navigator"]["xte_outer_m"], 15)
        self.assertEqual(self.tab.cfg["navigator"]["steer_look_s"], 1.2)
        self.assertEqual(self.tab.cfg["navigator"]["settle_s"], 0.5)

        self.tab.tune_inputs["xte_outer_m"].setText("5")
        self.tab.apply_tune()

        self.assertIn("Invalid COR outer", self.tab.tune_status.text())

    def test_corner_window_requires_max_above_min(self):
        self.tab.tune_inputs["corner_min_kmh"].setText("12")
        self.tab.tune_inputs["corner_max_kmh"].setText("25")
        self.tab.apply_tune()

        self.assertIn("applied", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["navigator"]["corner_max_kmh"], 25)

        self.tab.tune_inputs["corner_max_kmh"].setText("10")
        self.tab.apply_tune()

        self.assertIn("Invalid VEH max km/h", self.tab.tune_status.text())


class TestRouteEditor(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig()
        self.tab = MapTab(
            None,
            self.cfg,
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            app_cfg=self.cfg,
        )
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tab.preset_mgr = PresetManager(self._tmp.name, subdir="presets")
        self.tab.p_sel.clear()

    def test_apply_saves_to_the_selected_preset(self):
        mgr = self.tab.preset_mgr
        mgr.save_preset("r1", [])
        self.tab.p_sel.addItem("r1")
        self.tab.route_pts = [[1.0, 1.0]]

        self.tab._edit_btn.click()
        self.tab.route_pts = [[1.0, 1.0], [3.0, 3.0]]
        self.tab._apply_btn.click()

        self.assertEqual(mgr.load_preset("r1"), [[1.0, 1.0], [3.0, 3.0]])

    def test_route_controls_lock_while_driving(self):
        self.assertTrue(self.tab.p_sel.isEnabled())

        self.tab.driver = object()  # type: ignore[assignment]
        self.tab._set_follow_state(True)

        self.assertFalse(self.tab.p_sel.isEnabled())
        self.assertFalse(self.tab._edit_btn.isEnabled())

        self.tab.driver = None
        self.tab._set_follow_state(False)

        self.assertTrue(self.tab.p_sel.isEnabled())
        self.assertTrue(self.tab._edit_btn.isEnabled())

    def test_follow_refuses_while_editing(self):
        self.tab._edit_btn.click()

        self.tab.follow_toggle(silent=True)

        self.assertIsNone(self.tab.driver)
        self.assertIn("Finish editing", self.tab.routes_status.text())

    def test_reverse_is_refused_while_editing(self):
        self.tab.route_pts = [[0.0, 0.0], [1.0, 0.0]]
        self.tab._edit_btn.click()

        self.tab.routes_invert()

        self.assertEqual(self.tab.route_pts, [[0.0, 0.0], [1.0, 0.0]])
        self.assertIn("Finish editing", self.tab.routes_status.text())

    def test_editing_is_gated_by_the_edit_button(self):
        self.assertFalse(self.tab.map_widget.view._route_edit_mode)
        self.assertFalse(self.tab._edit_btn.isHidden())
        self.assertTrue(self.tab._apply_btn.isHidden())

        self.tab._edit_btn.click()

        self.assertTrue(self.tab.map_widget.view._route_edit_mode)
        self.assertTrue(self.tab._edit_btn.isHidden())
        self.assertFalse(self.tab._apply_btn.isHidden())
        self.assertFalse(self.tab._cancel_btn.isHidden())
        self.assertFalse(self.tab.p_sel.isEnabled())

        self.tab._apply_btn.click()

        self.assertFalse(self.tab.map_widget.view._route_edit_mode)
        self.assertFalse(self.tab._edit_btn.isHidden())
        self.assertTrue(self.tab._apply_btn.isHidden())
        self.assertTrue(self.tab.p_sel.isEnabled())

    def test_cancel_restores_the_route(self):
        self.tab.route_pts = [[10.0, 10.0], [20.0, 20.0]]

        self.tab._edit_btn.click()
        self.tab.route_pts = [[50.0, 50.0]]
        self.tab._cancel_btn.click()

        self.assertEqual(self.tab.route_pts, [[10.0, 10.0], [20.0, 20.0]])

    def test_apply_keeps_the_edited_route(self):
        self.tab.route_pts = [[10.0, 10.0]]

        self.tab._edit_btn.click()
        self.tab.route_pts = [[10.0, 10.0], [30.0, 30.0]]
        self.tab._apply_btn.click()

        self.assertEqual(self.tab.route_pts, [[10.0, 10.0], [30.0, 30.0]])

    def test_route_new_keeps_the_current_route_when_saving_fails(self):
        self.tab.route_pts = [[1.0, 1.0], [2.0, 2.0]]

        with (
            patch.object(self.tab, "_ask_preset_name", return_value="fresh"),
            patch.object(self.tab, "_save_route_to_preset", return_value=False),
        ):
            self.tab.route_new()

        self.assertEqual(self.tab.route_pts, [[1.0, 1.0], [2.0, 2.0]])

    def test_route_new_saves_empty_points_and_enters_edit_mode(self):
        self.tab.route_pts = [[1.0, 1.0]]
        saved: dict = {}

        def save(name: str) -> bool:
            saved["name"] = name
            saved["pts"] = list(self.tab.route_pts)
            return True

        with (
            patch.object(self.tab, "_ask_preset_name", return_value="fresh"),
            patch.object(self.tab, "_save_route_to_preset", side_effect=save),
        ):
            self.tab.route_new()

        self.assertEqual(saved["name"], "fresh")
        self.assertEqual(saved["pts"], [])
        self.assertEqual(self.tab.route_pts, [])
        self.assertTrue(self.tab.map_widget.view._route_edit_mode)
        self.assertTrue(self.tab._edit_btn.isHidden())


class TestPresetStartup(unittest.TestCase):
    def test_reload_selects_and_loads_the_last_preset(self):
        cfg = AppConfig()
        tab = MapTab(
            None,
            cfg,
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            app_cfg=cfg,
        )
        names = tab.preset_mgr.list_presets()
        if not names:
            self.skipTest("no presets in the repository")
        name = names[-1]

        cfg.navigator.last_preset = name
        tab.route_pts = []
        tab.preset_reload()

        self.assertEqual(tab.p_sel.currentText(), name)
        self.assertEqual(tab.route_pts, tab.preset_mgr.load_preset(name))

    def test_save_as_writes_the_route_under_the_asked_name(self):
        cfg = AppConfig()
        tab = MapTab(
            None,
            cfg,
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            app_cfg=cfg,
        )
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mgr = tab.preset_mgr = PresetManager(tmp.name, subdir="presets")
        tab.p_sel.clear()
        tab.route_pts = [[5.0, 6.0], [7.0, 8.0]]

        with patch.object(tab, "_ask_preset_name", return_value="route_x"):
            tab.preset_save()

        self.assertEqual(mgr.load_preset("route_x"), [[5.0, 6.0], [7.0, 8.0]])

    def test_invert_refuses_while_driving(self):
        cfg = AppConfig()
        tab = MapTab(
            None,
            cfg,
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            app_cfg=cfg,
        )
        tab.route_pts = [[0.0, 0.0], [10.0, 0.0]]
        tab.driver = object()  # type: ignore[assignment]

        tab.routes_invert(silent=True)

        self.assertEqual(tab.route_pts, [[0.0, 0.0], [10.0, 0.0]])
        self.assertIn("Stop the autopilot", tab.routes_status.text())


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


class _FakeLocator:
    def __init__(self) -> None:
        self.speed_rois: list[list[int] | None] = []
        self.mmap_rois: list[list[int]] = []

    def set_speed_roi(self, roi) -> None:
        self.speed_rois.append(roi)

    def set_roi(self, roi) -> None:
        self.mmap_rois.append(roi)


class TestSpeedRoiValidation(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig().to_dict()
        self.saved = 0
        self.loc = _FakeLocator()
        self.tab = RoiTab(
            None,
            self.cfg,
            save_cfg_fn=self._on_save,
            screen_cap_supplier=lambda: None,
            loc_thread_supplier=lambda: self.loc,
        )

    def _on_save(self):
        self.saved += 1

    def test_typed_speed_roi_is_validated_and_applied(self):
        self.tab.speed_inputs["w"].setText("5")
        self.tab.apply_speed_roi()
        self.assertIn("Error:", self.tab.speed_status_lbl.text())
        self.assertEqual(self.saved, 0)

        for name, value in (("x", "640"), ("y", "20"), ("w", "60"), ("h", "30")):
            self.tab.speed_inputs[name].setText(value)
        self.tab.apply_speed_roi()
        self.assertIn("OK:", self.tab.speed_status_lbl.text())
        self.assertEqual(self.cfg["capture"]["speed_roi"], [640, 20, 60, 30])
        self.assertEqual(self.loc.speed_rois[-1], [640, 20, 60, 30])
        self.assertEqual(self.tab.speed_vars["w"].get(), "60")
        self.assertEqual(self.saved, 1)

    def test_disable_clears_speed_roi(self):
        for name, value in (("x", "640"), ("y", "20"), ("w", "60"), ("h", "30")):
            self.tab.speed_inputs[name].setText(value)
        self.tab.apply_speed_roi()

        self.tab.disable_speed_roi()

        self.assertIsNone(self.cfg["capture"]["speed_roi"])
        self.assertEqual(self.loc.speed_rois[-1], None)
        self.assertEqual(self.tab.speed_status_lbl.text(), "disabled")


class TestMapScaleDisplay(unittest.TestCase):
    @staticmethod
    def _tab(store: Any) -> MapTab:
        return MapTab(
            None,
            AppConfig(),
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            map_store_supplier=lambda: store,
            map_name_supplier=lambda: "zestafona",
        )

    def test_route_length_uses_catalog_scale(self):
        class _Store:
            def px_per_m(self, name: str) -> float:
                return 2.0

        tab = self._tab(_Store())
        self.assertAlmostEqual(tab._map_px_per_m(), 2.0, delta=1e-9)

        tab.route_pts = [[0.0, 0.0], [200.0, 0.0]]
        self.assertIn("100 m", tab.routes_status.text())

    def test_route_length_falls_back_without_catalog_scale(self):
        tab = self._tab(object())

        self.assertEqual(tab._map_px_per_m(), 0.0)

        tab.route_pts = [[0.0, 0.0], [200.0, 0.0]]
        self.assertIn("100 m", tab.routes_status.text())

    def test_route_eta_uses_the_corner_planner(self):
        class _Store:
            def px_per_m(self, name: str) -> float:
                return 2.0

        tab = self._tab(_Store())
        tab.route_pts = [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0]]

        self.assertIn("planned", tab.routes_status.text())


class TestMapEmptyState(unittest.TestCase):
    def test_notice_reports_missing_assets_and_hides_when_ready(self):
        with tempfile.TemporaryDirectory() as tmp:

            class _Store:
                data_maps_dir = tmp

            tab = MapTab(
                None,
                AppConfig(),
                save_cfg_fn=lambda: None,
                loc_thread_supplier=lambda: None,
                map_store_supplier=lambda: _Store(),
                map_name_supplier=lambda: "zestafona",
            )

            self.assertFalse(tab._map_notice.isHidden())
            self.assertIn("not downloaded", tab._map_notice_lbl.text())

            for suffix in ("mu.npy", "feat.npz", "preview_16384.npy"):
                with open(os.path.join(tmp, f"zestafona_{suffix}"), "wb") as fh:
                    fh.write(b"x")
            tab.map_loaded()

            self.assertTrue(tab._map_notice.isHidden())


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
            patch.object(App, "_save_cfg", lambda _self: None),
        ):
            app = App(AppConfig())
            try:
                self.assertEqual(app.nb.count(), 2)
                self.assertIs(app.routes_tab, app.map_tab)
                self.assertIsNotNone(app._loc_thread)
                self.assertIsNotNone(app._hotkeys)
                self.assertEqual(app.nb.currentIndex(), 1)
                self.assertEqual(app.nb.tabText(1), "Map")
            finally:
                app.close()

    def test_app_remembers_the_window_geometry(self):
        from autopilot.ui.app import App
        from autopilot.vision.tracker import LiveLocator

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "ui.json")
            AppConfig().save(target)

            with (
                patch.object(LiveLocator, "start", lambda _self: None),
                patch.object(App, "_load_map_worker", lambda _self, _name: None),
            ):
                app = App(AppConfig.load(target))
                app.setGeometry(123, 45, 900, 700)
                app.close()

            ui = AppConfig.load(target).ui
            self.assertEqual(
                (ui.window_x, ui.window_y, ui.window_w, ui.window_h),
                (123, 45, 900, 700),
            )

    def test_offscreen_geometry_is_clamped_to_a_screen(self):
        from autopilot.ui.app import clamp_rect_to_screens

        app = QApplication.instance()
        assert isinstance(app, QApplication)
        rect = QRect(-30000, -30000, 900, 700)

        clamped = clamp_rect_to_screens(rect, app)

        screen = app.primaryScreen().availableGeometry()
        self.assertTrue(screen.intersects(clamped))
        self.assertGreaterEqual(clamped.width(), 400)
        self.assertGreaterEqual(clamped.height(), 300)

    def test_app_saves_to_the_loaded_config_path(self):
        from autopilot.ui.app import App
        from autopilot.vision import locator
        from autopilot.vision.tracker import LiveLocator

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "custom.json")
            AppConfig(capture=CaptureConfig(fps=11)).save(target)
            cfg = AppConfig.load(target).to_dict()

            with (
                patch.object(LiveLocator, "start", lambda _self: None),
                patch.object(App, "_load_map_worker", lambda _self, _name: None),
            ):
                app = App(cfg)
                try:
                    self.assertEqual(locator.get_store().config_path(), os.path.abspath(target))
                    app.cfg["capture"]["fps"] = 17
                    app._save_cfg()
                finally:
                    app.close()
                    locator.get_store().set_config_path(None)

            self.assertEqual(AppConfig.load(target).capture.fps, 17)


class TestMapDownloadUi(unittest.TestCase):
    def _app(self):
        from autopilot.ui.app import App
        from autopilot.vision.tracker import LiveLocator

        with (
            patch.object(LiveLocator, "start", lambda _self: None),
            patch.object(App, "_load_map_worker", lambda _self, _name: None),
            patch.object(App, "_save_cfg", lambda _self: None),
        ):
            return App(AppConfig())

    def test_download_button_follows_cache_status(self):
        app = self._app()
        try:
            with patch.object(
                app.roi_tab,
                "_get_map_cache_status",
                lambda _n: ("Not downloaded: run python tools/download_map.py x", theme.RED),
            ):
                app.roi_tab.cache_status_refresh()
                self.assertTrue(app.roi_tab._cache_download_btn.isEnabled())

            with patch.object(
                app.roi_tab, "_get_map_cache_status", lambda _n: ("Ready: mu OK", theme.GREEN)
            ):
                app.roi_tab.cache_status_refresh()
                self.assertFalse(app.roi_tab._cache_download_btn.isEnabled())
        finally:
            app.close()

    def test_hardware_indicator_reflects_the_port(self):
        from autopilot.ui import app as app_mod

        app = self._app()
        try:
            with patch.object(
                app_mod.list_ports, "comports", lambda: [SimpleNamespace(device="COM6")]
            ):
                app._update_hw_status()
            self.assertIn("COM6", app._status_hw.text())
            self.assertIn("●", app._status_hw.text())

            with patch.object(app_mod.list_ports, "comports", lambda: []):
                app._update_hw_status()
            self.assertIn("○", app._status_hw.text())
        finally:
            app.close()

    def test_download_worker_reports_success_and_failure(self):
        from autopilot.ui.tabs import roi_cache as roi_cache_mod
        from autopilot.vision import asset_sync

        app = self._app()
        try:
            app.roi_tab.on_map_rebuilt = None  # keep the test off the map reload thread
            done: list[str] = []
            failed: list[str] = []
            app.roi_tab.sig_download_done.connect(done.append)
            app.roi_tab.sig_download_failed.connect(failed.append)

            with (
                patch.object(asset_sync, "download_map", lambda *_a, **_k: True),
                patch.object(roi_cache_mod.QMessageBox, "information", lambda *_a, **_k: None),
            ):
                app.roi_tab._cache_download_worker("zestafona", {"repo": "x/y", "tag": "v"})

            self.assertEqual(done, ["zestafona"])
            self.assertFalse(app.roi_tab._cache_download_busy)

            with (
                patch.object(asset_sync, "download_map", lambda *_a, **_k: False),
                patch.object(roi_cache_mod.QMessageBox, "critical", lambda *_a, **_k: None),
            ):
                app.roi_tab._cache_download_worker("zestafona", {"repo": "x/y", "tag": "v"})

            self.assertEqual(len(failed), 1)
            self.assertFalse(app.roi_tab._cache_download_busy)
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
