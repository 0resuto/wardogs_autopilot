"""Input-validation tests driven through the real Qt widgets.

Tuning-panel and capture-zone values are typed into the actual QLineEdit
widgets and applied through the production handlers; config writes are stubbed
so the repository config.json is never touched. The App smoke test builds the
full main window with map loading and the locator thread stubbed out.
"""

import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QLayoutItem,
    QPushButton,
    QSizePolicy,
    QWidget,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import AppConfig, CaptureConfig  # noqa: E402
from autopilot.ui import theme  # noqa: E402
from autopilot.ui.flow_layout import FlowLayout  # noqa: E402
from autopilot.ui.presets import PresetManager  # noqa: E402
from autopilot.ui.tabs import logs_tab as logs_tab_mod  # noqa: E402
from autopilot.ui.tabs.logs_tab import LogsTab  # noqa: E402
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

    def test_fps_field_applies_and_validates(self):
        self.tab.fps_input.setText("30")
        self.tab.apply_tune()

        self.assertIn("applied", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["capture"]["fps"], 30)
        self.assertEqual(self.tab.app_cfg.capture.fps, 30)
        self.assertEqual(self.tab.fps_var.get(), "30")

        self.tab.fps_input.setText("999")
        self.tab.apply_tune()
        self.assertIn("Invalid fps", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["capture"]["fps"], 30)

    def test_engine_choice_applies_live_and_saves(self):
        self.assertEqual(self.tab.cfg["locator"].get("engine", "sift"), "sift")

        self.tab.engine_combo.setCurrentText("hybrid")

        self.assertEqual(self.tab.cfg["locator"]["engine"], "hybrid")
        self.assertEqual(self.tab.app_cfg.locator.engine, "hybrid")
        self.assertEqual(self.saved, 1)
        self.assertIn("engine: hybrid", self.tab.tune_status.text())

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

        self.assertIn("Invalid CEN ref", self.tab.tune_status.text())

    def test_corner_window_requires_max_above_min(self):
        self.tab.tune_inputs["corner_min_kmh"].setText("12")
        self.tab.tune_inputs["corner_max_kmh"].setText("25")
        self.tab.apply_tune()

        self.assertIn("applied", self.tab.tune_status.text())
        self.assertEqual(self.tab.cfg["navigator"]["corner_max_kmh"], 25)

        self.tab.tune_inputs["corner_max_kmh"].setText("10")
        self.tab.apply_tune()

        self.assertIn("Invalid VEH max km/h", self.tab.tune_status.text())


class TestParameterTooltips(unittest.TestCase):
    """Every tunable control explains itself and its direction of change."""

    DIRECTION_WORDS = ("lower", "higher", "increase", "decrease")

    def _assert_effect_tip(self, key: str, tip: str) -> None:
        self.assertTrue(tip.strip(), f"{key}: no hover text")
        self.assertTrue(
            any(word in tip for word in self.DIRECTION_WORDS),
            f"{key}: hover text does not say what changing the value does",
        )

    def test_tuning_fields_explain_themselves_on_hover(self):
        cfg = AppConfig()
        tab = MapTab(
            None,
            cfg,
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            app_cfg=cfg,
        )
        self.assertGreaterEqual(len(tab.tune_inputs), 30)
        for key, inp in tab.tune_inputs.items():
            self._assert_effect_tip(key, inp.toolTip())

    def test_capture_fields_explain_themselves_on_hover(self):
        cfg = AppConfig().to_dict()
        tab = RoiTab(
            None,
            cfg,
            save_cfg_fn=lambda: None,
            screen_cap_supplier=lambda: None,
            loc_thread_supplier=lambda: None,
        )
        self.assertEqual(len(tab.coord_inputs), 4)
        for group in (tab.coord_inputs, tab.center_inputs, tab.speed_inputs):
            for key, inp in group.items():
                self._assert_effect_tip(key, inp.toolTip())


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

    def test_combo_selection_loads_and_remembers_the_preset(self):
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
        mgr = PresetManager(tmp.name, subdir="presets")
        mgr.save_preset("a", [[1.0, 1.0]])
        mgr.save_preset("b", [[2.0, 2.0], [3.0, 3.0]])

        with patch("autopilot.ui.tabs.map_presets.PresetManager", lambda *a, **k: mgr):
            tab.route_pts = []
            tab.preset_reload()
            self.assertEqual(tab.p_sel.currentText(), "a")
            self.assertEqual(tab.route_pts, [[1.0, 1.0]])

            tab.p_sel.setCurrentText("b")

        self.assertEqual(tab.route_pts, [[2.0, 2.0], [3.0, 3.0]])
        self.assertEqual(tab.app_cfg.navigator.last_preset, "b")
        self.assertEqual(tab.cfg["navigator"]["last_preset"], "b")

    def test_preset_selection_updates_both_config_views(self):
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
        tab.preset_mgr = PresetManager(tmp.name, subdir="presets")
        tab.preset_mgr.save_preset("p1", [[1.0, 2.0]])
        tab.p_sel.clear()
        tab.p_sel.addItem("p1")
        tab.p_sel.setCurrentText("p1")

        tab.preset_load_sel()

        self.assertEqual(tab.cfg["navigator"]["last_preset"], "p1")
        self.assertEqual(tab.app_cfg.navigator.last_preset, "p1")

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

    def test_center_calibration_is_validated_and_applied(self):
        self.tab.center_inputs["center_dx"].setText("12.5")
        self.tab.center_inputs["center_dy"].setText("-8")
        self.tab.apply_center()

        self.assertIn("applied", self.tab.center_status_lbl.text())
        self.assertEqual(self.cfg["locator"]["center_dx"], 12.5)
        self.assertEqual(self.cfg["locator"]["center_dy"], -8.0)
        self.assertEqual(self.saved, 1)

        self.tab.center_inputs["center_dx"].setText("abc")
        self.tab.apply_center()
        self.assertIn("Error:", self.tab.center_status_lbl.text())
        self.assertEqual(self.saved, 1)

        self.tab.center_inputs["center_dx"].setText("500")
        self.tab.apply_center()
        self.assertIn("Error:", self.tab.center_status_lbl.text())
        self.assertEqual(self.cfg["locator"]["center_dx"], 12.5)
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

    def test_map_asset_state_distinguishes_missing_and_no_index(self):
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
            self.assertEqual(tab.map_asset_state(), "missing")

            for suffix in ("mu.npy", "preview_16384.npy"):
                with open(os.path.join(tmp, f"zestafona_{suffix}"), "wb") as fh:
                    fh.write(b"x")
            self.assertEqual(tab.map_asset_state(), "no_index")

            with open(os.path.join(tmp, "zestafona_feat.npz"), "wb") as fh:
                fh.write(b"x")
            self.assertEqual(tab.map_asset_state(), "ok")


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
                # persistent sidebar with its own sections
                self.assertEqual(app.side_tabs.count(), 3)
                self.assertEqual(app.side_tabs.tabText(1), "Map")
                self.assertEqual(app.side_tabs.tabText(2), "Logs")
                self.assertEqual(app.side_stack.count(), 3)
                self.assertIs(app.side_stack.widget(1), app.map_tab)
                self.assertIs(app.side_stack.widget(2), app.logs_tab)
                self.assertIs(app.routes_tab, app.map_tab)
                self.assertIsNotNone(app._loc_thread)
                self.assertIsNotNone(app._hotkeys)
                self.assertIsNotNone(app._override_keys)
                # the map canvas fills the right pane, the live capture preview
                # stays in the Capture section
                self.assertTrue(app.right_pane.isAncestorOf(app.map_tab.map_widget))
                self.assertFalse(app.right_pane.isAncestorOf(app.roi_tab.diagnostics_card))
                self.assertTrue(app.roi_tab.isAncestorOf(app.roi_tab.diagnostics_card))
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

    def test_save_cfg_persists_the_live_window_geometry(self):
        """Geometry must reach disk on any save, not only a clean close.

        A crash or a forced kill after a preset load used to leave the window
        rect from the previous session on disk, so the next start ignored the
        position the user had moved the window to.
        """
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
                try:
                    app.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
                    app.show()
                    app.setGeometry(222, 111, 903, 702)
                    app._save_cfg()
                    ui = AppConfig.load(target).ui
                finally:
                    app.close()

            self.assertEqual(
                (ui.window_x, ui.window_y, ui.window_w, ui.window_h),
                (222, 111, 903, 702),
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

    def test_save_keeps_foreign_config_changes(self):
        """A stale studio snapshot must not revert values written to disk.

        This is the two-instance / hand-edit case: the studio that did not make
        the change used to overwrite the whole file on Apply/close, so the
        change disappeared and the next start loaded the old defaults.
        """
        from autopilot.ui.app import App
        from autopilot.vision import locator
        from autopilot.vision.tracker import LiveLocator

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "cfg.json")
            AppConfig().save(target)
            cfg = AppConfig.load(target).to_dict()

            with (
                patch.object(LiveLocator, "start", lambda _self: None),
                patch.object(App, "_load_map_worker", lambda _self, _name: None),
            ):
                app = App(cfg)
                try:
                    foreign = AppConfig.load(target)
                    foreign.locator.center_dy = 9.0
                    foreign.save(target)

                    app.cfg["navigator"]["yaw_gain"] = 0.7
                    app._save_cfg()
                finally:
                    app.close()
                    locator.get_store().set_config_path(None)

            saved = AppConfig.load(target)
            self.assertEqual(saved.locator.center_dy, 9.0)
            self.assertEqual(saved.navigator.yaw_gain, 0.7)

    def test_save_persists_app_cfg_only_changes(self):
        """Settings set on the validated model must reach the file too.

        `navigator.last_preset` is written only on app_cfg by the preset
        handlers; the save used to rebuild that block from the cfg dict, so the
        selection was silently lost on the next start.
        """
        from autopilot.ui.app import App
        from autopilot.vision import locator
        from autopilot.vision.tracker import LiveLocator

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "cfg.json")
            AppConfig().save(target)
            cfg = AppConfig.load(target).to_dict()

            with (
                patch.object(LiveLocator, "start", lambda _self: None),
                patch.object(App, "_load_map_worker", lambda _self, _name: None),
            ):
                app = App(cfg)
                try:
                    app.app_cfg.navigator.last_preset = "route42"
                    app._save_cfg()
                finally:
                    app.close()
                    locator.get_store().set_config_path(None)

            self.assertEqual(AppConfig.load(target).navigator.last_preset, "route42")


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
            self.assertEqual(app._status_hw.property("state"), "green")

            with patch.object(app_mod.list_ports, "comports", lambda: []):
                app._update_hw_status()
            self.assertIn("○", app._status_hw.text())
            self.assertEqual(app._status_hw.property("state"), "off")
        finally:
            app.close()

    def test_preset_menu_exposes_the_actions(self):
        app = self._app()
        try:
            menu = app.map_tab._preset_menu_btn.menu()
            assert menu is not None
            texts = [a.text() for a in menu.actions() if not a.isSeparator()]
            self.assertEqual(
                texts,
                ["Load preset", "Save As…", "New route…", "Delete preset", "Reload list"],
            )
        finally:
            app.close()

    def test_capture_panels_show_placeholders_without_frames(self):
        app = self._app()
        try:
            app.roi_tab.get_loc = lambda: None
            app.roi_tab.update_preview()
            self.assertIn("no frames", app.roi_tab.preview_lbl.text())
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


class _FakeLogsSection:
    """Stand-in for the Logs section when the Map tab is wired on its own."""

    def __init__(self, nav_log: bool = False) -> None:
        self.nav_log = nav_log
        self.rejects: list[str] = []
        self.stop_calls = 0

    def nav_log_enabled(self) -> bool:
        return self.nav_log

    def set_last_reject(self, text: str) -> None:
        self.rejects.append(text)

    def stop_manual_record(self) -> None:
        self.stop_calls += 1


class TestMapDelegatesLogging(unittest.TestCase):
    @staticmethod
    def _tab(logs: _FakeLogsSection, cfg: AppConfig) -> MapTab:
        return MapTab(
            None,
            cfg,
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            logs_supplier=lambda: logs,
            app_cfg=cfg,
        )

    def test_nav_log_state_comes_from_the_logs_section(self):
        cfg = AppConfig()
        cfg.navigator.debug = False
        self.assertTrue(self._tab(_FakeLogsSection(nav_log=True), cfg)._nav_log_enabled())

        cfg.navigator.debug = True
        self.assertFalse(self._tab(_FakeLogsSection(nav_log=False), cfg)._nav_log_enabled())

    def test_nav_log_falls_back_to_the_config_without_the_logs_section(self):
        cfg = AppConfig()
        cfg.navigator.debug = True
        tab = MapTab(
            None,
            cfg,
            save_cfg_fn=lambda: None,
            loc_thread_supplier=lambda: None,
            app_cfg=cfg,
        )

        self.assertTrue(tab._nav_log_enabled())

    def test_reject_reason_is_forwarded_to_the_logs_section(self):
        logs = _FakeLogsSection()
        tab = self._tab(logs, AppConfig())

        tab.update_loc(
            {
                "pose": {"th": 90.0, "s": 1.0, "inl": 5},
                "map_px": [10.0, 20.0],
                "diag": {"reject": "vote_reject", "detail": "d" * 80},
            }
        )

        self.assertEqual(logs.rejects, ["vote_reject: " + "d" * 50])

        tab.update_loc({"pose": {"th": 0.0, "s": 0.0}, "map_px": [1.0, 1.0], "diag": {}})
        self.assertEqual(len(logs.rejects), 1)

    def test_follow_start_stops_the_manual_recorder_of_the_logs_section(self):
        logs = _FakeLogsSection()
        tab = self._tab(logs, AppConfig())
        tab.route_pts = [[0.0, 0.0], [100.0, 0.0]]

        with (
            patch.object(tab, "_make_kb", lambda _port: object()),
            patch("autopilot.navigation.follow.FollowDriver.start", lambda _self: None),
            patch("autopilot.navigation.follow.FollowDriver.stop", lambda _self: None),
        ):
            tab.follow_toggle(silent=True)
            self.assertEqual(logs.stop_calls, 1)
            self.assertIsNotNone(tab.driver)
            tab.emergency_stop()


class _FakeLogLocator:
    def __init__(self) -> None:
        self.collect: list[bool] = []

    def set_collect_fail_logs(self, enabled: bool) -> None:
        self.collect.append(enabled)

    def snapshot_debug(self):
        return None, None, None, None


class _FakeRecorder:
    """Stand-in for ManualDriveRecorder (no thread, no file)."""

    instances: list["_FakeRecorder"] = []

    def __init__(self, **_kwargs: Any) -> None:
        self.starts = 0
        self.stops = 0
        _FakeRecorder.instances.append(self)

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1


class TestLogsSection(unittest.TestCase):
    def setUp(self):
        _FakeRecorder.instances.clear()
        self.cfg = AppConfig()
        self.saved = 0
        self.loc = _FakeLogLocator()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tab = self._tab()

    def _tab(self) -> LogsTab:
        tab = LogsTab(
            None,
            self.cfg,
            save_cfg_fn=lambda: setattr(self, "saved", self.saved + 1),
            loc_thread_supplier=lambda: self.loc,
            app_cfg=self.cfg,
        )
        tab.output_dir = self.tmp.name
        tab.refresh_files()
        return tab

    def test_nav_log_switch_is_persisted(self):
        self.tab.nav_ck.setChecked(True)

        self.assertTrue(self.tab.nav_log_enabled())
        self.assertTrue(self.cfg.navigator.debug)
        self.assertTrue(self.tab.cfg["navigator"]["debug"])
        self.assertEqual(self.saved, 1)

        self.tab.nav_ck.setChecked(False)
        self.assertFalse(self.cfg.navigator.debug)
        self.assertEqual(self.saved, 2)

    def test_collect_fail_logs_reaches_the_locator(self):
        self.tab.collect_ck.setChecked(True)

        self.assertEqual(self.loc.collect, [True])
        self.assertTrue(self.cfg.debug.collect_fail_logs)
        self.assertTrue(self.tab.cfg["debug"]["collect_fail_logs"])
        self.assertEqual(self.saved, 1)

    def test_inventory_lists_files_and_counts_the_runs(self):
        for name in (
            "autopilot.log",
            "crash.log",
            "nav_dbg_20260101_101010.jsonl",
            "nav_dbg_20260101_111111.jsonl",
            "manual_dbg_20260101_121212.jsonl",
            "debug_fail_20260101_131313_001.json",
            "debug_fail_20260101_131313_002.txt",
        ):
            with open(os.path.join(self.tmp.name, name), "wb") as fh:
                fh.write(b"0123456789")
        os.makedirs(os.path.join(self.tmp.name, "snapshot_20260101_141414_000"))

        self.tab.refresh_files()

        rows = [lbl.text() for lbl in self.tab._file_rows]
        self.assertEqual(rows[:2], ["10 B", "10 B"])
        self.assertEqual(rows[2:], ["2", "1", "2", "1"])
        self.assertIn("nav_dbg_20260101_111111.jsonl", self.tab._file_rows[2].toolTip())
        for btn in self.tab._file_buttons:
            self.assertTrue(btn.isEnabled())
            self.assertTrue(str(btn.property("path")).endswith(".log"))

    def test_inventory_marks_missing_files(self):
        self.tab.refresh_files()

        self.assertEqual(self.tab._file_rows[0].text(), "missing")
        self.assertFalse(self.tab._file_buttons[0].isEnabled())
        self.assertEqual([lbl.text() for lbl in self.tab._file_rows[2:]], ["0", "0", "0", "0"])

    def test_last_reject_is_shown_and_copied(self):
        self.tab.set_last_reject("no_match_global: index miss")

        self.assertEqual(self.tab.dbg_text.text(), "no_match_global: index miss")
        self.tab.copy_debug()
        self.assertEqual(QApplication.clipboard().text(), "no_match_global: index miss")

    def test_snapshot_without_a_locator_reports_the_reason(self):
        self.tab.get_loc = lambda: None

        self.tab.save_debug_frame()

        self.assertIn("Locator not active", self.tab.save_status_lbl.text())
        self.assertTrue(self.tab.save_snap_btn.isEnabled())

    def test_snapshot_without_a_frame_reports_the_failure(self):
        self.tab.save_debug_frame()

        self.assertTrue(
            self._pump(lambda: "Save failed" in self.tab.save_status_lbl.text()),
            self.tab.save_status_lbl.text(),
        )
        self.assertTrue(self.tab.save_snap_btn.isEnabled())
        self.assertEqual(self.tab.save_snap_btn.text(), "Save frame")

    def _pump(self, predicate: Any, timeout: float = 3.0) -> bool:
        """Spin the Qt loop until a worker-thread signal lands (or timeout)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            QApplication.processEvents()
            if predicate():
                return True
            time.sleep(0.02)
        return bool(predicate())

    def test_manual_recording_starts_and_stops_from_the_logs_section(self):
        with patch.object(logs_tab_mod, "ManualDriveRecorder", _FakeRecorder):
            self.tab.manual_ck.setChecked(True)
            self.tab.refresh_files()
            self.assertEqual(len(_FakeRecorder.instances), 1)
            self.assertEqual(_FakeRecorder.instances[0].starts, 1)
            self.assertIn("recording your driving", self.tab.files_status.text())

            self.tab.stop_manual_record()

        self.assertEqual(_FakeRecorder.instances[0].stops, 1)
        self.assertFalse(self.tab.manual_ck.isChecked())
        self.assertNotIn("recording your driving", self.tab.files_status.text())

    def test_manual_recording_is_refused_without_a_locator(self):
        self.tab.get_loc = lambda: None

        self.tab.manual_ck.setChecked(True)

        self.assertFalse(self.tab.manual_ck.isChecked())


def _settle(rounds: int = 4) -> None:
    """Run the Qt layout passes so a hidden or new widget gets its geometry."""
    for _ in range(rounds):
        QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
        QApplication.processEvents()


def _item(flow: FlowLayout, index: int) -> QLayoutItem:
    item = flow.itemAt(index)
    assert item is not None
    return item


class TestFlowLayout(unittest.TestCase):
    """The wrapping layout: rows break instead of overflowing the column."""

    def _host(
        self, widths: list[int], heights: list[int] | None = None
    ) -> tuple[QWidget, FlowLayout]:
        host = QWidget()
        flow = FlowLayout(h_spacing=6, v_spacing=6)
        flow.setContentsMargins(0, 0, 0, 0)
        host.setLayout(flow)
        for i, width in enumerate(widths):
            btn = QPushButton(f"b{i}", host)
            btn.setFixedSize(width, (heights[i] if heights else 20))
            flow.addWidget(btn)
        # A hidden widget gets no layout pass, so the probe window must be shown.
        host.show()
        self.addCleanup(host.close)
        return host, flow

    def test_items_wrap_onto_the_next_row(self):
        host, flow = self._host([100, 100, 100])

        host.resize(220, 200)
        _settle()

        rows = sorted({_item(flow, i).geometry().top() for i in range(flow.count())})
        self.assertEqual(len(rows), 2)
        for i in range(flow.count()):
            self.assertLessEqual(_item(flow, i).geometry().right(), flow.geometry().right())
        del flow

    def test_single_item_never_exceeds_the_available_width(self):
        host, flow = self._host([400])

        host.resize(120, 200)
        _settle()

        item = _item(flow, 0)
        self.assertEqual(item.sizeHint().width(), 400)
        self.assertLessEqual(item.geometry().width(), flow.geometry().width())
        del flow

    def test_minimum_height_covers_the_stacked_layout(self):
        _, flow = self._host([100, 100], [20, 30])

        # Narrowest width: every item gets its own row, and the reported minimum
        # height must be the sum of those rows, otherwise a scroll area clips it.
        expected = 20 + flow.v_spacing + 30
        self.assertEqual(flow.minimumSize().height(), expected)
        self.assertEqual(flow.heightForWidth(flow.minimumSize().width()), expected)
        del flow

    def test_size_hint_does_not_pin_the_column_to_one_row(self):
        _, flow = self._host([100, 100])

        self.assertLessEqual(flow.sizeHint().width(), flow.minimumSize().width())
        del flow

    def test_one_row_hint_hugs_the_content(self):
        host = QWidget()
        flow = FlowLayout(h_spacing=6, v_spacing=0, one_row_hint=True)
        flow.setContentsMargins(0, 0, 0, 0)
        host.setLayout(flow)
        for i in range(2):
            btn = QPushButton(f"b{i}", host)
            btn.setFixedSize(100, 20)
            flow.addWidget(btn)
        host.show()
        self.addCleanup(host.close)

        # A floating toolbar wants the one-row width; it only wraps when it has
        # to, and the narrowest width stays reachable through minimumSize().
        self.assertEqual(flow.sizeHint().width(), 206)
        self.assertEqual(flow.minimumSize().width(), 100)
        del flow

    def test_expanding_items_share_the_row(self):
        host, flow = self._host([40, 40], [20, 20])
        for i in range(flow.count()):
            widget = _item(flow, i).widget()
            assert widget is not None
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )

        host.resize(400, 60)
        _settle()

        first = _item(flow, 0).geometry()
        second = _item(flow, 1).geometry()
        self.assertEqual(first.top(), second.top())
        self.assertLessEqual(second.right(), flow.geometry().right())
        del flow


class TestSidebarAdaptiveLayout(unittest.TestCase):
    """The Capture section must fit the 330..470 px sidebar column."""

    def _app(self):
        from autopilot.ui.app import App
        from autopilot.vision.tracker import LiveLocator

        with (
            patch.object(LiveLocator, "start", lambda _self: None),
            patch.object(App, "_load_map_worker", lambda _self, _name: None),
            patch.object(App, "_save_cfg", lambda _self: None),
        ):
            app = App(AppConfig())
        app.show()
        app.side_stack.setCurrentIndex(0)
        app.resize(1100, 760)
        return app

    def _resize_sidebar(self, app, width: int) -> None:
        app.splitter.setSizes([width, max(200, app.width() - width)])
        _settle()
        app.side_stack.setCurrentIndex(0)
        _settle()

    def test_capture_column_reaches_its_minimum_width(self):
        app = self._app()
        try:
            self._resize_sidebar(app, 330)

            self.assertLessEqual(app.roi_tab.width(), 330)
            self.assertLessEqual(app.roi_tab.diagnostics_card.width(), app.roi_tab.width())
        finally:
            app.close()

    def test_previews_wrap_and_never_overflow_the_card(self):
        app = self._app()
        try:
            roi = app.roi_tab
            for width in (330, 400, 470):
                self._resize_sidebar(app, width)
                card = roi.diagnostics_card
                rects = [roi.capture_panel.geometry(), roi.speed_panel.geometry()]
                for rect in rects:
                    self.assertLessEqual(rect.right(), card.width())
                    self.assertLessEqual(rect.bottom(), card.height())
                    self.assertGreaterEqual(rect.width(), 100)
                stacked = rects[0].top() != rects[1].top()
                self.assertEqual(stacked, rects[0].right() + 6 > rects[1].right())
        finally:
            app.close()

    def test_sidebar_cards_never_exceed_the_page_width(self):
        app = self._app()
        checked = 0
        try:
            for tab in range(app.side_stack.count()):
                app.side_stack.setCurrentIndex(tab)
                for width in (330, 470):
                    self._resize_sidebar(app, width)
                    app.side_stack.setCurrentIndex(tab)
                    _settle()
                    page = app.side_stack.currentWidget()
                    for card in page.findChildren(QGroupBox):
                        if card.parentWidget() is page:
                            checked += 1
                            self.assertLessEqual(card.width(), page.width(), card.title())
            self.assertGreater(checked, 0, "no sidebar cards were measured")
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
