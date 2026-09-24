"""Unit tests for decoupled UI modules: PresetManager, map_renderer, and debug_collage."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autopilot.ui.debug_collage import build_debug_collage, save_debug_snapshot
from autopilot.ui.map_renderer import (
    calc_fit_viewport,
    crop_map_viewport,
    native_to_screen,
    screen_to_native,
)
from autopilot.ui.presets import PresetManager


class TestPresetManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mgr = PresetManager(self.tmpdir.name, subdir="presets")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_preset_save_load_list_delete(self):
        pts = [[100.0, 200.0], [300.5, 400.25]]
        path = self.mgr.save_preset("route_a", pts)
        self.assertTrue(Path(path).exists())

        presets = self.mgr.list_presets()
        self.assertEqual(presets, ["route_a"])

        loaded = self.mgr.load_preset("route_a")
        self.assertEqual(loaded, pts)

        deleted = self.mgr.delete_preset("route_a")
        self.assertTrue(deleted)
        self.assertEqual(self.mgr.list_presets(), [])

    def test_preset_path_sanitization(self):
        malicious = "../../etc/passwd"
        safe_path = self.mgr.preset_path(malicious)
        expected_dir = str(self.mgr.presets_dir)
        self.assertEqual(str(Path(safe_path).parent), expected_dir)
        self.assertEqual(Path(safe_path).name, "passwd.json")


class TestMapRenderer(unittest.TestCase):
    def test_calc_fit_viewport(self):
        scale, ox, oy = calc_fit_viewport(1000.0, 500.0, 2000.0, 2000.0)
        self.assertAlmostEqual(scale, 0.25)
        self.assertAlmostEqual(ox, 250.0)
        self.assertAlmostEqual(oy, 0.0)

    def test_screen_native_roundtrip(self):
        disp = (0.5, 100.0, 50.0)
        thumb = 8.0
        orig_nx, orig_ny = 1234.0, 5678.0

        cx, cy = native_to_screen(disp, orig_nx, orig_ny, thumb)
        back_nx, back_ny = screen_to_native(disp, cx, cy, thumb)

        self.assertAlmostEqual(orig_nx, back_nx, places=5)
        self.assertAlmostEqual(orig_ny, back_ny, places=5)

    def test_crop_map_viewport(self):
        # Synthetic pyramid
        pyr = {512: np.full((512, 512), 128, dtype=np.uint8)}
        cropped = crop_map_viewport(
            s=0.5,
            ru=10.0,
            rv=10.0,
            rw=50.0,
            rh=50.0,
            pyr=pyr,
            map_size=4096,
            thumb_factor=8,
        )
        self.assertEqual(cropped.shape[2], 3)
        self.assertGreater(cropped.shape[0], 0)
        self.assertGreater(cropped.shape[1], 0)


class TestDebugCollage(unittest.TestCase):
    def test_build_debug_collage(self):
        mm = np.full((100, 100), 120, dtype=np.uint8)
        diag = {"mode": "test", "reject": None, "detail": "ok"}
        sheet = build_debug_collage(mm, None, None, None, diag)
        self.assertIsNotNone(sheet)
        self.assertEqual(sheet.shape[2], 3)
        self.assertGreater(sheet.shape[0], 300)

    def test_save_debug_snapshot_without_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mm = np.full((120, 160), 100, dtype=np.uint8)
            bgr = np.full((120, 160, 3), 100, dtype=np.uint8)
            mask = np.zeros((120, 160), dtype=bool)
            diag = {"mode": "index", "kp_pts": [(20, 20), (40, 40)], "inlier_pts": []}
            sheet, snap_dir, parts = save_debug_snapshot(
                tmp_dir, mm, bgr, mask, None, diag, map_name="zestafona"
            )
            self.assertTrue(os.path.isdir(snap_dir))
            self.assertIn("1_raw_capture.png", parts)
            self.assertIn("2_mask_overlay.png", parts)
            self.assertIn("3_sift_features.png", parts)
            self.assertNotIn("4_map_crop.png", parts)
            self.assertIn("state_log.txt", parts)
            self.assertIn("state.json", parts)
            self.assertTrue(os.path.exists(os.path.join(snap_dir, "1_raw_capture.png")))
            self.assertTrue(os.path.exists(os.path.join(snap_dir, "2_mask_overlay.png")))
            self.assertTrue(os.path.exists(os.path.join(snap_dir, "3_sift_features.png")))
            self.assertTrue(os.path.exists(os.path.join(snap_dir, "state_log.txt")))
            self.assertTrue(os.path.exists(os.path.join(snap_dir, "state.json")))

    def test_save_debug_snapshot_with_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mm = np.full((120, 160), 100, dtype=np.uint8)
            bgr = np.full((120, 160, 3), 100, dtype=np.uint8)
            mask = np.zeros((120, 160), dtype=bool)
            pose = {
                "map_x": 10000.0,
                "map_y": 12000.0,
                "th": 45.0,
                "s": 1.0,
                "inl": 15,
                "n_match": 30,
            }
            diag = {"mode": "index", "kp_pts": [(20, 20)], "inlier_pts": [(20, 20)]}
            sheet, snap_dir, parts = save_debug_snapshot(
                tmp_dir, mm, bgr, mask, pose, diag, map_name="zestafona"
            )
            self.assertTrue(os.path.isdir(snap_dir))
            self.assertIn("1_raw_capture.png", parts)
            self.assertIn("2_mask_overlay.png", parts)
            self.assertIn("3_sift_features.png", parts)
            self.assertIn("4_map_crop.png", parts)
            self.assertIn("state_log.txt", parts)
            self.assertIn("state.json", parts)
            self.assertTrue(os.path.exists(os.path.join(snap_dir, "4_map_crop.png")))


if __name__ == "__main__":
    unittest.main()
