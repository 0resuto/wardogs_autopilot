"""Regression tests for MapStore: feature-index cache lifecycle and rebuilds."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import autopilot.vision.featureindex as featureindex  # noqa: E402
import autopilot.vision.map_store as map_store_mod  # noqa: E402
from autopilot.vision.map_store import MapStore  # noqa: E402

MAPS = ("zestafona", "bakurani")


class TestIndexLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for name in MAPS:
            image = np.zeros((64, 64), np.uint8)
            image[16:48, 16:48] = 128
            cv2.imwrite(os.path.join(self.tmp.name, f"{name}_map.png"), image)
        self.store = MapStore(
            full_dir=self.tmp.name,
            data_maps_dir=self.tmp.name,
            default_map="zestafona",
        )
        self.loaded: list[SimpleNamespace] = []

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_load_index(self, name):
        idx = SimpleNamespace(
            name=name,
            gray_sig=self.store.gray_sig(),
            norm=map_store_mod._INDEX_NORM,
        )
        self.loaded.append(idx)
        return idx

    def _rebuild_patches(self):
        return (
            patch.object(MapStore, "build_previews", lambda _self, _full: None),
            patch.object(featureindex, "build_index", lambda _name, progress=False: None),
        )

    def test_rebuild_active_map_reloads_index(self):
        self.store.set_map("zestafona")

        previews_patch, build_patch = self._rebuild_patches()
        with patch.object(map_store_mod, "load_index", self._fake_load_index):
            first = self.store.get_index()
            self.assertIs(first, self.loaded[0])

            with previews_patch, build_patch:
                self.assertTrue(self.store.rebuild_map_cache("zestafona"))

            self.assertIsNone(self.store._g.get("idx"))

            second = self.store.get_index()
            self.assertIsNot(second, first)
            self.assertEqual(len(self.loaded), 2)
            self.assertEqual(second.name, "zestafona")

    def test_rebuild_other_map_keeps_active_map(self):
        self.store.set_map("zestafona")

        previews_patch, build_patch = self._rebuild_patches()
        with previews_patch, build_patch:
            self.store.rebuild_map_cache("bakurani")

        self.assertEqual(self.store.map_name(), "zestafona")
        self.assertIsNone(self.store._g.get("mu"))
        self.assertIsNone(self.store._g.get("idx"))

        with patch.object(map_store_mod, "load_index", self._fake_load_index):
            idx = self.store.get_index()
        self.assertEqual(idx.name, "zestafona")

    def test_failed_rebuild_restores_active_map(self):
        self.store.set_map("zestafona")

        def boom(_name, progress=False):
            raise RuntimeError("build failed")

        previews_patch, _build_patch = self._rebuild_patches()
        with (
            previews_patch,
            patch.object(featureindex, "build_index", boom),
            self.assertRaises(RuntimeError),
        ):
            self.store.rebuild_map_cache("bakurani")

        self.assertEqual(self.store.map_name(), "zestafona")


if __name__ == "__main__":
    unittest.main()
