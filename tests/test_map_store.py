"""Regression tests for MapStore: feature-index cache lifecycle and rebuilds."""

import json
import os
import sys
import tempfile
import time
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


class TestCatalogScale(unittest.TestCase):
    def test_catalog_scale_for_known_maps(self):
        store = MapStore()
        self.assertAlmostEqual(store.m_per_px("zestafona"), 0.498046875, delta=1e-9)
        self.assertAlmostEqual(store.px_per_m("zestafona"), 1.0 / 0.498046875, delta=1e-6)
        self.assertAlmostEqual(store.m_per_px("bakurani"), 0.99609375, delta=1e-9)
        self.assertAlmostEqual(store.px_per_m("bakurani"), 1.0 / 0.99609375, delta=1e-6)
        self.assertEqual(store.px_per_m("no_such_map"), 0.0)

    def test_catalog_artifact_sizes_match_disk(self):
        store = MapStore()
        for _name, entry in store._catalog().get("maps", {}).items():
            for fname, meta in (entry.get("artifacts") or {}).items():
                path = os.path.join(store.data_maps_dir, fname)
                if os.path.exists(path):
                    self.assertEqual(os.path.getsize(path), meta["size"], fname)

    def test_distributed_artifacts_are_consistent(self):
        """Catalog source, gray.txt and the index-embedded signature must agree."""
        store = MapStore()
        checked = 0
        for name in store._catalog().get("maps", {}):
            gray_txt = store.gray_sig_on_disk(name)
            if gray_txt is None:
                continue
            checked += 1
            entry = store._catalog()["maps"][name]
            self.assertEqual(entry.get("source"), gray_txt.split("|")[-1], name)
            self.assertEqual(store.gray_sig(name), gray_txt, name)
            feat = os.path.join(store.data_maps_dir, f"{name}_feat.npz")
            if os.path.exists(feat):
                with np.load(feat) as idx:
                    self.assertEqual(str(idx.get("gray_sig", [""])[0]), gray_txt, name)
                    self.assertEqual(str(idx.get("fmt", [""])[0]), "u8z", name)
        if checked == 0:
            self.skipTest("no map artifacts on this machine")


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

    def _fake_load_index(self, name, kind="sift"):
        idx = SimpleNamespace(
            name=name,
            kind=kind,
            gray_sig=self.store.gray_sig(),
            norm=map_store_mod._INDEX_NORM,
            fmt=map_store_mod._INDEX_FMT,
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

    def test_map_signature_invalidates_cache_key(self):
        sig1 = self.store.gray_sig()
        self.assertIn(self.store.map_signature(), sig1)

        time.sleep(0.02)
        png = os.path.join(self.tmp.name, "zestafona_map.png")
        cv2.imwrite(png, np.full((64, 64), 200, np.uint8))

        sig2 = self.store.gray_sig()
        self.assertNotEqual(sig1, sig2)

    def test_replaced_map_rebuilds_mu(self):
        with patch.object(MapStore, "build_previews", lambda _self, _full: None):
            self.store._g["mu"] = None
            mu1 = self.store.load_global_map()

            time.sleep(0.02)
            png = os.path.join(self.tmp.name, "zestafona_map.png")
            cv2.imwrite(png, np.zeros((64, 64), np.uint8))
            self.store._g["mu"] = None
            mu2 = self.store.load_global_map()

        self.assertLess(float(mu2.mean()), float(mu1.mean()))

    def test_index_with_legacy_format_is_rejected(self):
        self.store.set_map("zestafona")
        legacy = SimpleNamespace(
            name="zestafona",
            gray_sig=self.store.gray_sig(),
            norm=map_store_mod._INDEX_NORM,
            fmt=None,  # float32-era index without the storage-format marker
        )

        with patch.object(map_store_mod, "load_index", lambda _name, _kind="sift": legacy):
            self.assertIsNone(self.store.get_index())

    def test_set_config_path_reads_other_file(self):
        cfg_file = os.path.join(self.tmp.name, "custom.json")
        with open(cfg_file, "w", encoding="utf-8") as fh:
            json.dump({"locator": {"global_max_features": 12345}}, fh)

        self.store.set_config_path(cfg_file)
        self.assertEqual(self.store.config_path(), os.path.abspath(cfg_file))
        self.assertEqual(self.store.loc_cfg()["global_max_features"], 12345)

        time.sleep(0.02)
        with open(cfg_file, "w", encoding="utf-8") as fh:
            json.dump({"locator": {"global_max_features": 54321}}, fh)
        self.assertEqual(self.store.loc_cfg()["global_max_features"], 54321)

        self.store.set_config_path(None)
        self.assertTrue(self.store.config_path().endswith("config.json"))

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


class TestPreviewDerivation(unittest.TestCase):
    def _store(self, tmp: str) -> MapStore:
        store = MapStore(full_dir=tmp, data_maps_dir=tmp, default_map="mini")
        store.set_map("mini")
        return store

    def test_build_previews_skips_upscaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)

            store.build_previews(np.zeros((1024, 1024), np.uint8))

            self.assertTrue(os.path.exists(store.preview_path("mini", 1024)))
            self.assertFalse(os.path.exists(store.preview_path("mini", 2048)))
            self.assertFalse(os.path.exists(store.preview_path("mini", 16384)))

    def test_missing_levels_derive_from_the_shipped_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            top = np.zeros((1024, 1024), np.uint8)
            top[::2] = 200
            np.save(store.preview_path("mini", 1024), top)

            self.assertTrue(store.ensure_previews())

            previews = store.load_previews()
            assert previews is not None
            self.assertEqual(sorted(previews), [512, 1024])
            self.assertEqual(previews[512].shape, (512, 512))
            self.assertGreater(float(previews[512].mean()), 0.0)


class TestCompactIndexFormat(unittest.TestCase):
    def test_uint8_compressed_index_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            desc = np.arange(2 * 128, dtype=np.uint8).reshape(2, 128)
            np.savez_compressed(
                os.path.join(tmp, "mini_feat.npz"),
                ms=np.float32(2.6544),
                mu_h=64,
                mu_w=64,
                tile=32,
                gw=2,
                gh=2,
                gray_sig=np.array(["desat-1.000|ms=2.6544|1-2"], dtype="U128"),
                levels=np.asarray([1.0], np.float32),
                norm=np.array(["raw"], dtype="U32"),
                fmt=np.array([featureindex._INDEX_FMT], dtype="U16"),
                pts_lv0=np.zeros((2, 2), np.float32),
                desc_lv0=desc,
                tile_lv0=np.zeros(2, np.int32),
            )
            old_dir = featureindex._MAPS_DIR
            featureindex._MAPS_DIR = tmp
            try:
                idx = featureindex.load_index("mini")
            finally:
                featureindex._MAPS_DIR = old_dir

            self.assertIsNotNone(idx)
            assert idx is not None
            self.assertEqual(idx.fmt, featureindex._INDEX_FMT)
            self.assertEqual(idx._levels[0]["desc"].dtype, np.uint8)
            np.testing.assert_array_equal(idx._levels[0]["desc"], desc)


if __name__ == "__main__":
    unittest.main()
