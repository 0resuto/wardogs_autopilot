"""Unit tests for the map asset catalog tools (build + download)."""

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_tool(name: str):
    path = os.path.join(ROOT, "tools", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBuildMapAssets(unittest.TestCase):
    def test_refresh_entry_records_artifacts_and_source(self):
        tool = _load_tool("build_map_assets")
        with tempfile.TemporaryDirectory() as tmp:
            tool.DATA_MAPS = tmp
            suffixes = tool.artifact_suffixes({"size": [32768, 32768]})
            for suffix in suffixes:
                with open(os.path.join(tmp, f"zestafona{suffix}"), "wb") as f:
                    f.write(f"payload-{suffix}".encode())
            with open(os.path.join(tmp, "zestafona_gray.txt"), "w", encoding="utf-8") as f:
                f.write("desat-1.000|ms=2.6544|123-456")

            catalog: dict = {"maps": {"zestafona": {"files": {"zestafona_map.png": {}}}}}

            count, total = tool.refresh_entry(catalog, "zestafona")

            entry = catalog["maps"]["zestafona"]
            self.assertEqual(count, len(suffixes))
            self.assertGreater(total, 0)
            self.assertNotIn("files", entry)
            self.assertEqual(entry["source"], "123-456")
            meta = entry["artifacts"]["zestafona_mu.npy"]
            payload = b"payload-_mu.npy"
            self.assertEqual(meta["size"], len(payload))
            self.assertEqual(meta["sha256"], hashlib.sha256(payload).hexdigest())

    def test_refresh_entry_rejects_missing_artifact(self):
        tool = _load_tool("build_map_assets")
        with tempfile.TemporaryDirectory() as tmp:
            tool.DATA_MAPS = tmp
            with open(os.path.join(tmp, "zestafona_gray.txt"), "w", encoding="utf-8") as f:
                f.write("desat-1.000|ms=2.6544|123-456")
            catalog: dict = {"maps": {"zestafona": {}}}

            with self.assertRaises(SystemExit):
                tool.refresh_entry(catalog, "zestafona")


class TestArchiveDistribution(unittest.TestCase):
    def test_export_builds_a_zip_with_exactly_the_artifacts(self):
        tool = _load_tool("build_map_assets")
        with tempfile.TemporaryDirectory() as tmp:
            maps_dir = os.path.join(tmp, "maps")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(maps_dir)
            tool.DATA_MAPS = maps_dir
            suffixes = tool.artifact_suffixes({"size": [32768, 32768]})
            for suffix in suffixes:
                with open(os.path.join(maps_dir, f"zestafona{suffix}"), "wb") as f:
                    f.write(f"payload-{suffix}".encode())

            meta = tool.export_artifacts("zestafona", {"size": [32768, 32768]}, out_dir)

            zip_path = os.path.join(out_dir, "zestafona.zip")
            self.assertTrue(os.path.exists(zip_path))
            self.assertEqual(meta["name"], "zestafona.zip")
            self.assertEqual(meta["size"], os.path.getsize(zip_path))
            self.assertEqual(
                meta["sha256"], hashlib.sha256(open(zip_path, "rb").read()).hexdigest()
            )
            with zipfile.ZipFile(zip_path) as zf:
                self.assertEqual(sorted(zf.namelist()), sorted(f"zestafona{s}" for s in suffixes))

    def test_extract_archive_verifies_and_places_files(self):
        from autopilot.vision import asset_sync

        with tempfile.TemporaryDirectory() as tmp:
            payload = b"map-artifact"
            zip_path = os.path.join(tmp, "mini.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("mini_mu.npy", payload)
            info = {
                "artifacts": {
                    "mini_mu.npy": {
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                }
            }

            self.assertTrue(asset_sync.extract_archive("mini", info, zip_path, tmp))

            self.assertTrue(os.path.exists(os.path.join(tmp, "mini_mu.npy")))
            self.assertFalse(os.path.exists(os.path.join(tmp, ".extract_mini")))

    def test_extract_archive_rejects_corruption_and_unsafe_names(self):
        from autopilot.vision import asset_sync

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "mini.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("mini_mu.npy", b"corrupted")
                zf.writestr("evil.txt", b"x")

            info = {
                "artifacts": {
                    "mini_mu.npy": {
                        "size": 9,
                        "sha256": hashlib.sha256(b"expected").hexdigest(),
                    }
                }
            }
            self.assertFalse(asset_sync.extract_archive("mini", info, zip_path, tmp))
            self.assertFalse(os.path.exists(os.path.join(tmp, "mini_mu.npy")))

            unsafe = {
                "artifacts": {
                    "../evil.txt": {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
                }
            }
            self.assertFalse(asset_sync.extract_archive("mini", unsafe, zip_path, tmp))


class TestDownloadMapCatalog(unittest.TestCase):
    def test_repo_catalog_describes_derived_artifacts_only(self):
        from autopilot.vision import asset_sync

        catalog = asset_sync.load_catalog(os.path.join(ROOT, "data", "maps"))

        self.assertNotEqual(catalog.get("repo"), "owner/wardogs-autopilot")
        for name, info in catalog["maps"].items():
            self.assertNotIn("files", info)
            self.assertTrue(info.get("source"), name)
            artifacts = info.get("artifacts")
            self.assertTrue(artifacts, name)
            for fname, meta in artifacts.items():
                self.assertTrue(fname.startswith(name), fname)
                self.assertGreater(meta["size"], 0, fname)
                self.assertEqual(len(meta["sha256"]), 64, fname)
            archive = info.get("archive")
            self.assertTrue(archive, name)
            self.assertEqual(archive["name"], f"{name}.zip", name)
            self.assertGreater(archive["size"], 0, name)
            self.assertEqual(len(archive["sha256"]), 64, name)

    def test_verify_map_accepts_matching_and_rejects_corrupt(self):
        from autopilot.vision import asset_sync

        with tempfile.TemporaryDirectory() as tmp:
            data = b"artifact"
            path = os.path.join(tmp, "bakurani_mu.npy")
            with open(path, "wb") as f:
                f.write(data)
            catalog = {
                "maps": {
                    "bakurani": {
                        "artifacts": {
                            "bakurani_mu.npy": {
                                "size": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                            }
                        }
                    }
                }
            }

            self.assertTrue(asset_sync.verify_map("bakurani", catalog, tmp))

            with open(path, "wb") as f:
                f.write(b"corrupted")

            self.assertFalse(asset_sync.verify_map("bakurani", catalog, tmp))


if __name__ == "__main__":
    unittest.main()
