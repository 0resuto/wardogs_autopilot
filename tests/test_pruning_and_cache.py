"""Regression tests for debug-artifact pruning and map cache reporting.

The pruning checks call the production code (LiveLocator._prune_fail_files,
NavTelemetryLogger.open) with temporary directories; the map-cache checks read
the real data/maps tree and are skipped when the caches are not downloaded.
"""

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.telemetry import NavTelemetryLogger  # noqa: E402
from autopilot.ui import theme  # noqa: E402
from autopilot.ui.tabs.roi_tab import map_cache_status  # noqa: E402
from autopilot.vision.tracker import LiveLocator  # noqa: E402

MAP_NAMES = ("zestafona", "bakurani", "ozeti")


def _map_cache_ready(name: str) -> bool:
    data_dir = os.path.join(ROOT, "data", "maps")
    suffixes = ("mu.npy", "feat.npz", "gray.txt")
    if not all(os.path.exists(os.path.join(data_dir, f"{name}_{s}")) for s in suffixes):
        return False
    return any(
        os.path.exists(os.path.join(data_dir, f"{name}_preview_{sz}.npy"))
        for sz in (512, 1024, 2048, 4096, 8192, 16384)
    )


class TestArtifactPruning(unittest.TestCase):
    def test_fail_frame_pruning_keeps_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(12):
                with open(
                    os.path.join(tmp, f"debug_fail_{i:03d}.png"), "w", encoding="utf-8"
                ) as fh:
                    fh.write("x")

            LiveLocator._prune_fail_files(tmp, keep=5)

            remaining = sorted(p for p in os.listdir(tmp) if p.startswith("debug_fail_"))
            self.assertEqual(remaining, [f"debug_fail_{i:03d}.png" for i in range(7, 12)])

    def test_fail_frame_pruning_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("snapshot_dir", "autopilot.log", "debug_fail_a.png"):
                path = os.path.join(tmp, name)
                if name.endswith(".png"):
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write("x")
                else:
                    os.makedirs(path, exist_ok=True)

            LiveLocator._prune_fail_files(tmp, keep=0)

            self.assertFalse(os.path.exists(os.path.join(tmp, "debug_fail_a.png")))
            self.assertTrue(os.path.isdir(os.path.join(tmp, "snapshot_dir")))

    def test_nav_dbg_pruning_keeps_latest_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(15):
                with open(os.path.join(tmp, f"nav_dbg_{i:03d}.jsonl"), "w", encoding="utf-8") as fh:
                    fh.write("{}\n")

            logger = NavTelemetryLogger(dbg_target=True)
            logger.open([], {}, out_dir=tmp)
            try:
                files = sorted(
                    p for p in os.listdir(tmp) if p.startswith("nav_dbg_") and p.endswith(".jsonl")
                )
                self.assertEqual(len(files), 10)
                self.assertNotIn("nav_dbg_000.jsonl", files)
                assert logger._dbg_name is not None
                with open(logger._dbg_name, encoding="utf-8") as fh:
                    header = json.loads(fh.readline())
                self.assertEqual(header["kind"], "nav-log")
            finally:
                logger.close()

    def test_nav_dbg_explicit_path_is_not_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            keep = os.path.join(tmp, "keep.jsonl")
            with open(keep, "w", encoding="utf-8") as fh:
                fh.write("old\n")
            extra = os.path.join(tmp, "nav_dbg_000.jsonl")
            with open(extra, "w", encoding="utf-8") as fh:
                fh.write("{}\n")

            logger = NavTelemetryLogger(dbg_target=keep)
            logger.open([], {})
            try:
                self.assertTrue(os.path.exists(extra))
                with open(keep, encoding="utf-8") as fh:
                    self.assertEqual(json.loads(fh.readline())["kind"], "nav-log")
            finally:
                logger.close()


@unittest.skipUnless(
    all(_map_cache_ready(name) for name in MAP_NAMES), "map caches are not downloaded"
)
class TestMapCacheReporting(unittest.TestCase):
    def test_all_maps_report_ready(self):
        for name in MAP_NAMES:
            txt, color = map_cache_status(name)
            self.assertEqual(color, theme.GREEN, f"Map '{name}' status is not green: {txt}")
            self.assertIn("Ready", txt, f"Map '{name}' is not Ready: {txt}")

    def test_missing_map_is_reported(self):
        txt, color = map_cache_status("no_such_map")
        self.assertEqual(color, theme.RED)
        self.assertIn("Not downloaded", txt)

    def test_empty_name(self):
        txt, color = map_cache_status("")
        self.assertEqual(color, theme.TEXT_DIM)
        self.assertIn("No map", txt)

    def test_gray_signatures_match_config(self):
        """Cache signatures must carry the active palette and the map identity."""
        from autopilot.common.config import AppConfig

        conv = AppConfig.load("config.json").map.gray_conv
        maps_dir = os.path.join(ROOT, "data", "maps")
        for name in MAP_NAMES:
            with open(os.path.join(maps_dir, f"{name}_gray.txt"), encoding="utf-8") as fh:
                sig = fh.read().strip()
            self.assertTrue(sig.startswith(f"{conv}-"), f"{name}: {sig}")
            self.assertIn("|ms=", sig)


if __name__ == "__main__":
    unittest.main()
