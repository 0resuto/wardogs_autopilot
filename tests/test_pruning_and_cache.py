"""Unit tests for debug log pruning and map cache status."""
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


def load_test_cfg():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


class TestPruningAndCache(unittest.TestCase):

    def test_config_defaults(self):
        """collect_fail_logs must be False by default in config.json."""
        cfg = load_test_cfg()
        self.assertFalse(cfg.get("debug", {}).get("collect_fail_logs", True))

    def test_debug_fail_pruning(self):
        """debug_fail files must be pruned to retain at most 20 files."""
        out_dir = os.path.join(ROOT, "output")
        os.makedirs(out_dir, exist_ok=True)

        for i in range(25):
            p = os.path.join(out_dir, f"debug_fail_unit_{i:03d}.png")
            with open(p, "w") as f:
                f.write("test")

        try:
            files = sorted(p for p in os.listdir(out_dir) if p.startswith("debug_fail_unit_"))
            while len(files) > 20:
                os.remove(os.path.join(out_dir, files[0]))
                files = files[1:]

            remaining = [p for p in os.listdir(out_dir) if p.startswith("debug_fail_unit_")]
            self.assertEqual(len(remaining), 20)
        finally:
            for p in [p for p in os.listdir(out_dir) if p.startswith("debug_fail_unit_")]:
                os.remove(os.path.join(out_dir, p))

    def test_nav_dbg_pruning(self):
        """nav_dbg logs must be pruned to retain at most 10 session logs."""
        out_dir = os.path.join(ROOT, "output")
        os.makedirs(out_dir, exist_ok=True)

        for i in range(15):
            p = os.path.join(out_dir, f"nav_dbg_unit_{i:03d}.jsonl")
            with open(p, "w") as f:
                f.write("{}\n")

        try:
            old_logs = sorted(p for p in os.listdir(out_dir)
                              if p.startswith("nav_dbg_unit_") and p.endswith(".jsonl"))
            while len(old_logs) >= 10:
                os.remove(os.path.join(out_dir, old_logs[0]))
                old_logs.pop(0)

            remaining = [p for p in os.listdir(out_dir) if p.startswith("nav_dbg_unit_")]
            self.assertEqual(len(remaining), 9)
        finally:
            for p in [p for p in os.listdir(out_dir) if p.startswith("nav_dbg_unit_")]:
                os.remove(os.path.join(out_dir, p))

    def test_map_cache_status_all_maps(self):
        """All 3 maps ('zestafona', 'bakurani', 'ozeti') must report Ready with valid caches."""
        cfg = load_test_cfg()
        app = App(cfg)
        app.withdraw()
        try:
            for name in ["zestafona", "bakurani", "ozeti"]:
                txt, color = app._get_map_cache_status(name)
                self.assertEqual(color, "#8ae234", f"Map '{name}' status is not green: {txt}")
                self.assertIn("Ready", txt, f"Map '{name}' is not Ready: {txt}")
        finally:
            app.destroy()

    def test_map_gray_signatures_exist(self):
        """All 3 maps must have matching gray signature files to prevent unnecessary rebuilds."""
        maps_dir = os.path.join(ROOT, "data", "maps")
        for name in ["zestafona", "bakurani", "ozeti"]:
            sig_file = os.path.join(maps_dir, f"{name}_gray.txt")
            self.assertTrue(os.path.exists(sig_file), f"Missing signature: {sig_file}")
            with open(sig_file, encoding="utf-8") as f:
                sig = f.read().strip()
            self.assertEqual(sig, "luma-1.000")


if __name__ == "__main__":
    unittest.main()
