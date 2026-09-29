"""Build a route preset from a manual-driving recording (teach and repeat).

The recorded line follows the real road centre, so a preset built from it is
more accurate than the drawn vertices (measured mismatch: mostly <2 m, locally
up to ~7 m).

Usage:
    python tools/route_from_manual.py output/manual_dbg_*.jsonl
    python tools/route_from_manual.py output/manual_dbg_*.jsonl --map zestafona --name road_south
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import AppConfig  # noqa: E402
from autopilot.navigation.manual_record import build_route  # noqa: E402
from autopilot.ui.presets import PresetManager  # noqa: E402
from autopilot.vision.map_store import MapStore  # noqa: E402


def load_manual(path: str) -> tuple[dict, list[dict]]:
    """Read a manual_dbg JSONL: (header, rows)."""
    header: dict = {}
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") == "manual-log":
                header = rec
            else:
                rows.append(rec)
    return header, rows


def main() -> int:
    ap = argparse.ArgumentParser(prog="route_from_manual", description=__doc__)
    ap.add_argument("path", help="manual_dbg JSONL file")
    ap.add_argument("--map", default="", help="map name (default: header/config)")
    ap.add_argument("--name", default="", help="preset name (default: file stem)")
    ap.add_argument("--step-px", type=float, default=12.0, help="resample step in map px")
    ap.add_argument("--smooth", type=int, default=3, help="moving-average window (1=off)")
    args = ap.parse_args()

    header, rows = load_manual(args.path)
    if not rows:
        print("no rows in", args.path)
        return 1

    map_name = args.map or str((header.get("params") or {}).get("map") or "")
    if not map_name:
        try:
            map_name = AppConfig.load("config.json").map.name
        except Exception:
            map_name = "zestafona"
    name = args.name or os.path.splitext(os.path.basename(args.path))[0]

    points = build_route(rows, step_px=args.step_px, smooth_win=args.smooth)
    measured = sum(1 for r in rows if r.get("x") is not None and r.get("good"))
    if len(points) < 2:
        print("not enough measured poses (measured=%d)" % measured)
        return 1

    path = PresetManager(subdir=f"data/presets/{map_name}").save_preset(name, points)
    length_px = sum(
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        for a, b in zip(points, points[1:], strict=False)
    )
    px_per_m = MapStore().px_per_m(map_name) or 2.0
    print(
        "wrote %s\n  map=%s name=%s points=%d (from %d measured poses)\n"
        "  length=%.0f px (~%.0f m at %.2f px/m), step=%.0f px"
        % (
            path,
            map_name,
            name,
            len(points),
            measured,
            length_px,
            length_px / px_per_m,
            px_per_m,
            args.step_px,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
