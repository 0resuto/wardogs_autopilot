"""Fit vehicle calibration constants from nav_dbg JSONL runs.

Usage:
    python tools/calibrate_vehicle.py output/nav_dbg_*.jsonl
    python tools/calibrate_vehicle.py output/nav_dbg_*.jsonl --write

The runs come from the vehicle-tuning runbook (docs/vehicle-tuning.md):
straight acceleration, full braking and normal cornering with the Nav log
enabled. Fits are medians over episodes; --write updates yaw_gain and brake_g
in the config file atomically.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.common.config import AppConfig  # noqa: E402
from autopilot.navigation.calibration import (  # noqa: E402
    fit_brake,
    fit_px_per_m,
    fit_yaw_gain,
    format_report,
)
from autopilot.navigation.vehicle_model import VehicleModel  # noqa: E402


def load_runs(paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the run files: (header params, telemetry rows)."""
    params: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get("kind") == "nav-log":
                    params = row.get("params") or {}
                else:
                    rows.append(row)
    return params, rows


def current_yaw_gain(cfg: AppConfig, params: dict[str, Any]) -> float:
    """The gain the logs were produced with: config override or profile value."""
    if cfg.navigator.yaw_gain is not None:
        return float(cfg.navigator.yaw_gain)
    name = str(params.get("vehicle") or cfg.navigator.vehicle_profile or "")
    if name:
        try:
            return float(VehicleModel.load(name).yaw_gain)
        except (OSError, ValueError, KeyError):
            pass
    return 1.0


def expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and os.path.exists(pattern):
            matches = [pattern]
        paths.extend(matches)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(prog="calibrate_vehicle", description=__doc__)
    ap.add_argument("runs", nargs="+", help="nav_dbg JSONL files or glob patterns")
    ap.add_argument("--config", default="config.json")
    ap.add_argument(
        "--write",
        action="store_true",
        help="write the suggested yaw_gain and brake_g into the config file",
    )
    args = ap.parse_args()

    paths = expand(args.runs)
    if not paths:
        print("no run files matched")
        return 1

    params, rows = load_runs(paths)
    cfg = AppConfig.load(args.config)
    gain = current_yaw_gain(cfg, params)

    px_fit = fit_px_per_m(rows)
    brake_fit = fit_brake(rows)
    yaw_fit = fit_yaw_gain(rows, current_gain=gain)

    print(f"files: {len(paths)}")
    print(format_report(px_fit, brake_fit, yaw_fit, len(rows)))

    if args.write:
        changed = []
        if brake_fit.suggested_g is not None:
            cfg.navigator.brake_g = round(brake_fit.suggested_g, 3)
            changed.append(f"brake_g={cfg.navigator.brake_g}")
        if yaw_fit.suggested_gain is not None:
            cfg.navigator.yaw_gain = round(yaw_fit.suggested_gain, 3)
            changed.append(f"yaw_gain={cfg.navigator.yaw_gain}")
        if changed:
            cfg.save()
            print(f"wrote {args.config}: " + ", ".join(changed))
        else:
            print("nothing to write (not enough data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
