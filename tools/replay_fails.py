"""Batch-replay collected localization-failure frames and summarize the result.

After a live run with "collect fail logs" enabled, output/debug_fail_*.png holds
the frames the locator could not place, each with a <stem>.json/.txt sidecar
carrying the live context (previous position, reject reason, thresholds). This
tool feeds every frame through locator.global_pose exactly like the live
consumer and prints a triage table: whether the frame still fails, its reject
reason, keypoint/inlier counts and the wall time. Read-only unless --json.

Usage:
    python tools/replay_fails.py [--dir output] [--map NAME] [--no-prev]
                                 [--limit N] [--budget SEC] [--json]
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
for _p in (os.path.join(ROOT, "src"), TOOLS, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_match_debug as mmd  # noqa: E402

from autopilot.vision import locator  # noqa: E402


def _frames(out_dir, newest_first=False):
    fs = [p for p in glob.glob(os.path.join(out_dir, "debug_fail_*.png"))
          if not p.endswith("_rgb.png")]
    fs.sort(reverse=newest_first)
    return fs


def _mask_for(gray):
    mask = locator.make_mask()
    if mask.shape[:2] != gray.shape:
        mask = cv2.resize(mask.astype("uint8"), (gray.shape[1], gray.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    return mask


def main():
    ap = argparse.ArgumentParser(
        prog="replay_fails",
        description="Replay output/debug_fail_*.png frames and triage them.")
    ap.add_argument("--dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--map", default=mmd._map_name())
    ap.add_argument("--no-prev", action="store_true",
                    help="cold start (ignore the saved prev position)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N newest frames (0 = all)")
    ap.add_argument("--budget", type=float, default=None,
                    help="per-frame time budget in seconds (default: none)")
    ap.add_argument("--json", action="store_true",
                    help="write output/fail_report_<ts>.json")
    a = ap.parse_args()

    frames = _frames(a.dir, newest_first=bool(a.limit))
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        print("no debug_fail_*.png frames in %s" % a.dir)
        return

    locator.set_map(a.map)
    locator.load_global_map()
    print("map=%s  frames=%d  prev=%s  budget=%s"
          % (a.map, len(frames), "off" if a.no_prev else "from sidecar",
             a.budget))

    rows = []
    n_ok = 0
    for i, path in enumerate(frames, 1):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("  [%02d] cannot read %s" % (i, os.path.basename(path)))
            continue
        ctx = mmd._sibling_info(path) or {}
        prev = None if a.no_prev else mmd._parse_prev(ctx.get("prev"))
        t = time.time()
        pose, diag = locator.global_pose(
            gray, _mask_for(gray), prev_xy=prev, debug=True, budget=a.budget)
        dt = time.time() - t
        if pose is not None:
            n_ok += 1
            place = "OK  x=%.0f y=%.0f inl=%d" % (
                pose["map_x"], pose["map_y"], pose["inl"])
        else:
            place = "FAIL %s" % (diag.get("reject") or "?")
        rows.append(dict(
            frame=os.path.basename(path),
            prev=list(prev) if prev else None,
            live_reject=ctx.get("reject"),
            ok=pose is not None,
            reject=diag.get("reject"),
            inl=int(pose["inl"]) if pose else diag.get("inl1"),
            kp=diag.get("kp_mm"),
            n_match=pose.get("n_match") if pose else diag.get("good1"),
            map_x=float(pose["map_x"]) if pose else None,
            map_y=float(pose["map_y"]) if pose else None,
            secs=round(dt, 3),
        ))
        print("  [%02d] %-38s %-28s kp=%-5s %6.0fms"
              % (i, os.path.basename(path), place,
                 diag.get("kp_mm"), dt * 1000))

    print("\n%d/%d frames localize now (%.0f%%), %d still fail"
          % (n_ok, len(rows), 100.0 * n_ok / max(len(rows), 1),
             len(rows) - n_ok))
    tally = {}
    for r in rows:
        if not r["ok"]:
            tally[r["reject"]] = tally.get(r["reject"], 0) + 1
    if tally:
        print("remaining failures: %s"
              % ", ".join("%s=%d" % kv for kv in sorted(tally.items())))

    if a.json:
        out = os.path.join(a.dir, "fail_report_%s.json"
                           % time.strftime("%Y%m%d_%H%M%S"))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dict(map=a.map, total=len(rows), ok=n_ok,
                           budget=a.budget, no_prev=a.no_prev,
                           remaining=tally, rows=rows), f,
                      ensure_ascii=False, indent=2)
        print("wrote %s" % out)


if __name__ == "__main__":
    main()
