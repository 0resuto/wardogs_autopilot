"""Headless hybrid-engine regression: SIFT anchor + ECC tracking on a path.

Builds a synthetic minimap sequence (rotating player-up frames rendered from
the preview level, so a different resampling chain than the SIFT index) along
a known trajectory: a straight leg, then a turn. The first frame is anchored
with SIFT; the rest must be tracked by ECC within a few px and a fraction of
a degree. Prints per-frame error, anchor events and the ECC cost.

Usage: python tools/selfcheck_hybrid.py [--frames 60] [--yaw-deg-s 12]
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from autopilot.vision import locator  # noqa: E402

W, H = 336, 277
_FAILED = []


def _check(name, cond, extra=""):
    print(("PASS  %s" % name) if cond else ("FAIL  %s  %s" % (name, extra)))
    if not cond:
        _FAILED.append(name)


def _build_frame(preview, native_xy, heading_deg, ms):
    """Player-up minimap frame around native_xy with the given compass heading."""
    px, py = native_xy[0] / 2.0, native_xy[1] / 2.0  # preview = native / 2
    half_w, half_h = W * ms / 4.0, H * ms / 4.0
    x0, y0 = int(round(px - half_w)), int(round(py - half_h))
    crop = preview[y0 : y0 + int(round(H * ms / 2)), x0 : x0 + int(round(W * ms / 2))]
    mm = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    # measured convention: rotating the content by the cv2 angle yields the
    # same pose th (compass heading), so building with the heading is correct
    rot = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), heading_deg, 1.0)
    return cv2.warpAffine(mm, rot, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=60, help="number of frames (10 fps assumed)")
    ap.add_argument("--speed-mps", type=float, default=15.0)
    ap.add_argument("--yaw-deg-s", type=float, default=12.0, help="yaw after the straight leg")
    ap.add_argument("--straight-s", type=float, default=2.0)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="do not pace the frames to the 10 fps wall clock (anchor timing is time-based)",
    )
    args = ap.parse_args()

    from tools.selfcheck_features import _pick_center

    locator.set_map("zestafona")
    mu = locator.load_global_map()
    ms = locator._mini_scale("zestafona")
    px_per_m = locator.get_store().px_per_m("zestafona")
    preview = np.load(os.path.join(ROOT, "data", "maps", "zestafona_preview_16384.npy"))
    center = _pick_center(mu)
    if center is None:
        _check("textured center found", False, "no good-textured window")
        sys.exit(1)
    cx, cy = center
    locator.set_engine("hybrid")

    dt = 0.1
    native = np.array([cx * ms, cy * ms], np.float64)
    heading = 30.0
    frames = []
    truth = []
    for k in range(args.frames):
        frames.append(_build_frame(preview, native, heading, ms))
        truth.append((float(native[0]), float(native[1]), heading))
        if k * dt >= args.straight_s:
            heading = (heading + args.yaw_deg_s * dt) % 360.0
        step = args.speed_mps * dt * px_per_m
        native = native + step * np.array(
            [math.sin(math.radians(heading)), -math.cos(math.radians(heading))]
        )

    ui = np.zeros((H, W), bool)
    pos_errs, th_errs, ecc_ms = [], [], []
    anchors = 0
    prev_xy = None
    prev_th = 0.0
    next_due = time.perf_counter()
    for k, mm in enumerate(frames):
        if not args.fast:
            # pace the frames to the 10 fps capture cadence; the anchor period
            # (locator.hybrid_reanchor_s) is wall-clock based, so a slow first
            # anchor must not be compensated by racing through the rest
            now = time.perf_counter()
            if next_due > now:
                time.sleep(next_due - now)
            next_due = max(time.perf_counter(), next_due) + dt
        t0 = time.perf_counter()
        pose, diag = locator.global_pose(mm, ui, prev_xy=prev_xy, prev_th=prev_th, debug=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if pose is None:
            _check("frame %d pose" % k, False, "reject=%s" % diag.get("reject"))
            break
        if diag.get("anchor"):
            anchors += 1
        else:
            ecc_ms.append(dt_ms)
        gx, gy, gh = truth[k]
        pos_err = math.hypot(pose["map_x"] - gx, pose["map_y"] - gy)
        th_err = abs((pose["th"] - gh + 540.0) % 360.0 - 180.0)
        pos_errs.append(pos_err)
        th_errs.append(th_err)
        if k < 8 or k % 10 == 0 or diag.get("anchor"):
            print(
                "frame %3d  pos_err=%6.2f px  th_err=%5.2f deg  mode=%-8s %s"
                % (
                    k,
                    pos_err,
                    th_err,
                    diag.get("mode"),
                    "anchor" if diag.get("anchor") else "track",
                )
            )
        prev_xy = (pose["map_x"], pose["map_y"])
        prev_th = pose["th"]

    arr = np.asarray(pos_errs)
    th_arr = np.asarray(th_errs)
    # --fast runs the frames faster than real time, so the wall-clock anchor
    # period fires ~once: expect more drift than the real-time run
    mean_limit = 3.5 if args.fast else 2.0
    _check("all frames localized", len(pos_errs) == args.frames)
    _check(
        "mean position error under %.1f native px" % mean_limit,
        float(arr.mean()) < mean_limit,
        "mean=%.2f max=%.2f" % (arr.mean(), arr.max()),
    )
    _check(
        "max heading error under 2 deg",
        float(th_arr.max()) < 2.0,
        "max=%.2f" % th_arr.max(),
    )
    _check(
        "anchors only every hybrid_reanchor_s",
        anchors <= 1 + args.frames * 0.1 / 1.0 + 1,
        "anchors=%d of %d frames" % (anchors, args.frames),
    )
    if ecc_ms:
        print("ecc track: %.1f ms/frame avg (n=%d)" % (float(np.mean(ecc_ms)), len(ecc_ms)))
    print(
        "---- %d check(s) failed ----" % len(_FAILED) if _FAILED else "---- all checks passed ----"
    )
    sys.exit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()
