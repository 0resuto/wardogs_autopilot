"""Headless regression for the feature-index localization path.

Builds a synthetic minimap: a 337x278 window of the active map (mu cache)
rotated by ~35 deg and scaled 1.15, then asks global_pose where the player
is. Asserts the recovered position is within 60 native px of the truth.

Usage: python tools/selfcheck_features.py
config.json is NOT modified; the vote-gate scenarios from the consumer are
tested too.
"""

import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from autopilot.vision import locator  # noqa: E402

W, H = 337, 278
ANGLE = 35.0
SCALE = 1.15
TOL_NATIVE_PX = 60.0

_FAILED = []


def _check(name, cond, extra=""):
    print(("PASS  %s" % name) if cond else ("FAIL  %s  %s" % (name, extra)))
    if not cond:
        _FAILED.append(name)


def _crop(mu, cx, cy):
    return mu[cy - H // 2:cy - H // 2 + H, cx - W // 2:cx - W // 2 + W]


def _pick_center(mu):
    """A well-textured window center near the map center."""
    h, w = mu.shape
    best, best_kp = None, -1
    r = 700
    for cy in range(h // 2 - r, h // 2 + r + 1, 256):
        for cx in range(w // 2 - r, w // 2 + r + 1, 256):
            win = _crop(mu, cx, cy)
            if win.shape != (H, W):
                continue
            kp, _ = locator.SIFT.detectAndCompute(win, None)
            n = 0 if kp is None else len(kp)
            if n > best_kp:
                best, best_kp = (cx, cy), n
            if n > 80:
                best, best_kp = (cx, cy), n
                break
    return best


def _synthetic_mm(mu, cx, cy):
    """337x278 minimap of the map around (cx,cy), rotated 35deg, scaled 1.15."""
    crop = _crop(mu, cx, cy)
    R = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -ANGLE, 1.0)
    rot = cv2.warpAffine(crop, R, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    mm = cv2.resize(rot, (int(round(W * SCALE)), int(round(H * SCALE))),
                    interpolation=cv2.INTER_LINEAR)
    return mm


def _pose_ok(pose, diag, expected):
    if pose is None:
        return False, "pose is None (reject=%s %s)" % (
            diag.get("reject"), diag.get("detail"))
    if pose["inl"] < 4:
        return False, "inl=%d < 4" % pose["inl"]
    err = math.hypot(pose["map_x"] - expected[0], pose["map_y"] - expected[1])
    if err >= TOL_NATIVE_PX:
        return False, "pos err %.0f px (tol %.0f)" % (err, TOL_NATIVE_PX)
    return True, "err=%.0fpx inl=%d mode=%s" % (
        err, pose["inl"], diag.get("mode"))


def _run_mode(mu, cx, cy, mm, expected):
    print("--- feature-index run ---")
    locator._loc_cfg = lambda: dict(local_radius=450.0,
                                    radius_growth=1.6, global_max_features=60000,
                                    ratio=0.80, min_inl=4, min_inl_rate=0.0)
    ui = np.zeros(mm.shape[:2], bool)

    pose, diag = locator.global_pose(mm, ui, prev_xy=None, debug=True,
                                     budget=30.0)
    ok, info = _pose_ok(pose, diag, expected)
    _check("cold start found pose", ok, info)
    if ok:
        _check("cold start used the feature index", diag.get("mode") == "index",
               "mode=%s (expected index)" % diag.get("mode"))

    pose, diag = locator.global_pose(mm, ui, prev_xy=expected, debug=True,
                                     budget=30.0)
    ok, info = _pose_ok(pose, diag, expected)
    _check("hot start found pose", ok, info)
    if ok:
        _check("hot start used the feature index", diag.get("mode") == "index",
               "mode=%s (expected index)" % diag.get("mode"))


def _test_vote():
    """Consumer vote-gate: a single jump candidate is not published, three
    agreeing candidates are."""
    from autopilot.vision import tracker

    dbuf = tracker._vote_decide
    cases = [
        ([], 3, 300.0, (1000, 1000), False),                       # lone candidate
        ([(1000, 1000), (1010, 998)], 3, 300.0, (1005, 1003), True),   # 3 agree
        ([(0, 0), (5000, 0), (0, 5000)], 3, 300.0, (3, 2), False),   # scattered
        ([(0, 0), (5000, 0)], 3, 300.0, (1004, 1000), False),        # 2+1 far apart
    ]
    for buf, need, rad, new, want in cases:
        got = dbuf(list(buf), need, rad, new)
        _check("vote_decide cluster=%s want=%s" % (new, "accept" if want else "reject"),
               (got is not None) == want,
               "got=%s" % got)


def main():
    print("loading map cache + feature index...")
    mu = locator.load_global_map()
    ms = locator._mini_scale("zestafona")
    idx = locator._get_index()
    if idx is not None:
        print("index: %s features=%d" % (idx.name, idx.total_features()))
    else:
        print("WARNING: no feature index — build it first: "
              "python -m autopilot.vision.featureindex --build zestafona")

    center = _pick_center(mu)
    print("test center (mu): %s  native: %s" % (center,
                                                 (center[0] * ms, center[1] * ms)))
    if center is None:
        _check("textured center found", False, "no good-textured window")
        sys.exit(1)
    cx, cy = center
    mm = _synthetic_mm(mu, cx, cy)
    expected = (cx * ms, cy * ms)
    print("synthetic minimap: %s (s=%.2f) expected=(%.0f,%.0f)"
          % (mm.shape, SCALE, expected[0], expected[1]))

    _run_mode(mu, cx, cy, mm, expected)

    _test_vote()

    print("---- %d check(s) failed ----" % len(_FAILED) if _FAILED
          else "---- all checks passed ----")
    sys.exit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()
