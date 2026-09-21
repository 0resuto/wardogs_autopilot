"""Find a saved minimap frame on the map with the real localization pipeline
and render how that same area looks on the main map for the system.

Takes any saved raw frame (output/debug_mm_*.png, debug_raw_*.png or
debug_fail_*.png), runs locator.global_pose on it exactly like the live
consumer does (including the UI mask and the same normalizations), and writes
output/map_match_<t>.png with side-by-side panels:

  [map: RAW mu crop] [map: NORMALIZED _norm8] [mm: as SIFT sees it (fill+pct 2..98)]
  [mm: raw gray]     [mm: raw color]          [overview with player marker]

The green box on the normalized map panel shows where the rotated+scaled
minimap sits on the map; the red dot is the player. Comparing the left two
panels against the right ones shows whether the map-side filters (grayscale +
percentile stretch) match what the in-game minimap renders — the typical
reason "the same places" fail.

Usage:
    python tools/map_match_debug.py <frame.png> [more.png ...] [--map NAME] [--prev X,Y]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from autopilot.vision import locator  # noqa: E402

OUT = os.path.join(ROOT, "output")


def _read_kv_txt(path):
    """Parse a 'key=value' sidecar into a dict, or None when unreadable."""
    ctx = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    ctx[k.strip()] = v.strip()
    except OSError:
        return None
    return ctx


def _sibling_info(path):
    """Context next to a saved frame, or None.

    Handles debug_fail_<ts> frames (sidecars <stem>.json / <stem>.txt) and the
    live snapshot frames debug_mm_/debug_raw_/debug_sheet_/debug_collage_ whose
    context lives in debug_info_<ts>.txt / .json.
    """
    d = os.path.dirname(path)
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    cands = [stem + ".json", stem + ".txt"]
    for pre in ("debug_mm_", "debug_raw_", "debug_sheet_", "debug_collage_"):
        if base.startswith(pre):
            ts = stem[len(pre) :]
            cands += ["debug_info_" + ts + ".json", "debug_info_" + ts + ".txt"]
    for name in cands:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        if name.endswith(".json"):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                continue
        ctx = _read_kv_txt(p)
        if ctx is not None:
            return ctx
    return None


def _parse_prev(v):
    """Normalize a sidecar 'prev' value (list or '(x, y)' string) to (x, y)."""
    if v in (None, "-", "None", "", "[]"):
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            return None
    try:
        parts = str(v).strip("()[] ").split(",")
        if len(parts) != 2:
            return None
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None


def _map_name():
    try:
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        return (cfg.get("map") or {}).get("name", "zestafona")
    except (OSError, ValueError):
        return "zestafona"


def _fit_h(panel, hh):
    h, w = panel.shape[:2]
    if h == hh:
        return panel
    k = hh / float(h)
    return cv2.resize(panel, (max(1, int(round(w * k))), hh), interpolation=cv2.INTER_AREA)


def _pad_w(img, w):
    if img.shape[1] >= w:
        return img
    return cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1], cv2.BORDER_CONSTANT, value=30)


def _caption(img, text, color=(255, 255, 0)):
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _overview_marker(mu, cx_mu, cy_mu, title):
    prevs = locator.load_previews()
    if prevs:
        size = max(n for n in prevs if n <= 4096) if prevs else None
        img = prevs[size].copy()
        k = size / float(mu.shape[0])
    else:
        img = mu.copy()
        k = 1.0
    if img.dtype != np.uint8:
        img = locator._norm8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if cx_mu is not None:
        px, py = int(round(cx_mu * k)), int(round(cy_mu * k))
        cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
        cv2.circle(img, (px, py), 12, (0, 0, 255), 2)
        cv2.line(img, (px - 22, py), (px + 22, py), (0, 0, 255), 2)
        cv2.line(img, (px, py - 22), (px, py + 22), (0, 0, 255), 2)
    cv2.putText(img, title, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def _ncc(a, b):
    """Normalized cross-correlation of two same-shaped uint8 images."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    return float((a * b).sum() / den) if den > 0 else 0.0


# candidate color->gray conversions to rank against the captured minimap
GRAY_CANDS = (("luma", 1.0), ("equal", 1.0), ("equal", 0.95), ("equal", 0.9))


def compare_gray_sheet(gray, mask, pose, ms, open_sheet):
    """Rank map gray conversions against the captured minimap at the pose.

    Projects the map area around (map_x, map_y) into minimap space with the
    pose (scale s, rotation th), converts the projected map under every
    GRAY_CANDS (color->gray + gamma), normalizes both sides the same way
    (pct 2..98) and scores each candidate with NCC against the real minimap.
    The game's own near-monochrome minimap is the reference: the conversion
    with the highest NCC is what the matcher should use on the map side.
    """
    mm_sys = locator._fill_norm(gray, mask, (2, 98))
    hh, ww = mm_sys.shape
    s, th = pose["s"], pose["th"]
    r = max(ww, hh) / 2.0 * s + 20
    mu = locator._G["mu"]
    k = locator.color_map().shape[0] / float(mu.shape[0])
    cx = pose["map_x"] / ms
    cy = pose["map_y"] / ms
    rc = max(2, int(round(r * k)))
    cmap = locator.color_map()
    y0 = max(0, int(round(cy * k)) - rc)
    x0 = max(0, int(round(cx * k)) - rc)
    side = min(2 * rc, cmap.shape[0] - y0, cmap.shape[1] - x0)
    crop = cmap[y0 : y0 + side, x0 : x0 + side]
    c = np.cos(np.radians(th))
    t = np.sin(np.radians(th))
    cc, ss_ = c / s, t / s
    half = side / 2.0
    mx, my = ww / 2.0, hh / 2.0
    # map px (crop coords, center=half, half) -> minimap px
    tx = mx - (cc + ss_) * half
    ty = my - (-ss_ + cc) * half
    M = np.array([[cc, ss_, tx], [-ss_, cc, ty]], np.float64)

    results = []
    for conv, gamma in GRAY_CANDS:
        g = locator._bgr_to_gray(crop, conv, gamma)
        proj = cv2.warpAffine(g, M, (ww, hh), flags=cv2.INTER_AREA)
        pn = locator._fill_norm(proj, np.zeros_like(proj, bool), (2, 98))
        results.append((conv, gamma, _ncc(mm_sys, pn), pn))

    results.sort(key=lambda x: x[2], reverse=True)
    print("  gray conversion score (NCC vs real minimap, higher = closer):")
    for conv, gamma, ncc, _ in results:
        print(
            "    %-6s g=%.2f  NCC=%.3f%s"
            % (conv, gamma, ncc, "*" if (conv, gamma) == results[0][:2] else "")
        )

    panels = []
    for conv, gamma, ncc, pn in results:
        p = cv2.cvtColor(pn, cv2.COLOR_GRAY2BGR)
        _caption(
            p,
            "%s g=%.2f  NCC=%.3f" % (conv, gamma, ncc),
            (0, 255, 255) if (conv, gamma) == results[0][:2] else (255, 255, 0),
        )
        panels.append(_fit_h(p, hh))
    p_mm = cv2.cvtColor(mm_sys, cv2.COLOR_GRAY2BGR)
    _caption(p_mm, "real minimap (captured)", (0, 0, 255))
    row = np.hstack([p_mm] + panels)
    sheet = np.vstack([_pad_w(row, row.shape[1])])
    sheet = np.vstack([_pad_w(row, row.shape[1]), np.full((28, row.shape[1], 3), 0, np.uint8)])
    cv2.putText(
        sheet[-28:],
        "pose x=%.0f y=%.0f s=%.2f th=%.1f inl=%d  — "
        "projected map under each gray conv vs captured minimap"
        % (pose["map_x"], pose["map_y"], pose["s"], pose["th"], pose["inl"]),
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out_path = os.path.join(OUT, "graycmp_%s.png" % time.strftime("%H%M%S"))
    cv2.imwrite(out_path, sheet)
    print("  wrote %s" % out_path)
    if open_sheet:
        os.startfile(out_path)
    return out_path


def process(path, map_name, prev_xy, open_sheet, skip_gray_cmp=False):
    base = os.path.basename(path)
    ctx = _sibling_info(path)
    if prev_xy is None and ctx is not None:
        prev_xy = _parse_prev(ctx.get("prev"))
    locator.set_map(map_name)
    mu = locator.load_global_map()
    ms = locator._mini_scale()

    color = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if color is None or gray is None:
        print("cannot read %s (skipped)" % path)
        return None
    try:
        mask = locator.make_mask()
    except OSError:
        mask = np.zeros_like(gray, bool)
    if mask.shape[:2] != gray.shape:
        mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

    t0 = time.time()
    pose, diag = locator.global_pose(gray, mask, prev_xy=prev_xy, debug=True, budget=None)
    secs = time.time() - t0

    print("\n%s  (%dx%d  map=%s)" % (base, gray.shape[1], gray.shape[0], map_name))
    if ctx is not None:
        lmp = "live pose (%.0f,%.0f) good=%s mode=%s reject=%s prev=%s" % (
            float(ctx.get("map_x") or 0),
            float(ctx.get("map_y") or 0),
            ctx.get("good"),
            ctx.get("mode"),
            ctx.get("reject"),
            ctx.get("prev"),
        )
        print("  live snapshot context: %s" % lmp)
    if pose is not None:
        print(
            "  pose: x=%.0f y=%.0f s=%.2f th=%.1f inl=%d n_match=%d  "
            "mode=%s  (%.2fs)"
            % (
                pose["map_x"],
                pose["map_y"],
                pose["s"],
                pose["th"],
                pose["inl"],
                pose["n_match"],
                diag.get("mode"),
                secs,
            )
        )
        cx, cy = pose["map_x"] / ms, pose["map_y"] / ms
        s = pose["s"]
    else:
        cx = cy = None
        s = 1.0
        print("  NO pose: reject=%s  %s" % (diag.get("reject"), diag.get("detail")))
    print(
        "  mm_mean=%.0f mm_std=%.0f kp_mm=%d mask=%d%%"
        % (
            diag.get("mm_mean") or 0,
            diag.get("mm_std") or 0,
            diag.get("kp_mm") or 0,
            int(100 * (diag.get("mm_mask_frac") or 0)),
        )
    )

    if pose is not None and pose.get("inl", 0) > 0 and not skip_gray_cmp:
        compare_gray_sheet(gray, mask, pose, ms, open_sheet)

    # --- minimap side, exactly as the pipeline feeds SIFT (pass 1: 2..98)
    mm_sys = locator._fill_norm(gray, mask, (2, 98))
    p_mm_color = color.copy()
    _caption(p_mm_color, "minimap: RAW color", (0, 255, 255))
    p_mm_raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _caption(p_mm_raw, "minimap: RAW gray")
    p_mm_proc = cv2.cvtColor(mm_sys, cv2.COLOR_GRAY2BGR)
    _caption(p_mm_proc, "minimap: PROCESSED (as SIFT sees it)")

    # --- map side: the target window and the same normalize it gets
    blank = np.full((max(60, gray.shape[0]), max(60, gray.shape[1]), 3), 30, np.uint8)
    _caption(blank, "map: no pose — cannot show the area", (0, 0, 255))
    if cx is not None:
        rw = int(round(max(gray.shape[1], gray.shape[0]) / 2.0 * s)) + 20
        win, (y0, x0) = locator._crop_win(mu, cy, cx, rw, rw)
        p_map_raw = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
        _caption(p_map_raw, "map: RAW gray crop")
        p_map_norm = cv2.cvtColor(locator._norm8(win), cv2.COLOR_GRAY2BGR)
        th_r = np.radians(pose["th"]) if pose is not None else 0.0
        a, b = s * np.cos(th_r), s * np.sin(th_r)
        ccx, ccy = p_map_norm.shape[1] / 2.0, p_map_norm.shape[0] / 2.0
        box = np.array(
            [
                (
                    ccx + a * (-gray.shape[1] / 2) - b * (-gray.shape[0] / 2),
                    ccy + b * (-gray.shape[1] / 2) + a * (-gray.shape[0] / 2),
                ),
                (
                    ccx + a * (gray.shape[1] / 2) - b * (-gray.shape[0] / 2),
                    ccy + b * (gray.shape[1] / 2) + a * (-gray.shape[0] / 2),
                ),
                (
                    ccx + a * (gray.shape[1] / 2) - b * (gray.shape[0] / 2),
                    ccy + b * (gray.shape[1] / 2) + a * (gray.shape[0] / 2),
                ),
                (
                    ccx + a * (-gray.shape[1] / 2) - b * (gray.shape[0] / 2),
                    ccy + b * (-gray.shape[1] / 2) + a * (gray.shape[0] / 2),
                ),
            ],
            np.float32,
        ).reshape(-1, 1, 2)
        cv2.polylines(p_map_norm, [np.int32(box)], True, (0, 255, 0), 2)
        cv2.circle(p_map_norm, (int(ccx), int(ccy)), 5, (0, 0, 255), -1)
        _caption(p_map_norm, "map: PROCESSED crop + mm box")
        try:
            cmap = locator.color_map()
            k = cmap.shape[0] / float(mu.shape[0])
            y0 = max(0, int(round((cy - rw) * k)))
            x0 = max(0, int(round((cx - rw) * k)))
            r = int(round(2 * rw * k))
            r = min(r, cmap.shape[0] - y0, cmap.shape[1] - x0)
            p_map_color = cmap[y0 : y0 + r, x0 : x0 + r].copy()
            cv2.circle(p_map_color, (r // 2, r // 2), 5, (0, 0, 255), -1)
            _caption(p_map_color, "map: RAW color crop", (0, 255, 255))
        except Exception as exc:  # noqa: BLE001
            p_map_color = p_map_raw.copy()
            _caption(p_map_color, "map color unavailable: %s" % exc, (0, 0, 255))
    else:
        p_map_raw = p_map_norm = p_map_color = blank

    ov_txt = "overview (marker = player)"
    if cx is not None:
        ov_txt += "  x=%.0f y=%.0f" % (pose["map_x"], pose["map_y"])
    p_ov = _overview_marker(mu, cx, cy, ov_txt)

    top = np.hstack([_fit_h(p, 300) for p in (p_mm_color, p_mm_raw, p_mm_proc)])
    mid = np.hstack([_fit_h(p, 300) for p in (p_map_color, p_map_raw, p_map_norm)])
    bottom = _fit_h(p_ov, 300)
    rows = [top, mid]
    if cx is not None:
        rows.append(bottom)
    width = max(r.shape[1] for r in rows)
    rows = [_pad_w(r, width) for r in rows]
    sheet = np.vstack(rows)

    info = "mode=%s reject=%s detail=%s  prev=%s  (%.2fs)" % (
        diag.get("mode"),
        diag.get("reject"),
        diag.get("detail"),
        prev_xy if prev_xy else "-",
        secs,
    )
    bar = np.full((28, sheet.shape[1], 3), 0, np.uint8)
    cv2.putText(
        bar,
        info[: min(len(info), 150)],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    sheet = np.vstack([bar, sheet])

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "map_match_%s.png" % time.strftime("%H%M%S"))
    cv2.imwrite(out_path, sheet)
    print("wrote %s" % out_path)
    if open_sheet:
        os.startfile(out_path)  # Windows only — fine for this machine
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+", help="saved raw minimap frame(s) to localize")
    ap.add_argument("--map", default=_map_name())
    ap.add_argument("--prev", default=None, help="prev position 'x,y' in native map px (hot start)")
    ap.add_argument("--no-open", action="store_true", help="do not open the result sheet")
    ap.add_argument(
        "--no-gray-cmp", action="store_true", help="skip the gray-conversion NCC comparison sheet"
    )
    a = ap.parse_args()
    prev = None
    if a.prev:
        x, y = a.prev.split(",")
        prev = (float(x), float(y))
    for f in a.frames:
        process(f, a.map, prev, not a.no_open, a.no_gray_cmp)


if __name__ == "__main__":
    main()
