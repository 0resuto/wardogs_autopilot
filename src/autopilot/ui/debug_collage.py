"""Debug snapshot collage generator and persistence routines.

Produces 3x2 diagnostic sheets comparing raw & processed minimap captures against
full-map crops at the estimated pose.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from typing import Any

import cv2
import numpy as np

from ..common.log import get_logger
from ..vision import locator

logger = get_logger("debug_collage")


def _cap(img: np.ndarray, text: str, color: tuple[int, int, int] = (255, 255, 0)) -> None:
    """Draw a readable diagnostic label on the top-left of an image."""
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def build_debug_collage(
    mm: np.ndarray,
    mm_bgr: np.ndarray | None,
    ui: np.ndarray | None,
    pose: dict[str, Any] | None,
    diag: dict[str, Any],
) -> np.ndarray:
    """Build a 3x2 snapshot collage:
    COLOR minimap  | RAW minimap  | PROCESSED minimap (UI-fill + pct 2..98)
    COLOR map crop | RAW map crop | PROCESSED map crop (pct 2..98 + box)
    """
    hh = 300

    def fit(g: np.ndarray, h: int) -> np.ndarray:
        return cv2.resize(
            g,
            (max(1, int(round(g.shape[1] * h / g.shape[0]))), h),
            interpolation=cv2.INTER_AREA,
        )

    def pad(c: np.ndarray, wd: int) -> np.ndarray:
        if c.shape[1] >= wd:
            return c
        return cv2.copyMakeBorder(c, 0, 0, 0, wd - c.shape[1], cv2.BORDER_CONSTANT, value=40)

    def gray2bgr(g: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    p_mm_color = mm_bgr if mm_bgr is not None else gray2bgr(mm)
    _cap(
        p_mm_color,
        "minimap: RAW color" if mm_bgr is not None else "minimap: RAW color (gray fallback)",
        (0, 255, 255),
    )
    p_mm_raw = gray2bgr(mm)
    _cap(p_mm_raw, "minimap: RAW gray")
    p_mm_sys = gray2bgr(locator.fill_norm(mm, ui, (2, 98)))
    _cap(p_mm_sys, "minimap: PROCESSED (as SIFT sees it)")

    if pose is not None:
        mu = locator.load_global_map()
        ms = locator._mini_scale()
        cx, cy = pose["map_x"] / ms, pose["map_y"] / ms
        s, th = pose["s"], pose["th"]
        rw = int(round(max(mm.shape[1], mm.shape[0]) / 2.0 * s)) + 20
        win, _ = locator.crop_win(mu, cy, cx, rw, rw)
        p_map_raw = gray2bgr(win)
        _cap(
            p_map_raw,
            f"map: RAW gray crop  ({pose['map_x']:.0f}, {pose['map_y']:.0f})",
        )
        p_map_sys = gray2bgr(locator.norm8(win))
        th_r = np.radians(th)
        a, b = s * np.cos(th_r), s * np.sin(th_r)
        ccx, ccy = p_map_sys.shape[1] / 2.0, p_map_sys.shape[0] / 2.0
        box = np.array(
            [
                (
                    ccx + a * (-mm.shape[1] / 2) - b * (-mm.shape[0] / 2),
                    ccy + b * (-mm.shape[1] / 2) + a * (-mm.shape[0] / 2),
                ),
                (
                    ccx + a * (mm.shape[1] / 2) - b * (-mm.shape[0] / 2),
                    ccy + b * (mm.shape[1] / 2) + a * (-mm.shape[0] / 2),
                ),
                (
                    ccx + a * (mm.shape[1] / 2) - b * (mm.shape[0] / 2),
                    ccy + b * (mm.shape[1] / 2) + a * (mm.shape[0] / 2),
                ),
                (
                    ccx + a * (-mm.shape[1] / 2) - b * (mm.shape[0] / 2),
                    ccy + b * (-mm.shape[1] / 2) + a * (mm.shape[0] / 2),
                ),
            ],
            np.float32,
        ).reshape(-1, 1, 2)
        cv2.polylines(p_map_sys, [box.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(p_map_sys, (int(ccx), int(ccy)), 5, (0, 0, 255), -1)
        _cap(p_map_sys, "map: PROCESSED crop + mm box")
        try:
            cmap = locator.color_map()
            k = cmap.shape[0] / float(mu.shape[0])
            y0 = int(round((cy - rw) * k))
            x0 = int(round((cx - rw) * k))
            r = int(round(2 * rw * k))
            y0 = max(0, y0)
            x0 = max(0, x0)
            r = min(r, cmap.shape[0] - y0, cmap.shape[1] - x0)
            p_map_color = cmap[y0 : y0 + r, x0 : x0 + r].copy()
            cv2.circle(p_map_color, (r // 2, r // 2), 5, (0, 0, 255), -1)
            _cap(
                p_map_color,
                f"map: RAW color crop  ({pose['map_x']:.0f}, {pose['map_y']:.0f})",
                (0, 255, 255),
            )
        except Exception as exc:
            p_map_color = p_map_raw.copy()
            _cap(p_map_color, f"map color unavailable: {exc}", (0, 0, 255))
    else:
        blank = np.full((max(60, mm.shape[0]), max(60, mm.shape[1]), 3), 30, np.uint8)
        _cap(blank, "map: no pose — cannot show the area", (0, 0, 255))
        p_map_color = p_map_raw = p_map_sys = blank

    cells = [
        fit(p, hh) for p in (p_mm_color, p_mm_raw, p_mm_sys, p_map_color, p_map_raw, p_map_sys)
    ]
    width = max(c.shape[1] for c in cells)
    cells = [pad(c, width) for c in cells]
    sep = np.full((4, 3 * width, 3), 40, np.uint8)
    sheet = np.vstack([np.hstack(cells[0:3]), sep, np.hstack(cells[3:6])])

    info = (
        f"mode={diag.get('mode')} reject={diag.get('reject')} detail={diag.get('detail')} "
        f"inl={(pose or {}).get('inl', 0)} n_match={(pose or {}).get('n_match', 0)}"
    )
    bar = np.full((28, sheet.shape[1], 3), 0, np.uint8)
    cv2.putText(
        bar,
        info[: min(len(info), 200)],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, sheet])


def render_map_crop(
    pose: dict[str, Any],
    mm_shape: tuple[int, int] = (180, 240),
    crop_size: int = 600,
    map_name: str | None = None,
) -> np.ndarray | None:
    """Render a high-resolution map crop with vehicle position, heading, and minimap bounds."""
    try:
        map_x = float(pose["map_x"])
        map_y = float(pose["map_y"])
        th = float(pose.get("th", 0.0))
        s = float(pose.get("s", 1.0))
        inl = int(pose.get("inl", 0))

        # Try color map first, then grayscale mu fallback
        cmap = None
        try:
            cmap = locator.color_map()
        except Exception:
            cmap = None

        if cmap is None or not getattr(cmap, "size", 0):
            try:
                mu = locator.load_global_map()
                if mu is not None and getattr(mu, "size", 0):
                    cmap = cv2.cvtColor(mu, cv2.COLOR_GRAY2BGR)
            except Exception:
                cmap = None

        if cmap is None or not getattr(cmap, "size", 0):
            return None

        sz = locator.full_map_size(map_name) or (32768, 32768)
        full_w = float(sz[0] if isinstance(sz, (tuple, list)) else sz)
        k = cmap.shape[1] / full_w
        ms = locator._mini_scale(map_name)

        cx_cmap = map_x * k
        cy_cmap = map_y * k

        half = crop_size // 2
        x0 = int(round(cx_cmap - half))
        y0 = int(round(cy_cmap - half))
        x1 = x0 + crop_size
        y1 = y0 + crop_size

        crop = np.full((crop_size, crop_size, 3), (35, 35, 35), dtype=np.uint8)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(cmap.shape[1], x1)
        src_y1 = min(cmap.shape[0], y1)

        if src_x1 > src_x0 and src_y1 > src_y0:
            dst_x0 = src_x0 - x0
            dst_y0 = src_y0 - y0
            crop[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = cmap[
                src_y0:src_y1, src_x0:src_x1
            ]

        ccx = int(round(cx_cmap - x0))
        ccy = int(round(cy_cmap - y0))

        # 1. Minimap footprint box on the map
        mm_h, mm_w = mm_shape[:2]
        hw = (mm_w / 2.0) * (s * ms) * k
        hh = (mm_h / 2.0) * (s * ms) * k
        th_rad = math.radians(th)
        cos_t, sin_t = math.cos(th_rad), math.sin(th_rad)
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        box_pts = []
        for dx, dy in corners:
            bx = ccx + dx * cos_t - dy * sin_t
            by = ccy + dx * sin_t + dy * cos_t
            box_pts.append([int(round(bx)), int(round(by))])
        cv2.polylines(
            crop,
            [np.array(box_pts)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # 2. Vehicle target ring and center dot
        cv2.circle(crop, (ccx, ccy), 10, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(crop, (ccx, ccy), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        # 3. Directional heading arrow (0 deg = North/-Y, 90 deg = East/+X)
        arr_len = 40
        adx = math.sin(th_rad) * arr_len
        ady = -math.cos(th_rad) * arr_len
        arr_end = (int(round(ccx + adx)), int(round(ccy + ady)))
        cv2.arrowedLine(
            crop,
            (ccx, ccy),
            arr_end,
            (0, 255, 255),
            2,
            tipLength=0.3,
            line_type=cv2.LINE_AA,
        )

        # 4. Top info banner with coordinates and heading
        ov = crop.copy()
        cv2.rectangle(ov, (0, 0), (crop_size, 34), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.75, crop, 0.25, 0, crop)
        banner_text = f"POS: ({map_x:.0f}, {map_y:.0f})  HDG: {th:.1f}°  INL: {inl}  SCALE: {s:.3f}"
        cv2.putText(
            crop,
            banner_text,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return crop
    except Exception as exc:
        logger.warning("Failed to render map crop: %s", exc)
        return None


def format_state_log(
    timestamp: float,
    map_name: str | None,
    roi: list[int] | tuple[int, ...] | None,
    pose: dict[str, Any] | None,
    diag: dict[str, Any],
    lat_info: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Format human-readable log and machine-readable JSON dictionary."""
    iso_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    ms = int((timestamp % 1.0) * 1000)
    time_str = f"{iso_time}.{ms:03d}"

    is_localized = pose is not None and "map_x" in pose and "map_y" in pose
    kp_mm = diag.get("kp_mm", len(diag.get("kp_pts", [])))
    inl = pose.get("inl", diag.get("inl1", len(diag.get("inlier_pts", [])))) if pose else 0
    mode = diag.get("mode", "unknown")
    reject = diag.get("reject")
    detail = diag.get("detail", "")
    elapsed = lat_info.get("elapsed", 0.0) if lat_info else 0.0
    elapsed_ms = elapsed * 1000.0

    lines = [
        "=" * 60,
        "WARDOGS AUTOPILOT - DIAGNOSTIC SNAPSHOT",
        "=" * 60,
        f"Timestamp:       {time_str} (UNIX {timestamp:.3f})",
        f"Active Map:      {map_name or 'unknown'}",
        f"Capture ROI:     {list(roi) if roi else 'not set'}",
        "",
        "--- Localization Status ---",
        f"Status:          {'LOCALIZED' if is_localized else 'SEARCHING / UNLOCALIZED'}",
    ]
    if is_localized and pose is not None:
        lines.extend([
            f"Position (X, Y): ({pose['map_x']:.1f}, {pose['map_y']:.1f}) px",
            f"Heading:         {pose.get('th', 0.0):.1f}°",
            f"Scale:           {pose.get('s', 1.0):.3f}",
            f"Inliers:         {inl}",
            f"Total Matches:   {pose.get('n_match', 0)}",
        ])
    lines.extend([
        f"Proc Latency:    {elapsed_ms:.1f} ms",
        "",
        "--- Vision & SIFT Diagnostics ---",
        f"Mode:            {mode}",
        f"Keypoints Found: {kp_mm}",
        f"Inliers Count:   {inl}",
        f"Reject Reason:   {reject or 'None'}",
        f"Details:         {detail or 'OK'}",
        "=" * 60,
    ])
    text_log = "\n".join(lines) + "\n"

    json_payload = {
        "timestamp": timestamp,
        "datetime": time_str,
        "map_name": map_name,
        "roi": list(roi) if roi else None,
        "localized": is_localized,
        "pose": {
            "map_x": pose.get("map_x"),
            "map_y": pose.get("map_y"),
            "heading_deg": pose.get("th"),
            "scale": pose.get("s"),
            "inliers": inl,
            "n_match": pose.get("n_match"),
        }
        if is_localized and pose
        else None,
        "latency_ms": round(elapsed_ms, 2),
        "diag": {
            "mode": mode,
            "reject": reject,
            "detail": detail,
            "kp_count": kp_mm,
            "inl_count": inl,
            "mm_mean": diag.get("mm_mean"),
            "mm_std": diag.get("mm_std"),
            "mm_mask_frac": diag.get("mm_mask_frac"),
        },
    }

    return text_log, json_payload


def save_debug_snapshot(
    out_dir: str,
    mm: np.ndarray,
    bgr: np.ndarray | None,
    mask: np.ndarray | None,
    pose: dict[str, Any] | None,
    diag: dict[str, Any],
    lat_info: dict[str, Any] | None = None,
    map_name: str | None = None,
    roi: list[int] | tuple[int, ...] | None = None,
    max_history: int = 25,
) -> tuple[np.ndarray, str, list[str]]:
    """Save full debug snapshot suite to disk in a timestamped folder, prune old snapshots,
    and return (sheet, snapshot_dir, parts).
    """
    os.makedirs(out_dir, exist_ok=True)
    now = time.time()
    ms = int((now % 1.0) * 1000)
    base = f"{time.strftime('%Y%m%d_%H%M%S')}_{ms:03d}"
    snap_dir = os.path.join(out_dir, f"snapshot_{base}")
    os.makedirs(snap_dir, exist_ok=True)
    parts: list[str] = []

    # Prepare base color/gray frame
    frame = (
        bgr.copy()
        if (bgr is not None and getattr(bgr, "size", 0))
        else cv2.cvtColor(mm, cv2.COLOR_GRAY2BGR)
    )
    h, w = frame.shape[:2]

    # 1. Preview 1: Raw Capture
    raw_path = os.path.join(snap_dir, "1_raw_capture.png")
    cv2.imwrite(raw_path, frame)
    parts.append("1_raw_capture.png")

    # 2. Preview 2: Mask Overlay
    p2 = frame.copy()
    if mask is not None and getattr(mask, "size", 0):
        m = np.asarray(mask, bool)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        overlay = p2.copy()
        overlay[m] = (0, 30, 220)
        cv2.addWeighted(overlay, 0.45, p2, 0.55, 0, p2)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p2, cnts, -1, (0, 160, 255), 1)
    mask_path = os.path.join(snap_dir, "2_mask_overlay.png")
    cv2.imwrite(mask_path, p2)
    parts.append("2_mask_overlay.png")

    # 3. Preview 3: SIFT Keypoints & Inliers
    p3 = frame.copy()
    kp_pts = diag.get("kp_pts") or []
    inlier_pts = diag.get("inlier_pts") or []
    for pt in kp_pts:
        cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 255), -1)
    for pt in inlier_pts:
        cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 255, 0), -1)
        cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 6, (0, 200, 0), 1)
    sift_path = os.path.join(snap_dir, "3_sift_features.png")
    cv2.imwrite(sift_path, p3)
    parts.append("3_sift_features.png")

    # 4. Map crop with markings (only when position is determined)
    is_localized = pose is not None and "map_x" in pose and "map_y" in pose
    if is_localized and pose is not None:
        map_crop = render_map_crop(pose, mm_shape=(h, w), map_name=map_name)
        if map_crop is not None:
            crop_path = os.path.join(snap_dir, "4_map_crop.png")
            cv2.imwrite(crop_path, map_crop)
            parts.append("4_map_crop.png")

    # 5. State logs (human-readable text summary and structured JSON)
    text_log, json_payload = format_state_log(now, map_name, roi, pose, diag, lat_info)
    txt_path = os.path.join(snap_dir, "state_log.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_log)
    parts.append("state_log.txt")

    json_path = os.path.join(snap_dir, "state.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)
    parts.append("state.json")

    # 6. Consolidated 3x2 collage sheet
    sheet = build_debug_collage(mm, bgr, mask, pose, diag)
    collage_path = os.path.join(snap_dir, "collage.png")
    cv2.imwrite(collage_path, sheet)
    parts.append("collage.png")

    # 7. Prune older snapshot directories
    try:
        all_snaps = sorted(
            d
            for d in os.listdir(out_dir)
            if d.startswith("snapshot_") and os.path.isdir(os.path.join(out_dir, d))
        )
        while len(all_snaps) > max_history:
            old_dir = os.path.join(out_dir, all_snaps.pop(0))
            shutil.rmtree(old_dir, ignore_errors=True)
    except Exception as exc:
        logger.warning("Failed to prune old snapshots: %s", exc)

    return sheet, snap_dir, parts
