"""Debug snapshot collage generator and persistence routines.

Produces 3x2 diagnostic sheets comparing raw & processed minimap captures against
full-map crops at the estimated pose.
"""

from __future__ import annotations

import os
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

    cells = [fit(p, hh) for p in (p_mm_color, p_mm_raw, p_mm_sys, p_map_color, p_map_raw, p_map_sys)]
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


def save_debug_snapshot(
    out_dir: str,
    mm: np.ndarray,
    bgr: np.ndarray | None,
    mask: np.ndarray | None,
    pose: dict[str, Any] | None,
    diag: dict[str, Any],
    lat_info: dict[str, Any] | None = None,
    max_history: int = 10,
) -> tuple[np.ndarray, str, list[str]]:
    """Save full debug snapshot suite to disk, prune old files, and return (sheet, collage_path, parts)."""
    os.makedirs(out_dir, exist_ok=True)
    base = time.strftime("%H%M%S")
    parts: list[str] = []

    # 1. Raw grayscale minimap
    raw_path = os.path.join(out_dir, f"debug_mm_{base}.png")
    cv2.imwrite(raw_path, mm)
    parts.append(os.path.basename(raw_path))

    # 2. Raw color frame (if available)
    if bgr is not None:
        raw_bgr_path = os.path.join(out_dir, f"debug_raw_{base}.png")
        cv2.imwrite(raw_bgr_path, bgr)
        parts.append(os.path.basename(raw_bgr_path))

    # 3. Text context
    info_path = os.path.join(out_dir, f"debug_info_{base}.txt")
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"mode={diag.get('mode') or '-'}\n")
            f.write(f"reject={diag.get('reject') or '-'}\n")
            if lat_info:
                f.write(f"good={lat_info.get('good')}\n")
                if pose is not None:
                    f.write(f"map_x={pose['map_x']:.1f}\n")
                    f.write(f"map_y={pose['map_y']:.1f}\n")
                    f.write(
                        f"s={pose['s']:.3f} th={pose['th']:.1f} "
                        f"inl={pose['inl']} n_match={pose['n_match']}\n"
                    )
                f.write(f"prev={lat_info.get('prev_xy')}\n")
                f.write(f"attempt={lat_info.get('attempt', 0)}\n")
                f.write(f"elapsed={lat_info.get('elapsed', 0):.2f}\n")
        parts.append(os.path.basename(info_path))
    except Exception as exc:
        logger.warning("[debug_collage] debug info save failed: %s", exc)

    # 4. 3x2 Collage sheet
    sheet = build_debug_collage(mm, bgr, mask, pose, diag)
    collage_path = os.path.join(out_dir, f"debug_collage_{base}.png")
    cv2.imwrite(collage_path, sheet)
    parts.insert(0, os.path.basename(collage_path))

    # 5. Prune old debug snapshot files
    for prefix in ("debug_collage_", "debug_mm_", "debug_raw_", "debug_info_"):
        try:
            old_files = sorted(p for p in os.listdir(out_dir) if p.startswith(prefix))
            while len(old_files) > max_history:
                try:
                    os.remove(os.path.join(out_dir, old_files[0]))
                except OSError:
                    pass
                old_files.pop(0)
        except Exception:
            pass

    return sheet, collage_path, parts
