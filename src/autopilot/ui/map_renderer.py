"""Map viewport rendering, mipmap selection, and coordinate transforms.

Computes visible map subregions from preview pyramids and translates between
screen canvas coordinates and full-map native pixels.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .theme import RGB_CANVAS


def calc_fit_viewport(
    canvas_w: float, canvas_h: float, map_w: float, map_h: float
) -> tuple[float, float, float]:
    """Calculate scale and centered offsets (scale, offx, offy) to fit map on canvas."""
    scale = min(canvas_w / map_w, canvas_h / map_h)
    offx = (canvas_w - map_w * scale) / 2.0
    offy = (canvas_h - map_h * scale) / 2.0
    return scale, offx, offy


def screen_to_native(
    disp: tuple[float, float, float], cx: float, cy: float, thumb_factor: float
) -> tuple[float, float]:
    """Transform canvas (cx, cy) to native full-map pixel coordinates."""
    s, ox, oy = disp
    return (cx - ox) / s * thumb_factor, (cy - oy) / s * thumb_factor


def native_to_screen(
    disp: tuple[float, float, float], nx: float, ny: float, thumb_factor: float
) -> tuple[float, float]:
    """Transform native full-map pixel (nx, ny) to canvas coordinates."""
    s, ox, oy = disp
    return ox + (nx / thumb_factor) * s, oy + (ny / thumb_factor) * s


def crop_map_viewport(
    s: float,
    ru: float,
    rv: float,
    rw: float,
    rh: float,
    pyr: dict[int, np.ndarray],
    map_size: int,
    thumb_factor: int,
    bg_color: tuple[int, int, int] = RGB_CANVAS,
) -> np.ndarray:
    """Render BGR viewport of a map region (ru, rv, rw, rh in thumb units) at scale s.

    Selects the optimal preview pyramid mipmap level for sharpness and performance.
    Areas outside the map are filled with bg_color.
    """
    dw = max(1, int(round(rw * s)))
    dh = max(1, int(round(rh * s)))
    img = np.full((dh, dw, 3), bg_color, np.uint8)
    if rw <= 0 or rh <= 0:
        return img

    sizes = sorted(pyr.keys())
    hi_keep = s * 4096.0
    src_s = sizes[-1]
    for n in sizes:
        if n >= hi_keep:
            src_s = n
            break

    src = pyr[src_s]
    sz = map_size[0] if isinstance(map_size, (tuple, list)) else map_size
    lam = src_s / float(sz)
    x0 = ru * thumb_factor * lam
    y0 = rv * thumb_factor * lam
    x1 = (ru + rw) * thumb_factor * lam
    y1 = (rv + rh) * thumb_factor * lam
    xa = max(0.0, x0)
    ya = max(0.0, y0)
    xb = min(float(src_s), x1)
    yb = min(float(src_s), y1)
    if xb <= xa or yb <= ya:
        return img

    ia, ib = int(xa), int(math.ceil(xb))
    ja, jb = int(ya), int(math.ceil(yb))
    crop = src[ja:jb, ia:ib]
    la = (ia / lam / thumb_factor - ru) * s
    ta = (ja / lam / thumb_factor - rv) * s
    lw = (ib - ia) / lam / thumb_factor * s
    lh = (jb - ja) / lam / thumb_factor * s
    px0 = int(round(la))
    py0 = int(round(ta))
    pxc = max(1, int(round(lw)))
    pyc = max(1, int(round(lh)))

    head = cv2.resize(crop, (pxc, pyc), interpolation=cv2.INTER_LINEAR)
    if len(head.shape) == 2 or head.shape[2] == 1:
        head = cv2.cvtColor(head, cv2.COLOR_GRAY2BGR)
    h_head, w_head = head.shape[:2]

    dst_x0 = max(0, px0)
    dst_y0 = max(0, py0)
    dst_x1 = min(dw, px0 + w_head)
    dst_y1 = min(dh, py0 + h_head)

    src_x0 = max(0, -px0)
    src_y0 = max(0, -py0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        img[dst_y0:dst_y1, dst_x0:dst_x1] = head[src_y0:src_y1, src_x0:src_x1]

    return img
