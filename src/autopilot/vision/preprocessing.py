"""Image preprocessing routines for minimap capture and full-map crops.

Provides mask creation, color-to-grayscale conversions, percentile normalization,
and mirrored window cropping.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .. import PROJECT_ROOT

ROOT = PROJECT_ROOT
MASK_PATH = os.path.join(PROJECT_ROOT, "data", "masks", "mm_mask.png")


def make_mask(path: str | Path = MASK_PATH) -> np.ndarray:
    """Load the circular minimap mask from disk as a boolean array."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"mask not read: {path}")
    return mask > 0


def norm8(g: np.ndarray) -> np.ndarray:
    """Stretch 8-bit dynamic range between 2nd and 98th percentiles."""
    lo, hi = float(np.percentile(g, 2)), float(np.percentile(g, 98))
    return np.clip((g - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)


def bgr_to_gray(bgr: np.ndarray, conv: str = "luma", gamma: float = 1.0) -> np.ndarray:
    """Convert BGR map image to the grayscale palette approximating the in-game minimap.

    The in-game minimap is near-monochrome but its color->gray mapping does
    NOT match OpenCV's BGR2GRAY (BT.601 luma): yellow patches come out too
    bright and mid grays too dark under luma. 'equal' (channel average) plus
    an optional gamma lift tracks the game much closer:
      luma   : B*0.114 + G*0.587 + R*0.299 (OpenCV default)
      equal  : (B + G + R) / 3
      bt709  : B*0.0722 + G*0.7152 + R*0.2126
    gamma != 1.0 applies g' = 255 * (g/255)^gamma (gamma < 1 brightens mids).
    """
    if conv == "equal":
        g = bgr.astype(np.float32).mean(axis=2)
    elif conv == "bt709":
        g = (
            bgr[..., 2].astype(np.float32) * 0.2126
            + bgr[..., 1].astype(np.float32) * 0.7152
            + bgr[..., 0].astype(np.float32) * 0.0722
        )
    else:  # luma
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gamma != 1.0:
        g = 255.0 * np.power(np.clip(g, 0, 255) / 255.0, gamma)
    return np.clip(g, 0, 255).astype(np.uint8)


def fill_norm(mm: np.ndarray, ui_mask: np.ndarray | None, per: tuple[float, float]) -> np.ndarray:
    """Fill UI pixels with median of background, then normalize on percentiles."""
    m = mm.astype(np.float32)
    if ui_mask is not None and ui_mask.size:
        m[ui_mask] = float(np.median(m[~ui_mask]))
    lo, hi = float(np.percentile(m, per[0])), float(np.percentile(m, per[1]))
    return np.clip((m - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)


def shadow_fill_norm(mm: np.ndarray, ui_mask: np.ndarray | None) -> np.ndarray:
    """UI pixels -> median of the rest of the frame, then normalize on (2, 98)."""
    return fill_norm(mm, ui_mask, (2, 98))


def crop_win(
    mu: np.ndarray, cy: float, cx: float, rh: int, rw: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a window around (cx, cy) on `mu` with mirrored reflection at map edges.

    Returns (win, (y0, x0)) — the window's top-left corner in MU coordinates.
    """
    h, w = mu.shape
    y0, y1 = int(round(cy)) - rh, int(round(cy)) + rh
    x0, x1 = int(round(cx)) - rw, int(round(cx)) + rw
    pt, pb = max(0, -y0), max(0, y1 - h)
    pl, pr = max(0, -x0), max(0, x1 - w)
    img = mu[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)]
    img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REFLECT)
    return img, (int(y0), int(x0))
