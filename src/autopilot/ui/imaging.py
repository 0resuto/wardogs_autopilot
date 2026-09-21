"""Image -> Tk photo conversion helpers for the studio GUI."""

import base64
import tkinter as tk

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def to_photo(bgr: np.ndarray):
    """Convert a BGR frame to a Tk-compatible PhotoImage.

    Uses high-speed PIL ImageTk when available, falls back to uncompressed
    PPM stream, and finally base64 PNG as a safety net.
    """
    if bgr is None or bgr.size == 0:
        return tk.PhotoImage()

    h, w = bgr.shape[:2]
    if len(bgr.shape) == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if _HAS_PIL:
        try:
            return ImageTk.PhotoImage(Image.fromarray(rgb))
        except Exception:
            pass

    # Fast PPM P6 stream (supported natively by Tkinter PhotoImage)
    try:
        ppm = f"P6 {w} {h} 255\n".encode("ascii") + rgb.tobytes()
        return tk.PhotoImage(data=ppm)
    except Exception:
        pass

    # Safe fallback: PNG
    _ok, buf = cv2.imencode(".png", bgr)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))
