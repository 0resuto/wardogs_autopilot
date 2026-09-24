"""Image conversion helpers for PySide6 and backwards compatibility."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

try:
    from PIL import Image, ImageTk

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def to_qimage(img: np.ndarray | None) -> QImage:
    """Convert an OpenCV/numpy BGR, Grayscale, or BGRA array to a PySide6 QImage."""
    if img is None or img.size == 0:
        return QImage()

    h, w = img.shape[:2]
    if len(img.shape) == 2:
        return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8).copy()
    if img.shape[2] == 3:
        return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888).copy()
    if img.shape[2] == 4:
        return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_ARGB32).copy()
    return QImage()


def to_qpixmap(img: np.ndarray | None) -> QPixmap:
    """Convert an OpenCV/numpy array directly to a PySide6 QPixmap."""
    qimg = to_qimage(img)
    if qimg.isNull():
        return QPixmap()
    return QPixmap.fromImage(qimg)


def to_photo(bgr: np.ndarray | None) -> Any:
    """Legacy Tk PhotoImage helper maintained for backward compatibility with unit tests."""
    if bgr is None or bgr.size == 0:
        return None

    try:
        import tkinter as tk
    except ImportError:
        return None

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

    try:
        ppm = f"P6 {w} {h} 255\n".encode("ascii") + rgb.tobytes()
        return tk.PhotoImage(data=ppm)
    except Exception:
        pass

    _ok, buf = cv2.imencode(".png", bgr)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))
