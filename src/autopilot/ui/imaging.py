"""Image conversion helpers for PySide6."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


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
