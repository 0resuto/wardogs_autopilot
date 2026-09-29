"""Monochrome stroke icons drawn from inline Lucide-style SVG (kit parity).

The design kit uses Lucide icons; PySide6 ships QtSvg, so the same paths are
rendered at runtime in theme colors instead of shipping emoji or bitmap
glyphs. Icons get Normal/Active/Disabled pixmaps, so hover/disabled states
follow the palette automatically.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .theme import TEXT

_BODY = {
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "pause": (
        '<rect x="6" y="4" width="4" height="16" rx="1"/>'
        '<rect x="14" y="4" width="4" height="16" rx="1"/>'
    ),
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "trash": (
        '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>'
        '<path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>'
    ),
    "pencil": (
        '<path d="M21.2 6.8a2 2 0 0 0-4-4L3.8 16.2a2 2 0 0 0-.5.8l-1.3 4.4a.5.5 0 0 0 .6.6'
        'l4.4-1.3a2 2 0 0 0 .8-.5z"/><path d="m15 5 4 4"/>'
    ),
    "swap": (
        '<path d="M8 3 4 7l4 4"/><path d="M4 7h16"/><path d="m16 21 4-4-4-4"/><path d="M20 17H4"/>'
    ),
    "sliders": (
        '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/>'
        '<line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/>'
        '<line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/>'
        '<line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/>'
        '<line x1="16" x2="16" y1="18" y2="22"/>'
    ),
    "folder": (
        '<path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1'
        "-1.94 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 "
        '1.66.9H18a2 2 0 0 1 2 2v2"/>'
    ),
    "camera": (
        '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0'
        '-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>'
    ),
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "maximize": (
        '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>'
        '<path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>'
    ),
    "crosshair": (
        '<circle cx="12" cy="12" r="9"/>'
        '<line x1="21" x2="17" y1="12" y2="12"/><line x1="7" x2="3" y1="12" y2="12"/>'
        '<line x1="12" x2="12" y1="7" y2="3"/><line x1="12" x2="12" y1="21" y2="17"/>'
    ),
    "move": (
        '<polyline points="5 9 2 12 5 15"/><polyline points="9 5 12 2 15 5"/>'
        '<polyline points="15 19 12 22 9 19"/><polyline points="19 9 22 12 19 15"/>'
        '<line x1="2" x2="22" y1="12" y2="12"/><line x1="12" x2="12" y1="2" y2="22"/>'
    ),
    "disc": (
        '<circle cx="12" cy="12" r="9"/>'
        '<circle cx="12" cy="12" r="2.5" fill="{color}" stroke="none"/>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>'
    ),
    "refresh": (
        '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
        '<path d="M21 3v5h-5"/>'
        '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
        '<path d="M8 16H3v5"/>'
    ),
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
}

_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    "{body}</svg>"
)

_ACTIVE = "#ffffff"
_DISABLED = "#6b6d78"
_cache: dict[tuple[str, str, int], QIcon] = {}


def _pixmap(name: str, color: str, size: int) -> QPixmap:
    body = _BODY[name].replace("{color}", color)
    svg = _TEMPLATE.format(color=color, body=body)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return pix


def icon(name: str, color: str = TEXT, size: int = 16) -> QIcon:
    """QIcon with Normal/Active/Disabled pixmaps in the theme palette."""
    key = (name, color, size)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    qicon = QIcon()
    qicon.addPixmap(_pixmap(name, color, size), QIcon.Mode.Normal)
    qicon.addPixmap(_pixmap(name, _ACTIVE, size), QIcon.Mode.Active)
    qicon.addPixmap(_pixmap(name, _DISABLED, size), QIcon.Mode.Disabled)
    _cache[key] = qicon
    return qicon
