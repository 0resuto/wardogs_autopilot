"""Studio visual theme: "Cold Mirror" design kit for PySide6.

Tokens mirror the React UI kit (`design-kit`): bg #2b2d34, deep bg #1a1b1e,
surface #383a44, brand #e63946, text #eaeaea, plus the semantic telemetry
accents. Typography uses the vendored Exo 2 / Red Hat Mono fonts (falling back
to Segoe UI). Qt has no backdrop blur, so the glass surfaces are approximated
with translucent rgba fills and hairline borders - no proxy styles or bitmap
tricks.
"""

from __future__ import annotations

import os
from string import Template
from typing import Any

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QLabel

from .. import PROJECT_ROOT

# --- Palette (UI kit tokens) ---
BG = "#2b2d34"
BG_DEEP = "#1a1b1e"
SURFACE = "#383a44"
SURFACE_HI = "#4d505c"
BRAND = "#e63946"
TEXT = "#eaeaea"
TEXT_MUTED = "rgba(234,234,234,0.60)"
TEXT_DIM = "rgba(234,234,234,0.45)"
TEXT_FAINT = "rgba(234,234,234,0.28)"

BLUE = "#38BDF8"
GREEN = "#10B981"
RED = "#EF4444"
YELLOW = "#F59E0B"
PURPLE = "#A855F7"

# Glass surfaces (no real blur in Qt: translucent fills + hairlines)
GLASS = "rgba(56,58,68,0.40)"
GLASS_CARD = "rgba(56,58,68,0.30)"
GLASS_DEEP = "rgba(26,28,35,0.97)"
PANEL_BG = "rgba(26,27,30,0.85)"
CONTROL_BG = "rgba(56,58,68,0.40)"
BORDER = "rgba(234,234,234,0.10)"
BORDER_SOFT = "rgba(234,234,234,0.08)"
BORDER_STRONG = "rgba(234,234,234,0.18)"

ROUTE_GREEN = "#34D399"
RGB_CANVAS = (26, 27, 30)

_FONT_DIR = os.path.join(PROJECT_ROOT, "data", "fonts")
_FONT_FILES = ("Exo2.ttf", "RedHatMono.ttf")
_fonts_loaded = False


def _load_fonts() -> bool:
    """Register the vendored fonts; False without a live QApplication."""
    global _fonts_loaded
    if _fonts_loaded:
        return True
    if QApplication.instance() is None:
        return False  # QFontDatabase requires a QGuiApplication
    _fonts_loaded = True
    for fname in _FONT_FILES:
        path = os.path.join(_FONT_DIR, fname)
        if os.path.exists(path):
            QFontDatabase.addApplicationFont(path)
    return True


def ui_font_family() -> str:
    """Exo 2 when the vendored font is available, Segoe UI otherwise."""
    if not _load_fonts():
        return "Segoe UI"
    return "Exo 2" if "Exo 2" in set(QFontDatabase.families()) else "Segoe UI"


def mono_font_family() -> str:
    """Red Hat Mono for tabular numbers, with sane fallbacks."""
    if not _load_fonts():
        return "Consolas"
    families = set(QFontDatabase.families())
    for name in ("Red Hat Mono", "Consolas"):
        if name in families:
            return name
    return "monospace"


_QSS_TEMPLATE = Template(
    """
QWidget {
    color: $TEXT;
    font-family: "$FONT", "Segoe UI", sans-serif;
    font-size: 12px;
    selection-background-color: rgba(230,57,70,0.45);
    selection-color: #ffffff;
}

QMainWindow, QDialog, QMessageBox, QInputDialog, QWidget#CentralWidget {
    background-color: $BG_DEEP;
}

QToolTip {
    background-color: $GLASS_DEEP;
    color: $TEXT;
    border: 1px solid $BORDER;
    border-radius: 8px;
    padding: 6px 8px;
}

QTabWidget::pane {
    border: none;
    background: transparent;
}

QTabBar {
    qproperty-drawBase: 0;
}

QTabBar::tab {
    background-color: rgba(56,58,68,0.55);
    border: 1px solid $BORDER_SOFT;
    border-radius: 10px;
    padding: 6px 16px;
    margin-right: 6px;
    margin-bottom: 4px;
    color: $TEXT_MUTED;
    font-weight: 600;
}

QTabBar::tab:selected {
    background-color: rgba(230,57,70,0.20);
    border-color: rgba(230,57,70,0.55);
    color: $TEXT;
    font-weight: 700;
}

QTabBar::tab:hover:!selected {
    background-color: rgba(56,58,68,0.85);
    color: $TEXT;
}

QGroupBox {
    background-color: $GLASS_CARD;
    border: 1px solid $BORDER;
    border-radius: 12px;
    margin-top: 0px;
    padding: 28px 12px 10px 12px;
}

QGroupBox::title {
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 12px;
    top: 7px;
    padding: 0;
    color: $TEXT;
    font-weight: 700;
    background: transparent;
}

QPushButton {
    background-color: rgba(255,255,255,0.05);
    color: $TEXT;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 5px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: rgba(255,255,255,0.10);
    border-color: $BORDER_STRONG;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: rgba(255,255,255,0.14);
}

QPushButton:disabled {
    background-color: rgba(56,58,68,0.35);
    border-color: rgba(234,234,234,0.06);
    color: $TEXT_FAINT;
}

QPushButton#AccentButton {
    background-color: $BRAND;
    border: 1px solid rgba(230,57,70,0.55);
    color: $TEXT;
    font-weight: 700;
}

QPushButton#AccentButton:hover {
    background-color: #f04b57;
    color: #ffffff;
}

QPushButton#AccentButton:pressed {
    background-color: #c92c39;
}

QPushButton#AccentButton:disabled {
    background-color: rgba(230,57,70,0.35);
    border-color: rgba(230,57,70,0.25);
    color: rgba(234,234,234,0.50);
}

QPushButton#SuccessButton {
    background-color: rgba(16,185,129,0.18);
    border: 1px solid rgba(16,185,129,0.50);
    color: #a7f3d0;
    font-weight: 700;
    font-size: 13px;
    padding: 6px 16px;
}

QPushButton#SuccessButton:hover {
    background-color: rgba(16,185,129,0.30);
    color: #ffffff;
}

QPushButton#DangerButton {
    background-color: rgba(245,158,11,0.18);
    border: 1px solid rgba(245,158,11,0.50);
    color: #fcd34d;
    font-weight: 700;
    font-size: 13px;
    padding: 6px 16px;
}

QPushButton#DangerButton:hover {
    background-color: rgba(245,158,11,0.30);
    color: #ffffff;
}

QPushButton#EStopButton {
    background-color: rgba(239,68,68,0.20);
    border: 1px solid rgba(239,68,68,0.45);
    color: #fecaca;
    font-weight: 700;
    font-size: 12px;
    padding: 6px 14px;
}

QPushButton#EStopButton:hover {
    background-color: rgba(239,68,68,0.35);
    color: #ffffff;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: $CONTROL_BG;
    color: $TEXT;
    border: 1px solid rgba(234,234,234,0.15);
    border-radius: 10px;
    padding: 4px 6px;
    selection-background-color: rgba(230,57,70,0.45);
}

QLineEdit#TuneInput {
    padding: 3px 4px;
    border-radius: 8px;
}

QLineEdit:hover, QComboBox:hover {
    border-color: rgba(234,234,234,0.30);
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid $BRAND;
}

QComboBox::drop-down {
    border: none;
    width: 22px;
}

QComboBox QAbstractItemView {
    background-color: $GLASS_DEEP;
    border: 1px solid $BORDER_SOFT;
    border-radius: 8px;
    color: $TEXT;
    selection-background-color: rgba(230,57,70,0.25);
    outline: none;
}

QCheckBox {
    color: $TEXT;
    spacing: 8px;
    font-weight: 600;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 5px;
    background-color: rgba(56,58,68,0.80);
    border: 1px solid rgba(234,234,234,0.25);
}

QCheckBox::indicator:hover {
    border-color: rgba(230,57,70,0.60);
}

QCheckBox::indicator:checked {
    background-color: $BRAND;
    border-color: $BRAND;
}

QLabel {
    background: transparent;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: $SURFACE;
    border-radius: 3px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: $BRAND;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: $SURFACE;
    border-radius: 3px;
    min-width: 24px;
}

QScrollBar::handle:horizontal:hover {
    background: $BRAND;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QWidget#MapHud {
    background-color: $GLASS_CARD;
    border: 1px solid $BORDER_SOFT;
    border-radius: 10px;
}

QLabel#StatusBadge {
    background-color: rgba(56,58,68,0.60);
    border: 1px solid $BORDER;
    border-radius: 10px;
    padding: 2px 10px;
    color: $TEXT_MUTED;
    font-weight: 700;
    font-size: 11px;
}

QLabel#StatusBadge[state="green"] {
    background-color: rgba(16,185,129,0.18);
    border-color: rgba(16,185,129,0.50);
    color: #a7f3d0;
}

QLabel#StatusBadge[state="yellow"] {
    background-color: rgba(245,158,11,0.18);
    border-color: rgba(245,158,11,0.50);
    color: #fcd34d;
}

QLabel#StatusBadge[state="blue"] {
    background-color: rgba(56,189,248,0.18);
    border-color: rgba(56,189,248,0.50);
    color: #bae6fd;
}

QLabel#StatusBadge[state="red"] {
    background-color: rgba(239,68,68,0.20);
    border-color: rgba(239,68,68,0.45);
    color: #fecaca;
}

QMenu {
    background-color: $GLASS_DEEP;
    border: 1px solid $BORDER_SOFT;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    padding: 5px 18px;
    border-radius: 6px;
}

QMenu::item:selected {
    background-color: rgba(230,57,70,0.25);
}
"""
)


def build_qss() -> str:
    """Resolve the stylesheet template with the live font family and tokens."""
    return _QSS_TEMPLATE.substitute(
        FONT=ui_font_family(),
        BG=BG,
        BG_DEEP=BG_DEEP,
        SURFACE=SURFACE,
        BRAND=BRAND,
        TEXT=TEXT,
        TEXT_MUTED=TEXT_MUTED,
        TEXT_DIM=TEXT_DIM,
        TEXT_FAINT=TEXT_FAINT,
        GLASS=GLASS,
        GLASS_CARD=GLASS_CARD,
        GLASS_DEEP=GLASS_DEEP,
        CONTROL_BG=CONTROL_BG,
        BORDER=BORDER,
        BORDER_SOFT=BORDER_SOFT,
        BORDER_STRONG=BORDER_STRONG,
    )


def set_status_badge(label: QLabel, state: str, text: str | None = None) -> None:
    """Style a QLabel as a kit-style pill badge.

    `state` is one of 'green'/'yellow'/'blue'/'red' (semantic accents) or
    'off' for the neutral grey pill.
    """
    if text is not None:
        label.setText(text)
    if label.objectName() != "StatusBadge":
        label.setObjectName("StatusBadge")
    label.setProperty("state", state)
    label.style().unpolish(label)
    label.style().polish(label)


def windows_dark_mode() -> bool:
    """Windows app dark theme (AppsUseLightTheme == 0)."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        return True  # Default to dark


def apply_theme(app: QApplication, dark: bool = True) -> None:
    """Apply the Cold Mirror dark theme to the QApplication."""
    app.setFont(QFont(ui_font_family(), 9))
    if not dark:
        app.setStyleSheet("")
        return
    app.setStyleSheet(build_qss())
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG_DEEP))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_DEEP))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_DEEP))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(BRAND))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)


class ThemeManager:
    """ThemeManager compatibility helper for live OS theme tracking."""

    def __init__(self, root: Any = None, roi_status: Any = None) -> None:
        self._root = root
        self._roi_status = roi_status
        self._dark = True

    def apply(self) -> None:
        pass

    def start_poll(self) -> None:
        pass
