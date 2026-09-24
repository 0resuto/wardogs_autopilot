"""Studio visual theme: Fluent Dark / Modern Windows stylesheet for PySide6.

The palette constants live here so all widgets and canvas/scene components
reference a single source of truth.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

DARK_BG = "#1e1e1e"
DARK_PANEL = "#252526"
DARK_HI = "#383838"
DARK_FG = "#e2e2e2"
DARK_CANVAS = "#1e1e1e"
RGB_CANVAS = (30, 30, 30)

COLOR_ACCENT = "#0078d4"
COLOR_ACCENT_HOVER = "#106ebe"
COLOR_SUCCESS = "#107c41"
COLOR_SUCCESS_HOVER = "#0e6032"
COLOR_DANGER = "#d83b01"
COLOR_DANGER_HOVER = "#a80000"

DARK_QSS = """
QMainWindow, QWidget#CentralWidget {
    background-color: #1e1e1e;
    color: #e2e2e2;
    font-family: "Segoe UI", sans-serif;
    font-size: 9pt;
}

QTabWidget::pane {
    border: 1px solid #333333;
    background-color: #1e1e1e;
    top: -1px;
}

QTabBar::tab {
    background-color: #252526;
    color: #b0b0b0;
    padding: 8px 18px;
    font-weight: bold;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-bottom: 2px solid #0078d4;
}

QTabBar::tab:hover:!selected {
    background-color: #2d2d30;
    color: #e0e0e0;
}

QGroupBox {
    border: 1px solid #383838;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #88c0d0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}

QPushButton {
    background-color: #383838;
    color: #e2e2e2;
    border: 1px solid #484848;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #484848;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #2a2a2a;
}

QPushButton:disabled {
    background-color: #252526;
    color: #606060;
    border-color: #333333;
}

QPushButton#AccentButton {
    background-color: #0078d4;
    color: #ffffff;
    border: 1px solid #0078d4;
    font-weight: bold;
}

QPushButton#AccentButton:hover {
    background-color: #106ebe;
}

QPushButton#AccentButton:pressed {
    background-color: #004e8c;
}

QPushButton#SuccessButton {
    background-color: #107c41;
    color: #ffffff;
    border: 1px solid #107c41;
    font-weight: bold;
    font-size: 10pt;
    padding: 6px 16px;
}

QPushButton#SuccessButton:hover {
    background-color: #0e6032;
}

QPushButton#DangerButton {
    background-color: #d83b01;
    color: #ffffff;
    border: 1px solid #d83b01;
    font-weight: bold;
    font-size: 10pt;
    padding: 6px 16px;
}

QPushButton#DangerButton:hover {
    background-color: #a80000;
}

QPushButton#EStopButton {
    background-color: #a80000;
    color: #ffffff;
    border: 1px solid #a80000;
    font-weight: bold;
    font-size: 9pt;
    padding: 6px 14px;
}

QPushButton#EStopButton:hover {
    background-color: #850000;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #252526;
    color: #e2e2e2;
    border: 1px solid #383838;
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: #0078d4;
}

QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #0078d4;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #252526;
    color: #e2e2e2;
    border: 1px solid #383838;
    selection-background-color: #0078d4;
}

QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #383838;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #484848;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #1e1e1e;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #383838;
    min-width: 20px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #484848;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""


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
    """Apply modern Fluent Dark or Light theme to the QApplication."""
    app.setFont(QFont("Segoe UI", 9))
    if dark:
        app.setStyleSheet(DARK_QSS)
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(DARK_BG))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(DARK_FG))
        palette.setColor(QPalette.ColorRole.Base, QColor(DARK_PANEL))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(DARK_BG))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(DARK_PANEL))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(DARK_FG))
        palette.setColor(QPalette.ColorRole.Text, QColor(DARK_FG))
        palette.setColor(QPalette.ColorRole.Button, QColor(DARK_HI))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(DARK_FG))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_ACCENT))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        app.setPalette(palette)
    else:
        app.setStyleSheet("")


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
