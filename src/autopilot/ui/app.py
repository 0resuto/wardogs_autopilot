"""Studio: Visual tuning and control dashboard for WARDOGS autopilot (PySide6).

Coordinates the main window, map selector toolbar, global hotkeys,
and the three primary tabs: Capture Zone (ROI), Map Diagnostics, and Routes.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from serial.tools import list_ports

from ..common.config import (
    AppConfig,
    CaptureConfig,
    DebugConfig,
    LocatorConfig,
    MapConfig,
    NavigatorConfig,
    UiConfig,
    atomic_write_json,
)
from ..common.log import get_logger
from ..hardware.screen_capture import ScreenCapture
from ..vision import locator
from ..vision.tracker import LiveLocator
from .hotkeys import _HK_F6, _HK_F7, _HK_F8, HotkeyManager
from .tabs.map_tab import MapTab
from .tabs.roi_tab import RoiTab
from .theme import (
    RED,
    TEXT_MUTED,
    apply_theme,
    mono_font_family,
    set_status_badge,
)

logger = get_logger("ui.app")

_HOTKEY_HINT = "[F6] Follow   [F7] E-Stop   [F8] Reverse"


def clamp_rect_to_screens(rect: QRect, app: QApplication) -> QRect:
    """Keep a remembered window rect reachable on the current monitors.

    A window restored to a monitor that is gone would be invisible; if the
    saved rect (including its title bar) is not on any screen, it is centered
    on the primary screen at a clamped size.
    """
    screens = app.screens()
    for screen in screens:
        wa = screen.availableGeometry()
        visible = wa.intersected(rect)
        if visible.width() >= 150 and visible.height() >= 80 and visible.top() <= rect.top() + 40:
            return rect
    if not screens:
        return rect
    wa = app.primaryScreen().availableGeometry()
    w = max(400, min(rect.width(), wa.width()))
    h = max(300, min(rect.height(), wa.height()))
    return QRect(
        wa.x() + (wa.width() - w) // 2,
        wa.y() + (wa.height() - h) // 2,
        w,
        h,
    )


class App(QMainWindow):
    """Main dashboard application window coordinating UI components and services."""

    # Thread-safe Qt signal for background map loading
    sig_map_loaded = Signal(str, object, int)  # (map name, preview pyramid, size)

    def __init__(self, cfg: dict[str, Any] | AppConfig) -> None:
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            app = QApplication(sys.argv)
        self._app_instance = app
        apply_theme(app, dark=True)

        super().__init__()
        if isinstance(cfg, AppConfig):
            self.app_cfg = cfg
            self.cfg = cfg.to_dict()
        else:
            self.cfg = cfg
            try:
                self.app_cfg = AppConfig(**cfg)
            except Exception:
                self.app_cfg = AppConfig()

        self.setWindowTitle("WARDOGS minimap studio")
        self._restore_window_geometry()

        self._map_store = locator.get_store()
        self._map_store.set_config_path(self.app_cfg.cfg_path)
        self._map_name = self._cfg_map_name()
        locator.set_map(self._map_name)

        sz = locator.full_map_size(self._map_name)
        self._map_size = sz[0] if isinstance(sz, (tuple, list)) else (sz or 32768)
        self._thumb = 8

        cap_cfg = self.app_cfg.capture
        self._cap = ScreenCapture(monitor=cap_cfg.monitor, roi=cap_cfg.mmap_roi)
        self._mask = locator.make_mask()
        self._loc_thread = LiveLocator(cfg=self.app_cfg, mask=self._mask)
        self._loc_thread.start()

        self._hotkeys = HotkeyManager(self, self._on_global_hotkey)
        self._hotkeys.start()

        self._map_pyr: dict[int, np.ndarray] | None = None
        self._last_loc: dict[str, Any] | None = None
        self._last_state_sig: tuple[bool, str] | None = None

        self.sig_map_loaded.connect(self._on_map_loaded_ui)
        self._build_ui()

        # Load map in background thread
        threading.Thread(target=self._load_map_worker, args=(self._map_name,), daemon=True).start()

        # Main thread 20 Hz (50 ms) poll timer
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(50)

    def _restore_window_geometry(self) -> None:
        """Restore the last session's window rect, clamped to current screens."""
        ui = self.app_cfg.ui
        if ui.window_w > 0 and ui.window_h > 0:
            rect = clamp_rect_to_screens(
                QRect(ui.window_x, ui.window_y, ui.window_w, ui.window_h),
                self._app_instance,
            )
            self.setGeometry(rect)
            if ui.window_max:
                self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
            return
        self.resize(1100, 850)

    def _remember_window_geometry(self) -> None:
        """Persist the window rect (normal geometry when maximized)."""
        ui = self.app_cfg.ui
        rect = self.normalGeometry() if self.isMaximized() else self.geometry()
        ui.window_x, ui.window_y = rect.x(), rect.y()
        ui.window_w, ui.window_h = rect.width(), rect.height()
        ui.window_max = self.isMaximized()

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        # Top toolbar
        toolbar = QWidget(central)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(4, 2, 4, 2)
        tb_layout.setSpacing(8)

        # Left: Map selector
        lbl_map = QLabel("Map:", toolbar)
        lbl_map.setStyleSheet("font-weight: bold;")
        tb_layout.addWidget(lbl_map)

        self._map_sel = QComboBox(toolbar)
        self._map_sel.setFixedWidth(130)
        self._map_sel.addItems(locator.available_maps())
        self._map_sel.setCurrentText(self._map_name)
        self._map_sel.currentTextChanged.connect(self._on_map_changed)
        tb_layout.addWidget(self._map_sel)

        self._map_size_lbl = QLabel(f"{self._map_size}x{self._map_size}", toolbar)
        self._map_size_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8pt;")
        tb_layout.addWidget(self._map_size_lbl)

        tb_layout.addStretch()

        # Right: hardware, live status badges & hotkeys
        self._status_hw = QLabel("", toolbar)
        set_status_badge(self._status_hw, "off")
        self._status_hw.setToolTip("Arduino serial port (navigator.port)")
        tb_layout.addWidget(self._status_hw)

        self._status_loc = QLabel("", toolbar)
        set_status_badge(self._status_loc, "yellow", "○ SEARCHING")
        self._status_loc.setToolTip("Localization: SIFT index matcher")
        tb_layout.addWidget(self._status_loc)

        self._status_lat = QLabel("", toolbar)
        self._status_lat.setStyleSheet(
            f"color: {TEXT_MUTED}; font-family: '{mono_font_family()}'; "
            "font-size: 11px; min-width: 48px;"
        )
        tb_layout.addWidget(self._status_lat)

        self._status_nav = QLabel("", toolbar)
        set_status_badge(self._status_nav, "off", "○ IDLE")
        self._status_nav.setToolTip("Autopilot driver state")
        tb_layout.addWidget(self._status_nav)

        self._hotkey_lbl = QLabel(_HOTKEY_HINT, toolbar)
        self._hotkey_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9pt;")
        tb_layout.addWidget(self._hotkey_lbl)
        self._hotkeys.registration_changed.connect(self._on_hotkey_state)
        self._on_hotkey_state(self._hotkeys.is_ready())
        self._hw_check_at = 0.0
        self._update_hw_status()

        root_layout.addWidget(toolbar)

        # Tabs
        self.nb = QTabWidget(central)
        self.nb.currentChanged.connect(self._on_tab_changed)
        root_layout.addWidget(self.nb, stretch=1)

        self.roi_tab = RoiTab(
            self.nb,
            cfg=self.cfg,
            save_cfg_fn=self._save_cfg,
            screen_cap_supplier=lambda: self._cap,
            loc_thread_supplier=lambda: self._loc_thread,
            on_map_rebuilt=self._on_map_rebuilt,
        )
        self.nb.addTab(self.roi_tab, "Capture zone")

        self.map_tab = MapTab(
            self.nb,
            on_open_capture=lambda: self.nb.setCurrentIndex(0),
            cfg=self.cfg,
            save_cfg_fn=self._save_cfg,
            loc_thread_supplier=lambda: self._loc_thread,
            map_name_supplier=lambda: self._map_name,
            map_store_supplier=lambda: self._map_store,
            on_pick_roi=self.roi_tab.pick_roi,
            app_cfg=self.app_cfg,
        )
        self.routes_tab = self.map_tab  # Backward-compatibility alias
        self.nb.addTab(self.map_tab, "Map")
        # First run without map assets: land on the Capture tab where the
        # download action lives instead of showing an empty canvas.
        self.nb.setCurrentIndex(1 if self.map_tab.map_asset_state() == "ok" else 0)

    def _cfg_map_name(self) -> str:
        m = self.cfg.get("map")
        if isinstance(m, dict) and m.get("name"):
            return str(m["name"])
        return "zestafona"

    def _on_map_changed(self, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name or new_name == self._map_name:
            return
        self._apply_map(new_name)

    def _apply_map(self, name: str) -> None:
        self._map_name = name
        locator.set_map(name)
        self.cfg.setdefault("map", {})["name"] = name
        self._save_cfg()
        sz = locator.full_map_size(name)
        self._map_size = sz[0] if isinstance(sz, (tuple, list)) else (sz or 32768)
        self._map_size_lbl.setText(f"{self._map_size}x{self._map_size}")
        self.map_tab.map_name = name
        self.map_tab.preset_reload()
        self.map_tab.map_loading()
        threading.Thread(target=self._load_map_worker, args=(name,), daemon=True).start()

    def _on_map_rebuilt(self, name: str) -> None:
        if name == self._map_name:
            threading.Thread(target=self._load_map_worker, args=(name,), daemon=True).start()

    def _load_map_worker(self, name: str) -> None:
        try:
            sz = locator.full_map_size(name)
            map_size = sz[0] if isinstance(sz, (tuple, list)) else (sz or 32768)
            locator.ensure_previews()  # derive the smaller levels from the shipped top
            pyr = locator.load_previews()
            self.sig_map_loaded.emit(name, pyr, map_size)
        except Exception as exc:
            logger.error("Failed to load map '%s': %s", name, exc)
            self.sig_map_loaded.emit(name, None, 32768)

    def _on_map_loaded_ui(self, name: str, pyr: Any, map_size: int) -> None:
        if name != self._map_name:
            return
        self._map_pyr = pyr
        self._map_size = map_size

        self.map_tab.map_widget.scene.set_map(None, pyr, map_size=map_size, thumb=self._thumb)
        self.map_tab.map_widget.view.fit_view()
        self._map_size_lbl.setText(f"{map_size}x{map_size}")
        self.map_tab.map_loaded()

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            QTimer.singleShot(50, self.map_tab.map_widget.view.fit_view)

    def _update_hw_status(self) -> None:
        """Top-bar indicator: is the configured Arduino serial port present?"""
        port = str(getattr(self.app_cfg.navigator, "port", "") or "")
        present = False
        try:
            present = any(p.device.upper() == port.upper() for p in list_ports.comports())
        except Exception:
            present = False
        if present:
            set_status_badge(self._status_hw, "green", f"● Arduino {port}")
            self._status_hw.setToolTip(f"Arduino detected on {port}")
        else:
            set_status_badge(self._status_hw, "off", f"○ Arduino {port}")
            self._status_hw.setToolTip(
                f"{port} not found — the autopilot reports an error when you start Follow"
            )

    def _poll(self) -> None:
        """Periodic UI update loop at 20 Hz (50 ms)."""
        try:
            self.roi_tab.update_preview()
            latest = self._loc_thread.latest
            self._last_loc = latest
            self.map_tab.update_loc(latest)
            self.routes_tab.sync_driver_state()

            # Hardware presence is polled slowly (COM port enumeration)
            now = time.time()
            if now - self._hw_check_at >= 5.0:
                self._hw_check_at = now
                self._update_hw_status()

            # Update latency display next to status badge
            elapsed = latest.get("elapsed") if latest else None
            if elapsed is not None:
                self._status_lat.setText(f"{int(elapsed * 1000)} ms")
            else:
                self._status_lat.setText("")

            # Dirty checking for header badges (<0.1 us when unchanged)
            loc_active = bool(latest and latest.get("pose"))
            driver = self.routes_tab.driver
            nav_state = driver.state if driver is not None else "idle"
            state_sig = (loc_active, nav_state)
            if state_sig != self._last_state_sig:
                self._last_state_sig = state_sig
                if loc_active:
                    pose = latest.get("pose") if latest else None
                    inl = int(pose.get("inl", 0)) if pose else 0
                    set_status_badge(self._status_loc, "green", "● LIVE")
                    self._status_loc.setToolTip(f"Localization live: inl={inl}")
                else:
                    set_status_badge(self._status_loc, "yellow", "○ SEARCHING")
                    self._status_loc.setToolTip("Localization: searching for a pose")

                if nav_state == "idle":
                    set_status_badge(self._status_nav, "off", "○ IDLE")
                elif nav_state == "following":
                    set_status_badge(self._status_nav, "green", "● FOLLOWING")
                elif nav_state == "finished":
                    set_status_badge(self._status_nav, "blue", "✓ FINISHED")
                elif nav_state == "stopped":
                    set_status_badge(self._status_nav, "red", "🛑 STOPPED")
                else:
                    set_status_badge(self._status_nav, "blue", f"● {nav_state.upper()}")
        except Exception as exc:
            logger.debug("Poll exception: %s", exc)

    def _on_hotkey_state(self, ready: bool) -> None:
        if ready:
            self._hotkey_lbl.setText(_HOTKEY_HINT)
            self._hotkey_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9pt;")
        else:
            self._hotkey_lbl.setText("⚠ hotkeys F6/F7/F8 busy - retrying")
            self._hotkey_lbl.setStyleSheet(f"color: {RED}; font-size: 9pt;")

    def _on_global_hotkey(self, key_id: int) -> None:
        if key_id == _HK_F6:
            self.routes_tab.follow_toggle(silent=True)
        elif key_id == _HK_F7:
            self.routes_tab.emergency_stop()
        elif key_id == _HK_F8:
            self.routes_tab.routes_invert(silent=True)

    def _save_cfg(self) -> None:
        target = self.app_cfg.cfg_path
        try:
            app_cfg = AppConfig.load(target)
            if "capture" in self.cfg and isinstance(self.cfg["capture"], dict):
                app_cfg.capture = CaptureConfig(**self.cfg["capture"])
            if "locator" in self.cfg and isinstance(self.cfg["locator"], dict):
                app_cfg.locator = LocatorConfig(**self.cfg["locator"])
            if "map" in self.cfg and isinstance(self.cfg["map"], dict):
                app_cfg.map = MapConfig(**self.cfg["map"])
            if "navigator" in self.cfg and isinstance(self.cfg["navigator"], dict):
                app_cfg.navigator = NavigatorConfig(**self.cfg["navigator"])
            if "debug" in self.cfg and isinstance(self.cfg["debug"], dict):
                app_cfg.debug = DebugConfig(**self.cfg["debug"])
            app_cfg.ui = UiConfig(**self.app_cfg.ui.model_dump())
            app_cfg.save(target)
        except Exception:
            atomic_write_json(target, self.cfg)

    def closeEvent(self, event: Any) -> None:
        self._remember_window_geometry()
        self._save_cfg()
        self._hotkeys.stop()
        self.routes_tab.stop_manual_record()
        self.routes_tab.emergency_stop()
        self._loc_thread.stop()
        if hasattr(self, "_cap") and hasattr(self._cap, "close"):
            self._cap.close()
        event.accept()


def main(cfg: dict[str, Any] | AppConfig | None = None) -> int:
    """Launch the WARDOGS minimap studio GUI."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    if cfg is None:
        cfg = AppConfig.load("config.json")

    window = App(cfg)
    window.show()
    return app.exec()
