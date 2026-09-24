"""Studio: Visual tuning and control dashboard for WARDOGS autopilot (PySide6).

Coordinates the main window, map selector toolbar, global hotkeys,
and the three primary tabs: Capture Zone (ROI), Map Diagnostics, and Routes.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer, Signal
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

from .. import PROJECT_ROOT
from ..common.config import AppConfig
from ..common.log import get_logger
from ..hardware.screen_capture import ScreenCapture
from ..vision import locator
from ..vision.tracker import LiveLocator
from .hotkeys import _HK_F6, _HK_F7, HotkeyManager
from .map_renderer import crop_map_viewport
from .tabs.map_tab import MapTab
from .tabs.roi_tab import RoiTab
from .tabs.routes_tab import RoutesTab
from .theme import RGB_CANVAS, apply_theme

logger = get_logger("ui.app")


class App(QMainWindow):
    """Main dashboard application window coordinating UI components and services."""

    # Thread-safe Qt signal for background map loading
    sig_map_loaded = Signal(str, object, object, int)

    def __init__(self, cfg: dict[str, Any] | AppConfig) -> None:
        self._app_instance = QApplication.instance()
        if self._app_instance is None:
            self._app_instance = QApplication(sys.argv)
        apply_theme(self._app_instance, dark=True)

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
        self.resize(1100, 850)

        self._map_store = locator.get_store()
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

        self._map8: np.ndarray | None = None
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
        self._map_size_lbl.setStyleSheet("color: #808080; font-size: 8pt;")
        tb_layout.addWidget(self._map_size_lbl)

        tb_layout.addStretch()

        # Right: Live status indicators & Hotkeys
        self._status_loc = QLabel("○ SEARCHING", toolbar)
        self._status_loc.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 9pt;")
        tb_layout.addWidget(self._status_loc)

        self._status_lat = QLabel("", toolbar)
        self._status_lat.setStyleSheet("color: #88c0d0; font-size: 8pt; min-width: 48px;")
        tb_layout.addWidget(self._status_lat)

        self._status_nav = QLabel("○ IDLE", toolbar)
        self._status_nav.setStyleSheet("color: #808080; font-weight: bold; font-size: 9pt;")
        tb_layout.addWidget(self._status_nav)

        lbl_hk = QLabel("[F6] Follow   [F7] E-Stop", toolbar)
        lbl_hk.setStyleSheet("color: #a0a0a0; font-size: 8pt;")
        tb_layout.addWidget(lbl_hk)

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
            cfg=self.cfg,
            save_cfg_fn=self._save_cfg,
            loc_thread_supplier=lambda: self._loc_thread,
            on_pick_roi=self.roi_tab.pick_roi,
        )
        self.nb.addTab(self.map_tab, "Map")

        self.routes_tab = RoutesTab(
            self.nb,
            app_cfg=self.app_cfg,
            loc_thread_supplier=lambda: self._loc_thread,
            map_name_supplier=lambda: self._map_name,
            map_store_supplier=lambda: self._map_store,
        )
        self.nb.addTab(self.routes_tab, "Routes")

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
        self.routes_tab.preset_reload()
        threading.Thread(target=self._load_map_worker, args=(name,), daemon=True).start()

    def _on_map_rebuilt(self, name: str) -> None:
        if name == self._map_name:
            threading.Thread(target=self._load_map_worker, args=(name,), daemon=True).start()

    def _load_map_worker(self, name: str) -> None:
        try:
            sz = locator.full_map_size(name)
            map_size = sz[0] if isinstance(sz, (tuple, list)) else (sz or 32768)
            map8 = locator.color_map()
            pyr = locator.load_previews()
            self.sig_map_loaded.emit(name, map8, pyr, map_size)
        except Exception as exc:
            logger.error("Failed to load map '%s': %s", name, exc)

    def _on_map_loaded_ui(self, name: str, map8: Any, pyr: Any, map_size: int) -> None:
        if name != self._map_name:
            return
        self._map8 = map8
        self._map_pyr = pyr
        self._map_size = map_size

        self.map_tab.map_widget.scene.set_map(map8, pyr, map_size=map_size, thumb=self._thumb)
        self.routes_tab.map_widget.scene.set_map(map8, pyr, map_size=map_size, thumb=self._thumb)
        self.map_tab.map_widget.view.fit_view()
        self.routes_tab.map_widget.view.fit_view()
        self._map_size_lbl.setText(f"{map_size}x{map_size}")

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            QTimer.singleShot(50, self.map_tab.map_widget.view.fit_view)
        elif index == 2:
            QTimer.singleShot(50, self.routes_tab.map_widget.view.fit_view)

    def _poll(self) -> None:
        """Periodic UI update loop at 20 Hz (50 ms)."""
        try:
            self.roi_tab.update_preview()
            latest = self._loc_thread.latest
            self._last_loc = latest
            self.map_tab.update_loc(latest)
            self.routes_tab.sync_driver_state()

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
                    self._status_loc.setText("● LIVE")
                    self._status_loc.setStyleSheet("color: #8ae234; font-weight: bold; font-size: 9pt;")
                else:
                    self._status_loc.setText("○ SEARCHING")
                    self._status_loc.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 9pt;")

                if nav_state == "idle":
                    self._status_nav.setText("○ IDLE")
                    self._status_nav.setStyleSheet("color: #808080; font-weight: bold; font-size: 9pt;")
                elif nav_state == "following":
                    self._status_nav.setText("● FOLLOWING")
                    self._status_nav.setStyleSheet("color: #8ae234; font-weight: bold; font-size: 9pt;")
                elif nav_state == "finished":
                    self._status_nav.setText("✓ FINISHED")
                    self._status_nav.setStyleSheet("color: #88c0d0; font-weight: bold; font-size: 9pt;")
                elif nav_state == "stopped":
                    self._status_nav.setText("🛑 STOPPED")
                    self._status_nav.setStyleSheet("color: #ff3b3b; font-weight: bold; font-size: 9pt;")
                else:
                    self._status_nav.setText(f"● {nav_state.upper()}")
                    self._status_nav.setStyleSheet("color: #88c0d0; font-weight: bold; font-size: 9pt;")
        except Exception as exc:
            logger.debug("Poll exception: %s", exc)

    def _on_global_hotkey(self, key_id: int) -> None:
        if key_id == _HK_F6:
            self.routes_tab.follow_toggle(silent=True)
        elif key_id == _HK_F7:
            self.routes_tab.emergency_stop()

    def _save_cfg(self) -> None:
        try:
            from ..common.config import (
                AppConfig,
                CaptureConfig,
                DebugConfig,
                LocatorConfig,
                MapConfig,
                NavigatorConfig,
            )

            app_cfg = AppConfig.load("config.json")
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
            app_cfg.save("config.json")
        except Exception:
            with open(os.path.join(PROJECT_ROOT, "config.json"), "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)

    def closeEvent(self, event: Any) -> None:
        self._hotkeys.stop()
        self.routes_tab.emergency_stop()
        self._loc_thread.stop()
        if hasattr(self, "_cap") and hasattr(self._cap, "close"):
            self._cap.close()
        event.accept()

    # ---------- Backward compatibility proxies for unit tests ----------
    def withdraw(self) -> None:
        self.hide()

    def destroy(self) -> None:
        self._hotkeys.stop()
        self._loc_thread.stop()
        if hasattr(self, "_cap") and hasattr(self._cap, "close"):
            self._cap.close()
        self.close()

    def update(self) -> None:
        QApplication.processEvents()

    @property
    def _roi_vars(self) -> dict[str, Any]:
        return self.roi_tab.roi_vars

    @property
    def _roi_status(self) -> Any:
        class _StatusProxy:
            def __init__(self, lbl: QLabel) -> None:
                self._lbl = lbl

            def cget(self, prop: str) -> str:
                return self._lbl.text()

        return _StatusProxy(self.roi_tab.status_lbl)

    def _apply_roi(self) -> None:
        self.roi_tab.apply_roi()

    def _pick_roi(self) -> None:
        self.roi_tab.pick_roi()

    def _open_mask_folder(self) -> None:
        self.roi_tab.open_mask_folder()

    @property
    def _tune_vars(self) -> dict[str, Any]:
        return self.map_tab.tune_vars

    @property
    def _tune_status(self) -> Any:
        class _StatusProxy:
            def __init__(self, lbl: QLabel) -> None:
                self._lbl = lbl

            def cget(self, prop: str) -> str:
                return self._lbl.text()

        return _StatusProxy(self.map_tab.tune_status)

    def _apply_tune(self) -> None:
        self.map_tab.apply_tune()

    def _preset_path(self, name: str) -> str:
        return self.routes_tab.preset_mgr.preset_path(name)

    def _preset_dir(self) -> str:
        return str(self.routes_tab.preset_mgr.presets_dir)

    def _bg_crop(self, s: float, ru: float, rv: float, rw: float, rh: float) -> np.ndarray:
        pyr = self._map_pyr or ({self._map8.shape[0]: self._map8} if self._map8 is not None else {})
        return crop_map_viewport(s, ru, rv, rw, rh, pyr, self._map_size, self._thumb, RGB_CANVAS)

    def _get_map_cache_status(self, name: str) -> tuple[str, str]:
        return self.roi_tab._get_map_cache_status(name)

    @property
    def _driver(self) -> Any:
        return self.routes_tab.driver

    @_driver.setter
    def _driver(self, val: Any) -> None:
        self.routes_tab.driver = val

    @property
    def _route_pts(self) -> list[list[float]]:
        return self.routes_tab.route_pts

    @_route_pts.setter
    def _route_pts(self, val: list[list[float]]) -> None:
        self.routes_tab.route_pts = val

    # Canvas shims for existing test suite
    @property
    def _map_canvas(self) -> Any:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        return self._compat_canvas

    def _init_compat_canvas(self) -> None:
        class _CompatCanvas:
            def __init__(self) -> None:
                self._items: list[dict[str, Any]] = []
                self._next_id = 1
                self.width = 800
                self.height = 600

            def winfo_width(self) -> int:
                return self.width

            def winfo_height(self) -> int:
                return self.height

            def create_line(self, *args: Any, **kwargs: Any) -> int:
                item_id = self._next_id
                self._next_id += 1
                tags = kwargs.get("tags", "").split() if isinstance(kwargs.get("tags"), str) else [kwargs.get("tags")]
                self._items.append({"id": item_id, "type": "line", "tags": tags})
                return item_id

            def create_oval(self, *args: Any, **kwargs: Any) -> int:
                item_id = self._next_id
                self._next_id += 1
                tags = kwargs.get("tags", "").split() if isinstance(kwargs.get("tags"), str) else [kwargs.get("tags")]
                self._items.append({"id": item_id, "type": "oval", "tags": tags})
                return item_id

            def create_rectangle(self, *args: Any, **kwargs: Any) -> int:
                item_id = self._next_id
                self._next_id += 1
                tags = kwargs.get("tags", "").split() if isinstance(kwargs.get("tags"), str) else [kwargs.get("tags")]
                self._items.append({"id": item_id, "type": "rect", "tags": tags})
                return item_id

            def create_image(self, *args: Any, **kwargs: Any) -> int:
                item_id = self._next_id
                self._next_id += 1
                tags = kwargs.get("tags", "").split() if isinstance(kwargs.get("tags"), str) else [kwargs.get("tags")]
                self._items.append({"id": item_id, "type": "image", "tags": tags})
                return item_id

            def coords(self, *args: Any, **kwargs: Any) -> None:
                pass

            def delete(self, tag_or_id: Any) -> None:
                self._items = [
                    it for it in self._items
                    if it["id"] != tag_or_id and tag_or_id not in it["tags"]
                ]

            def tag_lower(self, tag_or_id: Any) -> None:
                matching = [it for it in self._items if it["id"] == tag_or_id or tag_or_id in it["tags"]]
                non_matching = [it for it in self._items if it["id"] != tag_or_id and tag_or_id not in it["tags"]]
                self._items = matching + non_matching

            def find_withtag(self, tag_or_id: Any) -> list[int]:
                return [it["id"] for it in self._items if it["id"] == tag_or_id or tag_or_id in it["tags"]]

            def find_all(self) -> list[int]:
                return [it["id"] for it in self._items]

        self._compat_canvas = _CompatCanvas()
        self._compat_disp: tuple[float, float, float] | None = None
        self._compat_fit_scale: float = 0.0
        self._compat_pending: tuple[float, float, float] | None = None

    def _bg_update(self, name: str, s: float, ox: float, oy: float) -> None:
        c = self._map_canvas
        c.delete("bg")
        c.create_image(ox, oy, tags="bg")
        c.tag_lower("bg")

    @property
    def _map_disp(self) -> tuple[float, float, float] | None:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        return self._compat_disp

    @_map_disp.setter
    def _map_disp(self, val: tuple[float, float, float] | None) -> None:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        self._compat_disp = val

    @property
    def _map_fit_scale(self) -> float:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        return self._compat_fit_scale

    @_map_fit_scale.setter
    def _map_fit_scale(self, val: float) -> None:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        self._compat_fit_scale = val

    @property
    def _map_pending(self) -> tuple[float, float, float] | None:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        return self._compat_pending

    @_map_pending.setter
    def _map_pending(self, val: tuple[float, float, float] | None) -> None:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        self._compat_pending = val

    def _map_zoom(self, e: Any) -> None:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        if self._compat_disp is None or self._compat_fit_scale <= 0:
            return
        cur = self._compat_pending if self._compat_pending is not None else self._compat_disp
        s, ox, oy = cur
        factor = 1.2 ** (e.delta / 120.0)
        hi = max(32.0, self._compat_fit_scale)
        ns = max(self._compat_fit_scale, min(s * factor, hi))
        ux = (e.x - ox) / s
        uy = (e.y - oy) / s
        self._compat_pending = (ns, e.x - ux * ns, e.y - uy * ns)

    def _map_flush_view(self) -> None:
        if not hasattr(self, "_compat_canvas"):
            self._init_compat_canvas()
        p = self._compat_pending
        if p is not None:
            self._compat_disp = p
            self._compat_pending = None


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
