"""Global hotkeys F6/F7 for the WARDOGS studio using Windows RegisterHotKey."""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..common.log import get_logger

logger = get_logger("hotkeys")

_WM_HOTKEY = 0x0312
_HK_MOD_NOREPEAT = 0x4000
_HK_F6 = 0x75
_HK_F7 = 0x76
_HK_IDS = (_HK_F6, _HK_F7)


class HotkeyManager(QObject):
    """Registers F6/F7 as global Windows hotkeys and forwards presses via Qt Signal."""

    hotkey_triggered = Signal(int)

    def __init__(self, parent: Any, on_hotkey: Callable[[int], None] | None = None) -> None:
        super().__init__(parent)
        if on_hotkey is not None:
            self.hotkey_triggered.connect(on_hotkey)
        self._thread: threading.Thread | None = None
        self._thread_id: int = 0

    def start(self) -> None:
        """Spawn the message pump thread."""
        if self._thread is not None:
            return
        t = threading.Thread(target=self._walk, daemon=True)
        self._thread = t
        t.start()

    def _walk(self) -> None:
        """Thread message pump: registers the hotkeys and emits Qt Signal on WM_HOTKEY."""
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()

        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        for vk in _HK_IDS:
            if not user32.RegisterHotKey(None, vk, _HK_MOD_NOREPEAT, vk):
                logger.warning("[studio] global hotkey F%s registration failed", "6" if vk == _HK_F6 else "7")

        while True:
            n = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if n <= 0:
                break
            if msg.message == _WM_HOTKEY:
                vk = int(msg.wParam)
                self.hotkey_triggered.emit(vk)

        for vk in _HK_IDS:
            user32.UnregisterHotKey(None, vk)

    def stop(self) -> None:
        """Wake the pump thread with WM_QUIT and join it."""
        t = self._thread
        if t is None or not t.is_alive():
            return
        ctypes.windll.user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)
        t.join(timeout=1.0)
        self._thread = None
