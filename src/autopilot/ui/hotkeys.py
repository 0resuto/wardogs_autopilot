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
_WM_TIMER = 0x0113
_WM_QUIT = 0x0012
_HK_MOD_NOREPEAT = 0x4000
_HK_F6 = 0x75
_HK_F7 = 0x76
_HK_F8 = 0x77
_HK_IDS = (_HK_F6, _HK_F7, _HK_F8)
_HK_RETRY_MS = 3000
_HK_TIMER_ID = 1


def _hk_label(vk: int) -> str:
    return {_HK_F6: "6", _HK_F7: "7", _HK_F8: "8"}.get(vk, str(vk))


class HotkeyManager(QObject):
    """Registers F6/F7 as global Windows hotkeys and forwards presses via Qt Signal.

    Registration is retried every few seconds while the keys are held by
    another process (e.g. a forgotten studio instance or a stale launcher):
    the hotkeys recover on their own instead of staying dead until restart.
    """

    hotkey_triggered = Signal(int)
    registration_changed = Signal(bool)

    def __init__(self, parent: Any, on_hotkey: Callable[[int], None] | None = None) -> None:
        super().__init__(parent)
        if on_hotkey is not None:
            self.hotkey_triggered.connect(on_hotkey)
        self._thread: threading.Thread | None = None
        self._thread_id: int = 0
        self._registered: set[int] = set()
        self._failed: set[int] = set()
        self._all_ok: bool | None = None

    def is_ready(self) -> bool:
        """True when both hotkeys are registered and will fire."""
        return self._all_ok is True

    def start(self) -> None:
        """Spawn the message pump thread (no-op while one is alive)."""
        if self._thread is not None and self._thread.is_alive():
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
        self._sync_registration(user32)
        timer_id = user32.SetTimer(None, _HK_TIMER_ID, _HK_RETRY_MS, None)
        if not timer_id:
            logger.warning("[studio] hotkey retry timer unavailable")

        try:
            while True:
                if user32.GetMessageW(ctypes.byref(msg), None, 0, 0) <= 0:
                    break
                if msg.message == _WM_HOTKEY:
                    self.hotkey_triggered.emit(int(msg.wParam))
                elif msg.message == _WM_TIMER:
                    self._sync_registration(user32)
        finally:
            if timer_id:
                user32.KillTimer(None, _HK_TIMER_ID)
            self._unregister_all(user32)

    def _sync_registration(self, user32: Any) -> None:
        """Register every missing hotkey; safe to call repeatedly (retry)."""
        for vk in _HK_IDS:
            if vk in self._registered:
                continue
            if user32.RegisterHotKey(None, vk, _HK_MOD_NOREPEAT, vk):
                self._registered.add(vk)
                logger.info("[studio] global hotkey F%s registered", _hk_label(vk))
            elif vk not in self._failed:
                self._failed.add(vk)
                logger.warning(
                    "[studio] global hotkey F%s registration failed "
                    "(possibly another studio instance) - retrying",
                    _hk_label(vk),
                )
        all_ok = len(self._registered) == len(_HK_IDS)
        if all_ok != self._all_ok:
            self._all_ok = all_ok
            self.registration_changed.emit(all_ok)

    def _unregister_all(self, user32: Any) -> None:
        for vk in _HK_IDS:
            if vk in self._registered:
                user32.UnregisterHotKey(None, vk)
        self._registered.clear()
        if self._all_ok is not False:
            self._all_ok = False
            self.registration_changed.emit(False)

    def stop(self) -> None:
        """Wake the pump thread with WM_QUIT and join it.

        On join timeout the handle is kept: dropping it would let `start()`
        spawn a twin pump next to the still-waking one.
        """
        t = self._thread
        if t is None:
            return
        if t.is_alive():
            tid = t.native_id or self._thread_id
            ctypes.windll.user32.PostThreadMessageW(tid, _WM_QUIT, 0, 0)
            t.join(timeout=1.0)
        if t.is_alive():
            logger.warning("[studio] hotkey pump thread did not stop within 1 s")
            return
        self._thread = None
