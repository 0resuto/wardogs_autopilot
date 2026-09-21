"""Global hotkeys F6/F7 for the studio.

RegisterHotKey delivers the keys even when the game (not the studio) has
focus: the OS itself waits for the press and posts a WM_HOTKEY message — no
hooks are installed and nothing is injected into the game.
"""

import ctypes
import queue
import threading
import tkinter as tk
from ctypes import wintypes

from ..common.log import get_logger

logger = get_logger("hotkeys")


_WM_HOTKEY = 0x0312
_HK_MOD_NOREPEAT = 0x4000
_HK_F6 = 0x75
_HK_F7 = 0x76
_HK_IDS = (_HK_F6, _HK_F7)


def _hk_label(vk: int) -> str:
    return "6" if vk == _HK_F6 else "7" if vk == _HK_F7 else str(vk)


def _get_last_error() -> int:
    try:
        return int(ctypes.windll.kernel32.GetLastError())
    except Exception:  # noqa: BLE001
        return -1


class HotkeyManager:
    """Registers F6/F7 as global hotkeys and forwards presses to a callback.

    root       — the Tk window whose event loop drives the queue polling.
    on_hotkey  — callable invoked for each hotkey event (once per press).
    poll_ms    — how often the UI thread drains the hotkey queue.
    """

    def __init__(self, root: tk.Tk, on_hotkey, poll_ms: int = 60) -> None:
        self._root = root
        self._on_hotkey = on_hotkey
        self._poll_ms = poll_ms
        self._queue: queue.Queue[int] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._poll_job: str | None = None

    def start(self) -> None:
        """Spawn the message pump thread and start polling the queue."""
        if self._thread is not None:
            return
        t = threading.Thread(target=self._walk, daemon=True)
        self._thread = t
        t.start()
        self._poll_job = self._root.after(self._poll_ms, self._poll)

    def _walk(self) -> None:
        """Thread message pump: registers the hotkeys and waits for WM_HOTKEY.

        RegisterHotKey with hwnd=None posts WM_HOTKEY to this thread's message
        queue; GetMessageW takes them and puts them into a queue for the UI
        thread (queue.Queue is thread-safe). WM_QUIT exits the loop (n==0).
        """
        user32 = ctypes.windll.user32
        msg = wintypes.MSG()
        # initialize the thread's message queue BEFORE RegisterHotKey
        # (otherwise there is nowhere to post WM_HOTKEY)
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        for vk in _HK_IDS:
            if not user32.RegisterHotKey(None, vk, _HK_MOD_NOREPEAT, vk):
                logger.warning(
                    "[studio] global hotkey F%s not registered (code %d)",
                    _hk_label(vk),
                    _get_last_error(),
                )

        while True:
            n = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if n <= 0:  # 0 — WM_QUIT, -1 — error
                break
            if msg.message == _WM_HOTKEY:
                self._queue.put(msg.wParam)
        for vk in _HK_IDS:
            user32.UnregisterHotKey(None, vk)

    def _poll(self) -> None:
        for _ in range(64):
            try:
                vk = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._on_hotkey(vk)
            except TypeError:
                self._on_hotkey()
            except Exception:  # noqa: BLE001
                pass
        self._poll_job = self._root.after(self._poll_ms, self._poll)

    def stop(self) -> None:
        """Wake the pump thread with WM_QUIT and join it."""
        t = self._thread
        if t is None:
            return
        ctypes.windll.user32.PostThreadMessageW(
            t.ident, 0x0012, 0, 0)  # WM_QUIT — wake up GetMessageW
        t.join(timeout=1.0)
