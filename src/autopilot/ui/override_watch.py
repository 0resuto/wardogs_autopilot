"""WASD/SPACE emergency-stop watch (Windows Raw Input).

F7 stops the autopilot from anywhere; the movement keys must do the same:
while the driver runs, a human press of W/A/S/D/SPACE (in the game or in any
other window) triggers the emergency stop. The Arduino Micro presses those
keys as a real USB HID keyboard, so a global key-state poll cannot tell its
presses from human ones. This watcher registers for Raw Input with
RIDEV_INPUTSINK (events arrive even while the game is in the foreground),
derives the Arduino's USB VID/PID from the configured serial port and ignores
every event whose source device matches it. Keys are only observed, never
swallowed: the human press still reaches the game, so the player takes over.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..common.log import get_logger

logger = get_logger("override")

_WM_INPUT = 0x00FF
_WM_KEYDOWN = 0x0100
_WM_SYSKEYDOWN = 0x0104
_WM_QUIT = 0x0012
_RID_INPUT = 0x10000003
_RID_INPUT_KEYBOARD = 1
_RIDI_DEVICENAME = 0x20000007
_RIDEV_INPUTSINK = 0x00000100
_HID_USAGE_PAGE_GENERIC = 0x01
_HID_USAGE_KEYBOARD = 0x06
_HWND_MESSAGE = wintypes.HWND(-3)
#: Win32 virtual keys of the override: W, A, S, D, SPACE.
_OVERRIDE_VKS = frozenset((0x57, 0x41, 0x53, 0x44, 0x20))
_MAX_RAW_INPUT = 4096


class _RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = (
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    )


class _RAWINPUTHEADER(ctypes.Structure):
    _fields_ = (
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    )


class _RAWKEYBOARD(ctypes.Structure):
    _fields_ = (
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    )


class _RAWINPUT(ctypes.Structure):
    _fields_ = (("header", _RAWINPUTHEADER), ("keyboard", _RAWKEYBOARD))


def arduino_hardware_id(port: str) -> str | None:
    """USB path tag (`VID_xxxx&PID_xxxx`) of the Arduino on `port`, if identifiable."""
    try:
        from serial.tools import list_ports

        infos = list(list_ports.comports())
    except Exception as exc:  # noqa: BLE001 - pyserial may be missing or the scan may fail
        logger.warning("[studio] override watch: COM port scan failed (%s)", exc)
        return None
    for info in infos:
        if str(info.device).upper() != str(port).upper():
            continue
        vid, pid = getattr(info, "vid", None), getattr(info, "pid", None)
        if vid is None or pid is None:
            return None
        return f"VID_{int(vid):04X}&PID_{int(pid):04X}"
    return None


class OverrideKeyWatcher(QObject):
    """Raw-Input watch: emits `override_pressed` for W/A/S/D/SPACE from non-Arduino devices.

    The pump thread is started by `arm()` and lives until `stop()`; `disarm()`
    only drops the Arduino filter, so re-arming is free. Registration is
    independent of the foreground window, and the keys are not swallowed.
    """

    override_pressed = Signal(int)

    def __init__(self, parent: Any, on_override: Callable[[int], None] | None = None) -> None:
        super().__init__(parent)
        if on_override is not None:
            self.override_pressed.connect(on_override)
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ignore_id: str | None = None
        self._failed_port: str | None = None
        self._device_names: dict[int, str] = {}

    def is_armed(self) -> bool:
        """True while events from non-Arduino keyboards are reported."""
        return self._ignore_id is not None

    def arm(self, port: str) -> bool:
        """Watch for override keys, ignoring the Arduino on `port`.

        Returns False (and disables the watch) when the port cannot be mapped
        to a USB device: without the VID/PID the Arduino's own key presses
        would be indistinguishable from human ones and would stop every run.
        A failed port is not rescanned until `disarm()` (the poll calls `arm()`
        repeatedly while the driver runs).
        """
        if port == self._failed_port:
            return False
        hw_id = arduino_hardware_id(port)
        if hw_id is None:
            self._failed_port = port
            self._ignore_id = None
            logger.warning(
                "[studio] override watch: Arduino on %s not identified - "
                "WASD/SPACE emergency stop disabled",
                port,
            )
            return False
        self._failed_port = None
        self._ignore_id = hw_id
        self.start()
        return True

    def disarm(self) -> None:
        """Stop reporting override keys (the pump thread keeps running)."""
        self._ignore_id = None
        self._failed_port = None

    def start(self) -> None:
        """Spawn the message pump thread (no-op while one is alive)."""
        if self._thread is not None and self._thread.is_alive():
            return
        t = threading.Thread(target=self._walk, daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        """Wake the pump thread with WM_QUIT and join it."""
        t = self._thread
        if t is None:
            return
        if t.is_alive():
            tid = t.native_id or self._thread_id
            ctypes.windll.user32.PostThreadMessageW(tid, _WM_QUIT, 0, 0)
            t.join(timeout=1.0)
        if t.is_alive():
            logger.warning("[studio] override watch: pump thread did not stop within 1 s")
            return
        self._thread = None

    def _walk(self) -> None:
        """Pump thread: hidden window + Raw Input, WM_INPUT handled in the queue."""
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()

        hwnd = self._create_window(user32)
        if not hwnd:
            logger.warning("[studio] override watch: raw input window unavailable")
            return
        device = _RAWINPUTDEVICE(
            _HID_USAGE_PAGE_GENERIC, _HID_USAGE_KEYBOARD, _RIDEV_INPUTSINK, hwnd
        )
        if not user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
            logger.warning("[studio] override watch: raw input registration failed")
            user32.DestroyWindow(hwnd)
            return

        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == _WM_INPUT:
                    self._on_raw_input(user32, msg.lParam)
                    continue
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            user32.DestroyWindow(hwnd)

    @staticmethod
    def _create_window(user32: Any) -> int:
        """Hidden message-only window that receives WM_INPUT on this thread."""
        user32.CreateWindowExW.restype = wintypes.HWND
        hwnd = user32.CreateWindowExW(
            0,
            "STATIC",
            "wardogs-override-watch",
            0,
            0,
            0,
            0,
            0,
            _HWND_MESSAGE,
            None,
            None,
            None,
        )
        return int(hwnd or 0)

    def _on_raw_input(self, user32: Any, lparam: int) -> None:
        """Emit `override_pressed` for a key-down that is not from the Arduino."""
        raw = wintypes.HANDLE(lparam)
        size = wintypes.UINT(0)
        header_size = ctypes.sizeof(_RAWINPUTHEADER)
        if user32.GetRawInputData(raw, _RID_INPUT, None, ctypes.byref(size), header_size) != 0:
            return
        if not 0 < size.value <= _MAX_RAW_INPUT:
            return
        buf = ctypes.create_string_buffer(size.value)
        if user32.GetRawInputData(raw, _RID_INPUT, buf, ctypes.byref(size), header_size) < 0:
            return
        data = ctypes.cast(buf, ctypes.POINTER(_RAWINPUT)).contents
        if data.header.dwType != _RID_INPUT_KEYBOARD:
            return
        kb = data.keyboard
        if kb.Message not in (_WM_KEYDOWN, _WM_SYSKEYDOWN) or kb.VKey not in _OVERRIDE_VKS:
            return
        ignore = self._ignore_id
        if not ignore:
            return
        if ignore in self._device_path(user32, data.header.hDevice).upper():
            return
        self.override_pressed.emit(int(kb.VKey))

    def _device_path(self, user32: Any, device: Any) -> str:
        """Cached RIDI_DEVICENAME for a raw-input device handle."""
        key = int(device.value or 0) if hasattr(device, "value") else int(device or 0)
        cached = self._device_names.get(key)
        if cached is not None:
            return cached
        size = wintypes.UINT(0)
        user32.GetRawInputDeviceInfoW(device, _RIDI_DEVICENAME, None, ctypes.byref(size))
        path = ""
        if size.value > 1:
            buf = ctypes.create_unicode_buffer(size.value)
            user32.GetRawInputDeviceInfoW(device, _RIDI_DEVICENAME, buf, ctypes.byref(size))
            path = buf.value
        self._device_names[key] = path
        return path
