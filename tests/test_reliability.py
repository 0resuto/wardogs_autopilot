"""Regression tests for reliability-critical components.

Covers, against the real production code:
1. Atomic config saving (AppConfig.save / atomic_write_json).
2. Concurrent map switching and index access in the locator store.
3. ScreenCapture resource closing.
4. ArduinoKeyDriver serial protocol: input drain and key-mask composition.
5. HotkeyManager Win32 pump: WM_HOTKEY dispatch, failures, unregistration.
"""

import ctypes
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from ctypes import wintypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import autopilot.common.config as config_mod  # noqa: E402
import autopilot.ui.hotkeys as hotkeys_mod  # noqa: E402
from autopilot.common.config import AppConfig, CaptureConfig, atomic_write_json  # noqa: E402
from autopilot.hardware.arduino_keyboard import ArduinoKeyDriver  # noqa: E402
from autopilot.hardware.screen_capture import ScreenCapture  # noqa: E402
from autopilot.ui.hotkeys import HotkeyManager  # noqa: E402
from autopilot.vision import locator  # noqa: E402
from autopilot.vision import map_store as map_store_mod  # noqa: E402


class TestAtomicConfigSave(unittest.TestCase):
    def test_save_writes_valid_json_without_leftover_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "config.json")
            AppConfig(capture=CaptureConfig(fps=25)).save(target)

            with open(target, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["capture"]["fps"], 25)
            self.assertEqual(os.listdir(tmp), ["config.json"])

    def test_failed_save_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "config.json")
            AppConfig().save(target)
            with open(target, encoding="utf-8") as fh:
                original = fh.read()

            def boom(*_args, **_kwargs):
                raise OSError("disk full")

            with (
                patch.object(config_mod.json, "dump", boom),
                self.assertRaises(OSError),
            ):
                AppConfig(capture=CaptureConfig(fps=30)).save(target)

            with open(target, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), original)
            self.assertEqual(os.listdir(tmp), ["config.json"])

    def test_atomic_write_json_supports_plain_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "payload.json")
            atomic_write_json(target, {"a": [1, 2, 3]})
            with open(target, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), {"a": [1, 2, 3]})


class TestLocatorConcurrency(unittest.TestCase):
    def test_concurrent_set_map_and_get_index(self):
        store = locator.get_store()
        known_maps = {"zestafona", "bakurani"}

        def fake_load_index(name):
            return SimpleNamespace(
                name=name,
                gray_sig=store.gray_sig(),
                norm=map_store_mod._INDEX_NORM,
            )

        errors: list[BaseException] = []
        stop = threading.Event()

        def switcher():
            try:
                while not stop.is_set():
                    locator.set_map("zestafona")
                    locator.set_map("bakurani")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                while not stop.is_set():
                    idx = locator._get_index()
                    if idx is not None and idx.name not in known_maps:
                        errors.append(AssertionError(f"unexpected index {idx.name!r}"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with patch.object(map_store_mod, "load_index", fake_load_index):
            threads = [
                threading.Thread(target=switcher),
                threading.Thread(target=reader),
                threading.Thread(target=reader),
            ]
            for t in threads:
                t.start()
            time.sleep(0.2)
            stop.set()
            for t in threads:
                t.join(timeout=5.0)

            locator.set_map("zestafona")
            self.assertEqual(errors, [])
            self.assertEqual(locator.map_name(), "zestafona")
            idx = locator._get_index()
            self.assertIsNotNone(idx)
            assert idx is not None
            self.assertEqual(idx.name, "zestafona")


class TestHardwareDrivers(unittest.TestCase):
    def test_screen_capture_close(self):
        cap = ScreenCapture(0, (0, 0, 100, 100))
        self.assertIsNotNone(cap._sct)
        cap.close()
        self.assertIsNone(cap._sct)
        cap.close()

    def test_screen_capture_grab_falls_back_on_bad_monitor_index(self):
        cap = ScreenCapture.__new__(ScreenCapture)
        cap.monitor_index = 99
        cap.region = None
        cap.monitors = [{"left": 0, "top": 0, "width": 10, "height": 10}]
        recorded: dict[str, int] = {}

        class _FakeSct:
            def grab(self, monitor):
                recorded.update(monitor)
                return np.zeros((4, 4, 4), np.uint8)

        cap._sct = _FakeSct()

        frame = cap.grab()
        self.assertEqual(frame.shape, (4, 4, 3))
        self.assertEqual(recorded["left"], 0)

    def test_screen_capture_grab_without_monitors_raises(self):
        cap = ScreenCapture.__new__(ScreenCapture)
        cap.monitor_index = 0
        cap.region = None
        cap.monitors = []
        cap._sct = object()

        with self.assertRaises(OSError):
            cap.grab()

    def test_arduino_key_driver_drains_input_buffer(self):
        mock_ser = MagicMock()
        mock_ser.in_waiting = 5

        driver = ArduinoKeyDriver.__new__(ArduinoKeyDriver)
        driver._ser = mock_ser
        driver._port = "TEST_PORT"
        driver._state = {}

        driver._write(0x01)
        mock_ser.reset_input_buffer.assert_called_once()
        mock_ser.write.assert_called_with(bytes([0x01]))

    def test_arduino_key_masks_match_firmware_protocol(self):
        mock_ser = MagicMock()
        driver = ArduinoKeyDriver.__new__(ArduinoKeyDriver)
        driver._ser = mock_ser
        driver._port = "TEST_PORT"
        driver._state = {}

        driver.set_state({"W": True, "SPACE": True})
        mock_ser.write.assert_called_with(bytes([0x11]))
        self.assertEqual(sorted(driver.held()), ["SPACE", "W"])

        driver.set_state({"A": True})
        mock_ser.write.assert_called_with(bytes([0x02]))
        self.assertEqual(driver.held(), ["A"])

        driver.release_all()
        mock_ser.write.assert_called_with(bytes([0x00]))
        self.assertEqual(driver.held(), [])


class TestHotkeyManager(unittest.TestCase):
    @staticmethod
    def _fake_windll(pending, registered, unregistered, register_ok=None):
        def get_message(lpmsg, _hwnd, _lo, _hi):
            if not pending:
                return 0
            message_id, vk = pending.pop(0)
            msg = ctypes.cast(lpmsg, ctypes.POINTER(wintypes.MSG)).contents
            msg.message = message_id
            msg.wParam = vk
            return 1

        def register(_hwnd, vk, _mods, _key):
            if register_ok is not None and not register_ok():
                return 0
            registered.append(vk)
            return 1

        user32 = SimpleNamespace(
            PeekMessageW=lambda *_args: 0,
            RegisterHotKey=register,
            UnregisterHotKey=lambda _hwnd, vk: unregistered.append(vk) or 1,
            GetMessageW=get_message,
            SetTimer=lambda *_args: 1,
            KillTimer=lambda *_args: 1,
        )
        kernel32 = SimpleNamespace(GetCurrentThreadId=lambda: 4242)
        return SimpleNamespace(user32=user32, kernel32=kernel32)

    def test_pump_dispatches_wm_hotkey_and_unregisters(self):
        received: list[int] = []
        registered: list[int] = []
        unregistered: list[int] = []
        fake = self._fake_windll(
            [
                (0x0312, hotkeys_mod._HK_F6),
                (0x0312, hotkeys_mod._HK_F7),
                (0x0312, hotkeys_mod._HK_F8),
            ],
            registered,
            unregistered,
        )

        manager = HotkeyManager(None, received.append)
        with patch.object(hotkeys_mod.ctypes, "windll", fake):
            manager._walk()

        self.assertEqual(received, [hotkeys_mod._HK_F6, hotkeys_mod._HK_F7, hotkeys_mod._HK_F8])
        self.assertEqual(registered, [hotkeys_mod._HK_F6, hotkeys_mod._HK_F7, hotkeys_mod._HK_F8])
        self.assertEqual(unregistered, [hotkeys_mod._HK_F6, hotkeys_mod._HK_F7, hotkeys_mod._HK_F8])

    def test_registration_failure_is_logged(self):
        received: list[int] = []
        registered: list[int] = []
        unregistered: list[int] = []
        fake = self._fake_windll([], registered, unregistered)
        fake.user32.RegisterHotKey = lambda *_args: 0

        manager = HotkeyManager(None, received.append)
        with (
            patch.object(hotkeys_mod.ctypes, "windll", fake),
            self.assertLogs("hotkeys", level="WARNING") as captured,
        ):
            manager._walk()

        self.assertEqual(received, [])
        self.assertTrue(
            any("registration failed" in line for line in captured.output), captured.output
        )

    def test_registration_is_retried_until_success(self):
        received: list[int] = []
        registered: list[int] = []
        unregistered: list[int] = []
        states: list[bool] = []
        attempts = {"n": 0}

        def register_ok() -> bool:
            attempts["n"] += 1
            return attempts["n"] > 2

        fake = self._fake_windll(
            [(hotkeys_mod._WM_TIMER, 1), (hotkeys_mod._WM_TIMER, 1)],
            registered,
            unregistered,
            register_ok=register_ok,
        )
        manager = HotkeyManager(None, received.append)
        manager.registration_changed.connect(states.append)

        with patch.object(hotkeys_mod.ctypes, "windll", fake):
            manager._walk()

        self.assertEqual(
            sorted(registered), sorted([hotkeys_mod._HK_F6, hotkeys_mod._HK_F7, hotkeys_mod._HK_F8])
        )
        self.assertIn(True, states)


if __name__ == "__main__":
    unittest.main()
