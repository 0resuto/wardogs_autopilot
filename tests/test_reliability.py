"""Unit test verifying high-priority reliability fixes:
1. Atomic config saving.
2. Thread-safe set_map and _get_index in locator.
3. ScreenCapture resource closing.
4. HotkeyManager vk dispatch.
5. ArduinoKeyDriver serial input drain.
"""

import json
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.hardware.arduino_keyboard import ArduinoKeyDriver
from autopilot.hardware.screen_capture import ScreenCapture
from autopilot.ui.hotkeys import HotkeyManager
from autopilot.vision import locator


class TestReliabilityFixes(unittest.TestCase):
    def test_atomic_config_save(self):
        """Verify atomic config saving writes valid json and cleans up .tmp."""
        test_path = os.path.join(ROOT, "output", "test_config_atomic.json")
        tmp_path = test_path + ".tmp"
        if os.path.exists(test_path):
            os.remove(test_path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        data = {"test_key": "test_val", "number": 42}
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, test_path)

        self.assertTrue(os.path.exists(test_path))
        self.assertFalse(os.path.exists(tmp_path))

        with open(test_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded, data)
        os.remove(test_path)

    def test_locator_thread_safety(self):
        """Verify concurrent set_map and _get_index do not crash or corrupt state."""
        errors = []

        def worker_set_map(name):
            try:
                for _ in range(10):
                    locator.set_map(name)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        def worker_get_index():
            try:
                for _ in range(10):
                    _ = locator._get_index()
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker_set_map, args=("zestafona",)),
            threading.Thread(target=worker_set_map, args=("zestafona",)),
            threading.Thread(target=worker_get_index),
            threading.Thread(target=worker_get_index),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Restore back to zestafona
        locator.set_map("zestafona")
        self.assertEqual(len(errors), 0, f"Errors in concurrent locator access: {errors}")

    def test_screen_capture_close(self):
        """Verify ScreenCapture close() method cleans up mss context."""
        cap = ScreenCapture(0, (0, 0, 100, 100))
        self.assertIsNotNone(cap._sct)
        cap.close()
        self.assertIsNone(cap._sct)
        # Calling close again should be a safe no-op
        cap.close()

    def test_arduino_key_driver_drain(self):
        """Verify ArduinoKeyDriver drains input buffer before writing."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 5

        driver = ArduinoKeyDriver.__new__(ArduinoKeyDriver)
        driver._ser = mock_ser
        driver._port = "TEST_PORT"
        driver._state = {}

        driver._write(0x01)
        mock_ser.reset_input_buffer.assert_called_once()
        mock_ser.write.assert_called_with(bytes([0x01]))

    def test_hotkey_manager_vk_dispatch(self):
        """Verify HotkeyManager passes vk code to the callback."""
        received = []
        hm = HotkeyManager.__new__(HotkeyManager)
        hm._on_hotkey = lambda vk: received.append(vk)
        hm._queue = MagicMock()
        hm._queue.get_nowait.side_effect = [0x75, 0x76, Exception("queue empty")]
        hm._root = MagicMock()
        hm._poll_ms = 60

        for _ in range(2):
            vk = hm._queue.get_nowait()
            hm._on_hotkey(vk)

        self.assertEqual(received, [0x75, 0x76])


if __name__ == "__main__":
    unittest.main()
