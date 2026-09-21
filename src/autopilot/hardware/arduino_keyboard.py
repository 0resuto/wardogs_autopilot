"""WASD input via an Arduino Micro (HID).

The Arduino emulates a USB keyboard and presses keys in hardware — its events
carry no LLKHF_INJECTED flag, which SendInput sets. The studio sends one byte
= key mask over serial (firmware arduino/keyboard_emulator.ino).
"""


class ArduinoKeyDriver:
    """Input via Arduino Micro (HID): the studio sends a byte mask to the COM port.

    The Arduino turns the mask into presses of a real USB keyboard — without the
    LLKHF_INJECTED flag that SendInput sets.

    Interface: set_state(keys) / release_all() / held().
    """

    # bit masks must match the firmware arduino/keyboard_emulator.ino
    MASK = {"W": 0x01, "A": 0x02, "S": 0x04, "D": 0x08, "SPACE": 0x10}
    RESET = 0xFF  # arduino: release everything and forget the state

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.2) -> None:
        import serial  # imported locally so the dependency stays optional

        self._port = port
        self._baud = baud
        # write_timeout=timeout: if the COM port / arduino hangs, a write won't
        # block the navigator thread forever — it fails with OSError after the
        # timeout, which the navigator already handles and stops sending keys
        # instead of freezing all of the control logic.
        self._ser = serial.Serial(port, baud, timeout=timeout, write_timeout=timeout)
        self._state: dict[str, bool] = {}
        # the firmware may release all keys left over from the previous run at boot
        try:
            self._ser.write(bytes([self.RESET]))
        except OSError:
            pass

    def set_state(self, keys: dict[str, bool]) -> None:
        mask = 0
        for key, want in keys.items():
            bit = self.MASK.get(key)
            if bit is None:
                continue
            if want:
                mask |= bit
            self._state[key] = want
        for key in self._state:
            if key not in keys:
                self._state[key] = False
        self._write(mask)

    def release_all(self) -> None:
        self._state = {}
        self._write(0)

    def held(self) -> list[str]:
        return [k for k, v in self._state.items() if v]

    def _write(self, mask: int) -> None:
        try:
            if getattr(self._ser, "in_waiting", 0):
                self._ser.reset_input_buffer()
            self._ser.write(bytes([mask]))
        except OSError as exc:
            raise OSError(
                "ArduinoKeyDriver: COM port %s unavailable — check that the "
                "arduino is connected and the port is correct" % self._port
            ) from exc

    def close(self) -> None:
        try:
            self.release_all()
        except OSError:  # noqa: BLE001
            pass
        try:
            self._ser.close()
        except OSError:  # noqa: BLE001
            pass
