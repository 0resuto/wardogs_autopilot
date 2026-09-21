// USB keyboard on Arduino Micro/Pro Micro (ATmega32U4).
// The studio sends one byte = key mask over Serial, the Arduino presses/
// releases the keys via HID (the input looks like a real USB keyboard).
//
//   bit 0 -> W
//   bit 1 -> A
//   bit 2 -> S
//   bit 3 -> D
//   bit 4 -> SPACE
//
// Garbage/noise protection ("key spam" on an idle connection without the
// studio):
//
// 1. Byte validation. Only an honest mask is accepted: 0x00..0x1F (WASD+SPACE).
//    Any value with bits above the 4th (0x20..0xFE) and 0xFF is garbage/reset:
//    all keys are released, nothing is remembered. A single "extra" byte on
//    the line no longer turns into a key press.
//
// 2. Watchdog. Keys are held ONLY while fresh valid commands arrive from the
//    studio (the navigator sends set_state every tick ~33 ms). If no valid
//    command arrives for longer than WATCHDOG_MS — release everything and
//    forget the state: no stuck pressed keys, no holds from a random byte.

#include <Keyboard.h>

const uint8_t MASK_W = 0x01;
const uint8_t MASK_A = 0x02;
const uint8_t MASK_S = 0x04;
const uint8_t MASK_D = 0x08;
const uint8_t MASK_SPACE = 0x10;

// bits above the 4th carry no data: everything above is garbage
const uint8_t MASK_VALID = (MASK_W | MASK_A | MASK_S | MASK_D | MASK_SPACE);

// a studio command pause longer than this = connection lost (the navigator
// sends every ~33 ms, ~6x margin). Keep a small margin so it does not trigger
// on a straight run, while 1 byte still fits both noise and flood
const unsigned long WATCHDOG_MS = 200;

uint8_t lastState = 0;
uint8_t sentState = 0;
unsigned long lastCmdMs = 0;

void reportState() {
  Serial.write(lastState);
}

void apply(uint8_t want) {
  uint8_t change = want ^ sentState;
  if (change & MASK_W) {
    if (want & MASK_W) Keyboard.press('w'); else Keyboard.release('w');
  }
  if (change & MASK_A) {
    if (want & MASK_A) Keyboard.press('a'); else Keyboard.release('a');
  }
  if (change & MASK_S) {
    if (want & MASK_S) Keyboard.press('s'); else Keyboard.release('s');
  }
  if (change & MASK_D) {
    if (want & MASK_D) Keyboard.press('d'); else Keyboard.release('d');
  }
  if (change & MASK_SPACE) {
    if (want & MASK_SPACE) Keyboard.press(' '); else Keyboard.release(' ');
  }
  sentState = want;
}

void releaseAll() {
  if (sentState) {
    Keyboard.releaseAll();
    sentState = 0;
  }
}

void setup() {
  Serial.begin(115200);
  Keyboard.begin();
  lastState = 0;
  sentState = 0;
  lastCmdMs = millis();
}

void loop() {
  // new state byte from the studio
  while (Serial.available() > 0) {
    uint8_t b = Serial.read();
    if (b == 0xFF || (b & ~MASK_VALID) != 0) {
      // 0xFF — explicit reset; other "extra" bytes are line garbage.
      // In both cases release everything and forget the state.
      releaseAll();
      lastState = 0;
    } else {
      lastState = b;
      lastCmdMs = millis();  // fresh command from the studio — feed the watchdog
      reportState();
    }
  }
  // connection lost / garbage without valid commands — do not keep keys held
  if ((unsigned long)(millis() - lastCmdMs) > WATCHDOG_MS) {
    releaseAll();
    lastState = 0;
  }
  apply(lastState);
}