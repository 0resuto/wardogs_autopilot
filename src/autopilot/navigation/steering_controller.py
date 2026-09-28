"""Steering controller with yaw-turn inertia and impulse micro-corrections.

Pulse durations get a light random jitter and micro-pulses vary by a tick, so
the injected input does not look like perfectly periodic machine taps.
"""

from __future__ import annotations

import random


def wrap180(deg: float) -> float:
    """Normalize an angle to [-180, +180) degrees."""
    return (deg + 180.0) % 360.0 - 180.0


class SteeringController:
    """Computes pulse steering commands with inertia anticipation."""

    def __init__(
        self,
        dead: float = 6.0,
        dead_off: float = 2.0,
        settle_t: float = 0.80,
        imp_k: float = 0.70,
        w_est: float = 18.0,
        t_min: float = 0.10,
        t_max: float = 0.50,
        pulse_on: int = 2,
        turn_deg: float = 25.0,
        hold_max: float = 8.0,
        jitter: float = 0.1,
    ) -> None:
        self.dead = max(float(dead), 6.0)
        self.dead_off = max(float(dead_off), 2.0)
        self.settle_t = float(settle_t)
        self.imp_k = float(imp_k)
        self.w_est = float(w_est)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.pulse_on = max(1, int(pulse_on))
        self.turn_deg = float(turn_deg)
        self.hold_max = float(hold_max)
        self.jitter = max(0.0, min(0.5, float(jitter)))

        self.steer = 0  # -1=A, 0=neutral, +1=D
        self.steer_ph = 0  # micro-pulse tick counter
        self.micro_ticks = self.pulse_on  # length of the current micro-pulse
        self.micro = False  # micro-tap mode
        self.hold = False  # continuous steering on large errors
        self.hold_err0 = 0.0  # |err| at hold-mode engagement
        self.big_n = 0  # consecutive big-error ticks (debounce)
        self.ang = 0.0  # heading angular velocity (deg/s)
        self.last_hd: float | None = None
        self.last_hd_t = 0.0
        self.settle_until = 0.0
        self.imp_end = 0.0
        self.press_t0 = 0.0
        self.press_h0 = 0.0
        self.hold_t0 = 0.0
        self.settle_mh = -1.0

    def reset(self) -> None:
        """Reset steering state to neutral."""
        self.steer = 0
        self.steer_ph = 0
        self.micro_ticks = self.pulse_on
        self.micro = False
        self.hold = False
        self.hold_err0 = 0.0
        self.big_n = 0
        self.ang = 0.0
        self.last_hd = None
        self.last_hd_t = 0.0
        self.settle_until = 0.0
        self.imp_end = 0.0

    def force_release(self, now: float, mh_t: float) -> None:
        """Release wheel immediately and trigger settle pause."""
        self.steer = 0
        self.settle_until = now + self.settle_t
        self.settle_mh = mh_t
        self.hold = False
        self.hold_err0 = 0.0

    def update_angular_velocity(self, now: float, heading: float) -> None:
        """Update estimated yaw rate from successive heading observations."""
        if self.last_hd is not None:
            delta = wrap180(heading - self.last_hd)
            dt = max(now - self.last_hd_t, 1e-3)
            self.ang = self.ang * 0.6 + (delta / dt) * 0.4
        self.last_hd = heading
        self.last_hd_t = now

    def calc_impulse(self, err: float, yaw_rate_max: float | None = None) -> float:
        """Steering impulse duration (seconds) for a heading error.

        With a model-provided yaw rate the duration follows the physically
        available turn rate; without it the legacy constant is used.
        """
        rate = self.w_est
        if yaw_rate_max is not None and yaw_rate_max > 1e-3:
            rate = float(yaw_rate_max)
        return max(
            self.t_min,
            min(self.t_max, abs(err) * self.imp_k / rate),
        )

    def _impulse_end(self, now: float, err: float, yaw_rate_max: float | None) -> float:
        """Jittered impulse end time (hold mode keeps its fixed safety timeout)."""
        if self.hold:
            return now + self.hold_max
        base = min(self.calc_impulse(abs(err), yaw_rate_max), self.t_max)
        return now + base * random.uniform(1.0 - self.jitter, 1.0 + self.jitter)

    def step(
        self,
        now: float,
        err: float,
        heading: float,
        mh: float | None,
        mh_t: float,
        yaw_rate_max: float | None = None,
    ) -> int:
        """Evaluate steering state machine and return active key command (-1=A, 0=None, +1=D)."""
        self.update_angular_velocity(now, heading)

        # Consecutive big-error debounce: a single-tick glitch (localization
        # flicker) must not engage continuous steering.
        self.big_n = self.big_n + 1 if abs(err) >= self.turn_deg else 0

        fresh = mh is None or mh_t > self.settle_mh or now > self.settle_until + 0.5

        if self.steer == 0:
            if now > self.settle_until and fresh:
                if err > self.dead:
                    self.steer = 1
                    self.micro = False
                    self.hold = self.big_n >= 2
                    self.hold_err0 = abs(err)
                    self.imp_end = self._impulse_end(now, err, yaw_rate_max)
                    self.press_t0 = now
                    self.press_h0 = heading
                    self.hold_t0 = now
                elif err < -self.dead:
                    self.steer = -1
                    self.micro = False
                    self.hold = self.big_n >= 2
                    self.hold_err0 = abs(err)
                    self.imp_end = self._impulse_end(now, err, yaw_rate_max)
                    self.press_t0 = now
                    self.press_h0 = heading
                    self.hold_t0 = now
        else:
            # Upgrade an in-progress impulse to continuous steering once a
            # big error persists across the debounce window (~2 ticks).
            if not self.hold and self.big_n >= 2:
                self.hold = True
                self.imp_end = now + self.hold_max

            rotated = abs(wrap180(heading - self.press_h0))
            press_age = now - self.press_t0
            small = err < self.dead_off if self.steer == 1 else err > -self.dead_off
            if self.hold:
                # Continuous steering: keep turning until the heading really
                # aligns with the bearing (not an impulse timeout).
                aligned = abs(err) < self.dead
                turned = rotated >= min(abs(err), self.hold_err0) * 0.8 + self.dead_off
                released = aligned or turned or now >= self.imp_end
            else:
                released = (
                    rotated >= abs(err) * 0.85 + self.dead_off
                    or (press_age > 0.45 and rotated < 3.0)
                    or small
                    or now >= self.imp_end
                )

            if released:
                self.force_release(now, mh_t)

        # Micro-corrections after settle
        if self.steer == 0 and now > self.settle_until and fresh:
            if err > self.dead_off:
                self.steer = 1
                self.steer_ph = 0
                self.micro = True
                self.micro_ticks = random.choice((self.pulse_on, self.pulse_on + 1))
                self.hold_t0 = now
                self.settle_mh = mh_t
            elif err < -self.dead_off:
                self.steer = -1
                self.steer_ph = 0
                self.micro = True
                self.micro_ticks = random.choice((self.pulse_on, self.pulse_on + 1))
                self.hold_t0 = now
                self.settle_mh = mh_t

        press_steer = False
        if self.steer != 0:
            if self.micro:
                if self.steer_ph < self.micro_ticks:
                    press_steer = True
                self.steer_ph += 1
                if self.steer_ph >= self.micro_ticks:
                    self.force_release(now, mh_t)
            else:
                press_steer = True

        if press_steer:
            return self.steer
        return 0
