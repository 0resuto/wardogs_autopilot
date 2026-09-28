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
        micro_settle_t: float = 0.30,
        imp_k: float = 0.70,
        w_est: float = 18.0,
        t_min: float = 0.10,
        t_max: float = 0.50,
        pulse_on: int = 3,
        turn_deg: float = 25.0,
        hold_max: float = 8.0,
        jitter: float = 0.1,
        ant_s: float = 0.25,
        pose_timeout: float = 0.6,
        settle_dist_m: float = 8.0,
    ) -> None:
        self.dead = max(float(dead), 6.0)
        self.dead_off = max(float(dead_off), 2.0)
        self.settle_t = float(settle_t)
        # Short pause after a micro tap / a target flip; the full settle_t is
        # only for a completed steering hold. 0.8 s used to leave the wheel
        # dead for ~20 m at speed while the error exploded.
        self.micro_settle_t = max(0.0, float(micro_settle_t))
        # Release anticipation: the wheel is released `ant_s` before the
        # measured heading reaches the aim, because the vehicle keeps yawing
        # through the pose latency + key-release delay. Without it every
        # correction overshoots and the loop relay-oscillates wall to wall.
        self.ant_s = max(0.0, min(1.0, float(ant_s)))
        # No steering continuation on a stale pose (frozen localization).
        self.pose_timeout = max(0.2, float(pose_timeout))
        # The pause after a completed hold is capped by distance: a fixed
        # 0.6 s is a 12 m open-loop coast at 20 m/s, which feeds the relay
        # limit cycle; at low speed the full settle is kept.
        self.settle_dist_m = max(1.0, float(settle_dist_m))
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
        self.last_lead = 0.0

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

    def force_release(self, now: float, mh_t: float, settle_t: float | None = None) -> None:
        """Release wheel immediately and trigger a settle pause."""
        self.steer = 0
        self.settle_until = now + (self.settle_t if settle_t is None else settle_t)
        self.settle_mh = mh_t
        self.hold = False
        self.hold_err0 = 0.0
        self.micro = False
        self.last_lead = 0.0

    def update_angular_velocity(self, now: float, heading: float, max_rate: float = 60.0) -> None:
        """Update the yaw-rate estimate from successive heading observations.

        The heading refreshes at the pose rate (~4-8 Hz), not every tick:
        updating on ticks decayed the estimate to zero between poses and made
        it useless for release anticipation. Only real heading changes count,
        and the instantaneous rate is clamped to the physical yaw authority so
        a localization jump cannot fake a spin.
        """
        if self.last_hd is None:
            self.last_hd = heading
            self.last_hd_t = now
            return
        delta = wrap180(heading - self.last_hd)
        if abs(delta) <= 1e-3:
            return
        dt = now - self.last_hd_t
        if not 1e-3 < dt < 1.5:
            self.last_hd = heading
            self.last_hd_t = now
            return
        inst = max(-max_rate, min(max_rate, delta / dt))
        self.ang = self.ang * 0.5 + inst * 0.5
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
        fresh_sample: bool = True,
        pose_age: float | None = None,
        mv_mps: float | None = None,
        heading_meas: float | None = None,
    ) -> int:
        """Evaluate steering state machine and return active key command (-1=A, 0=None, +1=D).

        The release tests use the *predicted* error `err - ang * ant_s`: the
        truck keeps yawing through the pose latency and the key-release delay,
        so a correction decided on the current (lagged) heading always
        overshoots. The yaw rate is estimated from the raw measured heading
        when available (`heading_meas`): the control heading is rate-limited
        and smoothed, and its rate lagged the real yaw by ~0.3 s exactly during
        the release decision. The debounce counts fresh pose samples, not
        ticks: with a ~23 Hz loop and ~8 Hz poses a single glitch used to
        engage a full-lock hold. A stale pose (frozen localization) releases
        the wheel.
        """
        rate_src = heading if heading_meas is None else heading_meas
        self.update_angular_velocity(
            now, rate_src, max_rate=max(15.0, 1.4 * (yaw_rate_max or 45.0))
        )

        pose_stale = pose_age is not None and pose_age > self.pose_timeout
        if pose_stale:
            if self.steer != 0:
                self.force_release(now, mh_t, 0.0)
            return 0

        # Consecutive big-error debounce over fresh pose samples.
        if fresh_sample:
            self.big_n = self.big_n + 1 if abs(err) >= self.turn_deg else 0

        fresh = mh is None or mh_t > self.settle_mh or now > self.settle_until + 0.5

        # Predicted error / rotation at the end of the control delay.
        lead = max(-45.0, min(45.0, self.ang * self.ant_s))
        self.last_lead = lead
        pred_err = err - lead

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
            # big error persists across the debounce window (~2 poses).
            if not self.hold and self.big_n >= 2:
                self.hold = True
                self.imp_end = now + self.hold_max

            rotated = abs(wrap180(heading - self.press_h0))
            # Rotation already underway in the commanded direction counts as
            # done; rotation against it delays the release.
            rotated_eff = rotated + lead * self.steer
            press_age = now - self.press_t0
            small = pred_err < self.dead_off if self.steer == 1 else pred_err > -self.dead_off
            # The target flipped to the other side (waypoint switch, bearing
            # glitch): holding the lock would keep driving away from it.
            opposite = (self.steer > 0 and err < -self.dead) or (self.steer < 0 and err > self.dead)
            if self.micro:
                # A micro tap is a fixed-length pulse: its tick counter is
                # handled by the press block below. Without this the stale
                # imp_end (never set on micro engagement) released the pulse
                # after a single tick, making every micro tap 4x too short.
                released = opposite
            elif self.hold:
                # Continuous steering: keep turning until the heading really
                # aligns with the bearing (not an impulse timeout).
                aligned = abs(pred_err) < self.dead
                turned = rotated_eff >= min(abs(pred_err), self.hold_err0) * 0.8 + self.dead_off
                released = aligned or turned or opposite or now >= self.imp_end
            else:
                released = (
                    rotated_eff >= abs(pred_err) * 0.85 + self.dead_off
                    or (press_age > 0.45 and rotated < 3.0)
                    or small
                    or opposite
                    or now >= self.imp_end
                )

            if released:
                if opposite:
                    # A target flip is not a finished correction: re-engage fast.
                    self.force_release(now, mh_t, self.micro_settle_t)
                elif mv_mps is not None and mv_mps > 0.5:
                    settle = min(self.settle_t, self.settle_dist_m / mv_mps)
                    self.force_release(now, mh_t, settle)
                else:
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

        pressed = self.steer
        press_steer = False
        if self.steer != 0:
            if self.micro:
                if self.steer_ph < self.micro_ticks:
                    press_steer = True
                self.steer_ph += 1
                if self.steer_ph >= self.micro_ticks:
                    # The release zeroes self.steer; the last tick of the pulse
                    # must still be pressed (it used to be swallowed, halving
                    # every micro tap).
                    self.force_release(now, mh_t, self.micro_settle_t)
            else:
                press_steer = True

        if press_steer:
            return pressed
        return 0
