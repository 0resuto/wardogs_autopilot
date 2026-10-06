"""Steering controller with yaw-turn inertia and impulse micro-corrections.

Pulse durations get a light random jitter and micro-pulses vary by a tick, so
the injected input does not look like perfectly periodic machine taps.
"""

from __future__ import annotations

import random

#: Impulse length from the reference manual drive
#: (output/manual_dbg_20260928_222519.jsonl): 0.03 s + 0.011 s/deg, i.e.
#: ~65 ms at 3 deg, ~125 ms at 10 deg, ~250 ms at 20 deg. The yaw model no
#: longer sets the pulse length: a full-lock rotation needs err / yaw_rate
#: seconds, which saturated every pulse at t_max, while the human re-taps
#: short pulses and lets the next sample close the rest.
IMP_BASE_S = 0.03
IMP_PER_DEG_S = 0.011
#: One control tick (~0.04 s at the ~24 Hz loop): micro taps are quantized by it.
MICRO_TICK_S = 0.04
#: Braking transfers weight to the front axle and sharpens the yaw response in
#: the game: at 60-90 km/h the measured yaw rate is p50/p90 = 0.2/7.4 deg/s
#: coasting vs 1.9/28.1 deg/s on SPACE (2026-10-01 23:06 run). While braking
#: the yaw clamp is raised by this factor and the impulse shortened, so the
#: release anticipation still matches the real rotation.
BRAKE_YAW_GAIN = 1.8
BRAKE_IMP_SCALE = 0.7
#: Continuous modulation by the off-centre urgency (0 at the inner edge, 1 at
#: the outer edge): the impulse grows to 2x and the settle shrinks to 0.6x at
#: the wall, so a long bend is held by firmer and more frequent taps instead
#: of the old two-variant (micro vs full) response.
URGENCY_IMP_GAIN = 1.0
URGENCY_TICKS = 2
URGENCY_SETTLE_CUT = 0.4
#: Spin recovery holds the wheel twice as long: the truck must come around,
#: not re-pulse every 2 s while it circles.
RECOVERY_HOLD_SCALE = 2.0
#: Above this speed an impulse never upgrades to a continuous hold: the
#: reference manual drive taps the wheel at speed (p50 0.13 s, p90 0.25 s at
#: 60+ km/h; its single 1.1 s press was a low-speed manoeuvre), while the
#: 2026-10-04 14:58 run held a full lock for ~1 s and yawed 25-50 deg per
#: correction - the weave that ended in the reported skid. Below the threshold
#: the hold stays: it is the low-speed yaw-authority fight the crawl needs.
HOLD_MAX_MPS = 10.0
#: Road turn ahead (deg, `PathTracker.calc_road_turn`) from which the wheel may
#: be held at any speed: through a bend the truck needs a sustained steer to
#: reach the grip-limited yaw the route asks for. The 2026-10-04 15:54 run
#: tapped through a 16 deg bend (~30% duty -> ~3 deg/s vs the route's 7-10),
#: the heading fell 10 deg behind and the truck cut the curve to xte 16 m.
#: A hold is only allowed here for an actual bend, not for a centering error.
CURVE_HOLD_DEG = 8.0
#: Rotation AGAINST the commanded direction by more than this (deg) sustained
#: for `WRONG_WAY_S` seconds means the truck is sliding/spinning on the locked
#: wheel (2026-10-03 02:19 run: half a minute of pinned lock while the heading
#: swung 180 deg the other way). The wheel is released instead of pinned.
WRONG_WAY_DEG = 8.0
WRONG_WAY_S = 0.3


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
        # Impulse calibration from the reference manual drive
        # (output/manual_dbg_20260928_222519.jsonl, 2026-09-28): at 79 km/h
        # (Ural, ~10 deg/s grip-limited yaw) the human taps the wheel for
        # ~65 ms at 3 deg of error, ~125 ms at 10 deg and ~250 ms at 20 deg,
        # re-tapping every ~0.4 s, and never holds a full-lock correction for
        # long. The old 0.7 gain targeted closing 70% of the error in one
        # pulse: every correction saturated at t_max=0.5 s, the duty was 2.4x
        # the human's and the truck ran p90 |err| 36 deg vs the human's 11.
        imp_k: float = 0.12,
        w_est: float = 18.0,
        t_min: float = 0.06,
        t_max: float = 0.30,
        pulse_on: int = 3,
        turn_deg: float = 25.0,
        hold_max: float = 2.0,
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
        self.curve_hold = False  # hold allowed because the route bends ahead
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

        Follows the reference manual drive (`IMP_BASE_S` + `IMP_PER_DEG_S` per
        degree), so the pulse length modulates with the error instead of
        clamping every small correction to one tick. `yaw_rate_max` stays in
        the signature for the callers and future shaping but does not stretch
        the pulse: the physical full-lock rate is what made corrections long.
        """
        return max(self.t_min, min(self.t_max, IMP_BASE_S + IMP_PER_DEG_S * abs(err)))

    def _micro_ticks_for(self, err: float, urgency: float = 0.0) -> int:
        """Micro-tap length in ticks, scaled with the error and the off-centre drift.

        The fixed 3-4 tick tap gave 2 deg and 6 deg the same correction; the
        reference drive modulates ~1 tick at 2 deg up to ~2-3 ticks at 6 deg,
        and a drifting car gets up to two extra ticks instead of the same tap.
        """
        base = int(round((IMP_BASE_S + IMP_PER_DEG_S * abs(err)) / MICRO_TICK_S))
        base += int(round(URGENCY_TICKS * max(0.0, min(1.0, urgency))))
        base = max(1, min(self.pulse_on + 1, base))
        return random.choice((base, min(base + 1, self.pulse_on + 1)))

    def _impulse_end(
        self,
        now: float,
        err: float,
        yaw_rate_max: float | None,
        scale: float = 1.0,
    ) -> float:
        """Jittered impulse end time (hold mode keeps its fixed safety timeout).

        `scale` combines the live modifiers: shorter while SPACE is held (the
        game yaws faster under braking) and longer with the off-centre urgency
        or a spin recovery.
        """
        if self.hold:
            return now + self.hold_max * scale
        base = min(self.calc_impulse(abs(err), yaw_rate_max), self.t_max) * scale
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
        braking: bool = False,
        center_urgency: float = 0.0,
        recovery: bool = False,
        curve: float | None = None,
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

        `center_urgency` (0..1, off-centre in the corridor) continuously
        stretches the impulse, adds micro ticks and shortens the settle.
        `recovery` (spun/backwards car) pins the wheel with a doubled hold
        timeout so the truck comes around instead of re-pulsing.
        """
        urgency = max(0.0, min(1.0, center_urgency))
        rate_src = heading if heading_meas is None else heading_meas
        max_rate = max(15.0, 1.4 * (yaw_rate_max or 45.0))
        if braking:
            max_rate *= BRAKE_YAW_GAIN
        self.update_angular_velocity(now, rate_src, max_rate=max_rate)
        impulse_scale = (BRAKE_IMP_SCALE if braking else 1.0) * (1.0 + URGENCY_IMP_GAIN * urgency)
        if recovery:
            impulse_scale *= RECOVERY_HOLD_SCALE
        settle_scale = 1.0 - URGENCY_SETTLE_CUT * urgency
        curve_hold = curve is not None and curve >= CURVE_HOLD_DEG
        self.curve_hold = curve_hold
        hold_ok = recovery or curve_hold or mv_mps is None or mv_mps <= HOLD_MAX_MPS

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
                    self.hold = (self.big_n >= 2 or recovery) and hold_ok
                    self.hold_err0 = abs(err)
                    self.imp_end = self._impulse_end(now, err, yaw_rate_max, impulse_scale)
                    self.press_t0 = now
                    self.press_h0 = heading
                    self.hold_t0 = now
                elif err < -self.dead:
                    self.steer = -1
                    self.micro = False
                    self.hold = (self.big_n >= 2 or recovery) and hold_ok
                    self.hold_err0 = abs(err)
                    self.imp_end = self._impulse_end(now, err, yaw_rate_max, impulse_scale)
                    self.press_t0 = now
                    self.press_h0 = heading
                    self.hold_t0 = now
        else:
            # Upgrade an in-progress impulse to continuous steering once a
            # big error persists across the debounce window (~2 poses).
            if not self.hold and (self.big_n >= 2 or recovery) and hold_ok:
                self.hold = True
                self.imp_end = now + self.hold_max * impulse_scale

            rotated = abs(wrap180(heading - self.press_h0))
            # Rotation already underway in the commanded direction counts as
            # done; rotation against it delays the release.
            rotated_eff = rotated + lead * self.steer
            press_age = now - self.press_t0
            small = pred_err < self.dead_off if self.steer == 1 else pred_err > -self.dead_off
            # The target flipped to the other side (waypoint switch, bearing
            # glitch): holding the lock would keep driving away from it.
            opposite = (self.steer > 0 and err < -self.dead) or (self.steer < 0 and err > self.dead)
            # The truck rotates opposite to the command for long enough: it is
            # sliding on the locked wheel, not turning. Pinning the lock only
            # feeds the spin.
            wrong_way = (
                press_age > WRONG_WAY_S
                and wrap180(heading - self.press_h0) * self.steer < -WRONG_WAY_DEG
            )
            if self.micro:
                # A micro tap is a fixed-length pulse: its tick counter is
                # handled by the press block below. Without this the stale
                # imp_end (never set on micro engagement) released the pulse
                # after a single tick, making every micro tap 4x too short.
                released = opposite or wrong_way
            elif self.hold:
                # Continuous steering: keep turning until the heading really
                # aligns with the bearing (not an impulse timeout). In a bend
                # the `turned` release would cut the hold every time the truck
                # catches up with the aim, which re-taps instead of tracking
                # the bend: the wheel stays pressed while the bend lasts.
                aligned = abs(pred_err) < self.dead
                turned = rotated_eff >= min(abs(pred_err), self.hold_err0) * 0.8 + self.dead_off
                released = (
                    aligned
                    or (turned and not self.curve_hold)
                    or opposite
                    or wrong_way
                    or now >= self.imp_end
                )
            else:
                released = (
                    rotated_eff >= abs(pred_err) * 0.85 + self.dead_off
                    or (press_age > 0.45 and rotated < 3.0)
                    or small
                    or opposite
                    or wrong_way
                    or now >= self.imp_end
                )

            if released:
                if opposite or wrong_way:
                    # A target flip or a slide is not a finished correction:
                    # re-engage fast once the truck settles.
                    self.force_release(now, mh_t, self.micro_settle_t * settle_scale)
                elif mv_mps is not None and mv_mps > 0.5:
                    settle = min(self.settle_t, self.settle_dist_m / mv_mps)
                    self.force_release(now, mh_t, settle * settle_scale)
                else:
                    self.force_release(now, mh_t, self.settle_t * settle_scale)

        # Micro-corrections after settle
        if self.steer == 0 and now > self.settle_until and fresh:
            if err > self.dead_off:
                self.steer = 1
                self.steer_ph = 0
                self.micro = True
                self.micro_ticks = self._micro_ticks_for(err, urgency)
                self.hold_t0 = now
                self.settle_mh = mh_t
            elif err < -self.dead_off:
                self.steer = -1
                self.steer_ph = 0
                self.micro = True
                self.micro_ticks = self._micro_ticks_for(err, urgency)
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
