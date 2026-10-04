"""Tests for the steering controller: model-aware impulse durations."""

import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import autopilot.navigation.steering_controller as steering_mod  # noqa: E402
from autopilot.navigation.steering_controller import SteeringController, wrap180  # noqa: E402


class TestSteeringImpulse(unittest.TestCase):
    def test_impulse_is_independent_of_the_yaw_model(self):
        """The pulse length follows the reference law, not err / yaw_rate.

        The physical full-lock rate stretched every correction to t_max; the
        human re-taps short pulses and lets the next sample close the rest.
        """
        ctrl = SteeringController()

        self.assertAlmostEqual(ctrl.calc_impulse(10.0, 100.0), 0.14, delta=0.01)
        self.assertAlmostEqual(ctrl.calc_impulse(10.0, 5.0), 0.14, delta=0.01)
        self.assertAlmostEqual(ctrl.calc_impulse(10.0), 0.14, delta=0.01)

    def test_impulse_is_clamped(self):
        ctrl = SteeringController()
        self.assertAlmostEqual(ctrl.calc_impulse(1.0, yaw_rate_max=500.0), ctrl.t_min, delta=1e-9)
        self.assertAlmostEqual(ctrl.calc_impulse(500.0, yaw_rate_max=10.0), ctrl.t_max, delta=1e-9)

    def test_wrap180(self):
        self.assertAlmostEqual(wrap180(190.0), -170.0, delta=1e-9)
        self.assertAlmostEqual(wrap180(-190.0), 170.0, delta=1e-9)

    def test_impulse_matches_the_reference_manual_drive(self):
        """Tap lengths follow the human reference at 79 km/h (Ural).

        The reference manual drive (output/manual_dbg_20260928_222519.jsonl)
        taps ~65 ms at 3 deg of error, ~125 ms at 10 deg and ~250 ms at
        20 deg; the old 0.7 gain saturated every correction at the 0.5 s cap
        and the duty was 2.4x the human's.
        """
        ctrl = SteeringController()
        yaw = 10.2  # Ural grip-limited yaw rate at 79 km/h

        self.assertAlmostEqual(ctrl.calc_impulse(3.0, yaw), 0.065, delta=0.02)
        self.assertAlmostEqual(ctrl.calc_impulse(10.0, yaw), 0.125, delta=0.03)
        self.assertAlmostEqual(ctrl.calc_impulse(20.0, yaw), 0.25, delta=0.05)
        self.assertLessEqual(ctrl.calc_impulse(60.0, yaw), ctrl.t_max)

    def test_hold_timeout_is_human_scale(self):
        ctrl = SteeringController()

        self.assertLessEqual(ctrl.hold_max, 2.0)


class TestBrakingYawAuthority(unittest.TestCase):
    """Weight transfer sharpens the game's yaw response while braking (SPACE).

    Measured at 60-90 km/h: p90 yaw 7.4 deg/s coasting vs 28.1 deg/s braking;
    the old clamp (1.4 * yaw_rate_max ~ 15-18 deg/s) rejected the real
    rotation and the release anticipation lagged behind the truck.
    """

    @staticmethod
    def _ang_after(braking: bool) -> float:
        ctrl = SteeringController()
        ctrl.step(
            now=100.0,
            err=10.0,
            heading=0.0,
            mh=0.0,
            mh_t=1.0,
            yaw_rate_max=10.0,
            heading_meas=0.0,
            braking=braking,
        )
        ctrl.step(
            now=100.05,
            err=10.0,
            heading=1.5,
            mh=0.0,
            mh_t=1.0,
            yaw_rate_max=10.0,
            heading_meas=1.5,
            braking=braking,
        )
        return ctrl.ang

    def test_braking_raises_the_yaw_clamp(self):
        self.assertGreater(self._ang_after(True), self._ang_after(False))

    def test_braking_shortens_the_impulse(self):
        with patch.object(steering_mod.random, "uniform", return_value=1.0):
            coast = SteeringController()
            coast.step(now=100.0, err=20.0, heading=0.0, mh=0.0, mh_t=1.0)
            brake = SteeringController()
            brake.step(now=100.0, err=20.0, heading=0.0, mh=0.0, mh_t=1.0, braking=True)

        self.assertAlmostEqual(coast.imp_end - 100.0, 0.25, delta=0.01)
        self.assertAlmostEqual(brake.imp_end - 100.0, 0.175, delta=0.01)


class TestUrgencyModulation(unittest.TestCase):
    """The off-centre urgency continuously scales amplitude and cadence."""

    def test_off_center_extends_the_impulse(self):
        with patch.object(steering_mod.random, "uniform", return_value=1.0):
            centered = SteeringController()
            centered.step(now=100.0, err=10.0, heading=0.0, mh=0.0, mh_t=1.0)
            urgent = SteeringController()
            urgent.step(now=100.0, err=10.0, heading=0.0, mh=0.0, mh_t=1.0, center_urgency=1.0)

        self.assertAlmostEqual(centered.imp_end - 100.0, 0.14, delta=0.01)
        self.assertAlmostEqual(urgent.imp_end - 100.0, 0.28, delta=0.02)

    def test_off_center_adds_micro_ticks(self):
        with patch.object(steering_mod.random, "choice", side_effect=lambda seq: max(seq)):
            centered = SteeringController(pulse_on=4)
            centered.step(now=100.0, err=4.0, heading=0.0, mh=0.0, mh_t=1.0)
            urgent = SteeringController(pulse_on=4)
            urgent.step(now=100.0, err=4.0, heading=0.0, mh=0.0, mh_t=1.0, center_urgency=1.0)

        self.assertGreater(urgent.micro_ticks, centered.micro_ticks)

    def test_recovery_pins_the_wheel_for_a_u_turn(self):
        ctrl = SteeringController()
        ctrl.step(now=100.0, err=150.0, heading=0.0, mh=0.0, mh_t=1.0, recovery=True)

        self.assertTrue(ctrl.hold)
        self.assertAlmostEqual(
            ctrl.imp_end - 100.0,
            ctrl.hold_max * steering_mod.RECOVERY_HOLD_SCALE,
            delta=1e-6,
        )


class TestWrongWayRelease(unittest.TestCase):
    """A truck sliding against the locked wheel must not keep the lock pinned."""

    def test_wheel_releases_when_the_truck_rotates_against_the_command(self):
        ctrl = SteeringController()
        ctrl.step(now=100.0, err=40.0, heading=0.0, mh=0.0, mh_t=1.0)
        ctrl.step(now=100.05, err=40.0, heading=-1.0, mh=0.0, mh_t=1.0)
        self.assertEqual(ctrl.steer, 1)

        out = ctrl.step(now=100.40, err=40.0, heading=-20.0, mh=0.0, mh_t=1.0)

        self.assertEqual(out, 0)
        self.assertEqual(ctrl.steer, 0)

    def test_rotation_with_the_command_keeps_the_lock(self):
        ctrl = SteeringController()
        ctrl.step(now=100.0, err=40.0, heading=0.0, mh=0.0, mh_t=1.0)
        ctrl.step(now=100.05, err=40.0, heading=1.0, mh=0.0, mh_t=1.0)
        self.assertTrue(ctrl.hold)

        # a few degrees of rotation in the commanded direction: not a slide
        out = ctrl.step(now=100.40, err=40.0, heading=2.0, mh=0.0, mh_t=1.0)

        self.assertEqual(out, 1)


class TestOppositeTargetRelease(unittest.TestCase):
    def test_hold_releases_when_target_flips(self):
        ctrl = SteeringController(turn_deg=25.0)

        ctrl.step(now=100.0, err=50.0, heading=0.0, mh=0.0, mh_t=1.0)
        steer = ctrl.step(now=100.05, err=50.0, heading=1.0, mh=0.0, mh_t=1.0)
        self.assertEqual(steer, 1)
        self.assertTrue(ctrl.hold)

        steer = ctrl.step(now=100.1, err=-30.0, heading=3.0, mh=0.0, mh_t=1.0)

        self.assertEqual(steer, 0)
        self.assertEqual(ctrl.steer, 0)

    def test_hold_keeps_turning_while_target_stays_on_the_same_side(self):
        ctrl = SteeringController(turn_deg=25.0)

        ctrl.step(now=100.0, err=50.0, heading=0.0, mh=0.0, mh_t=1.0)
        ctrl.step(now=100.05, err=50.0, heading=1.0, mh=0.0, mh_t=1.0)

        steer = ctrl.step(now=100.1, err=30.0, heading=3.0, mh=0.0, mh_t=1.0)

        self.assertEqual(steer, 1)
        self.assertEqual(ctrl.steer, 1)


class TestReleaseAnticipation(unittest.TestCase):
    @staticmethod
    def _release_heading(ant_s: float) -> float:
        ctrl = SteeringController(turn_deg=25.0, ant_s=ant_s)
        t = 100.0
        hd = 0.0
        ctrl.step(now=t, err=50.0, heading=hd, mh=0.0, mh_t=1.0)
        t += 0.05
        hd = 2.0
        ctrl.step(now=t, err=48.0, heading=hd, mh=0.0, mh_t=1.0)
        for _ in range(40):
            t += 0.05
            hd += 2.0  # steady ~40 deg/s yaw, the model-limited rate
            if ctrl.step(now=t, err=50.0 - hd, heading=hd, mh=0.0, mh_t=1.0) == 0:
                return hd
        return hd

    def test_release_anticipates_the_yaw_already_underway(self):
        self.assertLess(self._release_heading(0.25), self._release_heading(0.0))

    def test_zero_anticipation_matches_the_old_release(self):
        self.assertAlmostEqual(self._release_heading(0.0), 24.0, delta=3.0)


class TestSampleDebounce(unittest.TestCase):
    def test_hold_requires_two_fresh_samples_not_ticks(self):
        ctrl = SteeringController(turn_deg=25.0)

        ctrl.step(now=100.0, err=50.0, heading=0.0, mh=0.0, mh_t=1.0, fresh_sample=True)
        ctrl.step(now=100.05, err=50.0, heading=1.0, mh=0.0, mh_t=1.0, fresh_sample=False)
        self.assertFalse(ctrl.hold)

        ctrl.step(now=100.10, err=50.0, heading=2.0, mh=0.0, mh_t=1.1, fresh_sample=True)
        self.assertTrue(ctrl.hold)


class TestStalePose(unittest.TestCase):
    def test_stale_pose_releases_the_wheel(self):
        ctrl = SteeringController(turn_deg=25.0)
        ctrl.step(now=100.0, err=50.0, heading=0.0, mh=0.0, mh_t=1.0)
        ctrl.step(now=100.05, err=50.0, heading=1.0, mh=0.0, mh_t=1.0)
        self.assertEqual(ctrl.steer, 1)

        out = ctrl.step(now=100.10, err=50.0, heading=2.0, mh=0.0, mh_t=1.0, pose_age=0.9)

        self.assertEqual(out, 0)
        self.assertEqual(ctrl.steer, 0)

    def test_yaw_estimate_ignores_unphysical_steps(self):
        ctrl = SteeringController()
        ctrl.update_angular_velocity(100.0, 0.0, max_rate=45.0)
        ctrl.update_angular_velocity(100.05, 90.0, max_rate=45.0)  # glitch step

        self.assertLessEqual(abs(ctrl.ang), 45.0)


class TestMicroPulseTicks(unittest.TestCase):
    def test_all_ticks_of_a_micro_pulse_are_pressed(self):
        ctrl = SteeringController(pulse_on=3, micro_settle_t=0.3)
        with patch.object(steering_mod.random, "choice", return_value=3):
            outs = [
                ctrl.step(now=100.0 + i * 0.043, err=4.0, heading=0.0, mh=0.0, mh_t=1.0)
                for i in range(5)
            ]

        # the release zeroes the command but the last tick must still press
        self.assertEqual(outs[:3], [1, 1, 1])
        self.assertEqual(outs[3:], [0, 0])

    def test_yaw_estimate_prefers_the_raw_measured_heading(self):
        ctrl = SteeringController()
        ctrl.step(now=100.0, err=30.0, heading=0.0, mh=0.0, mh_t=1.0, heading_meas=0.0)
        ctrl.step(now=100.05, err=30.0, heading=0.5, mh=0.0, mh_t=1.0, heading_meas=5.0)

        # raw signal: 5 deg / 0.05 s = 100 deg/s clamped to 1.4 * 45
        self.assertGreater(ctrl.ang, 10.0)


class TestSettleWindows(unittest.TestCase):
    def test_micro_tap_release_uses_short_settle(self):
        ctrl = SteeringController(settle_t=0.8, micro_settle_t=0.3)
        with patch.object(steering_mod.random, "choice", return_value=1):
            ctrl.step(now=100.0, err=4.0, heading=0.0, mh=0.0, mh_t=1.0)

        self.assertEqual(ctrl.steer, 0)
        self.assertAlmostEqual(ctrl.settle_until, 100.3, delta=1e-6)

    def test_opposite_release_uses_short_settle(self):
        ctrl = SteeringController(settle_t=0.8, micro_settle_t=0.3, turn_deg=25.0)
        ctrl.step(now=100.0, err=50.0, heading=0.0, mh=0.0, mh_t=1.0)
        ctrl.step(now=100.05, err=50.0, heading=1.0, mh=0.0, mh_t=1.0)
        ctrl.step(now=100.1, err=-30.0, heading=3.0, mh=0.0, mh_t=1.0)

        self.assertEqual(ctrl.steer, 0)
        self.assertAlmostEqual(ctrl.settle_until, 100.4, delta=1e-6)

    def test_completed_hold_release_keeps_full_settle(self):
        ctrl = SteeringController(settle_t=0.8, micro_settle_t=0.3, turn_deg=25.0)
        ctrl.step(now=100.0, err=50.0, heading=0.0, mh=0.0, mh_t=1.0)
        ctrl.step(now=100.05, err=50.0, heading=1.0, mh=0.0, mh_t=1.0)
        # rotated 20 deg >= 0.8 * min(20, 50) + 2 = 18: finished correction
        ctrl.step(now=100.1, err=20.0, heading=20.0, mh=0.0, mh_t=1.0)

        self.assertEqual(ctrl.steer, 0)
        self.assertAlmostEqual(ctrl.settle_until, 100.1 + 0.8, delta=1e-6)


class TestInputJitter(unittest.TestCase):
    def test_impulse_duration_is_jittered(self):
        ctrl = SteeringController(jitter=0.1)
        with patch.object(steering_mod.random, "uniform", return_value=1.1) as rnd:
            ctrl.step(now=100.0, err=20.0, heading=0.0, mh=0.0, mh_t=1.0, yaw_rate_max=100.0)

        self.assertEqual(len(rnd.call_args_list), 1)
        low, high = rnd.call_args.args
        self.assertAlmostEqual(low, 0.9, delta=1e-9)
        self.assertAlmostEqual(high, 1.1, delta=1e-9)

        base = min(ctrl.t_max, ctrl.calc_impulse(20.0, 100.0))
        self.assertAlmostEqual(ctrl.imp_end, 100.0 + base * 1.1, delta=1e-6)

    def test_micro_pulse_length_varies(self):
        ctrl = SteeringController(pulse_on=2)
        with patch.object(steering_mod.random, "choice", return_value=3):
            ctrl.step(now=100.0, err=4.0, heading=0.0, mh=0.0, mh_t=1.0)

        self.assertEqual(ctrl.micro_ticks, 3)

    def test_hold_mode_keeps_fixed_safety_timeout(self):
        ctrl = SteeringController()
        ctrl.hold = True
        with patch.object(steering_mod.random, "uniform") as rnd:
            end = ctrl._impulse_end(now=10.0, err=40.0, yaw_rate_max=90.0)

        rnd.assert_not_called()
        self.assertAlmostEqual(end, 10.0 + ctrl.hold_max, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
