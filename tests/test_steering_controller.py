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
    def test_impulse_uses_model_yaw_rate(self):
        ctrl = SteeringController()
        expected = 30.0 * ctrl.imp_k / 90.0
        self.assertAlmostEqual(ctrl.calc_impulse(30.0, yaw_rate_max=90.0), expected, delta=1e-6)

    def test_impulse_falls_back_to_legacy_constant(self):
        ctrl = SteeringController()
        expected = min(ctrl.t_max, max(ctrl.t_min, 30.0 * ctrl.imp_k / ctrl.w_est))
        self.assertAlmostEqual(ctrl.calc_impulse(30.0), expected, delta=1e-6)

    def test_impulse_is_clamped(self):
        ctrl = SteeringController()
        self.assertAlmostEqual(ctrl.calc_impulse(1.0, yaw_rate_max=500.0), ctrl.t_min, delta=1e-9)
        self.assertAlmostEqual(ctrl.calc_impulse(500.0, yaw_rate_max=10.0), ctrl.t_max, delta=1e-9)

    def test_wrap180(self):
        self.assertAlmostEqual(wrap180(190.0), -170.0, delta=1e-9)
        self.assertAlmostEqual(wrap180(-190.0), 170.0, delta=1e-9)


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
