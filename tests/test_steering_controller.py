"""Tests for the steering controller: model-aware impulse durations."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


if __name__ == "__main__":
    unittest.main()
