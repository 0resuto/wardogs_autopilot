"""Tests for the vehicle calibration fits over nav_dbg telemetry rows."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.navigation.calibration import (  # noqa: E402
    fit_brake,
    fit_px_per_m,
    fit_yaw_gain,
)


def row(**kwargs) -> dict:
    base = dict(
        t=0.0,
        good=True,
        ocr=None,
        mv=0.0,
        keys="",
        steer=0,
        yaw_max=None,
        th_raw=None,
        heading=0.0,
    )
    base.update(kwargs)
    return base


class TestPxPerMFit(unittest.TestCase):
    def test_median_ratio(self):
        rows = [row(t=i * 0.1, ocr=36.0, mv=20.0) for i in range(20)]

        fit = fit_px_per_m(rows)

        assert fit.value is not None
        self.assertAlmostEqual(fit.value, 2.0, delta=1e-6)
        self.assertEqual(fit.samples, 20)

    def test_requires_enough_samples(self):
        rows = [row(t=i * 0.1, ocr=36.0, mv=20.0) for i in range(5)]
        self.assertIsNone(fit_px_per_m(rows).value)

    def test_filters_braking_slow_and_unmeasured_rows(self):
        rows = [row(ocr=36.0, mv=20.0, keys="SPACE") for _ in range(10)]
        rows += [row(ocr=5.0, mv=2.0) for _ in range(10)]
        rows += [row(ocr=36.0, mv=20.0, good=False) for _ in range(10)]

        self.assertIsNone(fit_px_per_m(rows).value)


class TestBrakeFit(unittest.TestCase):
    def test_deceleration_from_speed_slope(self):
        rows = [row(t=i * 0.1, ocr=60.0 - 10.0 * (i * 0.1), keys="SPACE") for i in range(21)]

        fit = fit_brake(rows)

        assert fit.decel_mps2 is not None
        assert fit.measured_g is not None
        assert fit.suggested_g is not None
        self.assertAlmostEqual(fit.decel_mps2, 10.0 / 3.6, delta=1e-6)
        self.assertAlmostEqual(fit.measured_g, (10.0 / 3.6) / 9.80665, delta=1e-6)
        self.assertAlmostEqual(fit.suggested_g, fit.measured_g * 0.9, delta=1e-6)

    def test_flat_speed_is_not_a_brake_episode(self):
        rows = [row(t=i * 0.1, ocr=60.0, keys="SPACE") for i in range(20)]
        self.assertIsNone(fit_brake(rows).measured_g)

    def test_space_does_not_leak_into_steering(self):
        rows = [row(t=i * 0.1, ocr=60.0, keys="SPACE", steer=1, yaw_max=40.0) for i in range(20)]
        self.assertIsNone(fit_yaw_gain(rows, current_gain=1.0).suggested_gain)


class TestYawGainFit(unittest.TestCase):
    @staticmethod
    def _episodes(count: int) -> list[dict]:
        rows: list[dict] = []
        t = 0.0
        for _ in range(count):
            for _i in range(11):
                rows.append(row(t=t, th_raw=(t * 30.0) % 360.0, steer=1, keys="D", yaw_max=40.0))
                t += 0.05
            rows.append(row(t=t, steer=0, keys="W", yaw_max=0.0))
            t += 0.05
        return rows

    def test_ratio_scales_the_current_gain(self):
        fit = fit_yaw_gain(self._episodes(3), current_gain=1.2)

        assert fit.ratio is not None
        assert fit.suggested_gain is not None
        self.assertAlmostEqual(fit.ratio, 0.75, delta=1e-6)
        self.assertAlmostEqual(fit.suggested_gain, 0.9, delta=1e-6)
        self.assertEqual(fit.episodes, 3)

    def test_requires_three_episodes(self):
        fit = fit_yaw_gain(self._episodes(2), current_gain=1.0)
        self.assertIsNone(fit.suggested_gain)


if __name__ == "__main__":
    unittest.main()
