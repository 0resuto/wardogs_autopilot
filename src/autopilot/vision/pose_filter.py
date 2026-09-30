"""Motion-compensated low-pass filter for measured player poses.

The SIFT + affine matcher has a per-frame noise floor of a few map pixels
(~1-2 m on the 0.5 m/px maps): the published marker visibly jitters even while
the vehicle drives smoothly. This is the alpha-beta filter that sits between
the matcher and the consumers (`LiveLocator.latest`).

The state is (position, velocity): each measurement first advances the
prediction by the last velocity and only the residual is blended in, so a
steady motion passes at ~zero lag while random noise is attenuated (position
std ~ sqrt(alpha / (2 - alpha)) of the raw match). `beta = alpha**2 / (2 -
alpha)` is the Benedict-Bordner choice: the position lag of a constant-velocity
target is zero in steady state.

Confirmed relocations and long measurement gaps reset the state instead of
gliding toward the new place: an excursion, a capture void or a vote-approved
re-acquisition must show up immediately.
"""

from __future__ import annotations

import math

#: Measurement gap (s) that restarts the filter: hold frames and vote-rejected
#: relocations must not be extrapolated with a stale velocity.
RESET_GAP_S = 1.0


class PoseSmoother:
    """Alpha-beta position filter; `alpha=0` disables smoothing (raw pass-through)."""

    def __init__(self, alpha: float = 0.5, reset_px: float = 100.0) -> None:
        self.alpha = 0.0
        self.beta = 0.0
        self.reset_px = 0.0
        self.configure(alpha, reset_px)
        self._has = False
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._t = 0.0

    def configure(self, alpha: float, reset_px: float) -> None:
        """Apply live tuning; alpha is clamped to [0, 1]."""
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.beta = self.alpha * self.alpha / (2.0 - self.alpha) if self.alpha > 0.0 else 0.0
        self.reset_px = max(0.0, float(reset_px))

    def reset(self) -> None:
        """Forget the state: the next measurement is published as measured."""
        self._has = False
        self._vx = 0.0
        self._vy = 0.0

    def update(self, x: float, y: float, t: float) -> tuple[float, float]:
        """Feed one measured pose; returns the smoothed position to publish."""
        x, y, t = float(x), float(y), float(t)
        if self.alpha <= 0.0:
            return x, y
        if not self._has:
            self._init(x, y, t)
            return x, y

        dt = t - self._t
        if dt <= 1e-6 or dt > RESET_GAP_S:
            self._init(x, y, t)
            return x, y

        pred_x = self._x + self._vx * dt
        pred_y = self._y + self._vy * dt
        rx, ry = x - pred_x, y - pred_y
        if self.reset_px > 0.0 and math.hypot(rx, ry) > self.reset_px:
            self._init(x, y, t)
            return x, y

        self._x = pred_x + self.alpha * rx
        self._y = pred_y + self.alpha * ry
        self._vx += self.beta * rx / dt
        self._vy += self.beta * ry / dt
        self._t = t
        return self._x, self._y

    def _init(self, x: float, y: float, t: float) -> None:
        self._x = x
        self._y = y
        self._t = t
        self._vx = 0.0
        self._vy = 0.0
        self._has = True
