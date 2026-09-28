"""Route-speed planning from curvature and braking capability.

The route is a polyline in map pixels; px_per_m converts it to meters. For
every vertex ahead the planner estimates the radius the vehicle will actually
drive and derives the speed that keeps lateral acceleration within budget:

    sharp vertex:  R = cut_m / (2*sin(dtheta/2))   (pure-pursuit corner cut)
    gentle curve:  R = arc_len / dtheta            (dense polyline samples)
    v_corner = sqrt(lat_accel * min(R_arc, R_cut))

The allowed speed now also accounts for braking needed to reach each
constraint:

    v_now = sqrt(v_corner^2 + 2*brake_decel*arc_ahead)

The final waypoint is planned as a v=0 constraint (smooth approach); the
result is clamped to the minimum corner speed and never exceeds the caller's
speed cap because the caller takes it as a minimum.
"""

from __future__ import annotations

import math

from .steering_controller import wrap180

G = 9.80665


class RouteSpeedPlanner:
    """Curvature and braking speed limits along the route, in meters."""

    def __init__(
        self,
        lat_accel_mps2: float,
        brake_decel_mps2: float,
        min_speed_kmh: float,
        lookahead_m: float = 200.0,
        cut_m: float = 15.0,
    ) -> None:
        self.lat_accel = max(0.1, float(lat_accel_mps2))
        self.brake_decel = max(0.1, float(brake_decel_mps2))
        self.min_speed_kmh = max(0.0, float(min_speed_kmh))
        self.lookahead_m = max(10.0, float(lookahead_m))
        self.cut_m = max(1.0, float(cut_m))

    def target_speed_kmh(
        self,
        mp: tuple[float, float],
        pts: list[tuple[float, float]],
        idx: int,
        px_per_m: float,
    ) -> float | None:
        """Planned speed (km/h), or None when nothing constrains the path."""
        if px_per_m <= 0.0 or not pts:
            return None
        idx = max(0, min(int(idx), len(pts) - 1))

        here = (float(mp[0]) / px_per_m, float(mp[1]) / px_per_m)
        vertices = [(float(x) / px_per_m, float(y) / px_per_m) for x, y in pts[idx:]]

        headings: list[float] = []
        distances: list[float] = []
        prev = here
        arc = 0.0
        for vx, vy in vertices:
            dx, dy = vx - prev[0], vy - prev[1]
            segment = math.hypot(dx, dy)
            arc += segment
            if segment > 1e-6:
                headings.append(math.degrees(math.atan2(dx, -dy)) % 360.0)
                distances.append(arc)
            prev = (vx, vy)

        target = float("inf")
        for k in range(1, len(headings)):
            corner_arc = distances[k - 1]
            if corner_arc > self.lookahead_m:
                break
            dtheta = math.radians(abs(wrap180(headings[k] - headings[k - 1])))
            if dtheta <= 1e-3:
                continue
            segment = distances[k] - corner_arc
            radius = min(
                segment / dtheta,
                self.cut_m / (2.0 * math.sin(dtheta / 2.0)),
            )
            v_corner = math.sqrt(self.lat_accel * radius)
            allowed = math.sqrt(v_corner * v_corner + 2.0 * self.brake_decel * corner_arc)
            target = min(target, allowed)

        if distances and distances[-1] <= self.lookahead_m:
            stop_allowed = math.sqrt(2.0 * self.brake_decel * max(distances[-1], 1e-6))
            target = min(target, stop_allowed)

        if not math.isfinite(target):
            return None
        return max(self.min_speed_kmh, target * 3.6)
