"""Path geometry, waypoint tracking, and trajectory calculations."""

from __future__ import annotations

import math
import time

from .steering_controller import wrap180


class PathTracker:
    """Tracks position relative to waypoints, cross-track error, and lookahead angles."""

    def __init__(
        self,
        pts: list[tuple[float, float]],
        arrive_r: float = 55.0,
        xte_m: float = 6.0,
        xte_outer_m: float | None = None,
        steer_look_s: float = 1.6,
        skip_ahead_m: float = 150.0,
        reacquire_gate_deg: float = 60.0,
        max_extra: float = 0.3,
        mh_dt: float = 0.5,
        mh_min_d: float = 10.0,
        mh_max_d: float = 30.0,
    ) -> None:
        self.pts = list(pts)
        self.idx = 0
        self.arrive_r = float(arrive_r)
        self.xte_m = float(xte_m)
        self.xte_outer_m = float(xte_outer_m if xte_outer_m is not None else 3.0 * xte_m)
        self.steer_look_s = float(steer_look_s)
        self.skip_ahead_m = max(0.0, float(skip_ahead_m))
        self.reacquire_gate_deg = max(10.0, min(120.0, float(reacquire_gate_deg)))
        self.max_extra = float(max_extra)
        # Cumulative route arc (pts never change after construction).
        self._cum = [0.0]
        for i in range(1, len(self.pts)):
            self._cum.append(
                self._cum[-1]
                + math.hypot(
                    self.pts[i][0] - self.pts[i - 1][0],
                    self.pts[i][1] - self.pts[i - 1][1],
                )
            )
        self._last_reacquire_t = -1e9

        self._samples: list[tuple[float, float, float]] = []  # (ts, x, y)
        self._mh: float | None = None
        self._mh_t = 0.0
        self._mv = 0.0
        self._mh_dt = float(mh_dt)
        self._mh_min_d = float(mh_min_d)
        self._mh_max_d = float(mh_max_d)

    @property
    def mv(self) -> float:
        return self._mv

    @property
    def mh(self) -> float | None:
        return self._mh

    @property
    def mh_t(self) -> float:
        return self._mh_t

    @property
    def samples(self) -> list[tuple[float, float, float]]:
        return self._samples

    def clean_stale_samples(self, now: float, max_age: float = 1.5) -> bool:
        """Prune samples older than max_age. Returns True if any sample remains."""
        if not self._samples:
            return False
        if now - self._samples[-1][0] > max_age:
            self._samples = [s for s in self._samples if now - s[0] <= max_age]
        return bool(self._samples)

    def lost_limit(self) -> float:
        """Max plausible speed for ghost-pose detection (px/s)."""
        return max(self._mv * 2.0 + 8.0, 50.0)

    def push_pose(self, ts: float, x: float, y: float) -> bool:
        """Append a fresh pose sample and update motion-derived course (M-heading)."""
        if self._samples and abs(ts - self._samples[-1][0]) < 1e-6:
            return False

        if self._samples:
            base = ts - self._mh_dt
            bx, by, bt = self._samples[0][1], self._samples[0][2], self._samples[0][0]
            for s in self._samples:
                if s[0] <= base:
                    bt, bx, by = s[0], s[1], s[2]
                else:
                    break
            dts = ts - bt
            d = math.hypot(x - bx, y - by)
            if dts > 1e-3:
                self._mv = d / dts
                self._mh_t = ts
                if d >= self._mh_min_d:
                    raw = math.degrees(math.atan2(x - bx, -(y - by))) % 360.0
                    if self._mh is None:
                        self._mh = raw
                    else:
                        dd = wrap180(raw - self._mh)
                        dd = max(-self._mh_max_d, min(self._mh_max_d, dd))
                        self._mh = (self._mh + dd * 0.5) % 360.0

        self._samples.append((ts, x, y))
        if len(self._samples) > 16:
            self._samples = self._samples[-16:]
        return True

    def pose_at(self, now: float) -> tuple[float, float] | None:
        """Estimate current position via linear interpolation/extrapolation."""
        n = len(self._samples)
        if n == 0:
            return None
        if n == 1:
            return (self._samples[0][1], self._samples[0][2])

        t0, x0, y0 = self._samples[-2]
        t1, x1, y1 = self._samples[-1]
        dt = t1 - t0
        if dt <= 1e-4:
            return (x1, y1)

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        d = now - t1
        if d < 0:
            k = (now - t0) / dt
            return (x0 + (x1 - x0) * k, y0 + (y1 - y0) * k)
        if d <= self.max_extra:
            return (x1 + vx * d, y1 + vy * d)
        return (x1, y1)

    def _seg_dir(self, i: int) -> float | None:
        """Bearing of segment [i, i+1], or None for a degenerate segment."""
        if i < 0 or i + 1 >= len(self.pts):
            return None
        sx = self.pts[i + 1][0] - self.pts[i][0]
        sy = self.pts[i + 1][1] - self.pts[i][1]
        if abs(sx) + abs(sy) <= 1e-9:
            return None
        return math.degrees(math.atan2(sx, -sy)) % 360.0

    def snap_to_nearest(
        self,
        mp: tuple[float, float],
        course_deg: float | None = None,
    ) -> int:
        """Entry point of the route at engage time: the nearest remaining point.

        Called once when the driver starts, so pressing F6 mid-route continues
        forward instead of always turning back to pts[0]. Strict in-order
        consumption (the out-and-back protection) still applies afterwards.
        A point already crossed is skipped using the same `_passed` rule.

        With a known course the entry refuses a point whose local route
        direction is more than `reacquire_gate_deg` off: on out-and-back
        routes the return leg lies at the same place and must not be picked
        (the car would U-turn at engage).
        """
        if not self.pts:
            return self.idx
        dists = [math.hypot(px - mp[0], py - mp[1]) for px, py in self.pts]
        nearest = min(range(len(self.pts)), key=lambda i: dists[i])
        best = nearest
        if course_deg is not None:
            near = max(dists[nearest] * 1.5, self.arrive_r * 2.0)
            candidates = []
            for i in range(len(self.pts) - 1):
                if dists[i] > near:
                    continue
                seg_dir = self._seg_dir(i)
                if seg_dir is None:
                    continue
                if abs(wrap180(seg_dir - course_deg)) > self.reacquire_gate_deg:
                    continue
                candidates.append(i)
            if candidates:
                best = min(candidates, key=lambda i: (round(dists[i], 1), i))
        self.idx = best
        while self.idx < len(self.pts) - 1 and self._passed(mp):
            self.idx += 1
        return self.idx

    def maybe_reacquire(
        self,
        mp: tuple[float, float],
        px_per_m: float,
        course_deg: float | None = None,
        now: float | None = None,
    ) -> bool:
        """Forward-only route re-acquisition after a big excursion.

        The active waypoint is normally consumed strictly in order. If the car
        ends up outside the outer corridor (spin, knock-out, missed jump), the
        index can trail far behind; this jumps it forward to the nearest route
        point whose segment is inside `skip_ahead_m` of route length, the car
        is close to it and driving along the route there (course gate). It
        never jumps to the final point and never backwards. The window plus
        the course gate keep out-and-back and self-overlapping routes from
        being cut, and the outside-the-corridor trigger leaves normal tracking
        untouched.
        """
        now = time.time() if now is None else now
        if self.skip_ahead_m <= 0.0 or px_per_m <= 0.0 or len(self.pts) < 3:
            return False
        if now - self._last_reacquire_t < 1.0 or course_deg is None:
            return False

        # Trigger only when the car is genuinely off the active segment.
        si = min(max(self.idx, 1), len(self.pts) - 1)
        ax, ay = self.pts[si - 1]
        sx = self.pts[si][0] - ax
        sy = self.pts[si][1] - ay
        l2 = sx * sx + sy * sy
        if l2 > 1e-6:
            xte = ((mp[0] - ax) * sy - (mp[1] - ay) * sx) / math.sqrt(l2)
            if abs(xte) <= self.xte_outer_m * px_per_m:
                return False

        limit = self._cum[self.idx] + self.skip_ahead_m * px_per_m
        best: int | None = None
        best_d = self.arrive_r * 4.0
        for k in range(self.idx + 1, len(self.pts) - 1):
            if self._cum[k] > limit:
                break
            ax, ay = self.pts[k - 1]
            sx = self.pts[k][0] - ax
            sy = self.pts[k][1] - ay
            l2 = sx * sx + sy * sy
            if l2 <= 1e-6:
                continue
            t = ((mp[0] - ax) * sx + (mp[1] - ay) * sy) / l2
            if not 0.0 <= t <= 1.0:
                continue
            d = abs((mp[0] - ax) * sy - (mp[1] - ay) * sx) / math.sqrt(l2)
            if d >= best_d:
                continue
            seg_dir = math.degrees(math.atan2(sx, -sy)) % 360.0
            if abs(wrap180(seg_dir - course_deg)) > self.reacquire_gate_deg:
                continue
            best, best_d = k, d
        if best is None:
            return False
        self.idx = best
        self._last_reacquire_t = now
        return True

    def _passed(self, mp: tuple[float, float]) -> bool:
        """True when the vehicle crossed the perpendicular through the current
        waypoint along the route, without wandering far off the corridor.

        At idx=0 there is no incoming segment, so the outgoing one is used with
        the perpendicular through pts[0] itself. Without this special case a
        run entering at idx=0 could never advance when it passed the start more
        than arrive_r away from it: idx stayed 0 for the whole run and every
        xte was measured against the first chord (observed as the car chasing
        a phantom and grinding along the terrain).
        """
        if self.idx == 0:
            if len(self.pts) < 2:
                return False
            ax, ay = self.pts[0]
            bx, by = self.pts[1]
            t_min = 0.0
        elif self.idx > 0:
            ax, ay = self.pts[self.idx - 1]
            bx, by = self.pts[self.idx]
            t_min = 1.0
        else:
            return False
        sx, sy = bx - ax, by - ay
        l2 = sx * sx + sy * sy
        if l2 <= 1e-6:
            return False
        t = ((mp[0] - ax) * sx + (mp[1] - ay) * sy) / l2
        if t < t_min:
            return False
        lateral = abs((mp[0] - ax) * sy - (mp[1] - ay) * sx) / math.sqrt(l2)
        return lateral <= self.arrive_r * 4.0

    def advance_waypoint(self, mp: tuple[float, float]) -> tuple[float, float, float, bool]:
        """Advance to the first waypoint that is neither reached nor passed.

        Waypoints are consumed strictly in order: picking the globally nearest
        remaining point used to cut out-and-back routes. The current waypoint
        is consumed when the vehicle is inside its arrival radius or has
        already crossed the perpendicular through it along the route (so a
        missed sample does not send the vehicle back). Returns
        (target_x, target_y, distance, arrived_at_end).
        """
        if self.idx >= len(self.pts):
            return 0.0, 0.0, 0.0, True

        while self.idx < len(self.pts):
            tx, ty = self.pts[self.idx]
            dist = math.hypot(tx - mp[0], ty - mp[1])
            if dist < self.arrive_r or self._passed(mp):
                self.idx += 1
                continue
            return tx, ty, dist, False

        tx, ty = self.pts[-1]
        return tx, ty, math.hypot(tx - mp[0], ty - mp[1]), True

    def calc_xte_and_bearing(
        self,
        mp: tuple[float, float],
        px_per_m: float,
    ) -> tuple[float, float, float]:
        """Calculate cross-track error, corridor limit, and pure pursuit target bearing.

        The aim point is the point on the polyline at arc distance `look` ahead
        of the car's projection, walked across as many segments as needed:
        dense taught lines (short chords) get the same lookahead as sparse
        drawn routes instead of aiming at the next vertex only.
        """
        # Lookahead in seconds of travel: a fixed 60 + 2.6*mv was ~4 s at
        # speed, which cut 4-6 m inside long curves and made the driver fight
        # the corridor; 1.0-2.0 s keeps a tight line. The floor is a distance
        # (8 m), not 40 px: at low speed 40 px was ~4 s and aimed past the
        # apex of tight corners.
        look = max(8.0 * px_per_m, min(240.0, self.steer_look_s * self._mv))
        if len(self.pts) < 2:
            pt = self.pts[0] if self.pts else (0.0, 0.0)
            bearing = math.degrees(math.atan2(pt[0] - mp[0], -(pt[1] - mp[1]))) % 360.0
            return 0.0, self.xte_m * px_per_m, bearing

        si = min(max(self.idx, 1), len(self.pts) - 1)
        ax3, ay3 = self.pts[si - 1]
        bx3, by3 = self.pts[si]
        sx3 = bx3 - ax3
        sy3 = by3 - ay3
        l2 = sx3 * sx3 + sy3 * sy3

        xte = 0.0
        if l2 > 1e-6:
            xte = ((mp[0] - ax3) * sy3 - (mp[1] - ay3) * sx3) / math.sqrt(l2)
            foot_t = ((mp[0] - ax3) * sx3 + (mp[1] - ay3) * sy3) / l2
            foot = (
                ax3 + min(max(foot_t, 0.0), 1.0) * sx3,
                ay3 + min(max(foot_t, 0.0), 1.0) * sy3,
            )
        else:
            foot = (bx3, by3)

        xte_lim = self.xte_m * px_per_m

        # Pure pursuit only. The old corridor mode aimed at the perpendicular
        # foot of the active segment once |xte| passed xte_lim, which pointed
        # the wheel up to ~90 deg across the route and spun the truck out; the
        # lookahead point corrects the cross-track error gradually
        # (correction ~ atan(xte / look)) and keeps the car near the centre.
        remaining = look
        cur = foot
        seg_dir: float | None = None
        ax2, ay2 = self.pts[-1]
        for i in range(si, len(self.pts)):
            nx, ny = self.pts[i]
            seg_len = math.hypot(nx - cur[0], ny - cur[1])
            if seg_len > 1e-9:
                seg_dir = math.degrees(math.atan2(nx - cur[0], -(ny - cur[1]))) % 360.0
            if seg_len >= remaining:
                k = remaining / seg_len if seg_len > 1e-9 else 0.0
                ax2 = cur[0] + k * (nx - cur[0])
                ay2 = cur[1] + k * (ny - cur[1])
                break
            remaining -= seg_len
            cur = (nx, ny)

        bearing = math.degrees(math.atan2(ax2 - mp[0], -(ay2 - mp[1]))) % 360.0
        if seg_dir is not None:
            bearing = self._corridor_correction(bearing, seg_dir, xte, px_per_m, look)
        return xte, xte_lim, bearing

    def _corridor_correction(
        self,
        bearing: float,
        seg_dir: float,
        xte: float,
        px_per_m: float,
        look: float,
    ) -> float:
        """Corridor cross-track guidance for the pure-pursuit bearing.

        The aim deviation relative to the segment direction mixes the
        cross-track part with the curvature lead over the lookahead; only the
        cross-track part is amplified (multiplying the lead commanded absurd
        headings in bends). The gain ramps continuously from the inner edge to
        the outer edge - the old hard steps let the loop ping-pong across the
        boundary - and the result stays clamped short of perpendicular. The
        speed side of the outer corridor is handled by the driver.
        """
        corr = wrap180(bearing - seg_dir)
        xte_m = abs(xte) / px_per_m if px_per_m > 0 else 0.0
        span = max(self.xte_outer_m - self.xte_m, 0.1)
        frac = max(0.0, min(1.0, (xte_m - self.xte_m) / span))
        gain = 1.0 + 1.2 * frac
        cap = 20.0 + 15.0 * frac
        corr_xte = math.degrees(math.atan2(xte, max(look, 1.0)))
        corr += (gain - 1.0) * corr_xte
        corr = max(-cap, min(cap, corr))
        return (seg_dir + corr) % 360.0

    def calc_road_turn(self, heading: float) -> tuple[float, float]:
        """Compute immediate turn angle and lookahead road turn curvature."""
        if self.idx + 1 < len(self.pts):
            out_h = (
                math.degrees(
                    math.atan2(
                        self.pts[self.idx + 1][0] - self.pts[self.idx][0],
                        -(self.pts[self.idx + 1][1] - self.pts[self.idx][1]),
                    )
                )
                % 360.0
            )
            if self.idx > 0:
                in_h = (
                    math.degrees(
                        math.atan2(
                            self.pts[self.idx][0] - self.pts[self.idx - 1][0],
                            -(self.pts[self.idx][1] - self.pts[self.idx - 1][1]),
                        )
                    )
                    % 360.0
                )
            else:
                in_h = heading
            turn_angle = abs(wrap180(out_h - in_h))
        else:
            in_h = heading
            turn_angle = 0.0

        ahead_t = 0.0
        acc = 0.0
        h_prev = in_h
        for i in range(self.idx + 1, len(self.pts)):
            h_next = (
                math.degrees(
                    math.atan2(
                        self.pts[i][0] - self.pts[i - 1][0],
                        -(self.pts[i][1] - self.pts[i - 1][1]),
                    )
                )
                % 360.0
            )
            ahead_t = max(ahead_t, abs(wrap180(h_next - h_prev)))
            h_prev = h_next
            acc += math.hypot(
                self.pts[i][0] - self.pts[i - 1][0],
                self.pts[i][1] - self.pts[i - 1][1],
            )
            if acc >= 200.0 or i >= self.idx + 6:
                break

        road_turn = max(turn_angle, ahead_t)
        return turn_angle, road_turn
