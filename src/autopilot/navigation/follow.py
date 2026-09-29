"""Autopilot route follower thread for WARDOGS.

Coordinates localization poses, route tracking, speed control, steering impulses,
and Arduino keyboard driver key injection.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Sequence
from typing import Any

from ..common.config import NavigatorConfig
from ..common.log import get_logger
from .path_tracker import PathTracker
from .speed_controller import SpeedController
from .speed_profile import G, RouteSpeedPlanner
from .steering_controller import SteeringController, wrap180
from .stop_logic import FinalStopMixin
from .telemetry import NavTelemetryLogger
from .vehicle_model import VehicleModel

logger = get_logger("follow")


class FollowDriver(FinalStopMixin, threading.Thread):
    """Route-following background driver thread.

    Takes a LiveLocator and a list of route points, computes target bearing
    and speed, steers via pulse impulses, and injects keys into the key driver.
    """

    def __init__(
        self,
        loc: Any,
        pts: Sequence[tuple[float, float]],
        arrive_r: float | None = None,
        dead: float | None = None,
        poll: float | None = None,
        kb: Any = None,
        brake_d: float | None = None,
        dead_off: float | None = None,
        speed_cap_kmh: float | None = None,
        xte_m: float | None = None,
        debug: bool | None = None,
        turn_deg: float | None = None,
        hold_max: float | None = None,
        stop_speed_kmh: float | None = None,
        stop_min_px_s: float | None = None,
        stop_confirm_s: float | None = None,
        stop_hold: float | None = None,
        stop_timeout: float | None = None,
        vehicle_model: Any = None,
        px_per_m: float | None = None,
        nav_cfg: NavigatorConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        if isinstance(nav_cfg, dict):
            try:
                self.nav_cfg = NavigatorConfig(**nav_cfg)
            except Exception:
                self.nav_cfg = NavigatorConfig()
        elif isinstance(nav_cfg, NavigatorConfig):
            self.nav_cfg = nav_cfg
        else:
            self.nav_cfg = NavigatorConfig()

        self.loc = loc
        self.pts = list(pts)
        self.arrive_r = float(arrive_r if arrive_r is not None else self.nav_cfg.arrive_r)
        base_dead = dead if dead is not None else self.nav_cfg.dead
        self.dead = max(float(base_dead), 6.0)
        self.poll = float(poll if poll is not None else self.nav_cfg.poll)
        self.brake_d = float(brake_d if brake_d is not None else self.nav_cfg.brake_d)
        cfg_dead_off = dead_off if dead_off is not None else self.nav_cfg.dead_off
        self.dead_off = max(
            float(cfg_dead_off if cfg_dead_off is not None else self.dead * 0.3), 2.0
        )
        self.speed_cap_kmh = float(
            speed_cap_kmh if speed_cap_kmh is not None else self.nav_cfg.speed_cap_kmh
        )
        self.xte_m = float(xte_m if xte_m is not None else self.nav_cfg.xte_m)
        self.xte_outer_m = max(self.xte_m, float(self.nav_cfg.xte_outer_m))
        self.steer_look_s = float(self.nav_cfg.steer_look_s)
        self.steer_lead_s = float(self.nav_cfg.steer_lead_s)
        self.skip_ahead_m = float(self.nav_cfg.skip_ahead_m)
        self.dbg = bool(debug if debug is not None else self.nav_cfg.debug)
        self.stop_speed_kmh = float(
            stop_speed_kmh if stop_speed_kmh is not None else self.nav_cfg.stop_speed_kmh
        )
        self.stop_min_px_s = float(
            stop_min_px_s if stop_min_px_s is not None else self.nav_cfg.stop_min_px_s
        )
        self.stop_confirm_s = float(
            stop_confirm_s if stop_confirm_s is not None else self.nav_cfg.stop_confirm_s
        )
        self.stop_hold = float(stop_hold if stop_hold is not None else self.nav_cfg.stop_hold)
        self.stop_timeout = float(
            stop_timeout if stop_timeout is not None else self.nav_cfg.stop_timeout
        )
        self.kb = kb
        self.vehicle_model = vehicle_model
        if self.vehicle_model is None and self.nav_cfg.vehicle_profile:
            try:
                self.vehicle_model = VehicleModel.load(self.nav_cfg.vehicle_profile)
                logger.info(
                    "[nav] vehicle profile %s loaded (%s)",
                    self.nav_cfg.vehicle_profile,
                    self.vehicle_model.name,
                )
            except (OSError, ValueError, KeyError) as exc:
                logger.warning(
                    "[nav] vehicle profile %r not loaded: %s",
                    self.nav_cfg.vehicle_profile,
                    exc,
                )
        if self.vehicle_model is not None and self.nav_cfg.yaw_gain is not None:
            self.vehicle_model.yaw_gain = float(self.nav_cfg.yaw_gain)

        # Sub-controllers
        self.planner = (
            RouteSpeedPlanner(
                lat_accel_mps2=self.nav_cfg.corner_lat_g * G,
                brake_decel_mps2=self.nav_cfg.brake_g * G,
                min_speed_kmh=self.nav_cfg.corner_min_kmh,
                lookahead_m=self.nav_cfg.plan_ahead_m,
                cut_m=self.nav_cfg.corner_cut_m,
            )
            if self.nav_cfg.speed_profile
            else None
        )
        self.path = PathTracker(
            self.pts,
            arrive_r=self.arrive_r,
            xte_m=self.xte_m,
            xte_outer_m=self.xte_outer_m,
            steer_look_s=self.steer_look_s,
            skip_ahead_m=self.skip_ahead_m,
        )
        self.speed_ctrl = SpeedController(
            speed_cap_kmh=self.speed_cap_kmh,
            brake_d=self.brake_d,
            corner_min_kmh=self.nav_cfg.corner_min_kmh,
            corner_max_kmh=self.nav_cfg.corner_max_kmh,
            px_per_m=float(px_per_m or 0.0),
        )
        self.steer_ctrl = SteeringController(
            dead=self.dead,
            dead_off=self.dead_off,
            settle_t=self.nav_cfg.settle_s,
            ant_s=self.steer_lead_s,
            turn_deg=(turn_deg if turn_deg is not None else self.nav_cfg.turn_deg),
            hold_max=(hold_max if hold_max is not None else self.nav_cfg.hold_max),
        )
        self.telemetry = NavTelemetryLogger(dbg_target=self.dbg)

        # Public state for UI polling
        self.state = "idle"
        self.last: dict[str, Any] = dict(
            idx=0, dist=0.0, bearing=0.0, err=0.0, heading=0.0, turn=0.0, speed=0.0
        )
        self.err: Exception | None = None
        self.speed = 0.0

        self._last_mp: tuple[float, float] | None = None
        self._last_t = 0.0
        self._last_measured_t: float | None = None
        self._last_speed_t: float | None = None
        self._speed_kmh = 0.0
        self._heading: float | None = None
        self._last_heading_t = 0.0
        self._lost = False
        self._route_snapped = False
        self._dbg_n = 0
        self._stop_ev = threading.Event()
        self._final_stop = False
        self._stop_s_t: float | None = None
        self._final_t0: float | None = None
        self._kb_closed = False

    @property
    def idx(self) -> int:
        return self.path.idx

    @idx.setter
    def idx(self, val: int) -> None:
        self.path.idx = val

    # Backward-compatible property facades
    @property
    def _samples(self) -> list[tuple[float, float, float]]:
        return self.path.samples

    @_samples.setter
    def _samples(self, val: list[tuple[float, float, float]]) -> None:
        self.path._samples = val

    @property
    def _mv(self) -> float:
        return self.path.mv

    @property
    def _mh(self) -> float | None:
        return self.path.mh

    @property
    def _mh_t(self) -> float:
        return self.path.mh_t

    @property
    def _ang(self) -> float:
        return self.steer_ctrl.ang

    @property
    def _steer(self) -> int:
        return self.steer_ctrl.steer

    @property
    def _micro(self) -> bool:
        return self.steer_ctrl.micro

    @property
    def _px_per_m(self) -> float:
        return self.speed_ctrl._px_per_m

    @property
    def _vmax_px(self) -> float:
        return self.speed_ctrl._vmax_px

    @property
    def _braking(self) -> bool:
        return self.speed_ctrl.is_braking

    def stop(self) -> None:
        """Signal thread to stop, release all held keys, and close the key driver."""
        self._stop_ev.set()
        try:
            if self.kb is not None:
                self.kb.release_all()
        except OSError:
            pass
        self.telemetry.close()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout=1.0)
        self._close_kb()

    def _close_kb(self) -> None:
        """Close the key driver's serial port exactly once (never raises)."""
        if self._kb_closed:
            return
        self._kb_closed = True
        close = getattr(self.kb, "close", None)
        if callable(close):
            try:
                close()
            except OSError:
                pass

    def _px_per_m_now(self) -> float:
        return self.speed_ctrl.px_per_m_now()

    def _kmh(self, px_s: float) -> float:
        return self.speed_ctrl.to_kmh(px_s)

    def _m(self, d_px: float) -> float:
        return self.speed_ctrl.to_meters(d_px)

    def _lost_limit(self) -> float:
        return self.path.lost_limit()

    def _push_pose(self, ts: float, x: float, y: float) -> bool:
        added = self.path.push_pose(ts, x, y)
        if added:
            ocr = self._speed_kmh if self._speed_fresh(ts) else None
            self.speed_ctrl.update_scale(self.path.mv, ocr_kmh=ocr)
        return added

    def _pose_at(self, now: float) -> tuple[float, float] | None:
        return self.path.pose_at(now)

    def _impulse(self, err: float) -> float:
        return self.steer_ctrl.calc_impulse(err)

    def _dbg_open(self) -> None:
        self.telemetry.open(self.pts, self._get_telemetry_params())

    def _dbg_tick(self, row: dict[str, Any]) -> None:
        self.telemetry.tick(row, self.pts, self._get_telemetry_params())

    def _dbg_flush(self) -> None:
        self.telemetry.flush()

    def _dbg_close(self) -> None:
        self.telemetry.close()

    def _wait(self, base: float) -> None:
        self._stop_ev.wait(base * random.uniform(0.75, 1.35))

    def _rel(self, state: str) -> None:
        try:
            if self.kb is not None:
                self.kb.release_all()
        except OSError:
            pass
        self.state = state

    def _update_speed(self, now: float, mp: tuple[float, float]) -> None:
        """Smoothed instantaneous speed estimate from successive poses."""
        prev = self._last_mp
        self._last_mp = (float(mp[0]), float(mp[1]))
        if prev is not None and self._last_t:
            dtp = max(now - self._last_t, 1e-3)
            inst = math.hypot(mp[0] - prev[0], mp[1] - prev[1]) / dtp
            self.speed = min(400.0, self.speed * 0.6 + inst * 0.4)
        else:
            self.speed = 0.0
        self._last_t = now

    def _smooth_heading(self, src: float, now: float, yaw_max: float | None) -> float:
        """Low-pass the heading source, rate-limited by the physical yaw.

        The motion-course source occasionally steps by tens of degrees (a
        localization glitch or a slip): feeding that into `err` made the relay
        flip sides. The truck cannot yaw faster than the grip-limited model
        value, so any faster change is rejected instead of tracked.
        """
        src = float(src) % 360.0
        dt = now - self._last_heading_t
        self._last_heading_t = now
        if self._heading is None or dt <= 1e-3 or dt > 0.5:
            self._heading = src
            return self._heading
        dd = wrap180(src - self._heading)
        lim = max(10.0, (yaw_max or 45.0) * 1.6) * dt
        step = max(-lim, min(lim, dd * 0.5))
        self._heading = (self._heading + step) % 360.0
        return self._heading

    def _speed_kmh_estimate(self, now: float) -> float:
        """Best available speed for model lookups: OCR when fresh, else the track."""
        if self._speed_fresh(now):
            return self._speed_kmh
        return self._kmh(self.path.mv)

    def _yaw_rate_max(self, speed_kmh: float) -> float | None:
        """Grip-limited yaw authority at a speed (None when no profile is loaded)."""
        if self.vehicle_model is None:
            return None
        lat_accel = (
            self.planner.lat_accel if self.planner is not None else self.nav_cfg.corner_lat_g * G
        )
        return self.vehicle_model.yaw_rate_max_deg_s(speed_kmh, lat_accel_mps2=lat_accel)

    def _route_target_kmh(self, mp: tuple[float, float]) -> float | None:
        """Planned corner/braking speed limit ahead (None when disabled)."""
        if self.planner is None:
            return None
        px_per_m = self._px_per_m_now()
        if px_per_m <= 0:
            return None
        return self.planner.target_speed_kmh(mp, self.pts, self.path.idx, px_per_m)

    def _speed_target(self, road_turn: float, plan_kmh: float | None) -> float:
        """Speed target in px/s: the planner caps cruise when it is enabled.

        The legacy vertex-angle heuristic applies the corner speed the moment
        any vertex enters its ~200 px horizon, which braked the whole approach
        to every waypoint; with the planner the baseline is cruise speed and
        the planned limit (distance- and brake-aware) does the capping.
        """
        if plan_kmh is not None:
            return min(
                self.speed_ctrl.calc_target_speed(0.0),
                self.speed_ctrl.from_kmh(plan_kmh),
            )
        return self.speed_ctrl.calc_target_speed(road_turn)

    def apply_vehicle_tuning(self, cfg: NavigatorConfig) -> None:
        """Apply live vehicle/planner tuning changed in the UI."""
        if self.vehicle_model is not None and cfg.yaw_gain is not None:
            self.vehicle_model.yaw_gain = float(cfg.yaw_gain)
        if self.planner is not None:
            self.planner.lat_accel = max(0.1, cfg.corner_lat_g * G)
            self.planner.brake_decel = max(0.1, cfg.brake_g * G)
            self.planner.min_speed_kmh = max(0.0, cfg.corner_min_kmh)
            self.planner.lookahead_m = max(10.0, cfg.plan_ahead_m)
            self.planner.cut_m = max(1.0, cfg.corner_cut_m)
        self.xte_m = max(0.0, float(cfg.xte_m))
        self.xte_outer_m = max(self.xte_m, float(cfg.xte_outer_m))
        self.steer_look_s = max(0.4, float(cfg.steer_look_s))
        self.path.xte_m = self.xte_m
        self.path.xte_outer_m = self.xte_outer_m
        self.path.steer_look_s = self.steer_look_s
        self.speed_ctrl.corner_min_kmh = max(0.0, float(cfg.corner_min_kmh))
        self.speed_ctrl.corner_max_kmh = max(
            self.speed_ctrl.corner_min_kmh, float(cfg.corner_max_kmh)
        )
        self.steer_ctrl.settle_t = max(0.1, min(2.0, float(cfg.settle_s)))
        self.steer_ctrl.ant_s = max(0.0, min(1.0, float(cfg.steer_lead_s)))
        self.steer_lead_s = self.steer_ctrl.ant_s
        self.skip_ahead_m = max(0.0, float(cfg.skip_ahead_m))
        self.path.skip_ahead_m = self.skip_ahead_m

    def _get_telemetry_params(self) -> dict[str, Any]:
        return dict(
            arrive_r=self.arrive_r,
            dead=self.dead,
            dead_off=self.dead_off,
            brake_d=self.brake_d,
            settle_t=self.steer_ctrl.settle_t,
            imp_k=self.steer_ctrl.imp_k,
            w_est=self.steer_ctrl.w_est,
            t_min=self.steer_ctrl.t_min,
            t_max=self.steer_ctrl.t_max,
            turn_deg=self.steer_ctrl.turn_deg,
            hold_max=self.steer_ctrl.hold_max,
            speed_cap_kmh=self.speed_cap_kmh,
            v_cruise=self.speed_ctrl.v_cruise,
            xte_m=self.xte_m,
            xte_outer_m=self.xte_outer_m,
            steer_look_s=self.path.steer_look_s,
            settle_s=self.steer_ctrl.settle_t,
            corner_max_kmh=self.speed_ctrl.corner_max_kmh,
            skip_ahead_m=self.path.skip_ahead_m,
            stop_speed_kmh=self.stop_speed_kmh,
            stop_min_px_s=self.stop_min_px_s,
            stop_confirm_s=self.stop_confirm_s,
            stop_hold=self.stop_hold,
            stop_timeout=self.stop_timeout,
            vehicle=self.vehicle_model.vehicle_id if self.vehicle_model else None,
            speed_profile=self.nav_cfg.speed_profile,
            corner_lat_g=self.nav_cfg.corner_lat_g,
            brake_g=self.nav_cfg.brake_g,
            corner_min_kmh=self.nav_cfg.corner_min_kmh,
            plan_ahead_m=self.nav_cfg.plan_ahead_m,
        )

    def run(self) -> None:
        """Main navigation control loop."""
        try:
            while not self._stop_ev.is_set():
                now = time.time()
                it = getattr(self.loc, "latest", None)
                new_sample, keep_going = self._ingest(now, it)
                if not keep_going:
                    continue

                if not self.path.clean_stale_samples(now, max_age=1.5):
                    self._rel("wait_pose")
                    self._wait(0.2)
                    continue

                mp = self.path.pose_at(now)
                if mp is None:
                    self._rel("wait_pose")
                    self._wait(0.2)
                    continue

                pose = it.get("pose") if it else None
                if pose is None:
                    self._rel("wait_pose")
                    self._wait(0.3)
                    continue

                course = (
                    self._heading
                    if self._heading is not None
                    else (float(pose["th"]) if pose.get("th") is not None else None)
                )

                self._enter_route(mp, course)
                self._maybe_reacquire(mp, course, now)

                # Advance waypoints
                final_seg = len(self.pts) > 1 and self.path.idx >= len(self.pts) - 1
                tx, ty, dist, arrived = self.path.advance_waypoint(mp)
                if arrived and not final_seg:
                    self._rel("arrived")
                    self._wait(0.3)
                    continue

                if (
                    final_seg
                    and not self._final_stop
                    and (arrived or dist < self._final_brake_dist())
                ):
                    self._begin_final_stop(now, dist)

                if self._final_stop:
                    self._tick_final_stop(now, it, mp, dist, new_sample)
                    continue

                self._drive_tick(now, it, pose, mp, tx, ty, dist, new_sample)
        finally:
            self.telemetry.close()
            try:
                if self.kb is not None:
                    self.kb.release_all()
            except OSError:
                pass
            self._close_kb()

    def _ingest(self, now: float, it: Any) -> tuple[bool, bool]:
        """Update OCR speed and the pose track from the latest locator frame.

        Returns (new_sample, keep_going): keep_going is False when a ghost pose
        was detected, so the caller should wait instead of driving.
        """
        if it is None:
            return False, True
        ts = it.get("ts", now)
        speed_ok = bool(it.get("speed_ok", False))
        speed_kmh = it.get("speed_kmh")
        if speed_ok and speed_kmh is not None:
            self._speed_kmh = float(speed_kmh)
            self._last_speed_t = ts
        # good=True marks a MEASURED pose; held (frozen) poses of a capture void
        # keep the last position with good=False. Held poses are only used for
        # aiming: feeding them to the track would fake a zero speed and a false
        # full stop.
        mpx = it.get("map_px")
        if mpx is None or not bool(it.get("good", False)):
            return False, True
        x, y = float(mpx[0]), float(mpx[1])
        self._last_measured_t = ts
        # Ghost pose detection
        if self.path.samples and ts > self.path.samples[-1][0]:
            lt, lx, ly = self.path.samples[-1]
            dts = ts - lt
            d = math.hypot(x - lx, y - ly)
            if dts > 1e-3 and d / dts > self.path.lost_limit():
                self._lost = True
                self._dbg_tick(
                    dict(
                        kind="lost",
                        t=now,
                        sa=round(now - lt, 2),
                        d=round(self._m(d), 1),
                        v_claim=round(d / dts, 1),
                    )
                )
                self._rel("lost")
                self._wait(0.25)
                return False, False
        new_sample = self._push_pose(ts, x, y)
        if new_sample:
            self._lost = False
        return new_sample, True

    def _enter_route(self, mp: tuple[float, float], course: float | None) -> None:
        """Route entry, once per run: engage at the nearest point instead of
        U-turning back to pts[0] when F6 is pressed mid-route."""
        if self._route_snapped:
            return
        self.path.snap_to_nearest(mp, course_deg=course)
        self._route_snapped = True
        px, py = self.pts[self.path.idx]
        logger.info(
            "[nav] route entry at point %d/%d (dist=%.0fm)",
            self.path.idx + 1,
            len(self.pts),
            self._m(math.hypot(px - mp[0], py - mp[1])),
        )

    def _maybe_reacquire(self, mp: tuple[float, float], course: float | None, now: float) -> None:
        """Forward re-acquisition after an excursion (outside the outer corridor
        only): jump the index to the nearest route point ahead instead of
        chasing a stale one."""
        if self.path.maybe_reacquire(mp, self._px_per_m_now(), course, now):
            logger.info(
                "[nav] re-acquired route at point %d/%d",
                self.path.idx + 1,
                len(self.pts),
            )
            self._dbg_tick(dict(kind="reacquire", t=round(now, 4), idx=self.path.idx))

    def _begin_final_stop(self, now: float, dist: float) -> None:
        """Enter the full-stop phase at the final waypoint."""
        self._final_stop = True
        self._final_t0 = now
        self._stop_s_t = None
        logger.info(
            "[nav] final waypoint %d in stop range (dist=%.0fm) sv=%.0fpx/s mv=%.0fpx/s",
            self.path.idx,
            self._m(dist),
            self.speed,
            self.path.mv,
        )

    def _tick_final_stop(
        self, now: float, it: Any, mp: tuple[float, float], dist: float, new_sample: bool
    ) -> None:
        """Brake with SPACE down to a full stop, then disable the autopilot.

        Reaching the arrival radius must trigger the stop as well: otherwise
        the waypoint index runs past the last point and the segment math
        (calc_xte_and_bearing) indexes out of range.
        """
        self._update_speed(now, mp)
        stop_state = self._stop_state(now)
        if stop_state != "braking":
            if stop_state == "timeout":
                pose_age = (
                    now - self._last_measured_t if self._last_measured_t is not None else -1.0
                )
                logger.warning(
                    "[nav] final stop timeout after %.1fs "
                    "(last measured pose %.1fs ago) — releasing keys",
                    self.stop_timeout,
                    pose_age,
                )
            self._finish()
            return
        self._rel("final_stop")
        try:
            if self.kb is not None:
                self.kb.set_state({"SPACE": True})
        except OSError as exc:
            self.err = exc
            self.state = "key_error"
            self._wait(0.3)
            return
        self.steer_ctrl.force_release(now, self.path.mh_t)
        self.err = None
        if self.dbg:
            self._dbg_tick(
                dict(
                    t=round(now, 4),
                    tick=self._dbg_n,
                    kind="final_stop",
                    pose_age=round(now - float(it["ts"]), 3) if it else 0.0,
                    sample_age=round(now - self.path.samples[-1][0], 3),
                    new_sample=bool(new_sample),
                    mode="S",
                    mv=round(self.path.mv, 1),
                    heading=round(self._heading or 0.0, 2),
                    speed=round(self.speed, 1),
                    ocr=round(self._speed_kmh, 1) if self._speed_fresh(now) else None,
                    dist=round(dist, 1),
                    keys="SPACE",
                    idx=self.path.idx,
                )
            )
        if self._dbg_n % 5 == 0:
            logger.info(
                "[nav] final-stop braking sv=%.1fpx/s mv=%.1fpx/s dist=%.0fm%s",
                self.speed,
                self.path.mv,
                self._m(dist),
                f" ocr={self._speed_kmh:.0f}km/h" if self._speed_fresh(now) else "",
            )
        self._dbg_n += 1
        self.last = dict(
            idx=self.path.idx,
            dist=dist,
            bearing=0.0,
            err=0.0,
            heading=self._heading or 0.0,
            turn=0.0,
            speed=self.speed,
        )
        self.state = "final_stop"
        self._wait(self.poll)

    def _drive_tick(
        self,
        now: float,
        it: Any,
        pose: dict[str, Any],
        mp: tuple[float, float],
        tx: float,
        ty: float,
        dist: float,
        new_sample: bool,
    ) -> None:
        """Normal driving tick: speed/heading estimation, steering and keys."""
        # Speed estimation from successive positions
        self._update_speed(now, mp)

        # Cross-track error & pure pursuit bearing
        xte, xte_lim, bearing = self.path.calc_xte_and_bearing(mp, self._px_per_m_now())

        # Heading calculation & smoothing
        mh_age = now - self.path.mh_t if self.path.mh is not None else 1e9
        mh_on = self.path.mh is not None and self.path.mv > 10.0 and mh_age < 1.5
        if self.path.mh is not None and mh_age < 3.0:
            heading_src = self.path.mh
        elif pose is not None:
            heading_src = float(pose["th"]) % 360.0
        else:
            heading_src = self.path.mh or 0.0

        # Steering state machine (speed-dependent yaw authority)
        yaw_max = self._yaw_rate_max(self._speed_kmh_estimate(now))
        heading = self._smooth_heading(heading_src, now, yaw_max)
        err = wrap180(bearing - heading)

        # Road geometry angles
        turn_angle, road_turn = self.path.calc_road_turn(heading)
        plan_kmh = self._route_target_kmh(mp)
        tgt_spd = self._speed_target(road_turn, plan_kmh)
        outside_outer = abs(xte) > self.xte_outer_m * self._px_per_m_now()
        if outside_outer:
            # Far outside the corridor: bleed speed while steering back.
            rejoin_kmh = max(self.nav_cfg.corner_min_kmh, 10.0)
            tgt_spd = min(tgt_spd, self.speed_ctrl.from_kmh(rejoin_kmh))

        # Age of the last MEASURED pose: the tracker republishes a
        # frozen position with a fresh frame timestamp during a capture
        # void, so the frame age alone cannot gate the wheel.
        steer_pose_age = (
            (now - self._last_measured_t) if self._last_measured_t is not None else None
        )
        raw_heading = float(pose["th"]) % 360.0 if pose.get("th") is not None else None
        steer_action = self.steer_ctrl.step(
            now,
            err,
            heading,
            self.path.mh,
            self.path.mh_t,
            yaw_rate_max=yaw_max,
            fresh_sample=bool(new_sample),
            pose_age=steer_pose_age,
            mv_mps=self._m(self.path.mv),
            heading_meas=raw_heading,
        )
        keys: dict[str, bool] = {}
        if steer_action == 1:
            keys["D"] = True
        elif steer_action == -1:
            keys["A"] = True

        # Throttle and braking evaluation
        gas_w, brake_space = self.speed_ctrl.decide_throttle_and_brake(
            mv=self.path.mv,
            tgt_spd=tgt_spd,
            steer=self.steer_ctrl.steer,
            micro=self.steer_ctrl.micro,
            road_turn=road_turn,
            turn_min=10.0,
            xte=xte,
            xte_lim=xte_lim,
            hold_window=not outside_outer,
        )

        if brake_space:
            # Keep A/D: dropping the wheel here left the vehicle unable
            # to catch a slide until it slowed down to the target.
            keys["SPACE"] = True
        elif gas_w:
            keys["W"] = True

        if self._dbg_n % 5 == 0:
            hmode = "M" if mh_on else "S"
            logger.info(
                "[nav] mp=%.0f,%.0f goal=%d (%.0f,%.0f) dist=%.0fm "
                "bearing=%.1f heading=%.1f err=%.1f xte=%.1fm "
                "turn=%.0f tgt=%.0fkm/h spar=%.0fkm/h(%.0f) "
                "keys=%s th=%s%s",
                mp[0],
                mp[1],
                self.path.idx,
                tx,
                ty,
                self._m(dist),
                bearing,
                heading,
                err,
                self._m(abs(xte)),
                turn_angle,
                self._kmh(tgt_spd),
                self._kmh(self.path.mv),
                self.path.mv,
                "".join(k for k in ("W", "A", "D", "SPACE") if keys.get(k)),
                hmode,
                f" ocr={self._speed_kmh:.0f}km/h" if self._speed_fresh(now) else "",
            )

        self._dbg_n += 1
        try:
            if self.kb is not None:
                self.kb.set_state(keys)
        except OSError as exc:
            self.err = exc
            self.state = "key_error"
            self._wait(0.3)
            return

        self.err = None
        if self.dbg:
            self._dbg_tick(
                dict(
                    t=round(now, 4),
                    tick=self._dbg_n,
                    pose_age=round(now - float(it["ts"]), 3) if it else 0.0,
                    sample_age=round(now - self.path.samples[-1][0], 3),
                    new_sample=bool(new_sample),
                    mode="M" if mh_on else "S",
                    mv=round(self.path.mv, 1),
                    mh=round(self.path.mh, 1) if self.path.mh is not None else None,
                    heading=round(heading, 2),
                    bearing=round(bearing, 2),
                    err=round(err, 2),
                    turn=round(turn_angle, 1),
                    roadT=round(road_turn, 1),
                    tgt=round(tgt_spd, 1),
                    dist=round(dist, 1),
                    xte=round(abs(xte), 1),
                    xte_m=round(self._m(abs(xte)), 1),
                    speed=round(self.speed, 1),
                    ocr=round(self._speed_kmh, 1) if self._speed_fresh(now) else None,
                    v=round(self.path.mv, 1),
                    pxm=round(self._px_per_m_now(), 3),
                    good=bool(it.get("good", False)) if it else False,
                    th_raw=round(float(pose["th"]), 2) if pose else None,
                    yaw_max=round(yaw_max, 1) if yaw_max else None,
                    plan_kmh=round(plan_kmh, 1) if plan_kmh is not None else None,
                    ang=round(self.steer_ctrl.ang, 2),
                    lead=round(self.steer_ctrl.last_lead, 1),
                    steer=self.steer_ctrl.steer,
                    micro=self.steer_ctrl.micro,
                    braking=bool(brake_space),
                    keys="".join(k for k in ("W", "A", "D", "SPACE") if keys.get(k)),
                    idx=self.path.idx,
                )
            )

        self.last = dict(
            idx=self.path.idx,
            dist=dist,
            bearing=bearing,
            err=err,
            heading=heading,
            turn=turn_angle,
            speed=self.speed,
        )
        self.state = "run"
        self._wait(self.poll)
