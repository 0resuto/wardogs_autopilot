"""Calibration fits for the vehicle model from nav_dbg JSONL runs.

The telemetry rows written by FollowDriver are enough to pin the absolute
scales that the physics pack cannot provide:

    px_per_m  = mv / (ocr / 3.6)            (steady, measured poses)
    brake_g   = |d(ocr)/dt| / 9.81 / 3.6    (SPACE episodes)
    yaw_gain *= measured_yaw / logged_yaw_max  (A/D episodes)

The fits are medians over episodes, so a single bad run cannot move them much.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .steering_controller import wrap180

G = 9.80665


@dataclass(frozen=True)
class PxPerMFit:
    value: float | None
    samples: int
    spread: float


@dataclass(frozen=True)
class BrakeFit:
    decel_mps2: float | None
    measured_g: float | None
    suggested_g: float | None
    episodes: int


@dataclass(frozen=True)
class YawFit:
    ratio: float | None
    episodes: int
    current_gain: float
    suggested_gain: float | None


def _num(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _has_key(row: dict[str, Any], name: str) -> bool:
    keys = str(row.get("keys", ""))
    if name == "SPACE":
        return "SPACE" in keys
    return name in keys.replace("SPACE", "")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of y over x."""
    n = len(points)
    if n < 2:
        return 0.0
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    denom = sum((p[0] - mean_x) ** 2 for p in points)
    if denom <= 1e-9:
        return 0.0
    num = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    return num / denom


def _episodes(
    rows: Iterable[dict[str, Any]],
    predicate,
) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if predicate(row):
            current.append(row)
        elif current:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def fit_px_per_m(rows: list[dict[str, Any]], min_speed_kmh: float = 15.0) -> PxPerMFit:
    """Pixel-per-meter scale from measured poses at a steady speed."""
    ratios: list[float] = []
    for row in rows:
        if not row.get("good"):
            continue
        if _has_key(row, "SPACE"):
            continue
        ocr = _num(row, "ocr")
        mv = _num(row, "mv")
        if ocr is None or mv is None or ocr < min_speed_kmh or mv <= 0.0:
            continue
        ratios.append(mv / (ocr / 3.6))

    if len(ratios) < 10:
        return PxPerMFit(None, len(ratios), 0.0)
    ratios.sort()
    spread = ratios[(3 * len(ratios)) // 4] - ratios[len(ratios) // 4]
    return PxPerMFit(_median(ratios), len(ratios), spread)


def fit_brake(rows: list[dict[str, Any]]) -> BrakeFit:
    """Braking deceleration from SPACE episodes (km/h per second)."""
    episodes = _episodes(rows, lambda r: _has_key(r, "SPACE"))
    decels: list[float] = []
    for episode in episodes:
        points = [
            (float(row["t"]), ocr)
            for row in episode
            if "t" in row and (ocr := _num(row, "ocr")) is not None
        ]
        if len(points) < 5 or points[-1][0] - points[0][0] < 0.4:
            continue
        slope_kmh_s = _slope(points)
        if slope_kmh_s < -1.0:
            decels.append(-slope_kmh_s / 3.6)

    if not decels:
        return BrakeFit(None, None, None, len(episodes))
    decel = _median(decels)
    measured_g = decel / G
    return BrakeFit(decel, measured_g, measured_g * 0.9, len(episodes))


def fit_yaw_gain(rows: list[dict[str, Any]], current_gain: float = 1.0) -> YawFit:
    """Yaw authority scale from A/D episodes compared with the logged yaw_max."""
    episodes = _episodes(rows, lambda r: _has_key(r, "A") or _has_key(r, "D"))
    ratios: list[float] = []
    for episode in episodes:
        points = [
            (float(row["t"]), heading)
            for row in episode
            if "t" in row
            and isinstance(
                heading := (
                    row.get("th_raw") if row.get("th_raw") is not None else row.get("heading")
                ),
                (int, float),
            )
        ]
        if len(points) < 5:
            continue
        duration = points[-1][0] - points[0][0]
        if duration < 0.3:
            continue
        steering = sum(1 for row in episode if row.get("steer"))
        if steering < 0.7 * len(episode):
            continue
        rotated = abs(wrap180(points[-1][1] - points[0][1]))
        if rotated < 3.0:
            continue
        yaws = [y for row in episode if (y := _num(row, "yaw_max")) is not None and y > 1.0]
        if not yaws:
            continue
        ratios.append((rotated / duration) / _median(yaws))

    if len(ratios) < 3:
        return YawFit(None, len(ratios), current_gain, None)
    ratio = _median(ratios)
    return YawFit(ratio, len(ratios), current_gain, current_gain * ratio)


def format_report(
    px_fit: PxPerMFit,
    brake_fit: BrakeFit,
    yaw_fit: YawFit,
    rows: int,
) -> str:
    """Human-readable calibration report."""
    lines = [f"nav_dbg rows analysed: {rows}"]

    if px_fit.value is None:
        lines.append(f"px_per_m: not enough samples ({px_fit.samples})")
    else:
        lines.append(f"px_per_m: {px_fit.value:.3f} (n={px_fit.samples}, IQR={px_fit.spread:.3f})")

    if (
        brake_fit.measured_g is None
        or brake_fit.suggested_g is None
        or brake_fit.decel_mps2 is None
    ):
        lines.append(f"brake_g: no usable SPACE episodes ({brake_fit.episodes})")
    else:
        lines.append(
            f"brake_g: measured {brake_fit.measured_g:.3f}g "
            f"({brake_fit.decel_mps2:.2f} m/s^2), suggest {brake_fit.suggested_g:.3f} "
            f"(episodes={brake_fit.episodes})"
        )

    if yaw_fit.suggested_gain is None:
        lines.append(f"yaw_gain: not enough steering episodes ({yaw_fit.episodes})")
    else:
        lines.append(
            f"yaw_gain: ratio {yaw_fit.ratio:.3f} -> suggest {yaw_fit.suggested_gain:.3f} "
            f"(current {yaw_fit.current_gain:.3f}, episodes={yaw_fit.episodes})"
        )
    return "\n".join(lines)
