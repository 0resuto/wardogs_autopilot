"""Strongly typed application configuration schemas using Pydantic.

Provides schema validation, field boundaries, and type safety for config.json.
Maintains full backward compatibility with legacy dictionary access via to_dict().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .. import PROJECT_ROOT

#: Tracked template that a fresh checkout is seeded from; the working config.json
#: itself is machine state (window geometry, tuned gains) and is git-ignored.
CONFIG_TEMPLATE_NAME = "config.default.json"


def resolve_config_path(path: str | Path) -> Path:
    """Absolute config path: relative paths are rooted at the project root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(PROJECT_ROOT) / candidate
    return candidate


def config_template_path() -> Path:
    """Absolute path of the tracked config template used to seed a fresh setup."""
    return Path(PROJECT_ROOT) / CONFIG_TEMPLATE_NAME


class CaptureConfig(BaseModel):
    """Minimap capture and monitor settings."""

    model_config = ConfigDict(extra="allow")

    monitor: int = Field(default=0, ge=0, description="Monitor index for screen capture")
    fps: int = Field(default=10, ge=1, le=60, description="Capture cadence in frames per second")
    mmap_roi: list[int] = Field(
        default_factory=lambda: [45, 1009, 336, 277],
        description="Minimap ROI box [x, y, width, height] on monitor",
    )
    speed_roi: list[int] | None = Field(
        default=None,
        description="Speedometer HUD ROI box [x, y, width, height] on monitor, "
        "or None to disable speed OCR",
    )

    @field_validator("mmap_roi")
    @classmethod
    def validate_roi(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("mmap_roi must have exactly 4 elements [x, y, w, h]")
        _, _, w, h = v
        if w < 10 or h < 10:
            raise ValueError(f"mmap_roi dimensions too small: w={w}, h={h} (minimum 10x10)")
        return v

    @field_validator("speed_roi")
    @classmethod
    def validate_speed_roi(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("speed_roi must have exactly 4 elements [x, y, w, h]")
        _, _, w, h = v
        if w < 8 or h < 8:
            raise ValueError(f"speed_roi dimensions too small: w={w}, h={h} (minimum 8x8)")
        return v


class LocatorConfig(BaseModel):
    """Vision matcher and localization parameters."""

    model_config = ConfigDict(extra="allow")

    local_radius: int = Field(default=450, ge=50, description="Tracking search radius (px)")
    radius_growth: float = Field(default=1.6, ge=1.0, le=5.0)
    global_max_features: int = Field(default=100000, ge=1000)
    max_kp_frame: int = Field(
        default=1200,
        ge=100,
        le=6000,
        description="Max SIFT keypoints kept from a captured frame (response-ranked)",
    )
    fast_clahe: bool = Field(
        default=False,
        description="Try a CLAHE-preprocessed fast pass first, fall back to the "
        "default normalization when it finds no pose",
    )
    ratio: float = Field(default=0.8, ge=0.1, le=1.0)
    min_inl: int = Field(default=4, ge=1)
    min_inl_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    track_radius: float = Field(default=900.0, ge=50.0)
    ratio_local: float = Field(default=0.85, ge=0.1, le=1.0)
    min_inl_local: int = Field(default=4, ge=1)
    min_inl_rate_local: float = Field(default=0.0, ge=0.0, le=1.0)
    ratio_global: float = Field(default=0.9, ge=0.1, le=1.0)
    min_inl_global: int = Field(default=5, ge=1)
    min_inl_rate_global: float = Field(default=0.0, ge=0.0, le=1.0)
    vote_need: int = Field(default=3, ge=1)
    vote_frames: int = Field(default=5, ge=1)
    vote_radius_px: int = Field(default=300, ge=10)
    jump_gate_px: int = Field(default=3000, ge=100)
    heading_gate_deg: float = Field(default=0.0, ge=0.0, le=180.0)
    vote_inl_skip: int = Field(default=40, ge=1)
    early_inl: int = Field(
        default=40,
        ge=0,
        description="Stop trying further scale-level candidates once a match "
        "reaches this inlier count (0 = exhaustive scan)",
    )
    hold_frames: int = Field(default=5, ge=0)
    center_dx: float = Field(
        default=0.0,
        ge=-1000.0,
        le=1000.0,
        description="Player-center calibration: horizontal offset (minimap px) of the "
        "in-game player marker from the capture ROI midpoint (+ = right in the captured "
        "frame). The pose reported on the map is the marker's location; a wrong center "
        "shows up as the position circling the true point while the minimap rotates",
    )
    center_dy: float = Field(
        default=0.0,
        ge=-1000.0,
        le=1000.0,
        description="Player-center calibration: vertical offset (minimap px) of the "
        "in-game player marker from the capture ROI midpoint (+ = down in the captured "
        "frame)",
    )


class MapConfig(BaseModel):
    """Active map metadata and display scaling."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="zestafona", description="Map name prefix")
    file: str = Field(default="zestafona_map.png", description="Base map file name in data/maps/")
    thumb_factor: int = Field(default=8, ge=1)
    size: int = Field(default=32768, ge=1024)
    gray_conv: str = Field(
        default="desat",
        description="Map palette: 'desat' (minimap material), 'luma', 'equal', 'bt709'",
    )
    gray_gamma: float = Field(default=1.0, ge=0.1, le=3.0)
    mini_scale: float = Field(
        default=2.6544, ge=0.1, le=10.0, description="Minimap to native map pixel scale"
    )


class NavigatorConfig(BaseModel):
    """Steering and route follower parameters."""

    model_config = ConfigDict(extra="allow")

    key_source: str = Field(default="arduino", description="'arduino' or virtual simulator")
    port: str = Field(default="COM6", description="Serial port for Arduino Micro")
    vehicle_profile: str = Field(
        default="ural",
        description="Physics profile name from data/vehicles ('' disables the model)",
    )
    yaw_gain: float | None = Field(
        default=None,
        ge=0.05,
        le=5.0,
        description="Steering yaw-authority scale override; None uses the profile value",
    )
    arrive_r: float = Field(default=25.0, ge=1.0, description="Waypoint arrival radius (px)")
    dead: float = Field(default=3.0, ge=0.0, description="Steering dead-zone (deg)")
    turn_deg: float = Field(
        default=25.0,
        ge=1.0,
        description="Heading error threshold for continuous steering (deg)",
    )
    hold_max: float = Field(
        default=8.0,
        ge=0.5,
        description="Safety timeout for continuous steering (s)",
    )
    speed_cap_kmh: float = Field(default=79.0, ge=1.0, description="Maximum driving speed in km/h")
    speed_profile: bool = Field(
        default=True,
        description="Plan corner/braking speed limits from the route geometry",
    )
    corner_lat_g: float = Field(
        default=0.35,
        ge=0.05,
        le=1.5,
        description="Lateral acceleration budget for corners (g)",
    )
    brake_g: float = Field(
        default=0.45,
        ge=0.05,
        le=2.0,
        description="Braking deceleration budget (g)",
    )
    corner_min_kmh: float = Field(
        default=12.0,
        ge=0.0,
        description="Lower edge of the steady-corner hold window (km/h)",
    )
    corner_max_kmh: float = Field(
        default=22.0,
        ge=0.0,
        description="Upper edge of the steady-corner hold window: inside it the driver "
        "neither accelerates nor brakes (km/h)",
    )
    plan_ahead_m: float = Field(
        default=200.0,
        ge=20.0,
        le=1000.0,
        description="Speed planning horizon along the route (meters)",
    )
    corner_cut_m: float = Field(
        default=15.0,
        ge=4.0,
        le=60.0,
        description="Distance over which a sharp vertex is rounded (meters)",
    )
    xte_m: float = Field(default=4.0, ge=0.0, description="Inner corridor half-width (meters)")
    xte_outer_m: float = Field(
        default=12.0,
        ge=4.0,
        description="Outer corridor half-width: past it the driver slows down and steers firmly",
    )
    steer_look_s: float = Field(
        default=1.6,
        ge=0.4,
        le=4.0,
        description="Steering lookahead in seconds of travel (lower = tighter line, more active)",
    )
    settle_s: float = Field(
        default=0.6,
        ge=0.1,
        le=2.0,
        description="Pause after a completed steering hold before the next one (s); "
        "shorter = more active corrections",
    )
    steer_lead_s: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Steering release anticipation in seconds: how much heading change is "
        "expected to arrive through the latency before the wheel takes effect",
    )
    skip_ahead_m: float = Field(
        default=150.0,
        ge=0.0,
        le=1000.0,
        description="Route re-acquisition window (m): outside the outer corridor the "
        "active point may jump forward within this route length (0 disables)",
    )

    @model_validator(mode="after")
    def _clamp_corridor_and_corner_windows(self) -> NavigatorConfig:
        """Keep window edges ordered even when config.json is edited by hand."""
        if self.xte_outer_m < self.xte_m:
            self.xte_outer_m = self.xte_m
        if self.corner_max_kmh < self.corner_min_kmh:
            self.corner_max_kmh = self.corner_min_kmh
        return self

    poll: float = Field(default=0.033, ge=0.001, description="Navigation tick rate in seconds")
    brake_d: float = Field(
        default=260.0,
        ge=1.0,
        description="Legacy braking-distance constant, used only when the speed planner is off",
    )
    dead_off: float | None = Field(default=None, description="Steering release dead-zone (deg)")
    stop_speed_kmh: float = Field(
        default=2.0,
        ge=0.5,
        description="Final-waypoint full-stop speed threshold (km/h, px floor applies)",
    )
    stop_min_px_s: float = Field(
        default=3.0,
        ge=0.0,
        description="Noise floor of the full-stop speed threshold (px/s)",
    )
    stop_confirm_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Max age of a measured pose to trust the full-stop decision (s)",
    )
    stop_hold: float = Field(
        default=0.8,
        ge=0.1,
        description="Seconds below stop speed before the autopilot disables itself",
    )
    stop_timeout: float = Field(
        default=5.0,
        ge=1.0,
        description="Hard timeout for final-waypoint braking before autopilot off (s)",
    )
    debug: bool = Field(default=False, description="Enable verbose nav logging")
    last_preset: str = Field(default="")


class DebugConfig(BaseModel):
    """Debugging options."""

    model_config = ConfigDict(extra="allow")

    collect_fail_logs: bool = Field(default=False)


class UiConfig(BaseModel):
    """Window geometry remembered between sessions (multi-monitor clamped)."""

    model_config = ConfigDict(extra="allow")

    window_x: int = Field(default=-1)
    window_y: int = Field(default=-1)
    window_w: int = Field(default=0, ge=0)
    window_h: int = Field(default=0, ge=0)
    window_max: bool = Field(default=False)


class AppConfig(BaseModel):
    """Root configuration for WARDOGS autopilot."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    locator: LocatorConfig = Field(default_factory=LocatorConfig)
    map: MapConfig = Field(default_factory=MapConfig)
    navigator: NavigatorConfig = Field(default_factory=NavigatorConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    cfg_path: str = Field(default="config.json", alias="_cfg_path")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a standard Python dictionary."""
        return self.model_dump(by_alias=True)

    @classmethod
    def load(cls, path: str | Path = "config.json") -> AppConfig:
        """Load and validate configuration from a JSON file.

        Relative paths are resolved against the project root (where
        config.json lives), so the app behaves the same regardless of the
        working directory. A missing file is created first (see
        `ensure_config_file`), so a fresh checkout starts with the maintained
        defaults instead of failing on an absent config.json.
        """
        cfg_path = ensure_config_file(path)
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls.model_validate(data)
        cfg.cfg_path = str(cfg_path)
        return cfg

    def save(self, path: str | Path | None = None) -> None:
        """Atomically save configuration back to JSON file.

        Writes to a sibling .tmp file, fsyncs it, then replaces the target with
        os.replace(), so a crash or full disk never leaves a truncated config.
        `_cfg_path` is runtime state (where this config was loaded from), not
        file content: it stays out of the dump, so a config opened on another
        machine never carries a foreign absolute path.
        """
        target_path = Path(path or self.cfg_path)
        atomic_write_json(target_path, self.model_dump(by_alias=True, exclude={"cfg_path"}))


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON to `path` via a temp file + os.replace so readers see a
    complete document even if the process dies mid-write."""
    target_path = Path(path)
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def ensure_config_file(path: str | Path) -> Path:
    """Create the working config at `path` from the template when it is absent.

    The working config.json is machine state and is not tracked; a fresh
    checkout would otherwise start without one. The file is seeded from the
    tracked `config.default.json` so the maintained defaults (capture ROI,
    active map, serial port, window size) stay in one reviewable place. A
    missing or invalid template falls back to the schema defaults: starting the
    app must never depend on a repository asset.

    An existing file is returned untouched, so a config.json that lives on this
    machine keeps its tuned values across updates.
    """
    target = resolve_config_path(path)
    if target.exists():
        return target

    payload: dict[str, Any] | None = None
    template = config_template_path()
    if template.exists():
        try:
            with open(template, encoding="utf-8") as f:
                data = json.load(f)
            payload = AppConfig.model_validate(data).model_dump(by_alias=True, exclude={"cfg_path"})
        except Exception:
            payload = None
    if payload is None:
        payload = AppConfig().model_dump(by_alias=True, exclude={"cfg_path"})

    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, payload)
    return target
