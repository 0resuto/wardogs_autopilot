"""Deterministic synthetic trajectories with known ground truth.

The bench never records real game frames: it renders player-up minimap frames
from a preview level of the map (`preview_16384`, native/2) along a known path,
so every localization engine can be scored against the exact position and
compass heading it was shown. The frame builder is the one verified by
`tools/selfcheck_hybrid.py`; rendering from the preview (instead of `mu`) keeps
a deliberate resampling gap against the feature index built from `mu`.

The trajectory model is intentionally simple and reproducible: constant speed
along the compass heading, a straight leg of `straight_s` seconds, then a
constant yaw rate (`mixed_s` alternates the yaw sign every 2 s). Optional
perturbations (UI mask, brightness, blur, noise) and flat "void" frames emulate
the capture problems the live pipeline has to survive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

#: Synthetic capture frame size (the app's default minimap ROI).
W, H = 336, 277

#: The shipped preview level is native/2, so mu px = native px / 2.
PREVIEW_FACTOR = 2.0

#: Window size `_pick_center` searches for (matches selfcheck_features.py).
_PICK_H, _PICK_W = 278, 337

#: Scenario names the bench knows about (see default_specs).
SCENARIO_NAMES = ("straight", "turns", "void", "mixed_s")

_SIFT: cv2.SIFT | None = None


def _sift() -> cv2.SIFT:
    """Detector used only to score the texture of candidate map windows."""
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT.create(nfeatures=6000, contrastThreshold=0.05, edgeThreshold=12)
    return _SIFT


def _crop(mu: np.ndarray, cx: int, cy: int, h: int = _PICK_H, w: int = _PICK_W) -> np.ndarray:
    return mu[cy - h // 2 : cy - h // 2 + h, cx - w // 2 : cx - w // 2 + w]


def _pick_center(mu: np.ndarray) -> tuple[int, int] | None:
    """A well-textured window center near the map center (mu px), or None."""
    h, w = mu.shape
    best, best_kp = None, -1
    r = 700
    for cy in range(h // 2 - r, h // 2 + r + 1, 256):
        for cx in range(w // 2 - r, w // 2 + r + 1, 256):
            win = _crop(mu, cx, cy)
            if win.shape != (_PICK_H, _PICK_W):
                continue
            kp, _ = _sift().detectAndCompute(win, None)
            n = 0 if kp is None else len(kp)
            if n > best_kp:
                best, best_kp = (cx, cy), n
            if n > 80:
                best, best_kp = (cx, cy), n
                break
    return best


def _build_frame(
    preview: np.ndarray, native_xy: tuple[float, float], heading_deg: float, ms: float
) -> np.ndarray:
    """Player-up minimap frame around native_xy with the given compass heading.

    Measured convention: rotating the content by the cv2 angle equal to the
    compass heading yields pose `th == heading`.
    """
    px, py = native_xy[0] / PREVIEW_FACTOR, native_xy[1] / PREVIEW_FACTOR
    half_w, half_h = W * ms / 4.0, H * ms / 4.0
    x0, y0 = int(round(px - half_w)), int(round(py - half_h))
    crop = preview[
        y0 : y0 + int(round(H * ms / 2)),
        x0 : x0 + int(round(W * ms / 2)),
    ]
    mm = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    rot = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), heading_deg, 1.0)
    return cv2.warpAffine(mm, rot, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _yaw_rate(spec: ScenarioSpec, t: float) -> float:
    """Heading rate (deg/s) at scenario time `t` for the scenario's profile."""
    if spec.name == "straight" or t < spec.straight_s:
        return 0.0
    if spec.name == "mixed_s":
        half = int((t - spec.straight_s) // 2.0)
        return spec.yaw_deg_s * (1.0 if half % 2 == 0 else -1.0)
    return spec.yaw_deg_s


@dataclass(frozen=True)
class ScenarioSpec:
    """One synthetic trajectory profile plus its capture perturbations."""

    name: str
    frames: int = 60
    fps: float = 10.0
    speed_mps: float = 15.0
    heading0: float = 30.0
    yaw_deg_s: float = 12.0
    straight_s: float = 2.0
    blur_sigma: float = 0.6
    noise_sigma: float = 2.0
    brightness: float = 1.0
    ui_mask: bool = False
    void_frames: tuple[int, ...] = ()
    seed: int = 0


@dataclass
class Scenario:
    """Rendered frames of a trajectory with their ground truth."""

    spec: ScenarioSpec
    frames: list[np.ndarray]
    truth: list[tuple[float, float, float]]
    dt: float
    mask: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frames)


def build_scenario(
    spec: ScenarioSpec,
    mu: np.ndarray,
    preview: np.ndarray,
    ms: float,
    center_mu: tuple[float, float],
    mask: np.ndarray | None = None,
    *,
    px_per_m: float,
) -> Scenario:
    """Render `spec` into frames around `center_mu` (mu px).

    `px_per_m` is the map's native-px-per-meter scale (MapStore.px_per_m); it
    turns the scenario speed into map pixels. `mask` is the UI overlay mask at
    its native resolution; it is resized to the frame and painted over when the
    spec asks for it.
    """
    dt = 1.0 / float(spec.fps)
    step = spec.speed_mps * dt * px_per_m
    rng = np.random.default_rng(spec.seed)
    ui = _resize_mask(mask) if mask is not None else None

    native = np.array([float(center_mu[0]) * ms, float(center_mu[1]) * ms], np.float64)
    heading = float(spec.heading0) % 360.0
    frames: list[np.ndarray] = []
    truth: list[tuple[float, float, float]] = []

    for k in range(spec.frames):
        frame = _build_frame(preview, (float(native[0]), float(native[1])), heading, ms)
        frame = _perturb(frame, spec, rng, ui)
        if k in spec.void_frames:
            fill = int(round(float(np.median(frames[k - 1] if k > 0 else frame))))
            frame = np.full_like(frame, fill)
        frames.append(frame)
        truth.append((float(native[0]), float(native[1]), heading))

        heading = (heading + _yaw_rate(spec, k * dt) * dt) % 360.0
        native = native + step * np.array(
            [math.sin(math.radians(heading)), -math.cos(math.radians(heading))]
        )

    return Scenario(spec=spec, frames=frames, truth=truth, dt=dt, mask=ui)


def _resize_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """UI mask as a boolean array at the frame resolution (like the producer)."""
    if mask is None:
        return None
    ui = np.asarray(mask)
    if ui.shape[:2] == (H, W):
        return ui > 0
    return cv2.resize(ui.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0


def _perturb(
    frame: np.ndarray, spec: ScenarioSpec, rng: np.random.Generator, ui: np.ndarray | None
) -> np.ndarray:
    """UI mask -> brightness -> blur -> noise, in that order."""
    out = frame
    if spec.ui_mask and ui is not None:
        out = out.copy()
        out[ui] = int(round(float(np.median(out))))
    if spec.brightness != 1.0:
        out = np.clip(out.astype(np.float32) * spec.brightness, 0, 255).astype(np.uint8)
    if spec.blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), spec.blur_sigma)
    if spec.noise_sigma > 0:
        noisy = rng.normal(0.0, spec.noise_sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noisy, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def default_specs(quick: bool = True) -> list[ScenarioSpec]:
    """The curated scenario set: quick = straight/turns/void, full adds mixed_s."""
    frames = 60 if quick else 150
    specs = [
        ScenarioSpec(name="straight", frames=frames, fps=10.0, speed_mps=15.0, yaw_deg_s=0.0),
        ScenarioSpec(name="turns", frames=frames, fps=10.0, speed_mps=15.0),
        ScenarioSpec(
            name="void", frames=frames, fps=10.0, speed_mps=15.0, void_frames=(20, 35, 50)
        ),
    ]
    if not quick:
        specs.append(ScenarioSpec(name="mixed_s", frames=frames, fps=10.0, speed_mps=15.0))
    return specs


def spec_for(name: str, **overrides) -> ScenarioSpec:
    """The curated spec for `name` with fields replaced (frames/fps/speed/...)."""
    for spec in default_specs(quick=False) + default_specs(quick=True):
        if spec.name == name:
            return replace(spec, **overrides)
    raise ValueError(f"unknown scenario: {name} (valid: {', '.join(SCENARIO_NAMES)})")
