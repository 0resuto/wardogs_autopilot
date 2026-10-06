"""Hover tooltips for every tunable parameter surfaced in the studio UI.

Each entry starts with a one-line summary that names the direction of change
(`lower` / `higher`, `increase` / `decrease`) and then explains the trade-off
with concrete example values, so a hover answers both "what is this" and "what
happens if I change it". The registry is keyed by config field name (plus a few
composite keys for ROI sub-fields), and the tuning panel, the Capture tab and
the benchmark window all read from here so the wording stays in one place.
"""

from __future__ import annotations

#: Tooltip per parameter. The first line is a complete, self-contained summary
#: (the bench hints reuse it); the lines below add detail with example values.
#: Plain text with explicit line breaks: the theme's QToolTip does not wrap
#: long single lines.
PARAM_TIPS: dict[str, str] = {
    # ------------------------------------------------------------- locator
    "engine": (
        "Localization engine: hybrid only (SIFT anchor + ECC frame-to-frame\n"
        "tracking; best with capture.fps 20-30)."
    ),
    "ratio_local": (
        "Lowe ratio test: lower = more but weaker matches; higher = stricter.\n"
        "Lower (0.75): accepts weaker pairs - survives low-texture frames, but\n"
        "wrong matches can pull the pose. Higher (0.9): only confident pairs -\n"
        "cleaner pose, yet feature-poor frames are rejected."
    ),
    "min_inl_local": (
        "Inliers required: lower = more frames pass; higher = only strong ones.\n"
        "Lower (2-3): more frames localized, noisier poses. Higher (8-16): only\n"
        "strong matches pass - steadier pose, more holds and dropouts."
    ),
    "min_inl_rate_local": (
        "Minimum inlier share (0 = off): lower = tolerates outliers; higher = cleaner.\n"
        "Lower: frames with many wrong matches still localize (noisier pose).\n"
        "Higher (0.3+): only clean matches are used - more rejections on weak frames."
    ),
    "track_radius": (
        "Search radius: lower = faster/riskier; higher = robust but slower.\n"
        "Lower (300 px): cannot lock onto a similar place; risky once the vehicle\n"
        "outruns it. Higher (2000 px): survives fast motion and long holds, but\n"
        "costs time and may latch onto a wrong spot."
    ),
    "max_kp_frame": (
        "Keypoints per frame: lower = faster; higher = robust in low texture.\n"
        "Lower (600): less matching work, lower latency; fewer points in empty\n"
        "terrain. Higher (2400): steadier in texture-poor areas, higher CPU cost."
    ),
    "smooth_alpha": (
        "Smoothing weight: lower = steadier/lag; higher = jitter included (0 = raw).\n"
        "Lower (0.2-0.35): steadier marker with a slight lag while accelerating.\n"
        "Higher (0.8): follows every match closely, outliers included - keep low\n"
        "on a shaky capture."
    ),
    "smooth_reset_px": (
        "Snap-back residual: lower = jumps on jitter; higher = glides longer.\n"
        "Lower: ordinary matcher noise moves the marker jerkily. Higher: real\n"
        "relocations are slowly glided to. Keep it above the matcher noise\n"
        "(~50-150 px) so confirmed relocations show up at once."
    ),
    "ransac_px": (
        "RANSAC inlier threshold: lower = tighter pose/fewer inliers; higher = pose-drag risk.\n"
        "Lower (2 px): lazy matches are rejected - may end up with too few\n"
        "inliers. Higher (6 px): textureless frames stay localized, but a wrong\n"
        "match can pull the pose."
    ),
    "ransac_fallback_px": (
        "Second chance for failed candidates: increase = more localizations (stray risk); 0 = strict.\n"
        "Increase (6-8): dubious matches get re-approved. Decrease/down to 0:\n"
        "safer, but more dropouts on weak frames."
    ),
    "hybrid_reanchor_s": (
        "Seconds between SIFT anchors: lower = precise/CPU-heavy; higher = cheap/drifts.\n"
        "Lower (0.5 s): precise, heavier. Higher (2-5 s): lighter, ECC drift\n"
        "accumulates. 0 = anchor only when the ECC step fails."
    ),
    "hybrid_min_cc": (
        "Minimum ECC correlation: lower = cheaper, drift risk; higher = stable, more anchors.\n"
        "Lower (0.3): trusts the frame-to-frame step longer. Higher (0.7):\n"
        "falls back to SIFT as soon as the crop looks different."
    ),
    "ratio_global": (
        "Lowe ratio for relocation: lower = more candidates (false risk); higher = safest/slowest.\n"
        "Lower (0.8): relocates sooner, may pick a false look-alike. Higher\n"
        "(0.95): rarely relocates, but almost never to a wrong place."
    ),
    "min_inl_global": (
        "Inliers needed: lower = recovers sooner; higher = safer/slower.\n"
        "Lower (3): quick re-acquisition, more false fixes. Higher (10+): only\n"
        "very sure matches - recovery after a loss takes longer."
    ),
    "min_inl_rate_global": (
        "Min inlier share (0 = off): lower = busy areas pass; higher = rejects them.\n"
        "Lower: wrong-but-busy places may pass. Higher (0.3+): true matches on\n"
        "low-texture terrain may never pass either."
    ),
    "vote_need": (
        "Agreeing frames: lower = fast/risky; higher = safe/slow.\n"
        "Lower (1-2): quick recovery, occasional false jump. Higher (5+): very\n"
        "safe, slow to re-acquire after losing the pose."
    ),
    "heading_gate_deg": (
        "Heading gate (0 = off): lower = stricter vs mirrored places; higher = permissive.\n"
        "Raise to ~60-90 to reject rotated/mirrored look-alikes; set 0 to let\n"
        "the vote decide instead."
    ),
    "vote_inl_skip": (
        "Vote-skip inliers: lower = trust strong frames sooner; higher = always vote.\n"
        "Lower (20): a strong single-frame match is accepted at once (faster,\n"
        "some risk). Higher (80): even strong matches wait for votes."
    ),
    "early_inl": (
        "Early-stop inliers (0 = scan every level): lower = snappier; higher = thorough.\n"
        "Lower (20-40): stops the scale scan early. Higher: checks all levels -\n"
        "best level found, more CPU per frame."
    ),
    "hold_frames": (
        "Hold frames: lower = honest 'lost' sooner; higher = keeps stale pose.\n"
        "Lower (0-2): the marker stops as soon as matching fails. Higher (10+):\n"
        "bridges short dropouts at a stationary vehicle."
    ),
    "center_dx": (
        "Player-center X offset (minimap px): increase = right; decrease = left.\n"
        "The reported pose is the marker's location; if the map marker circles\n"
        "the true spot while turning, the player arrow is off the ROI midpoint:\n"
        "align the magenta preview crosshair with it."
    ),
    "center_dy": (
        "Player-center Y offset (minimap px): increase = down; decrease = up.\n"
        "Same circle-drift fix as DX: move the magenta preview crosshair onto\n"
        "the in-game player arrow."
    ),
    # ------------------------------------------------------------ capture
    "fps": (
        "Capture rate: lower = less CPU; higher = smoother hybrid, more load.\n"
        "Lower (5-10 fps): fast motion changes more between frames. Higher\n"
        "(20-30 fps): the hybrid engine tracks smoother."
    ),
    "monitor": (
        "Capture monitor: pick the one the game runs on.\n"
        "The zone coordinates are monitor-local, so re-pick the minimap and\n"
        "speed zones after switching."
    ),
    "mmap_roi.x": (
        "Left edge of the minimap box (monitor px): increase = right; decrease = left.\n"
        "Re-pick the zone after moving the in-game minimap; a box that no longer\n"
        "covers it localizes poorly."
    ),
    "mmap_roi.y": (
        "Top edge of the minimap box (monitor px): increase = down; decrease = up.\n"
        "Keep the whole minimap inside the box at every in-game zoom level."
    ),
    "mmap_roi.w": (
        "Width of the minimap box (px): increase = more context/slower; decrease = clip risk.\n"
        "Fit the box to the full minimap: extra border is acceptable, missing\n"
        "minimap is not."
    ),
    "mmap_roi.h": (
        "Height of the minimap box (px): increase = more context/slower; decrease = clip risk.\n"
        "Keep the full minimap inside, including its frame."
    ),
    "speed_roi.x": (
        "Left edge of the speed OCR box (monitor px): increase = right; decrease = left.\n"
        "Wrap the digits only, not the units or labels."
    ),
    "speed_roi.y": (
        "Top edge of the speed OCR box (monitor px): increase = down; decrease = up.\n"
        "Wrap the digits only, not the units or labels."
    ),
    "speed_roi.w": (
        "Width of the speed OCR box (px): increase = units misread; decrease = digits clipped.\n"
        "Fit the digits tightly; the preview shows the OCR boxes and mask."
    ),
    "speed_roi.h": (
        "Height of the speed OCR box (px): increase = background misreads; decrease = digits clipped.\n"
        "Fit the digit height, not the whole HUD widget."
    ),
    # ---------------------------------------------------------- navigator
    "yaw_gain": (
        "Model authority scale: lower = longer pulses (wobble); higher = understeer risk.\n"
        "Lower (0.5-0.7): the model predicts less yaw per pulse. Higher (1.2+):\n"
        "short light pulses; calibrate with tools/calibrate_vehicle.py."
    ),
    "corner_lat_g": (
        "Lateral budget: lower = slower corners; higher = faster/slide risk.\n"
        "Planned speed v = sqrt(lat_g*9.81*R): lower (0.25) is safe, higher\n"
        "(0.5) is faster if the truck holds the line."
    ),
    "brake_g": (
        "Brake budget: lower = earlier braking; higher = later/harder.\n"
        "Lower (0.3): brakes early and gently. Higher (0.6): trusts the brakes -\n"
        "overshoots the corner entry if the real brakes are weaker."
    ),
    "corner_min_kmh": (
        "Corner floor: lower = can crawl; higher = may understeer wide.\n"
        "Lower (5-8 km/h) clears tight turns; higher (20+) keeps momentum but\n"
        "is too fast for hairpins."
    ),
    "corner_max_kmh": (
        "Corner ceiling: lower = conservative; higher = faster/riskier.\n"
        "Inside the min..max hold window the driver neither brakes nor\n"
        "accelerates, which stops pedal hunting."
    ),
    "plan_ahead_m": (
        "Planning horizon: lower = late/hard braking; higher = early/smooth (may slow too soon).\n"
        "Lower (80-120 m): reacts close to the corner. Higher (300+ m):\n"
        "anticipates and eases off early."
    ),
    "corner_cut_m": (
        "Vertex rounding: lower = follows the polyline; higher = cuts the apex.\n"
        "Lower (5-10 m): safe line, slower bends. Higher (25 m): carries speed\n"
        "through click corners, may clip obstacles."
    ),
    "xte_m": (
        "Centering dead band: below it the pose jitter gets no wheel command.\n"
        "lower (0.3-0.8 m) = tighter line, slightly busier wheel; higher\n"
        "(1.5+ m) = lets the car drift before correcting."
    ),
    "xte_outer_m": (
        "Centering reference: the wheel, urgency and speed response ramp from\n"
        "the dead band up to this and saturate past it (one gradient corridor).\n"
        "lower (2-3 m) = firm early corrections, slows sooner when off line;\n"
        "higher (6+ m) = calmer, tolerates wide lines."
    ),
    "steer_look_s": (
        "Steering lookahead: lower = tight/active; higher = smooth/cuts bends.\n"
        "Lower (0.8-1.2 s): aims close - tight line, busy wheel. Higher\n"
        "(2.5+ s): aims far - smooth, cuts the inside of curves."
    ),
    "settle_s": (
        "Steering pause: lower = busy wheel; higher = smooth but can drift.\n"
        "Lower (0.2 s): frequent small corrections. Higher (1.0+ s): no\n"
        "steering during the pause; at speed it is distance-capped (~8 m)."
    ),
    "steer_lead_s": (
        "Release anticipation: lower = holds longer; higher = releases earlier.\n"
        "Raise if the truck systematically overshoots (heading still arrives\n"
        "through the latency); lower if it releases too early and drifts."
    ),
    "skip_ahead_m": (
        "Rejoin window (0 = off): lower = small cut-offs only; higher = may skip route.\n"
        "Only when outside the outer corridor the active point may jump forward\n"
        "to the nearest route point within this route length. Bigger windows\n"
        "recover from larger detours, but may skip junctions."
    ),
    # --------------------------------------------------------------- bench
    "bench.frames": (
        "Frames per scenario: lower = quick smoke test; higher = tighter stats.\n"
        "Lower (20-60): fast feedback. Higher (150-300): smaller confidence\n"
        "intervals, longer runs."
    ),
    "bench.fps": (
        "Synthetic rate: lower = studio default; higher = heavier tracking.\n"
        "Lower (10 Hz): matches the app default. Higher (30 Hz): more frames\n"
        "per second of motion (the Full matrix runs 10 and 30 Hz)."
    ),
    "bench.speed": (
        "Synthetic speed: lower = easy/small steps; higher = stresses tracking.\n"
        "Lower (5-10 m/s): small inter-frame motion. Higher (15-20 m/s): big\n"
        "steps between frames (the Full matrix runs 10 and 20 m/s)."
    ),
}


def tip_for(key: str, fallback: str = "") -> str:
    """Tooltip for a parameter, or `fallback` when the key is not registered."""
    return PARAM_TIPS.get(str(key), fallback)


def summary_for(key: str, fallback: str = "") -> str:
    """First (direction-of-change) line of a parameter tooltip."""
    text = tip_for(key, fallback)
    return text.split("\n", 1)[0] if text else fallback
