"""Extract a trimmed, app-ready vehicle profile from the physics data pack.

The pack (reverse-engineered cooked assets) lives outside the repository; this
tool keeps only the high-confidence dynamics fields and derives the wheel
radius from the validated top-speed identity:

    v_max = max_rpm / (top_gear * final_drive) / 60 * 2*pi*r

Usage:
    python tools/extract_vehicle_profile.py --vehicle WHL_07 --out data/vehicles/ural.json
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = r"C:\Users\ORESUTO_PC\Desktop\wardogs-vehicle-physics"

CALIBRATION_DEFAULTS = {
    "yaw_gain": 1.0,
    "wheelbase_m": 3.8,
    "steer_angle_max_deg": 35.0,
    "drive_efficiency": 0.9,
    # Gray-box longitudinal calibration: accel = drive_accel_scale * F_drive
    # - resist_a - resist_b * v^2 - brake_decel on SPACE. The brake and yaw
    # values are refitted from recorded runs by
    # .agents/pending/calibrate_vehicle_predictor.py; the drive scale is provisional from
    # the reference manual drive (~1.7 m/s^2 at 60-70 km/h) and should be
    # refitted with a controlled throttle-step test on flat ground.
    "drive_accel_scale": 0.00022,  # m/s^2 per model force unit
    "resist_a_mps2": 0.15,
    "resist_b": 0.0003,
    "brake_decel_mps2": 5.0,
    "brake_yaw_gain": 1.8,  # measured p90 yaw under SPACE at speed
    # Measured net acceleration/deceleration tables (km/h -> m/s^2), fitted
    # from controlled manual recordings by the pending predictor calibrator.
    # Empty means "use the physics force model above".
    "drive_accel_curve": [],
    "coast_decel_curve": [],
}


def curve_pairs(keys):
    """(x, y) pairs from either {'t','v'} or {'Time','Value'} curve keys."""
    pairs = []
    for key in keys:
        if not isinstance(key, dict):
            continue
        if "Time" in key and "Value" in key:
            pairs.append([float(key["Time"]), float(key["Value"])])
        elif "t" in key and "v" in key:
            pairs.append([float(key["t"]), float(key["v"])])
    return pairs


def extract(source, vehicle_id):
    path = os.path.join(source, "vehicles", f"vehicle_{vehicle_id}.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    et = data.get("engine_transmission", {})
    gears = [float(g) for g in et.get("gear_ratios_in_file_order", [])]
    if len(gears) < 2:
        raise SystemExit(f"{vehicle_id}: no gear ratios in {path}")
    forward = gears[:-1]
    reverse = gears[-1]
    final_drive = float(et["final_drive_ratio"])
    idle_rpm = float(et.get("idle_rpm") or 0.0)
    max_rpm = float(et.get("max_rpm") or 0.0)
    if max_rpm <= 0:
        raise SystemExit(f"{vehicle_id}: max rpm is not stored; cannot derive wheel radius")

    stats = data.get("stats_ui", {}).get("values_in_order") or []
    max_speed_kmh = float(stats[0]) if stats else 0.0

    top_gear = min(forward)
    wheel_radius = (max_speed_kmh / 3.6) * top_gear * final_drive / (2.0 * math.pi * max_rpm / 60.0)

    curves = data.get("curves", {})
    torque = curve_pairs(curves.get("torque", {}).get("keys", []))
    steer_limit = curve_pairs(curves.get("steer_limit", {}).get("keys", []))
    steer_speed = curve_pairs(curves.get("steer_speed", {}).get("keys", []))

    tire_peak = 0.0
    tire_slip: list[list[float]] = []
    for slip in data.get("data_assets", {}).get("tire_model", {}).get("slip_curves", []):
        samples = slip.get("samples", [])
        tire_slip = [[round(float(c), 4), round(float(f), 3)] for c, f in samples[::2]]
        for _slip, force in samples:
            tire_peak = max(tire_peak, abs(float(force)))
    if tire_slip and tire_slip[-1][0] != float(
        data["data_assets"]["tire_model"]["slip_curves"][0]["samples"][-1][0]
    ):
        last = data["data_assets"]["tire_model"]["slip_curves"][0]["samples"][-1]
        tire_slip.append([round(float(last[0]), 4), round(float(last[1]), 3)])

    # The game's own client-prediction constants (class defaults, identical
    # across all wheeled vehicles; field names need a .usmap schema).
    prediction_raw = data.get("data_assets", {}).get("prediction", {}).get("float_tokens", [])
    prediction_values = [round(float(v), 4) for _o, v in prediction_raw if abs(float(v)) > 1e-9]

    profile = {
        "vehicle_id": vehicle_id,
        "name": data.get("name", vehicle_id),
        "source": "wardogs-vehicle-physics pack (cooked asset decode)",
        "confidence": (
            "gears/rpm/torque/steer curves: high; wheel radius: derived; "
            "brake/yaw: fitted from recorded runs; drive accel: provisional"
        ),
        "max_speed_kmh": max_speed_kmh,
        "wheel_radius_m": round(wheel_radius, 4),
        "gears_forward": forward,
        "gear_reverse": reverse,
        "final_drive": final_drive,
        "idle_rpm": idle_rpm,
        "max_rpm": max_rpm,
        "torque_nm": torque,
        "steer_limit_pct": steer_limit,
        "steer_speed_mult": steer_speed,
        "brake_engagement": curve_pairs(curves.get("brake_engagement", {}).get("keys", [])),
        "clutch_engagement": curve_pairs(curves.get("clutch_engagement", {}).get("keys", [])),
        "throttle_engagement": curve_pairs(curves.get("throttle_engagement", {}).get("keys", [])),
        "tire_slip_curve": tire_slip,
        "tire_peak_force": tire_peak,
        "game_prediction_defaults": {
            "values": prediction_values,
            "note": "DA_PredictionSettings_* class defaults (identical for all "
            "wheeled vehicles); labels require the game's .usmap schema",
        },
        "calibration": dict(CALIBRATION_DEFAULTS),
    }
    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="physics pack directory")
    ap.add_argument("--vehicle", default="WHL_07", help="vehicle id, e.g. WHL_07")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "vehicles", "ural.json"))
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.source, "vehicles")):
        raise SystemExit(f"physics pack not found under {args.source}")

    profile = extract(args.source, args.vehicle)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)

    r = profile["wheel_radius_m"]
    v_check = (
        profile["max_rpm"]
        / (min(profile["gears_forward"]) * profile["final_drive"])
        / 60.0
        * 2.0
        * math.pi
        * r
        * 3.6
    )
    print(f"wrote {args.out}")
    print(
        f"{profile['name']}: gears={profile['gears_forward']} final={profile['final_drive']} "
        f"r={r:.3f}m top_check={v_check:.1f}km/h (declared {profile['max_speed_kmh']:.1f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
