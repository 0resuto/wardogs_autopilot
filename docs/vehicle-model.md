# Ural vehicle model - physics sanity

Source: the reverse-engineered pack (`wardogs-vehicle-physics`), trimmed into
`data/vehicles/ural.json` by `tools/extract_vehicle_profile.py`.

## Validated against the pack's own stats

| Check | Result |
| --- | --- |
| Top-speed identity `max_rpm / (top_gear * final_drive) / 60 * 2*pi*r` | 79.0 km/h = the declared `stats_ui` value (r = 0.649 m, which is also present as a wheel token) |
| Gear ratios + final drive 7.32 | match the real YaMZ final drive |
| Torque 900 at 1200 rpm, max 2600 rpm | in the range of the real YaMZ-236/238 diesel |

## Envelope cross-checks vs a real Ural-like truck

| Quantity | Model | Real-world reference | Note |
| --- | --- | --- | --- |
| Top speed | 79 km/h | 80-85 km/h | matches |
| Braking 79 -> 0 at 0.45g | ~54 m | ~40-60 m loaded | plausible; `brake_g` is a tuning knob |
| Lateral 0.35g | R ~56 m at 50 km/h, ~140 m at 79 km/h | trucks ~50-70 m at 50 km/h | plausible |
| Full-lock turn radius | R = L / tan(35 deg) ~ 5.4 m (rear axle) | ~10 m outer | `steer_angle_max_deg` is a nominal knob |
| 1st gear drive force | ~51 kN at peak torque | exceeds tire grip | wheelspin at full throttle in low gear is expected; the game likely assists |

Mass is not in the pack: the effective mass is pinned by an in-game
acceleration run and folded into the calibration constants.

## Grip-limited yaw rate

The kinematic (bicycle) yaw rate is only an upper bound - the tires cap it:

    yaw_max(v) = min( v * tan(steer_angle_max * steer_limit(v)) / L, a_lat / v )

Defaults: L = 3.8 m, steer_angle_max = 35 deg, a_lat = 0.35g.

| Speed | Bicycle model | Grip cap | Effective |
| --- | --- | --- | --- |
| 5 km/h | 13.9 deg/s | 141.6 deg/s | 13.9 (geometry) |
| 50 km/h | 90.6 deg/s | 14.2 deg/s | 14.2 (grip) |
| 79 km/h | 108.3 deg/s | 9.0 deg/s | 9.0 (grip) |

Without the cap the controller assumes a six-fold higher turn rate at mid
speeds and releases the wheel far too early. The cap is applied in
`FollowDriver._yaw_rate_max` from the planner's lateral budget
(`corner_lat_g`), so steering pulses follow the real grip envelope.
