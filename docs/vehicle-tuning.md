# Vehicle tuning runbook

The vehicle model (`data/vehicles/ural.json`) gives the right *shape* of the
dynamics (gearing, torque, steering authority), but the absolute scales —
steering gain, cornering budget, braking budget — must be pinned in game.

Open **Map -> Tuning -> Vehicle** to edit the values; **Apply** updates a
running autopilot live and writes them to `config.json`.

| Field | Meaning | Default |
| --- | --- | --- |
| `gain` (`yaw_gain`) | Scales the model's steering authority | 1.0 |
| `lat g` (`corner_lat_g`) | Lateral grip budget for corners | 0.35 |
| `brake g` (`brake_g`) | Braking deceleration budget | 0.45 |
| `min km/h` (`corner_min_kmh`) | Slowest planned speed in a corner | 12 |
| `ahead m` (`plan_ahead_m`) | Speed planning horizon | 200 |
| `cut m` (`corner_cut_m`) | Distance used to round a sharp vertex | 15 |

`vehicle_profile` and `speed_profile` are restart-level switches in
`config.json` (the Tuning panel does not toggle them live).

## Prerequisites

- The speedometer OCR zone is set up (Capture zone -> Speedometer) and shows a
  live value in the diagnostic panel.
- The **Nav log** checkbox in Map -> Tuning is enabled; runs land in
  `output/nav_dbg_*.jsonl` (replay with `python tools/nav_dbg.py --tail`).
- F6 starts following, F7 is the emergency stop.

## Run 1 - straight acceleration (px_per_m)

1. Pick a long straight road; place a 2-point route of roughly 1 km.
2. Start with F6 and let the truck accelerate. Stop with F7 before the end.
3. In the log, compare `ocr` (km/h) with the position-derived `mv` (px/s) while
   the speed is steady: `px_per_m = mv / (ocr / 3.6)`. The average over several
   seconds is what the planner uses for corner radii and distances.

## Run 2 - full braking (brake g)

1. Use a route whose final waypoint is reached at 50-70 km/h so the final-stop
   phase engages (it logs `kind="final_stop"` lines with `ocr`).
2. From the log, take the slope of `ocr` over time during the SPACE phase:
   `brake_mps2 = (kmh0 - kmh1) / 3.6 / dt`. Set `brake g` to roughly that value
   divided by 9.81, minus a safety margin (start 10-15% lower).

## Run 3 - steering authority (yaw_gain)

1. Drive any route with moderate curves for a few minutes with the nav log on.
2. In the log, take ticks where `keys` contains `A` or `D` and compare the
   heading change per second (from consecutive `heading` values) with the
   logged `yaw_max` for the same tick.
3. If the measured yaw is consistently lower than `yaw_max`, reduce
   `gain` by that ratio; if the truck wobbles left/right around the line,
   raise it slightly. Typical starting points: 0.5-0.7.

## Run 4 - corner and braking budgets

1. Build a test route with one 90-degree turn and one hairpin.
2. Watch `plan_kmh` and the actual line in the log:
   - drifts wide or lifts wheels -> lower `lat g` and/or `min km/h`;
   - slows far too early -> raise `lat g`;
   - cuts the apex of a sharp click corner -> lower `cut m`;
   - crawls through gentle bends -> raise `ahead m`.
3. A comfortable truck default is around `lat g` 0.3-0.4 and `brake g`
   0.4-0.5; adjust from there.

## Automatic fitting

After a few logged runs:

```bash
python tools/calibrate_vehicle.py output/nav_dbg_*.jsonl
python tools/calibrate_vehicle.py output/nav_dbg_*.jsonl --write
```

The report gives `px_per_m`, measured/suggested `brake_g` and a suggested
`yaw_gain`; `--write` stores the suggestions into `config.json` (`yaw_gain`
and `brake_g`, atomically). `px_per_m` is informational: the runtime keeps
estimating the scale by itself.

## Log fields worth watching

`ocr` (measured km/h), `plan_kmh` (planned limit), `tgt` (final target px/s),
`yaw_max` (model authority deg/s), `keys`, `heading`, `th_raw`, `mv`, `xte_m`,
`good` (pose measured), `dist`.
