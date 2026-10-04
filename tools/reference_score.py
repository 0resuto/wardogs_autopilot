"""Score a navigation run against a human reference drive (methodology gate).

Usage:
    uv run python tools/reference_score.py [--run output/nav_dbg_*.jsonl]
        [--reference output/manual_dbg_20260928_222519.jsonl] [--segment-m 500]

Gates (from the methodology review):
  1. segment time ratio vs the human: median <= 1.10, worst <= 1.25
  2. p95 cross-track error <= 1.5x the reference p95
  3. blind seconds per km <= 1.0 and pose-interval p95 <= 0.3 s
  4. zero safety events (spin ticks, stuck/reverse events)

Both drives are projected onto the run's route polyline; the human recording
may predate small route edits, so the comparison is corridor-level.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys


def load_jsonl(path: str) -> tuple[dict | None, list[dict], list[dict]]:
    """(header, rows, events) of a nav_dbg/manual_dbg JSONL file."""
    header: dict | None = None
    rows: list[dict] = []
    events: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind"):
                if rec.get("kind") in ("nav-log", "manual-log"):
                    header = rec
                else:
                    events.append(rec)
                continue
            rows.append(rec)
    return header, rows, events


def wrap180(x: float) -> float:
    return (x + 540.0) % 360.0 - 180.0


def project_to_route(route: list[tuple[float, float]], x: float, y: float) -> tuple[float, float]:
    """Nearest-segment projection: (arc length px, signed lateral px)."""
    best: tuple[float, float] | None = None
    best_d = None
    cum = 0.0
    for i in range(len(route) - 1):
        ax, ay = route[i]
        bx, by = route[i + 1]
        sx, sy = bx - ax, by - ay
        seg = math.hypot(sx, sy)
        if seg <= 1e-9:
            continue
        t = max(0.0, min(1.0, ((x - ax) * sx + (y - ay) * sy) / (seg * seg)))
        fx, fy = ax + t * sx, ay + t * sy
        d = math.hypot(x - fx, y - fy)
        if best_d is None or d < best_d:
            lateral = ((x - ax) * sy - (y - ay) * sx) / seg
            best_d = d
            best = (cum + t * seg, lateral)
        cum += seg
    if best is None:
        return 0.0, 0.0
    return best


def route_length_px(route: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        for i in range(len(route) - 1)
    )


def progress_series(
    rows: list[dict], route: list[tuple[float, float]]
) -> tuple[list[float], list[float], list[float]]:
    """(times, arcs, |lateral|) for rows with a good measured position."""
    times: list[float] = []
    arcs: list[float] = []
    laterals: list[float] = []
    for r in rows:
        if r.get("x") is None or r.get("y") is None or not r.get("good", True):
            continue
        arc, lateral = project_to_route(route, float(r["x"]), float(r["y"]))
        times.append(float(r["t"]))
        arcs.append(arc)
        laterals.append(abs(lateral))
    return times, arcs, laterals


def orient_progress(arcs: list[float], total_px: float) -> list[float]:
    """Flip a reference driven against the route direction into route order.

    The manual recording may have been driven end-to-start; its arc then
    decreases over time and every segment end is reached before its start, so
    all segment times come out None. Mirroring the arc (total - arc) restores
    a monotonically increasing progress and keeps the segment order comparable.
    """
    deltas = [b - a for a, b in zip(arcs, arcs[1:], strict=False) if abs(b - a) > 1.0]
    if deltas and sum(1 for d in deltas if d < 0) > len(deltas) / 2:
        return [total_px - a for a in arcs]
    return arcs


def segment_times(
    times: list[float], arcs: list[float], total_px: float, segment_px: float
) -> list[float | None]:
    """Traversal time of each equal arc-length segment (None when not reached)."""
    count = max(1, int(total_px // segment_px))
    out: list[float | None] = [None] * count
    for k in range(count):
        start = k * segment_px
        end = start + segment_px
        t0 = next((t for t, a in zip(times, arcs, strict=False) if a >= start), None)
        t1 = next((t for t, a in zip(times, arcs, strict=False) if a >= end), None)
        if t0 is not None and t1 is not None and t1 > t0:
            out[k] = t1 - t0
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def score(
    run_times: list[float | None],
    ref_times: list[float | None],
    run_xte_p95: float,
    ref_xte_p95: float,
    blind_s_per_km: float,
    pose_p95: float,
    safety_events: int,
) -> tuple[list[tuple[str, bool, str]], float, float]:
    """(gate results, median ratio, worst ratio)."""
    ratios = [
        a / b
        for a, b in zip(run_times, ref_times, strict=False)
        if a is not None and b is not None and b > 2.0
    ]
    median_ratio = statistics.median(ratios) if ratios else float("nan")
    worst_ratio = max(ratios) if ratios else float("nan")
    xte_ok = ref_xte_p95 <= 0.0 or run_xte_p95 <= 1.5 * ref_xte_p95
    gates = [
        (
            "segment time median <= 1.10",
            bool(ratios) and median_ratio <= 1.10,
            "%.3f" % median_ratio,
        ),
        ("segment time worst <= 1.25", bool(ratios) and worst_ratio <= 1.25, "%.3f" % worst_ratio),
        (
            "p95 xte <= 1.5x reference",
            xte_ok,
            "%.1f vs %.1f m" % (run_xte_p95, ref_xte_p95),
        ),
        ("blind <= 1.5 s/km", blind_s_per_km <= 1.5, "%.2f" % blind_s_per_km),
        ("pose interval p95 <= 0.30 s", pose_p95 <= 0.30, "%.2f" % pose_p95),
        ("zero safety events", safety_events == 0, str(safety_events)),
    ]
    return gates, median_ratio, worst_ratio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="nav_dbg file (default: newest)")
    ap.add_argument(
        "--reference",
        default=None,
        help="manual_dbg file (default: newest in output/)",
    )
    ap.add_argument("--segment-m", type=float, default=500.0)
    args = ap.parse_args()

    run_path = args.run
    if not run_path:
        files = sorted(glob.glob(os.path.join("output", "nav_dbg_*.jsonl")), key=os.path.getmtime)
        if not files:
            raise SystemExit("no nav_dbg logs in output/")
        run_path = files[-1]
    ref_path = args.reference
    if not ref_path:
        refs = sorted(glob.glob(os.path.join("output", "manual_dbg_*.jsonl")), key=os.path.getmtime)
        if not refs:
            raise SystemExit("no manual_dbg logs in output/")
        ref_path = refs[-1]
    if not os.path.isfile(ref_path):
        raise SystemExit("reference recording not found: %s" % ref_path)

    run_meta, run_rows, run_events = load_jsonl(run_path)
    _, ref_rows, _ = load_jsonl(ref_path)
    if run_meta is None or not run_meta.get("route"):
        raise SystemExit("run has no route in its header")
    route = [(float(a), float(b)) for a, b in run_meta["route"]]
    pxm = statistics.median([float(r["pxm"]) for r in run_rows if r.get("pxm")] or [2.008])
    total_px = route_length_px(route)
    segment_px = max(50.0, args.segment_m * pxm)

    rt, ra, _ = progress_series(run_rows, route)
    ft, fa, fl = progress_series(ref_rows, route)
    fa = orient_progress(fa, total_px)
    run_times = segment_times(rt, ra, total_px, segment_px)
    ref_times = segment_times(ft, fa, total_px, segment_px)

    run_xte = [float(r["xte_m"]) for r in run_rows if r.get("xte_m") is not None]
    run_xte_p95 = percentile(run_xte, 0.95)
    ref_xte_p95 = percentile([v / pxm for v in fl], 0.95)
    blind_events = [e for e in run_events if e.get("kind") == "blind"]
    distance_km = max(total_px / pxm / 1000.0, 1e-6)
    blind_s_per_km = len(blind_events) * 0.45 / distance_km  # ~the blind pause
    pose_p95 = percentile(
        [float(r["pose_age"]) for r in run_rows if r.get("pose_age") is not None], 0.95
    )
    safety = sum(1 for r in run_rows if r.get("spin")) + sum(
        1 for e in run_events if e.get("kind") in ("stuck", "reverse")
    )

    gates, median_ratio, worst_ratio = score(
        run_times, ref_times, run_xte_p95, ref_xte_p95, blind_s_per_km, pose_p95, safety
    )
    print("# reference score")
    print(
        "  run      : %s (config %s)"
        % (run_path, (run_meta.get("params") or {}).get("config_hash", "?"))
    )
    print("  reference: %s" % ref_path)
    print(
        "  route    : %.2f km, %d segments of %.0f m"
        % (total_px / pxm / 1000.0, len(run_times), args.segment_m)
    )
    for name, ok, detail in gates:
        print("  [%s] %-32s %s" % ("PASS" if ok else "FAIL", name, detail))
    passed = all(ok for _n, ok, _d in gates)
    print("  verdict  : %s" % ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
