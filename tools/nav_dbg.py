"""Analyze the navigator JSONL log (output/nav_dbg_*.jsonl).

Usage:
    python tools/nav_dbg.py [path-to-file]
    python tools/nav_dbg.py --summary    # KPI: slow share, heading source,
                                         # frozen mh, overshoot, gas-cut steering
    python tools/nav_dbg.py --tail 120   # show last N ticks
    python tools/nav_dbg.py --bursts 40  # table of last N steering corrections

By default it takes the newest log in output/.
Purpose: see how far the steering decision lags behind the physics — pose state
age (pose_age), steering hold time (hold), sync with new samples (new_sample),
and overshoot after release (extreme err in the settle window).
"""

import argparse
import glob
import json
import os
import sys


def load(path: str):
    meta = None
    rows = []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "nav-log":
                meta = rec
            elif rec.get("kind"):
                events.append(rec)
            else:
                rows.append(rec)
    return meta, rows, events


def wrap180_deg(x: float) -> float:
    return (x + 540.0) % 360.0 - 180.0


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (0.0, 0.0, 0.0)
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    p95 = s[min(n - 1, int(n * 0.95))]
    return (mean, p95, max(s))


def pct(s, t):
    return 100.0 * s / t if t else 0.0


def row_vkmh(row: dict) -> float | None:
    """Track speed in km/h: the logged value, else px/s with the logged scale."""
    v = row.get("vkmh")
    if v is not None:
        return float(v)
    mv = row.get("v")
    pxm = row.get("pxm") or 0.0
    if mv is None or pxm <= 0.0:
        return None
    return float(mv) * 3.6 / float(pxm)


def overshoots(bursts: list) -> list:
    """Signed extreme error in the release window after each correction."""
    out = []
    for _b, w in bursts:
        if w:
            out.append(max(min(r["err"] for r in w), -min(r["err"] for r in w), key=abs))
    return out


def steering_runs(rows: list) -> list:
    """Consecutive ticks with a non-zero steering command (one correction each)."""
    runs = []
    current = []
    for r in rows:
        if r["steer"] != 0:
            current.append(r)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def print_summary(rows: list, bursts: list, slow_kmh: float) -> None:
    """Compact run KPI: the numbers that separate control faults from noise."""
    n = len(rows)
    slow = sum(1 for r in rows if (v := row_vkmh(r)) is not None and v < slow_kmh)
    sources: dict[str, int] = {}
    for r in rows:
        src = r.get("head_src") or ("mh" if r.get("mode") == "M" else "pose")
        sources[src] = sources.get(src, 0) + 1
    src_txt = "  ".join("%s=%.0f%%" % (k, pct(v, n)) for k, v in sorted(sources.items()))

    checked = disagree = 0
    disagree_max = 0.0
    for r in rows:
        # The control heading (what the wheel actually follows), not the raw
        # mh: the raw value is expected to be stale while the pose steers.
        if r.get("heading") is None or r.get("th_raw") is None:
            continue
        checked += 1
        d = abs(wrap180_deg(r["heading"] - r["th_raw"]))
        disagree_max = max(disagree_max, d)
        if d > 20.0:
            disagree += 1

    ages = [r["mh_age"] for r in rows if r.get("mh_age") is not None]
    stale = sum(1 for a in ages if a > 1.0)

    held_nogas = low_nogas = 0
    for r in rows:
        keys = r.get("keys") or ""
        if ("A" in keys or "D" in keys) and "W" not in keys:
            held_nogas += 1
            v = row_vkmh(r)
            if v is not None and v < slow_kmh:
                low_nogas += 1

    over = overshoots(bursts)
    o = stats(over) if over else (0.0, 0.0, 0.0)

    brake = sum(1 for r in rows if r.get("braking") or "SPACE" in (r.get("keys") or ""))
    print("# summary")
    print("  slow ticks <%.0f km/h:   %d/%d (%.0f%%)" % (slow_kmh, slow, n, pct(slow, n)))
    print("  brake (SPACE) ticks:    %d/%d (%.0f%%)" % (brake, n, pct(brake, n)))
    print("  heading source:         %s" % src_txt)
    if ages:
        print("  mh idle >1 s (pose steers): %d ticks (%.0f%%)" % (stale, pct(stale, len(ages))))
    if checked:
        print(
            "  control vs map >20:     %d/%d (%.0f%%), max %.0f deg"
            % (disagree, checked, pct(disagree, checked), disagree_max)
        )
    print(
        "  wheel held, gas cut:    %d/%d (%.0f%%), of them slow: %d"
        % (held_nogas, n, pct(held_nogas, n), low_nogas)
    )
    print("  corrections: %d; overshoot |err| mean=%.1f p95=%.1f max=%.1f deg" % (len(bursts), *o))


def bursts_of(rows):
    """Split into steering corrections (continuous steer != 0). Each correction
    carries a window after it (settle) to measure the overshoot."""
    bursts = []
    i = 0
    n = len(rows)
    while i < n:
        if rows[i]["steer"] != 0:
            start = i
            while i < n and rows[i]["steer"] != 0:
                i += 1
            end = i - 1  # last tick with the steering held
            b = rows[start : end + 1]
            # window after release: until the next hold, capped at 1.2 s
            w = []
            j = end + 1
            while j < n and rows[j]["steer"] == 0 and rows[j]["t"] - rows[end]["t"] <= 1.2:
                w.append(rows[j])
                j += 1
            bursts.append((b, w))
        else:
            i += 1
    return bursts


def main():
    ap = argparse.ArgumentParser(description="Analyze nav JSONL")
    ap.add_argument(
        "path", nargs="?", default=None, help="log file; default: newest one in output/"
    )
    ap.add_argument("--tail", type=int, default=0, help="print last N ticks as a table")
    ap.add_argument(
        "--bursts", type=int, default=0, help="print last N steering corrections (0=none)"
    )
    ap.add_argument("--no-bursts", action="store_true", help="do not count corrections")
    ap.add_argument(
        "--summary",
        action="store_true",
        help="print KPI summary (slow share, heading source, frozen course, overshoot)",
    )
    ap.add_argument(
        "--slow-kmh",
        type=float,
        default=36.0,
        help="slow-speed threshold for the summary (default 36: below it the "
        "motion course freezes at the default 10 px / 0.5 s window)",
    )
    args = ap.parse_args()

    path = args.path
    if not path:
        files = sorted(glob.glob(os.path.join("output", "nav_dbg_*.jsonl")), key=os.path.getmtime)
        if not files:
            print('no logs in output/ (enable "Nav log" in the studio)')
            return 1
        path = files[-1]
        print("# log:", path)
    meta, rows, events = load(path)
    if not rows:
        print("log is empty")
        return 1

    dur = rows[-1]["t"] - rows[0]["t"]
    print(
        "# ticks=%d  duration=%.1f s  (~%.0f ticks/s)"
        % (len(rows), dur, len(rows) / max(dur, 1e-3))
    )

    pose_age = [r["pose_age"] for r in rows if r.get("pose_age") is not None]
    sample_age = [r["sample_age"] for r in rows if r.get("sample_age") is not None]
    pa = stats(pose_age)
    sa = stats(sample_age)
    print("pose age pose_age:       mean=%.2f  p95=%.2f  max=%.2f s" % pa)
    print("sample age sample_age:   mean=%.2f  p95=%.2f  max=%.2f s" % sa)
    lost = [e for e in events if e.get("kind") == "lost"]
    if lost:
        print("localization losses (pose jump rejected): %d times" % len(lost))
        for e in lost[:12]:
            claim = e.get("v_claim", 0.0)
            claim_txt = claim if isinstance(claim, str) else "%.0f px/s" % claim
            print(
                "    t=%5.2f  pause=%s  offset~%.1fm  claimed=%s"
                % (e["t"] - rows[0]["t"], e.get("sa"), e.get("d", 0.0), claim_txt)
            )
        if len(lost) > 12:
            print("    ... and %d more" % (len(lost) - 12))
    blind = [e for e in events if e.get("kind") == "blind"]
    if blind:
        print(
            "blind (stale pose, keys released): %d times, pauses %s"
            % (len(blind), ", ".join("%.2f" % e.get("sa", 0.0) for e in blind[:8]))
        )
    for kind in ("stuck", "reverse"):
        items = [e for e in events if e.get("kind") == kind]
        if items:
            print(
                "%s events: %d (first at t=%.1f s)"
                % (kind, len(items), items[0]["t"] - rows[0]["t"])
            )
    if rows[0].get("mh_age") is not None:
        ma = stats([r["mh_age"] for r in rows if r.get("mh_age") is not None])
        print("M-course age mh_age:    mean=%.2f  p95=%.2f  max=%.2f s" % ma)

    modes = {}
    steer_ticks = 0
    for r in rows:
        modes[r.get("mode", "?")] = modes.get(r.get("mode", "?"), 0) + 1
        if r["steer"] != 0:
            steer_ticks += 1
    ms = "  ".join("%s=%d (%.0f%%)" % (k, v, pct(v, len(rows))) for k, v in sorted(modes.items()))
    print("course source (th): %s" % ms)

    runs = steering_runs(rows)
    run_dur = [b[-1]["t"] - b[0]["t"] for b in runs]
    print(
        "steering held: %d ticks (%.0f%%), %d corrections, avg hold=%.2f max=%.2f s"
        % (
            steer_ticks,
            pct(steer_ticks, len(rows)),
            len(runs),
            sum(run_dur) / max(len(run_dur), 1),
            max(run_dur, default=0.0),
        )
    )

    bursts = bursts_of(rows) if (not args.no_bursts or args.summary) else []
    if not args.no_bursts:
        b_all, w_all = [], []
        for b, w in bursts:
            b_all.append(b)
            w_all.append(w)
        over = overshoots(bursts)
        o = stats(over) if over else (0.0, 0.0, 0.0)
        print("steering corrections: %d" % len(b_all))
        print("overshoot in release window (|err|): mean=%.1f p95=%.1f max=%.1f deg" % o)
        # steering sync with new pose samples: was the run "blind"
        blind = 0
        held = 0
        got = 0
        has_fresh = any("fresh" in r for r in rows)
        for b in b_all:
            if any(r["new_sample"] for r in b):
                got += 1
            if has_fresh:
                blind += sum(1 for r in b if not r.get("fresh", True))
            else:
                blind += sum(1 for r in b if r["pose_age"] > 0.5)
            held += len(b)
        print(
            "holds during which a new sample arrived: %d/%d "
            "(%.0f%%)" % (got, len(b_all), pct(got, len(b_all)))
        )
        if has_fresh:
            print(
                "blind hold (no fresh sample): %d ticks out of %d "
                "(%.0f%%)" % (blind, held, pct(blind, held))
            )
        else:
            print(
                "blind hold (pose_age>0.5 s): %d ticks out of %d (%.0f%%)"
                % (blind, held, pct(blind, held))
            )

        if args.bursts:
            print("\n# last %d corrections:" % args.bursts)
            print(
                "%6s %5s %5s %5s %5s %6s %6s %6s %6s %7s %6s %5s %5s %5s %4s"
                % (
                    "t,s",
                    "err?",
                    "err!",
                    "hold",
                    "imp",
                    "dHead",
                    "extrERR",
                    "dErr?",
                    "fresh",
                    "poseAge",
                    "sampA",
                    "mode",
                    "keys",
                    "settle",
                    "new",
                )
            )
            n = len(b_all)
            for bi in range(max(0, n - args.bursts), n):
                b, w = b_all[bi], w_all[bi]
                e0, e1 = b[0]["err"], b[-1]["err"]
                h = b[-1]["t"] - b[0]["t"]
                imp = max((r.get("imp", r.get("hold_left", 0.0)) for r in b), default=0.0)
                dh = wrap180_deg(b[-1]["heading"] - b[0]["heading"])
                ext = max(abs(r["err"]) for r in w) if w else 0.0
                dd = wrap180_deg(e1 - e0)
                has_fresh = "fresh" in b[0]
                fresh_n = sum(1 for r in b if r.get("fresh", True)) if has_fresh else None
                pa_m = max(r["pose_age"] for r in b)
                sa_m = max(r["sample_age"] for r in b)
                new = sum(1 for r in b if r["new_sample"])
                print(
                    "%6.2f %5.1f %5.1f %5.2f %5.2f %6.1f %6.1f %6.1f %6s "
                    "%7.2f %6.2f %5s %5s %5.2f %4d"
                    % (
                        b[0]["t"] - rows[0]["t"],
                        e0,
                        e1,
                        h,
                        imp,
                        dh,
                        ext,
                        dd,
                        "%d/%d" % (fresh_n, len(b)) if fresh_n is not None else "-",
                        pa_m,
                        sa_m,
                        b[0]["mode"],
                        b[0]["keys"],
                        max(r.get("settle", 0.0) for r in w) if w else 0.0,
                        new,
                    )
                )

    if args.summary:
        print_summary(rows, bursts, args.slow_kmh)

    if args.tail:
        print(
            "\n# last %d ticks (err column: deviation, + steer D, - steer A,"
            " . neutral, * new sample, STALE if pose older than 0.5 s)" % args.tail
        )
        print(
            "%6s %5s %5s %6s %5s %5s %6s %6s %5s %7s %6s %5s %s %s"
            % (
                "t,s",
                "err",
                "head",
                "steer",
                "hold",
                "v",
                "km/h",
                "tgt",
                "xte",
                "settle",
                "poseA",
                "mode",
                "keys",
                "ru",
            )
        )
        for r in rows[-args.tail :]:
            st = "A" if r["steer"] == -1 else ("D" if r["steer"] == 1 else ".")
            st += "*" if r["new_sample"] else " "
            flag = " STALE" if r.get("fresh") is False else ""
            ru = "R" if r.get("runaway") else ""
            print(
                "%6.2f %5.1f %5.1f %6s %5s %5.0f %5.0f %6.0f %5s %6.2f "
                "%7.2f %5s %s%s %s"
                % (
                    r["t"] - rows[0]["t"],
                    r["err"],
                    r["heading"],
                    st,
                    "H" if r.get("hold") else ".",
                    r.get("v", 0.0),
                    row_vkmh(r) or 0.0,
                    r.get("tgt", 0.0),
                    "--" if r.get("xte_m") is None else "%.1fm" % r["xte_m"],
                    r.get("settle", 0.0),
                    r["pose_age"],
                    r["mode"],
                    r["keys"],
                    flag,
                    ru,
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
