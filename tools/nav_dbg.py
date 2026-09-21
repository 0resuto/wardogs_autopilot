"""Analyze the navigator JSONL log (output/nav_dbg_*.jsonl).

Usage:
    python tools/nav_dbg.py [path-to-file]
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
            print(
                "    t=%5.2f  pause=%s  offset~%.1fm  claimed speed="
                "%.0f px/s"
                % (e["t"] - rows[0]["t"], e.get("sa"), e.get("d", 0.0), e.get("v_claim", 0.0))
            )
        if len(lost) > 12:
            print("    ... and %d more" % (len(lost) - 12))
    if rows[0].get("mh_age") is not None:
        ma = stats([r["mh_age"] for r in rows if r.get("mh_age") is not None])
        print("M-course age mh_age:    mean=%.2f  p95=%.2f  max=%.2f s" % ma)

    modes = {}
    steer_ticks = 0
    hold_all = []
    for r in rows:
        modes[r.get("mode", "?")] = modes.get(r.get("mode", "?"), 0) + 1
        if r["steer"] != 0:
            steer_ticks += 1
            hold_all.append(r.get("hold", 0.0))
    ms = "  ".join("%s=%d (%.0f%%)" % (k, v, pct(v, len(rows))) for k, v in sorted(modes.items()))
    print("course source (th): %s" % ms)
    print(
        "steering held: %d ticks (%.0f%%), avg hold duration=%.2f s"
        % (steer_ticks, pct(steer_ticks, len(rows)), sum(hold_all) / max(len(hold_all), 1))
    )

    if not args.no_bursts:
        bursts = bursts_of(rows)
        b_all, w_all = [], []
        for b, w in bursts:
            b_all.append(b)
            w_all.append(w)
        over = []
        for _b, w in bursts:
            if w:
                over.append(max(min(r["err"] for r in w), -min(r["err"] for r in w), key=abs))
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
                imp = max((r.get("imp", 0.0) for r in b), default=0.0)
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
                "%6.2f %5.1f %5.1f %6s %5.2f %5.0f %5.0f %6.0f %5s %6.2f "
                "%7.2f %5s %s%s %s"
                % (
                    r["t"] - rows[0]["t"],
                    r["err"],
                    r["heading"],
                    st,
                    r.get("hold", 0.0),
                    r.get("v", 0.0),
                    r.get("vkmh", 0.0),
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
