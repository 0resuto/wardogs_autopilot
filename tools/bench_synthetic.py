"""CLI for the synthetic localization benchmark.

Thin wrapper over `autopilot.bench`: builds the scenario frames once per
(fps, speed) combination, reuses them for every engine and configuration, and
writes results.csv, per_frame/*.jsonl, report.md and (optionally) plots/ into
`output/bench_<timestamp>`. config.json is never touched — sweep values reach
the engine through an in-memory overlay.

Examples:
    uv run python tools/bench_synthetic.py --quick --no-plot
    uv run python tools/bench_synthetic.py --engines orb --scenarios straight --frames 20
    uv run python tools/bench_synthetic.py --engines xfeat --sweep xfeat_min_cos=0.75,0.82,0.88
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot import PROJECT_ROOT  # noqa: E402
from autopilot.bench import (  # noqa: E402
    ENGINES,
    SCENARIO_NAMES,
    BenchSkip,
    RunResult,
    available_engines,
    config_hash,
    default_specs,
    expand_oat,
    load_inputs,
    param_registry,
    parse_sweep,
    plot_results,
    run_case,
    spec_for,
    write_csv,
    write_jsonl,
    write_report,
    xfeat_skip_reason,
)
from autopilot.vision import locator  # noqa: E402


def _parse_list(raw: str, cast) -> list:
    """Comma separated CLI list -> list of `cast` values."""
    return [cast(v.strip()) for v in str(raw or "").split(",") if v.strip()]


def _build_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", dest="map_name", default="", help="map name (default: config's)")
    ap.add_argument("--engines", default="", help="comma separated (default: all available)")
    ap.add_argument("--scenarios", default="", help="comma separated (default: quick set)")
    ap.add_argument("--frames", type=int, default=0, help="frames per scenario (default 60/150)")
    ap.add_argument("--fps", default="10", help="comma separated scenario rates")
    ap.add_argument("--speed", default="15", help="comma separated speeds in m/s")
    ap.add_argument(
        "--sweep", action="append", default=[], metavar="KEY=V1,V2", help="repeatable OAT sweep"
    )
    ap.add_argument("--budget", type=float, default=3.0, help="per-frame search budget (s)")
    ap.add_argument(
        "--full", action="store_true", help="150 frames, 4 scenarios, fps 10/30, 10/20 m/s"
    )
    ap.add_argument("--quick", action="store_true", help="default behaviour (explicit no-op)")
    ap.add_argument("--live", action="store_true", help="also run the real LiveLocator per engine")
    ap.add_argument("--plot", dest="plot", action="store_true", default=None)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    ap.add_argument("--out", default="", help="output dir (default output/bench_<time>)")
    return ap


def _wants_plot(args: argparse.Namespace) -> bool:
    if args.plot is not None:
        return bool(args.plot)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _build_args().parse_args(argv)
    full = bool(args.full)
    frames = int(args.frames) if args.frames else (150 if full else 60)
    scenarios = _parse_list(args.scenarios, str) or [s.name for s in default_specs(quick=not full)]
    unknown = [s for s in scenarios if s not in SCENARIO_NAMES]
    if unknown:
        print(f"error: unknown scenario(s): {', '.join(unknown)}")
        print(f"       valid: {', '.join(SCENARIO_NAMES)}")
        return 2
    fps_list = _parse_list(args.fps, float)
    speed_list = _parse_list(args.speed, float)
    if full and args.fps == "10":
        fps_list = [10.0, 30.0]
    if full and args.speed == "15":
        speed_list = [10.0, 20.0]

    try:
        sweeps = [parse_sweep(expr) for expr in args.sweep]
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    map_name = args.map_name or str(locator.get_store().map_name())
    engines = _parse_list(args.engines, str) or available_engines(map_name)
    bad = [e for e in engines if e not in ENGINES]
    if bad:
        print(f"error: unknown engine(s): {', '.join(bad)} (valid: {', '.join(ENGINES)})")
        return 2
    skipped = [e for e in ENGINES if e not in engines and not args.engines]
    if skipped:
        print(f"skip: engines unavailable for {map_name}: {', '.join(skipped)}")
    if not engines:
        print(f"skip: no engine index available for map '{map_name}'")
        print("      run: python tools/download_map.py <map>")
        return 0
    # An explicitly requested engine still has to be runnable: xfeat needs its
    # index, both models and a GPU EP, and a missing one is a skip, not a crash.
    for engine in list(engines):
        reason = xfeat_skip_reason(map_name) if engine == "xfeat" else None
        if reason is None:
            continue
        print(f"skip: {engine}: {reason}")
        engines.remove(engine)
    if not engines:
        print("skip: no requested engine can run on this machine")
        return 0

    out_dir = args.out or os.path.join(
        PROJECT_ROOT, "output", "bench_" + time.strftime("%Y%m%d_%H%M%S")
    )
    per_frame_dir = os.path.join(out_dir, "per_frame")
    os.makedirs(per_frame_dir, exist_ok=True)

    store = locator.get_store()
    locator.set_map(map_name)
    print(f"map: {map_name}   engines: {', '.join(engines)}   scenarios: {', '.join(scenarios)}")
    try:
        inputs = load_inputs(map_name)
    except BenchSkip as exc:
        print(f"skip: {exc}")
        return 0
    if inputs.mask_missing:
        print("note: data/masks/mm_mask.png missing - scenarios run without the UI mask")

    configs = expand_oat(sweeps, {})
    results: list[RunResult] = []
    per_frame: dict[str, list] = {}
    failures: list[str] = []

    for fps in fps_list:
        for speed in speed_list:
            built = inputs.render_all(
                [
                    spec_for(name, frames=frames, fps=float(fps), speed_mps=float(speed))
                    for name in scenarios
                ]
            )
            tag = f"fps{fps:g}_speed{speed:g}"
            for engine in engines:
                # A sweep of engine A's keys must not leak into engine B's runs.
                known = param_registry(engine)
                engine_configs = [{k: v for k, v in cfg.items() if k in known} for cfg in configs]
                for scen in built:
                    for cfg in engine_configs:
                        label = f"{engine}/{scen.spec.name}/{tag}"
                        if cfg:
                            label += " " + format_config(cfg)
                        t0 = time.perf_counter()
                        try:
                            result, records = run_case(
                                engine, scen, cfg, map_name, budget=args.budget
                            )
                        except Exception as exc:  # noqa: BLE001 - one broken case is reported
                            failures.append(f"{label}: {type(exc).__name__}: {exc}")
                            traceback.print_exc()
                            continue
                        results.append(result)
                        per_frame[f"{engine}_{scen.spec.name}_{config_hash(cfg)}_{tag}.jsonl"] = (
                            records
                        )
                        print(
                            "  %-52s loc %5.1f%%  err %6.2f px  %6.1f ms  (%.1fs)"
                            % (
                                label,
                                result.localized_pct,
                                result.pos_err_px_mean,
                                result.latency_ms_mean,
                                time.perf_counter() - t0,
                            )
                        )

    if not results:
        print("no runs completed" + ("" if not failures else " (see the errors above)"))
        return 1 if failures else 0

    write_csv(results, os.path.join(out_dir, "results.csv"))
    for name, records in per_frame.items():
        write_jsonl(records, os.path.join(per_frame_dir, name))
    m_per_px = store.m_per_px(map_name)
    report_path = write_report(
        results,
        os.path.join(out_dir, "report.md"),
        {
            "map": map_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "full" if full else "quick",
            "engines": engines,
            "scenarios": scenarios,
            "frames": frames,
            "fps": ",".join("%g" % v for v in fps_list),
            "speed_mps": ",".join("%g" % v for v in speed_list),
            "m_per_px": m_per_px,
        },
    )

    if args.live:
        for engine in engines:
            try:
                live = _run_live(engine, built[0], map_name, store, inputs.mask, frames)
            except Exception as exc:  # noqa: BLE001 - integration smoke must not abort
                failures.append(f"live/{engine}: {type(exc).__name__}: {exc}")
                continue
            print(
                "  live %-8s %5.1f fps  %d/%d frames"
                % (engine, live["fps"], live["done"], live["frames"])
            )
            _append_live(out_dir, engine, live)

    if _wants_plot(args):
        plots = plot_results(results, os.path.join(out_dir, "plots"))
        for path in plots:
            print("  plot: %s" % os.path.relpath(path, ROOT))

    _print_summary(results)
    print("\nreport: %s" % os.path.relpath(report_path, ROOT))
    print("csv:    %s" % os.path.relpath(os.path.join(out_dir, "results.csv"), ROOT))
    if failures:
        print("\n%d case(s) failed:" % len(failures))
        for line in failures:
            print("  " + line)
        return 1
    return 0


def format_config(config: dict) -> str:
    return " ".join("%s=%s" % (k, config[k]) for k in sorted(config))


def _append_live(out_dir: str, engine: str, live: dict) -> None:
    with open(os.path.join(out_dir, "live.txt"), "a", encoding="utf-8") as fh:
        fh.write(
            "%s fps=%.1f frames=%d done=%d dropped=%d%s\n"
            % (
                engine,
                live["fps"],
                live["frames"],
                live["done"],
                live["dropped"],
                f" error={live['error']}" if live.get("error") else "",
            )
        )


def _run_live(engine: str, scenario, map_name: str, store, ui_mask, frames: int) -> dict:
    """Integration smoke: the real LiveLocator fed by the synthetic frames.

    `LiveLocator.attempt` counts the frames it actually localized; the capture
    producer drops the stale ones, so `frames - attempt` is the number of frames
    the synthetic run offered that never reached the engine.
    """
    import cv2

    from autopilot.common.config import AppConfig
    from autopilot.vision.tracker import LiveLocator

    app_cfg = AppConfig()
    app_cfg.map.name = map_name
    app_cfg.locator.engine = engine
    app_cfg.capture.fps = int(round(scenario.spec.fps))
    h, w = scenario.frames[0].shape[:2]
    app_cfg.capture.mmap_roi = [0, 0, w, h]

    pool = list(scenario.frames)[: int(frames)]
    total = len(pool)
    state = {"i": 0}

    def source() -> np.ndarray:
        frame = pool[min(state["i"], total - 1)]
        state["i"] += 1
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    mask = ui_mask if ui_mask is not None else np.zeros((h, w), bool)
    loc = LiveLocator(cfg=app_cfg, mask=mask, frame_source=source)
    store.set_map(map_name)
    locator.set_engine(engine)
    t0 = time.perf_counter()
    loc.start()
    deadline = t0 + scenario.dt * total + 5.0
    while time.perf_counter() < deadline:
        if loc.attempt >= total and state["i"] >= total:
            break
        time.sleep(min(scenario.dt, 0.05))
    elapsed = max(1e-6, time.perf_counter() - t0)
    done = int(loc.attempt)
    error = loc.error
    loc.stop()
    loc.join(timeout=5.0)
    return {
        "fps": done / elapsed,
        "frames": total,
        "done": done,
        "dropped": max(0, total - done),
        "error": error or "",
    }


def _print_summary(results: list[RunResult]) -> None:
    """Fixed-width console table of every run."""
    head = "%-8s %-9s %5s %7s %7s %8s %7s %7s %6s %5s %5s" % (
        "engine",
        "scenario",
        "fps",
        "loc%",
        "err px",
        "p95 px",
        "th deg",
        "ms",
        "fps_out",
        "vote",
        "hold",
    )
    print()
    print(head)
    print("-" * len(head))
    for r in sorted(results, key=lambda r: (r.engine, r.scenario, r.fps, sorted(r.config.items()))):
        print(
            "%-8s %-9s %5.0f %7.1f %7.2f %8.2f %7.2f %7.1f %6.1f %5d %5d"
            % (
                r.engine,
                r.scenario,
                r.fps,
                r.localized_pct,
                r.pos_err_px_mean,
                r.pos_err_px_p95,
                r.heading_err_deg_mean,
                r.latency_ms_mean,
                r.achieved_fps,
                r.vote_rejects,
                r.holds,
            )
        )


if __name__ == "__main__":
    sys.exit(main())
