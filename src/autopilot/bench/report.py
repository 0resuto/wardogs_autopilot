"""Markdown report and optional matplotlib plots for a bench run.

`write_report` renders what the numbers mean (one table per engine, one table
per swept parameter, the top reject reasons). `plot_results` is best-effort:
matplotlib is imported inside the function, never at module import time, and a
missing install only prints how to add it.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .runner import RunResult

#: Columns of the per-engine and per-parameter tables (short, fixed order).
_TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("loc%", "localized_pct", "%.1f"),
    ("err px", "pos_err_px_mean", "%.2f"),
    ("p95 px", "pos_err_px_p95", "%.2f"),
    ("max px", "pos_err_px_max", "%.2f"),
    ("err m", "pos_err_m_mean", "%.2f"),
    ("th deg", "heading_err_deg_mean", "%.2f"),
    ("ms", "latency_ms_mean", "%.1f"),
    ("fps", "achieved_fps", "%.1f"),
    ("vote", "vote_rejects", "%d"),
    ("hold", "holds", "%d"),
    ("anch", "anchors", "%d"),
)

_MISSING_MPL = (
    "plots skipped: run with: uv run --with matplotlib python tools/bench_synthetic.py ..."
)


def swept_keys(results: Sequence[RunResult], engine: str) -> list[str]:
    """Parameter keys `engine` was actually swept over.

    A key counts only when every one of the engine's runs carries it *and* the
    runs disagree on its value: a one-off override (a single run nudged for any
    other reason) is not a sweep and must not grow a parameter table.
    """
    configs = [r.config for r in results if r.engine == engine]
    if len(configs) < 2:
        return []
    common = set(configs[0])
    for cfg in configs[1:]:
        common &= set(cfg)
    return [k for k in sorted(common) if len({json_scalar(cfg[k]) for cfg in configs}) > 1]


def json_scalar(value: Any) -> str:
    """A value as a stable, sortable string (for grouping and labels)."""
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _value_sort_key(value: str) -> tuple[int, float, str]:
    """Order parameter values: the unset '-' first, then numbers, then text."""
    if value == "-":
        return (0, 0.0, value)
    try:
        return (1, float(value), value)
    except ValueError:
        return (2, 0.0, value)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _table(rows: Sequence[tuple[Sequence[str], RunResult]], header: Sequence[str]) -> list[str]:
    """A markdown table; each row pairs its label cells with a RunResult."""
    head = "| " + " | ".join([*header, *(label for label, _key, _fmt in _TABLE_COLUMNS)]) + " |"
    sep = "| " + " | ".join(["---"] * (len(header) + len(_TABLE_COLUMNS))) + " |"
    lines = [head, sep]
    for labels, result in rows:
        cells = [
            fmt % float(getattr(result, key)) if fmt != "%d" else fmt % int(getattr(result, key))
            for _label, key, fmt in _TABLE_COLUMNS
        ]
        lines.append("| " + " | ".join([*(str(v) for v in labels), *cells]) + " |")
    return lines


def write_report(results: Sequence[RunResult], path: str, meta: dict[str, Any]) -> str:
    """Render the markdown report of `results`; returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    lines: list[str] = [
        "# Synthetic localization benchmark",
        "",
        f"- map: `{meta.get('map', '?')}`",
        f"- generated: {meta.get('timestamp') or time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- mode: {meta.get('mode', 'quick')}",
        f"- engines: {', '.join(meta.get('engines') or sorted({r.engine for r in results}))}",
        f"- scenarios: {', '.join(meta.get('scenarios') or sorted({r.scenario for r in results}))}",
        f"- frames/scenario: {meta.get('frames', '?')}   fps: {meta.get('fps', '?')}"
        f"   speed: {meta.get('speed_mps', '?')} m/s",
        f"- map scale: {meta.get('m_per_px', '?')} m/px"
        f"   warm-up is excluded from the latencies below",
        "",
        "Errors are published-vs-truth (the smoother is part of the pipeline),"
        " measured on frames that produced a fresh pose; `loc%` counts those"
        " frames, held and vote-rejected ones are excluded.",
        "",
    ]

    if not results:
        lines += ["_no runs were recorded_", ""]

    for engine in _ordered_engines(results):
        runs = [r for r in results if r.engine == engine]
        lines += [f"## {engine}", ""]
        base = _default_config(runs)
        default_runs = [r for r in runs if r.config == base]
        shown = default_runs or runs[:1]
        rows = [
            (
                [
                    r.scenario,
                    "%.0f" % r.fps,
                    "%.0f" % r.speed_mps,
                ],
                r,
            )
            for r in sorted(shown, key=lambda r: (r.scenario, r.fps))
        ]
        lines += _table(rows, ("scenario", "fps", "speed"))
        lines += ["", "```", f"config: {base}", "```", ""]

        for key in swept_keys(results, engine):
            lines += [f"### {engine}: {key}", ""]
            by_value: dict[str, list[RunResult]] = defaultdict(list)
            for run in runs:
                by_value[json_scalar(run.config.get(key))].append(run)
            merged = {
                value: RunResult(
                    engine=engine,
                    scenario=f"{len(group)} scenario(s)",
                    fps=_mean([r.fps for r in group]),
                    speed_mps=_mean([r.speed_mps for r in group]),
                    localized_pct=_mean([r.localized_pct for r in group]),
                    pos_err_px_mean=_mean([r.pos_err_px_mean for r in group]),
                    pos_err_px_median=_mean([r.pos_err_px_median for r in group]),
                    pos_err_px_p95=_mean([r.pos_err_px_p95 for r in group]),
                    pos_err_px_max=_mean([r.pos_err_px_max for r in group]),
                    pos_err_m_mean=_mean([r.pos_err_m_mean for r in group]),
                    heading_err_deg_mean=_mean([r.heading_err_deg_mean for r in group]),
                    heading_err_deg_max=_mean([r.heading_err_deg_max for r in group]),
                    latency_ms_mean=_mean([r.latency_ms_mean for r in group]),
                    latency_ms_p95=_mean([r.latency_ms_p95 for r in group]),
                    achieved_fps=_mean([r.achieved_fps for r in group]),
                    vote_rejects=int(_mean([r.vote_rejects for r in group])),
                    holds=int(_mean([r.holds for r in group])),
                    anchors=int(_mean([r.anchors for r in group])),
                )
                for value, group in by_value.items()
            }
            values = sorted(merged, key=_value_sort_key)
            rows = [([value], merged[value]) for value in values]
            lines += _table(rows, (key,))
            lines += ["", "```", f"mean over {len(values)} distinct value(s)", "```", ""]

        tally: dict[str, int] = defaultdict(int)
        for run in runs:
            for reason, count in run.reject_reasons.items():
                tally[reason] += count
        top = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        lines += ["top rejects: " + (", ".join(f"`{k}` {v}" for k, v in top) or "none"), ""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _ordered_engines(results: Sequence[RunResult]) -> list[str]:
    from .runner import ENGINES

    seen = {r.engine for r in results}
    return [e for e in ENGINES if e in seen] + sorted(seen - set(ENGINES))


def _default_config(runs: Sequence[RunResult]) -> dict[str, Any]:
    """The config dict shared by most runs (the empty override when unswept)."""
    if not runs:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        counts[_config_key(run.config)] += 1
    best = max(counts, key=lambda k: (counts[k], k == "{}"))
    for run in runs:
        if _config_key(run.config) == best:
            return dict(run.config)
    return {}


def _config_key(config: dict[str, Any]) -> str:
    import json

    return json.dumps(dict(config), sort_keys=True, default=str)


# ---------------------------------------------------------------------- plots


def plot_results(results: Sequence[RunResult], out_dir: str) -> list[str]:
    """Render the standard plots into `out_dir`; [] when matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(_MISSING_MPL)
        return []

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    written += _plot_latency(results, out_dir, plt)
    written += _plot_param_curves(results, out_dir, plt)
    written += _plot_success_fps(results, out_dir, plt)
    return written


def _save(fig: Any, out_dir: str, name: str, plt: Any) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_latency(results: Sequence[RunResult], out_dir: str, plt: Any) -> list[str]:
    engines = _ordered_engines(results)
    if not engines:
        return []
    means = [
        _mean(
            [
                r.latency_ms_mean
                for r in results
                if r.engine == engine
                and r.config == _default_config([r for r in results if r.engine == engine])
            ]
        )
        or _mean([r.latency_ms_mean for r in results if r.engine == engine])
        for engine in engines
    ]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(engines, means, color="#4c8bf5")
    ax.set_ylabel("mean latency (ms)")
    ax.set_title("Frame latency by engine (default config)")
    ax.grid(axis="y", alpha=0.3)
    return [_save(fig, out_dir, "latency_by_engine.png", plt)]


def _plot_param_curves(results: Sequence[RunResult], out_dir: str, plt: Any) -> list[str]:
    written: list[str] = []
    for engine in _ordered_engines(results):
        runs = [r for r in results if r.engine == engine]
        for key in swept_keys(results, engine):
            points: dict[float, list[RunResult]] = defaultdict(list)
            for run in runs:
                raw = run.config.get(key)
                if raw is None:
                    continue
                points[float(raw)].append(run)
            if len(points) < 2:
                continue
            xs = sorted(points)
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(xs, [_mean([r.pos_err_px_mean for r in points[x]]) for x in xs], "o-")
            ax.set_xlabel(key)
            ax.set_ylabel("pos err mean (px)")
            ax.set_title(f"{engine}: position error vs {key}")
            ax.grid(alpha=0.3)
            written.append(_save(fig, out_dir, f"error_vs_param_{engine}_{key}.png", plt))
    return written


def _plot_success_fps(results: Sequence[RunResult], out_dir: str, plt: Any) -> list[str]:
    engines = _ordered_engines(results)
    if not engines:
        return []
    fig, ax = plt.subplots(figsize=(6, 3.2))
    plotted = False
    for engine in engines:
        runs = [r for r in results if r.engine == engine]
        points: dict[float, list[float]] = defaultdict(list)
        for run in runs:
            points[run.fps].append(run.localized_pct)
        xs = sorted(points)
        if len(xs) < 2:
            continue
        ax.plot(xs, [_mean(points[x]) for x in xs], "o-", label=engine)
        plotted = True
    if not plotted:
        plt.close(fig)
        return []
    ax.set_xlabel("scenario fps")
    ax.set_ylabel("localized frames (%)")
    ax.set_title("Localization success vs frame rate")
    ax.grid(alpha=0.3)
    ax.legend()
    return [_save(fig, out_dir, "success_vs_fps.png", plt)]
