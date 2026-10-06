"""Run orchestration: engine x scenario x config -> metrics, CSV, per-frame JSONL.

A "run" is one (engine, scenario, config) triple: a fresh `BenchTracker` is
built for it, the engine's warm-up is paid once and reported separately, then
every frame is scored against the scenario ground truth. Nothing here writes to
`config.json` — the sweep values reach the engine through `ConfigOverlay` only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..vision import featureindex, locator
from .params import ConfigOverlay
from .scenario import Scenario
from .tracker import BenchTracker, FrameRecord

#: Engine the bench runs (hybrid anchors with the SIFT index).
ENGINES = ("hybrid",)

#: Index kind each engine needs ("hybrid" anchors with SIFT).
_INDEX_KIND = {"hybrid": "sift"}

#: Stable CSV columns (report tables and plots read the dataclass, not the file).
CSV_COLUMNS = (
    "engine",
    "scenario",
    "fps",
    "speed_mps",
    "frames",
    "localized_pct",
    "pos_err_px_mean",
    "pos_err_px_median",
    "pos_err_px_p95",
    "pos_err_px_max",
    "pos_err_m_mean",
    "heading_err_deg_mean",
    "heading_err_deg_max",
    "latency_ms_mean",
    "latency_ms_p95",
    "achieved_fps",
    "vote_rejects",
    "holds",
    "anchors",
    "warmup_ms",
    "config",
    "reject_reasons",
)


@dataclass
class RunResult:
    """Aggregated scoreboard of one run."""

    engine: str
    scenario: str
    config: dict[str, Any] = field(default_factory=dict)
    frames: int = 0
    localized_pct: float = 0.0
    pos_err_px_mean: float = 0.0
    pos_err_px_median: float = 0.0
    pos_err_px_p95: float = 0.0
    pos_err_px_max: float = 0.0
    pos_err_m_mean: float = 0.0
    heading_err_deg_mean: float = 0.0
    heading_err_deg_max: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    achieved_fps: float = 0.0
    vote_rejects: int = 0
    holds: int = 0
    anchors: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    warmup_ms: float = 0.0
    fps: float = 0.0
    speed_mps: float = 0.0
    m_per_px: float = 0.0

    def sweep_key(self, key: str) -> Any:
        """The swept value of `key` for this run ('-' when not overridden)."""
        value = self.config.get(key)
        return "-" if value is None else value


def _index_ready(map_name: str, kind: str) -> bool:
    """True when the feature index of `kind` is present (cached per map+kind)."""
    key = (map_name, kind)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = featureindex.load_index(map_name, kind) is not None
    return _INDEX_CACHE[key]


_INDEX_CACHE: dict[tuple[str, str], bool] = {}


def available_engines(map_name: str) -> list[str]:
    """Engines runnable for `map_name`: the SIFT index must be present."""
    return [e for e in ENGINES if _index_ready(map_name, _INDEX_KIND[e])]


def config_hash(config: Mapping[str, Any] | None) -> str:
    """Short stable id of a config dict (used in the per-frame file names)."""
    payload = json.dumps(dict(config or {}), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]  # noqa: S324 - id, not crypto


def run_case(
    engine: str,
    scenario: Scenario,
    overrides: Mapping[str, Any] | None = None,
    map_name: str | None = None,
    *,
    realtime: bool | None = None,
    budget: float | None = 3.0,
    warmup: bool = True,
) -> tuple[RunResult, list[FrameRecord]]:
    """Run one (engine, scenario, config) case and score it.

    `realtime=None` paces hybrid runs to the wall clock (its re-anchor period
    is time based) and leaves the other engines as fast as possible.
    """
    overrides = dict(overrides or {})
    store = locator.get_store()
    if map_name:
        locator.set_map(str(map_name))
    locator.set_engine("hybrid")

    with ConfigOverlay(store, overrides):
        cfg = store.loc_cfg()
        pace = (str(engine) == "hybrid") if realtime is None else bool(realtime)
        bench = BenchTracker(engine, cfg, realtime=pace, budget=budget)

        warmup_ms = 0.0
        if warmup:
            bench.reset()
            t0 = time.perf_counter()
            bench.step(0, scenario.frames[0], scenario.mask, 0.0, scenario.truth[0])
            warmup_ms = (time.perf_counter() - t0) * 1000.0
            bench.reset()

        records: list[FrameRecord] = []
        for idx, frame in enumerate(scenario.frames):
            records.append(
                bench.step(idx, frame, scenario.mask, idx * scenario.dt, scenario.truth[idx])
            )

    store_name = store.map_name()
    result = RunResult(
        engine=str(engine),
        scenario=scenario.spec.name,
        config=overrides,
        frames=len(records),
        fps=float(scenario.spec.fps),
        speed_mps=float(scenario.spec.speed_mps),
        m_per_px=float(store.m_per_px(store_name)),
        warmup_ms=round(warmup_ms, 3),
    )
    _aggregate(result, records)
    return result, records


def _aggregate(result: RunResult, records: Sequence[FrameRecord]) -> None:
    """Fill the metrics of `result` from its per-frame records."""
    total = len(records)
    if total == 0:
        return
    measured = [r for r in records if r.new_sample and r.err_px is not None]
    latency = np.asarray([r.latency_ms for r in records], np.float64)
    result.localized_pct = round(100.0 * len(measured) / total, 2)
    result.latency_ms_mean = round(float(latency.mean()), 3)
    result.latency_ms_p95 = round(float(np.percentile(latency, 95)), 3)
    result.achieved_fps = (
        round(1000.0 / result.latency_ms_mean, 2) if result.latency_ms_mean else 0.0
    )

    if measured:
        err = np.asarray([r.err_px for r in measured], np.float64)
        th = np.asarray([r.th_err_deg for r in measured if r.th_err_deg is not None], np.float64)
        result.pos_err_px_mean = round(float(err.mean()), 3)
        result.pos_err_px_median = round(float(np.median(err)), 3)
        result.pos_err_px_p95 = round(float(np.percentile(err, 95)), 3)
        result.pos_err_px_max = round(float(err.max()), 3)
        if th.size:
            result.heading_err_deg_mean = round(float(th.mean()), 3)
            result.heading_err_deg_max = round(float(th.max()), 3)
    if result.m_per_px > 0.0:
        result.pos_err_m_mean = round(result.pos_err_px_mean * result.m_per_px, 4)

    result.vote_rejects = sum(1 for r in records if r.vote_reject)
    result.holds = sum(1 for r in records if r.hold)
    result.anchors = sum(1 for r in records if r.anchor)
    tally: Counter[str] = Counter()
    for rec in records:
        if rec.reject:
            tally[str(rec.reject)] += 1
    result.reject_reasons = dict(sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------- writers


def write_csv(results: Sequence[RunResult], path: str) -> str:
    """Write the results table (stable columns); returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for result in results:
            row = asdict(result)
            row["config"] = json.dumps(result.config, sort_keys=True)
            row["reject_reasons"] = json.dumps(result.reject_reasons, sort_keys=True)
            writer.writerow([_csv_value(row.get(col)) for col in CSV_COLUMNS])
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_jsonl(records: Sequence[FrameRecord], path: str) -> str:
    """Write one JSON object per frame; returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.as_json(), sort_keys=True))
            fh.write("\n")
    return path
