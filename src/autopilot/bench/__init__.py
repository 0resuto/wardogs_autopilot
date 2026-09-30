"""Synthetic benchmark for the localization engines.

Runs `sift` / `orb` / `xfeat` / `hybrid` over deterministic synthetic
trajectories with a known ground truth, sweeps curated tuning parameters one at
a time and scores every frame. Nothing here changes production tracking logic
and nothing is written outside `output/bench_*` (except what the caller asks
for).

Public API:

    ScenarioSpec, Scenario, build_scenario, default_specs, spec_for
    BenchSkip, ScenarioInputs, load_inputs, render_scenarios
    ParamSpec, ENGINE_PARAMS, SHARED_PARAMS, parse_sweep, expand_oat, ConfigOverlay
    BenchTracker, FrameRecord
    RunResult, available_engines, xfeat_skip_reason, run_case, write_csv, write_jsonl
    write_report, plot_results
"""

from __future__ import annotations

from .assets import (
    PREVIEW_LEVEL,
    BenchSkip,
    ScenarioInputs,
    load_inputs,
    render_scenarios,
)
from .params import (
    ENGINE_PARAMS,
    SHARED_PARAMS,
    ConfigOverlay,
    ParamSpec,
    expand_oat,
    param_registry,
    parse_sweep,
    sweep_keys,
)
from .report import plot_results, swept_keys, write_report
from .runner import (
    CSV_COLUMNS,
    ENGINES,
    RunResult,
    available_engines,
    config_hash,
    run_case,
    write_csv,
    write_jsonl,
    xfeat_skip_reason,
)
from .scenario import (
    SCENARIO_NAMES,
    H,
    Scenario,
    ScenarioSpec,
    W,
    build_scenario,
    default_specs,
    spec_for,
)
from .tracker import BenchTracker, FrameRecord

__all__ = [
    "CSV_COLUMNS",
    "ENGINE_PARAMS",
    "ENGINES",
    "H",
    "PREVIEW_LEVEL",
    "SCENARIO_NAMES",
    "SHARED_PARAMS",
    "W",
    "BenchSkip",
    "BenchTracker",
    "ConfigOverlay",
    "FrameRecord",
    "ParamSpec",
    "RunResult",
    "Scenario",
    "ScenarioInputs",
    "ScenarioSpec",
    "available_engines",
    "build_scenario",
    "config_hash",
    "default_specs",
    "expand_oat",
    "load_inputs",
    "param_registry",
    "parse_sweep",
    "plot_results",
    "render_scenarios",
    "run_case",
    "spec_for",
    "sweep_keys",
    "swept_keys",
    "write_csv",
    "write_jsonl",
    "write_report",
    "xfeat_skip_reason",
]
