"""Tests for the benchmark window: worker thread, plan expansion, export, wiring.

Nothing here touches the real engines: the worker, the window and the Logs tab
all take their `runner`, `loader` and `probe` by injection, so the GUI paths
(stop, skip, per-frame export, checkbox state) are exercised in-process.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from typing import Any, cast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autopilot.bench import (  # noqa: E402
    ENGINES,
    BenchSkip,
    FrameRecord,
    RunResult,
    Scenario,
    ScenarioInputs,
    ScenarioSpec,
    config_hash,
    expand_oat,
    load_inputs,
    param_registry,
)
from autopilot.common.config import AppConfig  # noqa: E402
from autopilot.ui.bench_window import (  # noqa: E402
    BenchPlan,
    BenchWindow,
    BenchWorker,
    param_hint,
    record_key,
)
from autopilot.ui.tabs.logs_tab import LogsTab  # noqa: E402

_QT_APP = QApplication.instance() or QApplication([])

#: Set by `_slow_runner` while it is inside the run, so the test can race it.
_SLOW_STARTED = False


def _spin(ms: int = 20) -> None:
    """Let the event loop deliver queued signals for a bounded time."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    _QT_APP.processEvents()


def _wait_until(predicate, timeout: float = 20.0) -> bool:
    """Pump the event loop until `predicate()` is true or `timeout` expires."""
    deadline = time.perf_counter() + timeout
    while not predicate() and time.perf_counter() < deadline:
        _spin()
    _QT_APP.processEvents()
    return bool(predicate())


def _inputs(map_name: str = "fake") -> ScenarioInputs:
    """A `ScenarioInputs` over a small synthetic map (no files touched)."""
    mu = np.random.default_rng(3).integers(0, 255, (600, 640), np.uint8)
    return ScenarioInputs(
        map_name=map_name,
        mu=mu,
        preview=mu,
        mini_scale=1.0,
        center_mu=(320.0, 300.0),
        px_per_m=2.0,
        mask=None,
    )


class _FakeRecord:
    """Stand-in for `FrameRecord` (only `as_json` is used by the exporters)."""

    def __init__(self, engine: str, idx: int) -> None:
        self.engine = engine
        self.idx = idx

    def as_json(self) -> dict[str, Any]:
        return {"engine": self.engine, "idx": self.idx}


def _result(engine: str = "sift", scenario: str = "straight", **kw: Any) -> RunResult:
    """One plausible `RunResult`; callers override the fields they assert on."""
    fields: dict[str, Any] = {
        "engine": engine,
        "scenario": scenario,
        "config": {},
        "frames": 2,
        "localized_pct": 90.0,
        "pos_err_px_mean": 1.25,
        "pos_err_px_p95": 2.5,
        "pos_err_m_mean": 0.006,
        "heading_err_deg_mean": 1.5,
        "latency_ms_mean": 12.0,
        "achieved_fps": 83.0,
        "vote_rejects": 1,
        "holds": 2,
        "anchors": 3,
    }
    fields.update(kw)
    return RunResult(**fields)


def _fake_runner(engine, scenario, overrides, map_name, **kw) -> tuple[RunResult, list]:
    """One deterministic `RunResult` plus two fake per-frame records."""
    result = _result(engine, scenario.spec.name, config=dict(overrides or {}))
    return result, [_FakeRecord(engine, i) for i in range(2)]


def _slow_runner(engine, scenario, overrides, map_name, **kw):
    """A runner slow enough to still be in flight when the window closes."""
    global _SLOW_STARTED
    _SLOW_STARTED = True
    time.sleep(1.5)
    return _fake_runner(engine, scenario, overrides, map_name, **kw)


def _boom_probe(_name: str) -> list[str]:
    raise OSError("no index on this machine")


def _empty_scenario(name: str = "straight", frames: int = 2) -> Scenario:
    return Scenario(
        spec=ScenarioSpec(name=name, frames=frames),
        frames=[np.zeros((4, 4), np.uint8)] * frames,
        truth=[(0.0, 0.0, 0.0)] * frames,
        dt=0.1,
    )


class TestBenchPlan(unittest.TestCase):
    """Plan expansion: the pure, Qt-free half of the window."""

    def test_jobs_multiply_engines_scenarios_and_configs(self) -> None:
        plan = BenchPlan(
            engines=["xfeat", "sift"],
            specs=[ScenarioSpec(name="straight", frames=2), ScenarioSpec(name="turns", frames=2)],
            sweeps=[("xfeat_min_cos", [0.7, 0.8])],
        )
        scenarios = {spec.name: _empty_scenario(spec.name) for spec in plan.specs}

        jobs = plan.jobs(scenarios)

        self.assertEqual(plan.job_count(), 8)  # 2 engines x 2 scenarios x 2 values
        self.assertEqual(len(jobs), 8)
        self.assertEqual(jobs[0][0], "xfeat")
        self.assertEqual(jobs[0][1].spec.name, "straight")
        self.assertEqual(jobs[0][2], {"xfeat_min_cos": 0.7})
        # An xfeat sweep must not leak into sift's runs, which have no such key.
        self.assertTrue(all(not cfg for e, _s, cfg in jobs if e == "sift"))
        xfeat_cfgs = {tuple(sorted(cfg.items())) for e, _s, cfg in jobs if e == "xfeat"}
        self.assertEqual(
            xfeat_cfgs,
            {
                (("xfeat_min_cos", 0.7),),
                (("xfeat_min_cos", 0.8),),
            },
        )

    def test_a_shared_key_reaches_every_engine_that_knows_it(self) -> None:
        plan = BenchPlan(
            engines=["sift", "xfeat"],
            specs=[ScenarioSpec(name="straight", frames=2)],
            sweeps=[("smooth_alpha", [0.35, 0.8])],
        )
        scenarios = {"straight": _empty_scenario()}

        jobs = plan.jobs(scenarios)

        self.assertTrue(all(cfg for _e, _s, cfg in jobs))

    def test_job_count_is_zero_for_an_empty_selection(self) -> None:
        self.assertEqual(BenchPlan().job_count(), 0)
        self.assertEqual(BenchPlan(engines=["sift"]).job_count(), 0)
        self.assertEqual(BenchPlan(specs=[ScenarioSpec(name="straight")]).job_count(), 0)

    def test_expand_oat_keeps_the_baseline(self) -> None:
        configs = expand_oat([("ratio_local", [0.7, 0.8])], {})
        self.assertEqual(configs, [{"ratio_local": 0.7}, {"ratio_local": 0.8}])


class TestBenchWorker(unittest.TestCase):
    """The worker: results, per-frame records, per-case failures, stop, skip."""

    def _plan(self, engines: list[str], **kw: Any) -> BenchPlan:
        return BenchPlan(engines=engines, specs=[ScenarioSpec(name="straight", frames=2)], **kw)

    def test_worker_streams_results_and_keeps_records(self) -> None:
        worker = BenchWorker(
            self._plan(["sift", "orb"]), "fake", runner=_fake_runner, loader=_inputs
        )
        seen: list[Any] = []
        progress: list[tuple[int, int]] = []
        errors: list[str] = []
        worker.result.connect(seen.append)
        worker.progress.connect(lambda done, total: progress.append((done, total)))
        worker.failed.connect(errors.append)

        worker.run()  # synchronous: the same body the thread runs

        self.assertEqual([r.engine for r in seen], ["sift", "orb"])
        self.assertEqual(errors, [])
        self.assertEqual(progress[-1], (2, 2))
        self.assertEqual(len(worker.records), 2)
        self.assertEqual(len(worker.records[record_key(seen[0])]), 2)

    def test_worker_emits_finished_even_when_nothing_ran(self) -> None:
        finished: list[bool] = []
        worker = BenchWorker(BenchPlan(), "fake", runner=_fake_runner, loader=_inputs)
        worker.runs_finished.connect(lambda: finished.append(True))

        worker.run()

        self.assertEqual(finished, [True])

    def test_worker_reports_one_failure_and_keeps_going(self) -> None:
        def flaky(engine, scenario, overrides, map_name, **kw):
            if engine == "sift":
                raise RuntimeError("engine exploded")
            return _fake_runner(engine, scenario, overrides, map_name, **kw)

        worker = BenchWorker(self._plan(["sift", "orb"]), "fake", runner=flaky, loader=_inputs)
        seen: list[Any] = []
        errors: list[str] = []
        worker.result.connect(seen.append)
        worker.failed.connect(errors.append)

        worker.run()

        self.assertEqual([r.engine for r in seen], ["orb"])
        self.assertEqual(len(errors), 1)
        self.assertIn("engine exploded", errors[0])
        self.assertIn("sift/straight", errors[0])

    def test_worker_stops_between_cases(self) -> None:
        worker = BenchWorker(self._plan(list(ENGINES)), "fake", runner=_fake_runner, loader=_inputs)
        seen: list[Any] = []
        errors: list[str] = []
        worker.result.connect(seen.append)
        worker.failed.connect(errors.append)

        worker.request_stop()
        worker.run()

        self.assertEqual(seen, [])
        self.assertEqual(errors, ["stopped after 0/4 case(s)"])
        self.assertTrue(worker.stopping)

    def test_worker_turns_a_missing_asset_into_a_skip(self) -> None:
        def missing(_map_name: str) -> ScenarioInputs:
            raise BenchSkip("map cache for 'x' is unavailable")

        worker = BenchWorker(self._plan(["sift"]), "x", runner=_fake_runner, loader=missing)
        seen: list[Any] = []
        errors: list[str] = []
        worker.result.connect(seen.append)
        worker.failed.connect(errors.append)

        worker.run()

        self.assertEqual(seen, [])
        self.assertEqual(errors, ["skip: map cache for 'x' is unavailable"])

    def test_worker_reports_an_unexpected_loader_error(self) -> None:
        def broken(_map_name: str) -> ScenarioInputs:
            raise ValueError("corrupt npz")

        worker = BenchWorker(self._plan(["sift"]), "x", runner=_fake_runner, loader=broken)
        errors: list[str] = []
        worker.failed.connect(errors.append)

        worker.run()

        self.assertEqual(errors, ["cannot start: ValueError: corrupt npz"])

    def test_worker_uses_the_package_loader_by_default(self) -> None:
        self.assertIs(BenchWorker(BenchPlan(), "x").loader, load_inputs)


class TestBenchWindow(unittest.TestCase):
    """Widget state: engine probe, sweep rows, table, charts, export, close."""

    def setUp(self) -> None:
        self.window = self._window()
        self.assertTrue(
            _wait_until(lambda: bool(self.window.available)), "the probe never returned"
        )

    def _window(self, probe=lambda _name: ["sift", "orb"]) -> BenchWindow:
        window = BenchWindow(map_name="fake", runner=_fake_runner, probe=probe, loader=_inputs)
        self.addCleanup(self._close, window)
        return window

    @staticmethod
    def _close(window: BenchWindow) -> None:
        worker = window.worker
        if worker is not None:
            worker.request_stop()
            worker.wait(20000)
            window.worker = None
        window.close()
        _QT_APP.processEvents()

    def test_probe_ticks_available_engines_and_disables_the_rest(self) -> None:
        boxes = self.window.engine_boxes
        self.assertEqual(self.window.available, ["sift", "orb"])
        self.assertTrue(boxes["sift"].isChecked() and boxes["sift"].isEnabled())
        self.assertTrue(boxes["orb"].isChecked() and boxes["orb"].isEnabled())
        self.assertFalse(boxes["xfeat"].isChecked() or boxes["xfeat"].isEnabled())
        self.assertIn("not available", boxes["xfeat"].toolTip())
        self.assertIn("ratio_local", boxes["sift"].toolTip())
        self.assertEqual(self.window.selected_engines(), ["sift", "orb"])
        self.assertIn("unavailable", self.window.status_lbl.text())

    def test_probe_failure_disables_every_engine(self) -> None:
        window = self._window(probe=_boom_probe)
        self.assertTrue(_wait_until(lambda: "failed" in window.status_lbl.text()))
        self.assertFalse(any(box.isEnabled() for box in window.engine_boxes.values()))
        self.assertEqual(window.selected_engines(), [])

    def test_probe_is_off_the_gui_thread(self) -> None:
        window = self._window()
        self.assertTrue(_wait_until(lambda: bool(window.available)))
        self.assertIsNone(window._probe)

    def test_quick_and_full_presets(self) -> None:
        self.assertTrue(self.window.quick_radio.isChecked())
        self.assertEqual(self.window.frames_spin.value(), 60)
        self.assertEqual(self.window.rates(), [(10.0, 15.0)])
        self.assertTrue(self.window.fps_spin.isEnabled())

        self.window.full_radio.setChecked(True)

        self.assertEqual(self.window.frames_spin.value(), 150)
        self.assertEqual(
            self.window.rates(), [(10.0, 10.0), (10.0, 20.0), (30.0, 10.0), (30.0, 20.0)]
        )
        # In Full mode the rate matrix is the CLI's, so the spins are locked.
        self.assertFalse(self.window.fps_spin.isEnabled())

    def test_default_scenarios_and_run_button(self) -> None:
        self.assertEqual(self.window.selected_scenarios(), ["straight", "turns", "void"])
        self.assertTrue(self.window.run_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())
        self.assertFalse(self.window.export_btn.isEnabled())

        for name in ("turns", "void", "mixed_s"):
            self.window.scenario_boxes[name].setChecked(False)

        self.assertEqual(self.window.selected_scenarios(), ["straight"])
        self.assertEqual(self.window._job_count(), 2)
        self.window.scenario_boxes["straight"].setChecked(False)
        self.assertEqual(self.window._job_count(), 0)
        self.assertFalse(self.window.run_btn.isEnabled())

    def test_sweep_rows_add_and_remove(self) -> None:
        self.assertEqual(self.window.sweep_specs(), [])
        self.window.sweep_key.setCurrentText("ratio_local")
        self.window.sweep_values.setText("0.7,0.8")

        self.window.add_sweep_row()

        self.assertEqual(self.window.sweep_specs(), [("ratio_local", [0.7, 0.8])])
        self.assertIn("sweeping", self.window.sweep_list_lbl.text())
        # 2 engines x 3 scenarios x 2 swept values
        self.assertEqual(self.window._job_count(), 12)

        row = self.window.sweep_rows[0]
        self.window.remove_sweep_row(row)

        self.assertEqual(self.window.sweep_rows, [])
        self.assertIn("no sweeps", self.window.sweep_list_lbl.text())
        self.assertEqual(self.window._job_count(), 6)

    def test_a_broken_sweep_row_is_ignored(self) -> None:
        self.window.sweep_key.setCurrentText("ratio_local")
        self.window.sweep_values.setText("0.7,0.8")
        self.window.add_sweep_row()

        self.window.sweep_rows[0].value_box.setText("nonsense")

        self.assertEqual(self.window.sweep_specs(), [])
        self.window.sweep_rows[0].value_box.setText("")
        self.assertEqual(self.window.sweep_specs(), [])

    def test_add_sweep_row_refuses_an_empty_or_bad_selection(self) -> None:
        self.window.sweep_values.setText("")
        self.window.add_sweep_row()
        self.assertEqual(self.window.sweep_rows, [])
        self.assertIn("pick a key", self.window.status_lbl.text())

        self.window.sweep_key.setCurrentText("ratio_local")
        self.window.sweep_values.setText("abc")
        self.window.add_sweep_row()

        self.assertEqual(self.window.sweep_rows, [])
        self.assertIn("bad value", self.window.status_lbl.text())

    def test_sweep_row_remove_button_drops_the_row(self) -> None:
        self.window.sweep_key.setCurrentText("ratio_local")
        self.window.sweep_values.setText("0.7")
        self.window.add_sweep_row()
        self.assertEqual(self.window.sweep_box.count(), 1)
        row = self.window.sweep_rows[0]

        row._remove()  # the button is wired to this slot

        _QT_APP.processEvents()
        self.assertEqual(self.window.sweep_rows, [])
        self.assertEqual(self.window.sweep_box.count(), 0)

    def test_results_fill_the_table_and_the_charts(self) -> None:
        results = [
            _result("sift", "straight"),
            _result("sift", "turns", latency_ms_mean=20.0),
            _result("orb", "straight", latency_ms_mean=8.0),
        ]

        for result in results:
            self.window._on_result(result)

        table = self.window.table
        self.assertEqual(table.rowCount(), 3)
        self.assertEqual(_cell_text(table, 0, 0), "sift")
        self.assertEqual(_cell_text(table, 0, 2), "-")
        self.assertEqual(_cell_text(table, 0, 8), "12")
        self.assertEqual(_cell_text(table, 1, 8), "20")
        self.assertTrue(self.window.export_btn.isEnabled())
        if self.window.charts_ok:
            # One bar set per engine, plus one line per swept key with 2+ values.
            self.assertEqual(set(self.window.latency_sets), {"sift", "orb"})
            self.assertEqual(len(self.window.error_series), 0)
            self.window._on_result(
                _result("sift", "straight", config={"ratio_local": 0.8}, pos_err_px_mean=1.25)
            )
            self.window._on_result(
                _result("sift", "turns", config={"ratio_local": 0.7}, pos_err_px_mean=2.5)
            )
            self.assertEqual(len(self.window.error_series), 1)
            self.assertEqual(self.window.error_series[0].name(), "ratio_local")

    def test_cell_formats_numbers_and_the_config(self) -> None:
        result = _result(
            "sift",
            "straight",
            config={"ratio_local": 0.8},
            localized_pct=90.0,
            pos_err_px_mean=1.234,
            vote_rejects=3,
        )

        self.assertEqual(BenchWindow._cell(result, None).text(), "ratio_local=0.8")
        self.assertEqual(BenchWindow._cell(result, "engine").text(), "sift")
        self.assertEqual(BenchWindow._cell(result, "scenario").text(), "straight")
        self.assertEqual(BenchWindow._cell(result, "localized_pct").text(), "90")
        self.assertEqual(BenchWindow._cell(result, "pos_err_px_mean").text(), "1.234")
        self.assertEqual(BenchWindow._cell(result, "vote_rejects").text(), "3")
        self.assertEqual(BenchWindow._cell(_result(), None).text(), "-")

    def test_numeric_cells_sort_numerically(self) -> None:
        for engine, latency in (("sift", 12.0), ("orb", 8.0), ("xfeat", 100.0)):
            self.window._on_result(_result(engine, "straight", latency_ms_mean=latency))

        self.window.table.sortItems(8, Qt.SortOrder.DescendingOrder)

        self.assertEqual(
            [_cell_text(self.window.table, row, 8) for row in range(3)],
            ["100", "12", "8"],  # lexicographic sorting would give 100, 12, 8 too
        )
        self.window.table.sortItems(8, Qt.SortOrder.AscendingOrder)
        self.assertEqual(
            [_cell_text(self.window.table, row, 8) for row in range(3)],
            ["8", "12", "100"],
        )

    def test_export_writes_csv_report_and_per_frame(self) -> None:
        result = _result("sift", "straight")
        self.window.results = [result]
        self.window.records = {
            record_key(result): cast(
                "list[FrameRecord]", [_FakeRecord("sift", 0), _FakeRecord("sift", 1)]
            )
        }

        with tempfile.TemporaryDirectory() as tmp:
            report = self.window.write_export(tmp)

            self.assertTrue(os.path.exists(report))
            self.assertTrue(os.path.exists(os.path.join(tmp, "results.csv")))
            files = os.listdir(os.path.join(tmp, "per_frame"))
            self.assertEqual(files, ["sift_straight_%s.jsonl" % _hash_of(result)])
            with open(os.path.join(tmp, "per_frame", files[0]), encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual([r["idx"] for r in rows], [0, 1])
            with open(report, encoding="utf-8") as fh:
                self.assertIn("sift", fh.read())

    def test_export_without_results_is_a_no_op(self) -> None:
        self.window.results = []
        self.window.export_btn.setEnabled(False)
        self.window.export_results()  # must not open a dialog
        self.assertFalse(self.window.export_btn.isEnabled())

    def test_start_run_streams_results_into_the_window(self) -> None:
        self.window.scenario_boxes["turns"].setChecked(False)
        self.window.scenario_boxes["void"].setChecked(False)

        self.window.start_run()

        self.assertTrue(_wait_until(lambda: self.window.worker is None))
        self.assertEqual([r.engine for r in self.window.results], ["sift", "orb"])
        self.assertEqual(self.window.table.rowCount(), 2)
        self.assertEqual(len(self.window.records), 2)
        self.assertEqual(self.window.progress.value(), 2)
        self.assertIn("2 run(s) done", self.window.status_lbl.text())
        self.assertTrue(self.window.run_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())

    def test_stop_run_asks_the_worker_to_quit(self) -> None:
        global _SLOW_STARTED
        _SLOW_STARTED = False
        self.window.runner = _slow_runner
        for name in ("turns", "void"):
            self.window.scenario_boxes[name].setChecked(False)

        self.window.start_run()

        self.assertTrue(_wait_until(lambda: _SLOW_STARTED, 10.0))
        self.window.stop_run()
        worker = self.window.worker
        assert worker is not None
        self.assertTrue(worker.stopping)
        self.assertIn("stopping", self.window.status_lbl.text())
        self.assertTrue(_wait_until(lambda: self.window.worker is None, 30.0))
        # Both engines were in the plan, but the stop lands between cases.
        self.assertLessEqual(len(self.window.results), 2)

    def test_start_run_with_nothing_selected_reports_it(self) -> None:
        for box in self.window.engine_boxes.values():
            box.setChecked(False)

        self.window.start_run()

        self.assertIsNone(self.window.worker)
        self.assertIn("nothing to run", self.window.status_lbl.text())

    def test_close_waits_for_the_worker_instead_of_destroying_it(self) -> None:
        global _SLOW_STARTED
        _SLOW_STARTED = False
        self.window.runner = _slow_runner
        self.window.start_run()
        self.assertTrue(_wait_until(lambda: _SLOW_STARTED, 10.0))

        self.window.close()  # must stop and join, never kill a running QThread

        _QT_APP.processEvents()
        self.assertFalse(_SLOW_STARTED is False)
        self.assertIsNone(self.window.worker)
        self.assertIsNotNone(self.window.results)


class TestLogsTabWiring(unittest.TestCase):
    """The Logs section owns the launcher for every `output/` writer."""

    def _tab(self, **kw: Any) -> LogsTab:
        cfg = AppConfig()
        cfg.map.name = "zestafona"
        tab = LogsTab(None, cfg, lambda: None, lambda: None, lambda: "zestafona", **kw)
        self.addCleanup(tab.deleteLater)
        return tab

    def test_logs_section_has_the_bench_launcher(self) -> None:
        tab = self._tab()
        self.assertTrue(tab.bench_btn.isEnabled())
        self.assertIn("bench_synthetic.py", tab.bench_btn.toolTip())

    def test_the_button_constructs_the_window_with_a_stub(self) -> None:
        """The plan's Qt check: the button exists and opens a window, no run."""
        seen: list[str] = []

        def factory(map_name: str) -> Any:
            seen.append(map_name)
            return BenchWindow(
                map_name=map_name,
                runner=_fake_runner,
                probe=lambda _n: ["sift"],
                loader=_inputs,
            )

        tab = self._tab(bench_factory=factory)

        tab.bench_btn.click()
        window = cast(BenchWindow, tab._bench_window)

        self.assertEqual(seen, ["zestafona"])
        self.assertIsNone(window.worker)
        self.assertEqual(window.results, [])
        self.addCleanup(window.close)
        self.assertIs(window, tab.open_bench_window())

    def test_bench_button_opens_one_window_for_the_configured_map(self) -> None:
        tab = self._tab()

        window = tab.open_bench_window()
        self.addCleanup(window.close)
        self.assertIsInstance(window, BenchWindow)
        self.assertEqual(window.map_name, "zestafona")
        self.assertIs(window, tab.open_bench_window())

    def test_bench_window_is_recreated_after_it_is_destroyed(self) -> None:
        tab = self._tab()
        first = tab.open_bench_window()
        first.close()
        _QT_APP.processEvents()

        second = tab.open_bench_window()
        self.addCleanup(second.close)
        self.assertIsNot(first, second)
        self.assertIs(tab._bench_window, second)


class TestScenarioInputs(unittest.TestCase):
    """The preflight the CLI and the studio both load the map through."""

    def test_render_uses_the_injected_arrays(self) -> None:
        inputs = _inputs()
        spec = ScenarioSpec(name="straight", frames=2, fps=10.0, speed_mps=5.0)

        scenario = inputs.render(spec)

        self.assertEqual(len(scenario.frames), 2)
        self.assertEqual(len(scenario.truth), 2)
        self.assertEqual(scenario.dt, 0.1)
        self.assertEqual(len(inputs.render_all([spec, spec])), 2)

    def test_missing_map_raises_bench_skip(self) -> None:
        with self.assertRaises(BenchSkip) as ctx:
            load_inputs("definitely_not_a_map")
        self.assertIn("definitely_not_a_map", str(ctx.exception))

    def test_engines_and_registry_stay_in_one_place(self) -> None:
        self.assertEqual(set(ENGINES), {"sift", "orb", "xfeat", "hybrid"})
        self.assertIn("ratio_local", param_registry("sift"))

    def test_param_hint_lists_the_keys_and_what_changing_them_does(self) -> None:
        hint = param_hint("sift")
        self.assertIn("ratio_local", hint)
        self.assertIn("lower", hint)
        self.assertIn("higher", hint)


class TestNoProductionSideEffects(unittest.TestCase):
    """Opening the window must not write config.json or move the map."""

    def setUp(self) -> None:
        self.config_path = os.path.join(ROOT, "config.json")
        self.before = (
            os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else None,
            open(self.config_path, "rb").read() if os.path.exists(self.config_path) else b"",
        )

    def test_opening_the_window_leaves_config_untouched(self) -> None:
        window = BenchWindow(
            map_name="fake", runner=_fake_runner, probe=lambda _n: ["sift"], loader=_inputs
        )
        self.addCleanup(window.close)
        self.assertTrue(_wait_until(lambda: bool(window.available)))

        window.add_sweep_row() if window.sweep_values.text() else None
        window.start_run()
        self.assertTrue(_wait_until(lambda: window.worker is None))

        self.assertEqual(self.before[1], open(self.config_path, "rb").read())
        self.assertEqual(self.before[0], os.path.getmtime(self.config_path))

    def test_export_only_touches_the_chosen_directory(self) -> None:
        window = BenchWindow(
            map_name="fake", runner=_fake_runner, probe=lambda _n: ["sift"], loader=_inputs
        )
        self.addCleanup(window.close)
        self.assertTrue(_wait_until(lambda: bool(window.available)))
        before = sorted(os.listdir(os.path.join(ROOT, "output")))

        with tempfile.TemporaryDirectory() as tmp:
            result = _result("sift", "straight")
            window.results = [result]
            window.records = {record_key(result): []}
            window.write_export(tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "report.md")))

        self.assertEqual(before, sorted(os.listdir(os.path.join(ROOT, "output"))))


def _hash_of(result: RunResult) -> str:
    return config_hash(result.config)


def _cell_text(table: Any, row: int, col: int) -> str:
    """`QTableWidget.item()` returns None for an empty cell; fail loudly here."""
    item = table.item(row, col)
    assert item is not None, f"no item at ({row}, {col})"
    return item.text()


if __name__ == "__main__":
    unittest.main()
