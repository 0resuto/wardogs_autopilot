"""Interactive benchmark window (PySide6): synthetic runs over the real engines.

The window is a thin shell over `autopilot.bench`: it collects a selection of
engines, scenarios, rates and OAT sweep rows, then a worker thread runs the same
`runner.run_case` calls the CLI runs and streams the `RunResult`s into a table
while two QtCharts views update.

Every piece of bench work happens off the GUI thread. `run_case` drives the
locator singletons (map, engine, feature index), which the studio thread must not
touch while a run is in flight, and both the engine availability probe (which
imports onnxruntime, ~4 s the first time) and the frame rendering are slow
enough to freeze the UI.

The window is non-modal and opens from the Logs section, because that section
owns every control that writes into `output/`.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QLineSeries,
    QValueAxis,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import PROJECT_ROOT
from ..bench import (
    ENGINES,
    SCENARIO_NAMES,
    BenchSkip,
    FrameRecord,
    RunResult,
    Scenario,
    ScenarioSpec,
    available_engines,
    config_hash,
    expand_oat,
    load_inputs,
    param_registry,
    parse_sweep,
    run_case,
    spec_for,
    write_csv,
    write_jsonl,
    write_report,
)
from .icons import icon
from .param_tips import summary_for, tip_for
from .theme import BLUE, GREEN, RED, TEXT_DIM, TEXT_MUTED, YELLOW

#: Table columns: (header, RunResult attribute; None = a computed cell).
_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("engine", "engine"),
    ("scenario", "scenario"),
    ("config", None),
    ("loc%", "localized_pct"),
    ("err px", "pos_err_px_mean"),
    ("p95 px", "pos_err_px_p95"),
    ("err m", "pos_err_m_mean"),
    ("th deg", "heading_err_deg_mean"),
    ("ms", "latency_ms_mean"),
    ("fps", "achieved_fps"),
    ("vote", "vote_rejects"),
    ("hold", "holds"),
    ("anch", "anchors"),
)

#: Columns shown with 2 decimals (everything else is an int or a label).
_FLOAT_COLUMNS = frozenset(
    {
        "localized_pct",
        "pos_err_px_mean",
        "pos_err_px_p95",
        "pos_err_m_mean",
        "heading_err_deg_mean",
        "latency_ms_mean",
        "achieved_fps",
    }
)

#: Short descriptions of the curated scenarios (shown as tooltips).
SCENARIO_HINTS = {
    "straight": "constant heading - baseline precision and latency",
    "turns": "straight, then a constant yaw rate - steering-like motion",
    "mixed_s": "straight, then the yaw sign flips every 2 s - heading sweep",
    "void": "turns plus flat frames at 20/35/50 - hold and recovery",
}

#: How long closeEvent waits for the in-flight case before refusing to close.
_CLOSE_WAIT_MS = 15000


def charts_available() -> bool:
    """QtCharts ships with PySide6, but its absence must not break the studio."""
    try:
        import PySide6.QtCharts  # noqa: F401
    except ImportError:
        return False
    return True


def param_hint(engine: str) -> str:
    """Tooltip listing the keys one engine can be swept over and what they do."""
    registry = param_registry(engine)
    if not registry:
        return "no sweepable parameters for this engine"
    lines = ["Sweepable keys (add them in the sweep card below):"]
    lines += [f"- {key}: {summary_for(key, spec.label)}" for key, spec in registry.items()]
    return "\n".join(lines)


def config_label(config: dict[str, Any]) -> str:
    """`a=1 b=2` for the table cell / log line (empty for the shipped config)."""
    return " ".join("%s=%s" % (k, config[k]) for k in sorted(config))


def record_key(result: RunResult) -> str:
    """Stable id of one run, used to pair a result with its per-frame records."""
    return "%s|%s|%s" % (result.engine, result.scenario, config_hash(result.config))


@dataclass
class BenchPlan:
    """What to run: engines x scenarios x OAT configs (no Qt, no map access)."""

    engines: list[str] = field(default_factory=list)
    specs: list[ScenarioSpec] = field(default_factory=list)
    sweeps: list[tuple[str, list[Any]]] = field(default_factory=list)

    def job_count(self) -> int:
        """Upper bound on the number of cases (empty configs count as one)."""
        configs = max(1, len(expand_oat(self.sweeps, {})))
        return len(self.engines) * len(self.specs) * configs

    def jobs(self, scenarios: dict[str, Scenario]) -> list[tuple[str, Scenario, dict[str, Any]]]:
        """`(engine, scenario, config)` triples in a stable, reproducible order.

        A sweep of engine A's keys must not leak into engine B's runs, so each
        config is filtered down to the keys that engine's registry knows.
        """
        configs = expand_oat(self.sweeps, {})
        out: list[tuple[str, Scenario, dict[str, Any]]] = []
        for engine in self.engines:
            known = param_registry(engine)
            for spec in self.specs:
                scenario = scenarios[spec.name]
                for config in configs:
                    out.append((engine, scenario, {k: v for k, v in config.items() if k in known}))
        return out


class ProbeWorker(QThread):
    """Asks `available_engines` off the GUI thread (onnxruntime import is slow)."""

    done = Signal(object, str)

    def __init__(
        self,
        map_name: str,
        probe: Callable[[str], list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.map_name = str(map_name)
        self.probe = probe or available_engines

    def run(self) -> None:
        try:
            self.done.emit(list(self.probe(self.map_name)), "")
        except Exception as exc:  # noqa: BLE001 - an unreadable index is a skip
            self.done.emit([], f"{type(exc).__name__}: {exc}")


class BenchWorker(QThread):
    """Runs the bench plan off the GUI thread and streams the results back.

    The map assets are loaded and the frames rendered once, inside the thread, so
    the GUI never blocks and the locator singletons are only touched by this one
    thread while a run is in flight. Stopping is checked between cases, so the
    in-flight case always finishes and reports.
    """

    progress = Signal(int, int)
    result = Signal(object)
    failed = Signal(str)
    runs_finished = Signal()

    def __init__(
        self,
        plan: BenchPlan,
        map_name: str,
        budget: float = 3.0,
        runner: Callable[..., tuple[RunResult, list[FrameRecord]]] = run_case,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.map_name = str(map_name)
        self.budget = float(budget)
        self.runner = runner
        self.loader = loader or load_inputs
        self.records: dict[str, list[FrameRecord]] = {}
        self._stop = threading.Event()

    def request_stop(self) -> None:
        """Ask the worker to stop after the case it is currently inside."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def prepare(self) -> list[tuple[str, Scenario, dict[str, Any]]]:
        """Load the map assets and render the frames. Raises `BenchSkip`."""
        inputs = self.loader(self.map_name)
        scenarios = {spec.name: inputs.render(spec) for spec in self.plan.specs}
        return self.plan.jobs(scenarios)

    def run(self) -> None:
        total = self.plan.job_count()
        self.progress.emit(0, total)
        try:
            try:
                jobs = self.prepare()
            except BenchSkip as exc:
                self.failed.emit("skip: %s" % exc)
                return
            except Exception as exc:  # noqa: BLE001 - a broken asset is a message
                self.failed.emit("cannot start: %s: %s" % (type(exc).__name__, exc))
                return
            self.progress.emit(0, len(jobs))
            for done, (engine, scenario, config) in enumerate(jobs, start=1):
                if self._stop.is_set():
                    self.failed.emit("stopped after %d/%d case(s)" % (done - 1, len(jobs)))
                    break
                label = "%s/%s%s" % (
                    engine,
                    scenario.spec.name,
                    (" " + config_label(config)) if config else "",
                )
                try:
                    result, records = self.runner(
                        engine, scenario, config, self.map_name, budget=self.budget
                    )
                except Exception as exc:  # noqa: BLE001 - one case must not kill the thread
                    self.failed.emit("%s: %s: %s" % (label, type(exc).__name__, exc))
                    continue
                self.records[record_key(result)] = list(records)
                self.result.emit(result)
                self.progress.emit(done, len(jobs))
        finally:
            self.runs_finished.emit()


class _SweepRow:
    """One `key = values` line of the sweep card, with its own remove button."""

    def __init__(
        self,
        parent: QWidget,
        keys: Sequence[str],
        key: str,
        values: str,
        on_remove: Callable[[_SweepRow], None],
    ) -> None:
        self._on_remove = on_remove
        self.widget = QWidget(parent)
        self.key_box = QComboBox(self.widget)
        self.key_box.addItems(list(keys))
        self.key_box.setCurrentText(key)
        self.key_box.setToolTip(tip_for(key))
        self.key_box.currentTextChanged.connect(
            lambda selected: self.key_box.setToolTip(tip_for(selected))
        )
        self.value_box = QLineEdit(values, self.widget)

        drop = QPushButton(self.widget)
        drop.setIcon(icon("x"))
        drop.setFixedWidth(30)
        drop.setToolTip("Remove this sweep row")
        drop.clicked.connect(self._remove)

        row = QHBoxLayout(self.widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.key_box, 1)
        row.addWidget(QLabel("=", self.widget))
        row.addWidget(self.value_box, 1)
        row.addWidget(drop)

    def _remove(self) -> None:
        self._on_remove(self)

    def spec(self) -> tuple[str, list[Any]] | None:
        """`(key, values)` when the row parses, else None (it is ignored)."""
        key = self.key_box.currentText()
        raw = self.value_box.text().strip()
        if not key or not raw:
            return None
        try:
            return parse_sweep("%s=%s" % (key, raw))
        except ValueError:
            return None


class BenchWindow(QWidget):
    """Non-modal window: pick engines/scenarios/rates, sweep, run, export."""

    def __init__(
        self,
        parent: QWidget | None = None,
        map_name: str | None = None,
        runner: Callable[..., tuple[RunResult, list[FrameRecord]]] = run_case,
        probe: Callable[[str], list[str]] | None = None,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(None)  # top-level window: the studio keeps running
        self._parent_ref = parent  # kept so the window can reach the studio later
        self.setWindowTitle("Localization benchmark")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(1180, 740)

        self.runner = runner
        self.loader = loader or load_inputs
        self.map_name = str(map_name or self._config_map())
        self.results: list[RunResult] = []
        self.records: dict[str, list[FrameRecord]] = {}
        self.worker: BenchWorker | None = None
        self._probe: ProbeWorker | None = None
        self.sweep_rows: list[_SweepRow] = []
        self.available: list[str] = []

        self._build_ui()
        self._probe_engines(probe or available_engines)

    @staticmethod
    def _config_map() -> str:
        """The map the studio is configured for (read once, never written)."""
        from ..common.config import AppConfig

        return str(AppConfig().map.name)

    # ------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        head = QHBoxLayout()
        map_lbl = QLabel("map: %s" % self.map_name, self)
        map_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        head.addWidget(map_lbl)
        head.addStretch()
        self.status_lbl = QLabel("idle", self)
        self.status_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9pt;")
        head.addWidget(self.status_lbl)
        root.addLayout(head)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(self._engine_card(), 1)
        controls.addWidget(self._scenario_card(), 1)
        controls.addWidget(self._budget_card(), 1)
        root.addLayout(controls)

        root.addWidget(self._sweep_card())

        actions = QHBoxLayout()
        self.run_btn = QPushButton("Run", self)
        self.run_btn.setIcon(icon("play"))
        self.run_btn.setToolTip("Run the selected engines x scenarios x sweep values")
        self.run_btn.clicked.connect(self.start_run)
        actions.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setIcon(icon("stop"))
        self.stop_btn.setToolTip("Stop after the case that is in flight")
        self.stop_btn.clicked.connect(self.stop_run)
        actions.addWidget(self.stop_btn)

        self.export_btn = QPushButton("Export", self)
        self.export_btn.setIcon(icon("folder"))
        self.export_btn.setToolTip("Write results.csv, report.md and per_frame/*.jsonl")
        self.export_btn.clicked.connect(self.export_results)
        actions.addWidget(self.export_btn)
        actions.addStretch()
        root.addLayout(actions)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m")
        root.addWidget(self.progress)

        bottom = QHBoxLayout()
        bottom.addWidget(self._table(), 3)
        bottom.addWidget(self._charts(), 2)
        root.addLayout(bottom, 1)

    def _engine_card(self) -> QGroupBox:
        card = QGroupBox("Engines", self)
        box = QVBoxLayout(card)
        self.engine_boxes: dict[str, QCheckBox] = {}
        for engine in ENGINES:
            ck = QCheckBox(engine, card)
            ck.setEnabled(False)  # enabled by _apply_probe()
            ck.setToolTip(param_hint(engine))
            ck.toggled.connect(self._update_buttons)
            box.addWidget(ck)
            self.engine_boxes[engine] = ck
        box.addStretch()
        return card

    def _scenario_card(self) -> QGroupBox:
        card = QGroupBox("Scenarios", self)
        box = QVBoxLayout(card)
        self.scenario_boxes: dict[str, QCheckBox] = {}
        for name in SCENARIO_NAMES:
            ck = QCheckBox(name, card)
            ck.setToolTip(SCENARIO_HINTS.get(name, ""))
            ck.toggled.connect(self._update_buttons)
            box.addWidget(ck)
            self.scenario_boxes[name] = ck
        for quick in ("straight", "turns", "void"):
            self.scenario_boxes[quick].setChecked(True)

        mode = QHBoxLayout()
        self.quick_radio = QRadioButton("Quick", card)
        self.quick_radio.setChecked(True)
        self.quick_radio.setToolTip("60 frames at fps 10 / 15 m/s")
        self.full_radio = QRadioButton("Full", card)
        self.full_radio.setToolTip("150 frames at fps 10+30 and 10+20 m/s")
        for radio in (self.quick_radio, self.full_radio):
            radio.toggled.connect(self._on_mode_changed)
            mode.addWidget(radio)
        box.addLayout(mode)
        box.addStretch()
        return card

    def _budget_card(self) -> QGroupBox:
        card = QGroupBox("Scenario budget", self)
        grid = QGridLayout(card)
        self.frames_spin = QSpinBox(card)
        self.frames_spin.setRange(2, 2000)
        self.frames_spin.setValue(60)
        self.frames_spin.setToolTip(tip_for("bench.frames"))
        self.fps_spin = QSpinBox(card)
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(10)
        self.fps_spin.setToolTip(tip_for("bench.fps"))
        self.speed_spin = QSpinBox(card)
        self.speed_spin.setRange(1, 90)
        self.speed_spin.setValue(15)
        self.speed_spin.setToolTip(tip_for("bench.speed"))
        for row, (label, spin) in enumerate(
            (("frames", self.frames_spin), ("fps", self.fps_spin), ("speed m/s", self.speed_spin))
        ):
            grid.addWidget(QLabel(label, card), row, 0)
            grid.addWidget(spin, row, 1)
            spin.valueChanged.connect(self._update_buttons)
        return card

    def _sweep_card(self) -> QGroupBox:
        card = QGroupBox("Parameter sweep (one value per key, OAT)", self)
        box = QVBoxLayout(card)
        self.sweep_box = QVBoxLayout()
        box.addLayout(self.sweep_box)

        self.sweep_key = QComboBox(card)
        self.sweep_key.addItems(list(param_registry()))
        self.sweep_key.setToolTip(tip_for(self.sweep_key.currentText()))
        self.sweep_key.currentTextChanged.connect(
            lambda selected: self.sweep_key.setToolTip(tip_for(selected))
        )
        self.sweep_values = QLineEdit(card)
        self.sweep_values.setPlaceholderText("e.g. 0.75,0.82,0.88")
        self.sweep_values.setFixedWidth(220)
        add_btn = QPushButton("Add", card)
        add_btn.setToolTip("Add this key with its values to the sweep")
        add_btn.clicked.connect(self.add_sweep_row)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("key", card))
        picker.addWidget(self.sweep_key, 1)
        picker.addWidget(QLabel("values", card))
        picker.addWidget(self.sweep_values)
        picker.addWidget(add_btn)
        picker.addStretch()
        box.addLayout(picker)

        self.sweep_list_lbl = QLabel("", card)
        self.sweep_list_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8pt;")
        self.sweep_list_lbl.setWordWrap(True)
        box.addWidget(self.sweep_list_lbl)
        return card

    def _table(self) -> QTableWidget:
        table = QTableWidget(0, len(_COLUMNS), self)
        table.setHorizontalHeaderLabels([title for title, _key in _COLUMNS])
        table.setSortingEnabled(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table = table
        return table

    def _charts(self) -> QWidget:
        """Two QtCharts views: mean latency per engine, error vs a swept key."""
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 0, 0, 0)
        self.charts_ok = charts_available()
        if not self.charts_ok:
            lbl = QLabel("PySide6.QtCharts is unavailable", panel)
            lbl.setStyleSheet(f"color: {YELLOW};")
            box.addWidget(lbl)
            return panel

        self.latency_chart = QChart()
        self.latency_series = QBarSeries()
        self.latency_sets: dict[str, QBarSet] = {}
        self.latency_chart.addSeries(self.latency_series)
        self.latency_chart.setTitle("Mean latency by engine (ms)")
        self.latency_chart.legend().setVisible(False)
        self.latency_axis_x = QBarCategoryAxis()
        self.latency_axis_x.setTitleText("engine")
        self.latency_axis_y = QValueAxis()
        self.latency_axis_y.setTitleText("ms")
        self.latency_chart.addAxis(self.latency_axis_x, Qt.AlignmentFlag.AlignBottom)
        self.latency_chart.addAxis(self.latency_axis_y, Qt.AlignmentFlag.AlignLeft)
        self.latency_series.attachAxis(self.latency_axis_x)
        self.latency_series.attachAxis(self.latency_axis_y)
        self.latency_view = QChartView(self.latency_chart, self)
        self.latency_view.setMinimumHeight(180)
        box.addWidget(self.latency_view, 1)

        self.error_chart = QChart()
        self.error_chart.setTitle("Position error vs swept value")
        self.error_axis_x = QValueAxis()
        self.error_axis_x.setTitleText("value")
        self.error_axis_y = QValueAxis()
        self.error_axis_y.setTitleText("err px")
        self.error_chart.addAxis(self.error_axis_x, Qt.AlignmentFlag.AlignBottom)
        self.error_chart.addAxis(self.error_axis_y, Qt.AlignmentFlag.AlignLeft)
        self.error_chart.legend().setVisible(False)
        self.error_series: list[QLineSeries] = []
        self.error_view = QChartView(self.error_chart, self)
        self.error_view.setMinimumHeight(180)
        box.addWidget(self.error_view, 1)
        return panel

    # ------------------------------------------------------------- selection

    def _probe_engines(self, probe: Callable[[str], list[str]]) -> None:
        """Ask which engines can run here, without blocking the studio thread."""
        self._say("probing engines...", TEXT_DIM)
        self._probe = ProbeWorker(self.map_name, probe)
        self._probe.done.connect(self._apply_probe)
        self._probe.start()

    def _apply_probe(self, usable: object, error: str = "") -> None:
        """Tick the engines that can run here; name the ones that cannot."""
        names = {str(name) for name in usable} if isinstance(usable, (list, tuple, set)) else set()
        self.available = [e for e in ENGINES if e in names]
        for engine, ck in self.engine_boxes.items():
            ok = engine in self.available
            ck.setEnabled(ok)
            ck.setChecked(ok)
            ck.setToolTip(param_hint(engine) if ok else "not available for this map")
        self._probe = None
        if error:
            self._say("engine probe failed: %s" % error, YELLOW)
            return
        missing = [e for e in ENGINES if e not in self.available]
        self._say("unavailable: %s" % ", ".join(missing) if missing else "ready", TEXT_DIM)
        self._update_buttons()

    def selected_engines(self) -> list[str]:
        return [e for e in ENGINES if self.engine_boxes[e].isChecked()]

    def selected_scenarios(self) -> list[str]:
        return [s for s in SCENARIO_NAMES if self.scenario_boxes[s].isChecked()]

    def sweep_specs(self) -> list[tuple[str, list[Any]]]:
        """The complete sweep rows as `(key, values)`; broken rows are skipped."""
        specs = (row.spec() for row in self.sweep_rows)
        return [spec for spec in specs if spec is not None]

    def add_sweep_row(self) -> None:
        key = self.sweep_key.currentText()
        raw = self.sweep_values.text().strip()
        if not key or not raw:
            self._say("pick a key and at least one value first", YELLOW)
            return
        try:
            parse_sweep("%s=%s" % (key, raw))
        except ValueError as exc:
            self._say(str(exc), RED)
            return
        row = _SweepRow(self, list(param_registry()), key, raw, self.remove_sweep_row)
        self.sweep_rows.append(row)
        self.sweep_box.addWidget(row.widget)
        self._refresh_sweep_label()
        self._update_buttons()

    def remove_sweep_row(self, row: _SweepRow) -> None:
        """Drop a sweep row (its remove button calls this, then deletes it)."""
        self.sweep_rows = [r for r in self.sweep_rows if r is not row]
        self.sweep_box.removeWidget(row.widget)
        row.widget.deleteLater()
        self._refresh_sweep_label()
        self._update_buttons()

    def _refresh_sweep_label(self) -> None:
        specs = self.sweep_specs()
        self.sweep_list_lbl.setText(
            "sweeping: %s"
            % ", ".join("%s=%s" % (k, ",".join(_fmt(v) for v in vs)) for k, vs in specs)
            if specs
            else "no sweeps - every engine runs with the shipped config"
        )

    def _on_mode_changed(self, _checked: bool = False) -> None:
        full = self.full_radio.isChecked()
        self.frames_spin.setValue(150 if full else 60)
        self._refresh_budget()
        self._update_buttons()

    def _refresh_budget(self) -> None:
        """Quick = one rate; Full = the CLI's fps 10/30 x 10/20 m/s matrix."""
        full = self.full_radio.isChecked()
        self.fps_spin.setEnabled(not full)
        self.speed_spin.setEnabled(not full)
        if full:
            self.fps_spin.setValue(10)
            self.speed_spin.setValue(10)

    def rates(self) -> list[tuple[float, float]]:
        """The (fps, speed) pairs to run: the Full matrix, or the single pair."""
        fps, speed = float(self.fps_spin.value()), float(self.speed_spin.value())
        if not self.full_radio.isChecked():
            return [(fps, speed)]
        return [(f, s) for f in (10.0, 30.0) for s in (10.0, 20.0)]

    def _job_count(self) -> int:
        names = self.selected_scenarios()
        if not names or not self.selected_engines():
            return 0
        specs = [
            spec_for(name, frames=int(self.frames_spin.value()), fps=fps, speed_mps=speed)
            for fps, speed in self.rates()
            for name in names
        ]
        return BenchPlan(
            engines=self.selected_engines(), specs=specs, sweeps=self.sweep_specs()
        ).job_count()

    def _plan(self) -> BenchPlan:
        """Snapshot the current selection as a `BenchPlan` (no map access)."""
        frames = int(self.frames_spin.value())
        specs = [
            spec_for(name, frames=frames, fps=fps, speed_mps=speed)
            for fps, speed in self.rates()
            for name in self.selected_scenarios()
        ]
        return BenchPlan(engines=self.selected_engines(), specs=specs, sweeps=self.sweep_specs())

    def _update_buttons(self, *_args: object) -> None:
        if not hasattr(self, "run_btn"):  # still assembling the UI
            return
        running = self.worker is not None
        self.run_btn.setEnabled(not running and self._job_count() > 0)
        self.stop_btn.setEnabled(running)
        self.export_btn.setEnabled(bool(self.results))

    def _say(self, text: str, color: str = TEXT_MUTED) -> None:
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color: {color}; font-size: 9pt;")

    # ------------------------------------------------------------------ run

    def make_worker(self, plan: BenchPlan) -> BenchWorker:
        worker = BenchWorker(plan, self.map_name, runner=self.runner, loader=self.loader)
        worker.progress.connect(self._on_progress)
        worker.result.connect(self._on_result)
        worker.failed.connect(lambda msg: self._say(msg, YELLOW))
        worker.runs_finished.connect(self._on_finished)
        return worker

    def start_run(self) -> None:
        if self.worker is not None:
            return
        plan = self._plan()
        if not plan.engines or not plan.specs:
            self._say("nothing to run: tick at least one engine and one scenario", YELLOW)
            return
        self.results = []
        self.records = {}
        self.table.setRowCount(0)
        self.progress.setRange(0, max(1, plan.job_count()))
        self.progress.setValue(0)
        if self.charts_ok:
            self._clear_latency()
            for series in self.error_series:
                self.error_chart.removeSeries(series)
            self.error_series = []

        self.worker = self.make_worker(plan)
        self._update_buttons()
        self._say("running %d case(s)..." % plan.job_count(), BLUE)
        self.worker.start()

    def stop_run(self) -> None:
        if self.worker is None:
            return
        self.worker.request_stop()
        self.stop_btn.setEnabled(False)
        self._say("stopping after the current case...", YELLOW)

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_result(self, result: object) -> None:
        if not isinstance(result, RunResult):
            return
        self.results.append(result)
        self._append_row(result)
        self._update_charts()
        self._update_buttons()

    def _on_finished(self) -> None:
        if self.worker is not None:
            self.records = dict(self.worker.records)
        self.worker = None
        self._update_buttons()
        self._say(
            "%d run(s) done" % len(self.results) if self.results else "no runs completed",
            GREEN if self.results else YELLOW,
        )

    # ---------------------------------------------------------------- table

    def _append_row(self, result: RunResult) -> None:
        table = self.table
        # Sorting must be off while a row is filled in: every `setItem` would
        # otherwise re-sort the view and the remaining cells land on another row.
        sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        try:
            row = table.rowCount()
            table.insertRow(row)
            for col, (_title, key) in enumerate(_COLUMNS):
                table.setItem(row, col, self._cell(result, key))
        finally:
            table.setSortingEnabled(sorting)
        table.resizeRowsToContents()

    @staticmethod
    def _cell(result: RunResult, key: str | None) -> QTableWidgetItem:
        if key is None:
            item = QTableWidgetItem(config_label(result.config) or "-")
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return item
        value = getattr(result, key)
        if isinstance(value, str):
            # `setData` rewrites the text for a numeric role, so a string cell
            # keeps its label exactly as given.
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return item
        # Numeric cells sort numerically: `setData(EditRole, ...)` is both what
        # the delegate paints and what the header sort compares.
        item = QTableWidgetItem()
        item.setData(
            Qt.ItemDataRole.EditRole, float(value) if key in _FLOAT_COLUMNS else int(value)
        )
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    def _clear_latency(self) -> None:
        for bar_set in self.latency_sets.values():
            bar_set.remove(0, bar_set.count())
            self.latency_series.remove(bar_set)
        self.latency_sets = {}
        while self.latency_axis_x.count():
            self.latency_axis_x.remove(self.latency_axis_x.at(0))

    def _update_charts(self) -> None:
        if not getattr(self, "charts_ok", False):
            return
        by_engine: dict[str, list[float]] = {}
        for res in self.results:
            by_engine.setdefault(res.engine, []).append(float(res.latency_ms_mean))
        self._clear_latency()
        means: list[float] = []
        for engine in ENGINES:
            if engine not in by_engine:
                continue
            mean = sum(by_engine[engine]) / len(by_engine[engine])
            means.append(mean)
            bar_set = QBarSet(engine)
            bar_set.setColor(QColor(BLUE))
            bar_set.append(mean)
            self.latency_series.append(bar_set)
            self.latency_sets[engine] = bar_set
            self.latency_axis_x.append(engine)
        self.latency_axis_y.setRange(0.0, (max(means) if means else 1.0) * 1.2 + 1.0)

        for series in self.error_series:
            self.error_chart.removeSeries(series)
        self.error_series = []
        xs: list[float] = []
        ys: list[float] = []
        for key in sorted({k for res in self.results for k in res.config}):
            points = sorted(
                (float(res.config[key]), float(res.pos_err_px_mean))
                for res in self.results
                if key in res.config
            )
            if len({p[0] for p in points}) < 2:
                continue
            series = QLineSeries()
            series.setName(key)
            for x, y in points:
                series.append(x, y)
            series.attachAxis(self.error_axis_x)
            series.attachAxis(self.error_axis_y)
            self.error_chart.addSeries(series)
            self.error_series.append(series)
            xs.extend(p[0] for p in points)
            ys.extend(p[1] for p in points)
        if self.error_series and xs:
            self.error_axis_x.setRange(min(xs), max(xs))
            self.error_axis_y.setRange(0.0, max(ys) * 1.2 + 1.0)
            self.error_chart.legend().setVisible(True)

    # -------------------------------------------------------------- export

    def export_results(self) -> None:
        """Ask for a directory and write the CSV/report/per-frame files there."""
        if not self.results:
            return
        start = os.path.join(PROJECT_ROOT, "output")
        chosen = QFileDialog.getExistingDirectory(self, "Export benchmark results", start)
        if not chosen:
            return
        try:
            self.write_export(chosen)
        except OSError as exc:
            self._say("export failed: %s" % exc, RED)
            return
        self._say("exported to %s" % os.path.relpath(chosen, PROJECT_ROOT), GREEN)

    def write_export(self, out_dir: str) -> str:
        """Write results.csv, report.md and per_frame/*.jsonl; returns the report."""
        os.makedirs(out_dir, exist_ok=True)
        write_csv(self.results, os.path.join(out_dir, "results.csv"))
        per_frame = os.path.join(out_dir, "per_frame")
        for result in self.results:
            name = "%s_%s_%s.jsonl" % (
                result.engine,
                result.scenario,
                config_hash(result.config),
            )
            write_jsonl(self.records.get(record_key(result), []), os.path.join(per_frame, name))
        return write_report(
            self.results,
            os.path.join(out_dir, "report.md"),
            {
                "map": self.map_name,
                "mode": "full" if self.full_radio.isChecked() else "quick",
                "engines": self.selected_engines(),
                "scenarios": self.selected_scenarios(),
                "frames": int(self.frames_spin.value()),
                "fps": ",".join("%g" % fps for fps, _ in self.rates()),
                "speed_mps": ",".join("%g" % speed for _, speed in self.rates()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    # -------------------------------------------------------------- closing

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        """Never destroy a QThread that is still inside `run_case`."""
        if self._probe is not None:
            self._probe.wait(_CLOSE_WAIT_MS)
            self._probe = None
        worker = self.worker
        if worker is not None:
            worker.request_stop()
            if not worker.wait(_CLOSE_WAIT_MS):
                self._say("a case is still running - press Stop, then close", YELLOW)
                event.ignore()
                return
        super().closeEvent(event)


def _fmt(value: Any) -> str:
    return "%g" % value if isinstance(value, (int, float)) else str(value)
