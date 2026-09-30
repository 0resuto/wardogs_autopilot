"""Logs section of the studio sidebar: every control that writes to `output/`.

Everything that produces a file under `output/` lives here, so the Capture and
Map sections stay about their own job:

* the flight-recorder switches (nav JSONL, manual-drive recording, fail dumps),
* the last localization reject with a copy button (what the fail dumps explain),
* the diagnostic snapshot writer (preview frames + `state_log.txt`),
* the localization benchmark window (`results.csv`, `report.md`, `per_frame/`),
* the inventory of what is currently in `output/` with one-click opening.

The log files themselves are opened with the system default application, so the
app never has to embed a text viewer.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import PROJECT_ROOT, crashlog
from ...common.config import AppConfig
from ...navigation.manual_record import ManualDriveRecorder
from ..debug_collage import save_debug_snapshot
from ..flow_layout import FlowLayout
from ..icons import icon
from ..theme import (
    BLUE,
    CONTROL_BG,
    GREEN,
    RED,
    TEXT,
    TEXT_DIM,
    TEXT_MUTED,
    YELLOW,
    mono_font_family,
)
from .common import compact_label

# Rows of the "Log files" card: (label, filename, tooltip).
_STATIC_LOGS: tuple[tuple[str, str, str], ...] = (
    ("App log", "autopilot.log", "Application log (rotating, 5 MB x 3)"),
    ("Crash log", "crash.log", "Fatal exceptions and native crashes"),
)

# Counting rows: (label, pattern, tooltip). `_` is the wildcard for the
# timestamped files written into output/.
_COUNTED_LOGS: tuple[tuple[str, str, str], ...] = (
    (
        "Nav runs",
        "nav_dbg_*.jsonl",
        "Flight recorder of the autopilot (replay: python tools/nav_dbg.py --tail)",
    ),
    (
        "Manual runs",
        "manual_dbg_*.jsonl",
        "Recorded manual driving (your W/A/S/D/SPACE presses and poses)",
    ),
    (
        "Fail dumps",
        "debug_fail_*",
        "Localization failures saved while 'Collect fail logs' is on",
    ),
    ("Snapshots", "snapshot_*", "Diagnostic snapshots written by 'Save frame'"),
)


def _human_size(size: int) -> str:
    """Short size label for the log inventory."""
    if size <= 0:
        return "empty"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def _count_matching(names: list[str], pattern: str) -> int:
    """Count entries whose name matches a simple `prefix*suffix` pattern."""
    prefix, _, suffix = pattern.partition("*")
    return sum(1 for name in names if name.startswith(prefix) and name.endswith(suffix))


def _last_matching(names: list[str], pattern: str) -> str:
    """Newest entry matching a simple `prefix*suffix` pattern (`''` if none)."""
    prefix, _, suffix = pattern.partition("*")
    hits = [n for n in names if n.startswith(prefix) and n.endswith(suffix)]
    return max(hits) if hits else ""


def _default_bench_window(map_name: str) -> Any:
    """Build the real benchmark window for the studio's active map."""
    from ..bench_window import BenchWindow

    return BenchWindow(map_name=map_name)


class LogsTab(QWidget):
    """Sidebar section owning every `output/` writer and the log inventory."""

    # Thread-safe Qt signals (the snapshot writer runs in a worker thread)
    sig_save_done = Signal(str)
    sig_save_failed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None,
        cfg: dict[str, Any] | AppConfig,
        save_cfg_fn: Callable[[], None],
        loc_thread_supplier: Callable[[], Any],
        map_name_supplier: Callable[[], str] | None = None,
        route_supplier: Callable[[], list[list[float]]] | None = None,
        app_cfg: AppConfig | None = None,
        bench_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(cfg, AppConfig):
            self.app_cfg = cfg
            self.cfg = cfg.to_dict()
        else:
            self.cfg = cfg
            self.app_cfg = app_cfg or AppConfig()

        self.save_cfg = save_cfg_fn
        self.get_loc = loc_thread_supplier
        self.get_map_name = map_name_supplier or (
            lambda: str(self.cfg.get("map", {}).get("name", "zestafona"))
        )
        self.get_route = route_supplier or (lambda: [])
        self._bench_factory = bench_factory or _default_bench_window

        self.output_dir = os.path.join(PROJECT_ROOT, "output")
        self._manual_rec: ManualDriveRecorder | None = None
        self._snap_busy = False
        self._last_snapshot_dir: str | None = None
        self._bench_window: QWidget | None = None

        self.sig_save_done.connect(self._on_save_done)
        self.sig_save_failed.connect(self._on_save_failed)

        self._build_ui()

        # The inventory is only rescanned while the section is open, so the
        # 20 Hz poll never touches the filesystem.
        self._scan_timer = QTimer(self)
        self._scan_timer.setInterval(2000)
        self._scan_timer.timeout.connect(self._on_scan_timer)
        self._scan_timer.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._build_logging_card(layout)
        self._build_snapshot_card(layout)
        self._build_bench_card(layout)
        self._build_files_card(layout)
        layout.addStretch()  # keep the sidebar content top-aligned

    def _card(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        card = QGroupBox(title, self)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 14, 12, 12)
        card_layout.setSpacing(8)
        return card, card_layout

    def _hint(self, text: str, parent: QWidget) -> QLabel:
        lbl = QLabel(text, parent)
        lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        lbl.setWordWrap(True)
        compact_label(lbl)
        return lbl

    def _build_logging_card(self, layout: QVBoxLayout) -> None:
        card, card_layout = self._card("Logging")
        card_layout.addWidget(
            self._hint(
                "Everything recorded while driving lands in output/. "
                "Replay a nav run with: python tools/nav_dbg.py --tail",
                card,
            )
        )

        self.nav_ck = QCheckBox("Nav log", card)
        self.nav_ck.setToolTip(
            "Log every autopilot tick to output/nav_dbg_<time>.jsonl\n(keeps the 10 newest runs)"
        )
        self.nav_ck.setChecked(bool(self.app_cfg.navigator.debug))
        self.nav_ck.toggled.connect(self.apply_nav_log)
        card_layout.addWidget(self.nav_ck)

        self.manual_ck = QCheckBox("Record my driving", card)
        self.manual_ck.setToolTip(
            "Log your own W/A/S/D/SPACE presses and poses to "
            "output/manual_dbg_<time>.jsonl while driving by hand"
        )
        self.manual_ck.toggled.connect(self.toggle_manual_record)
        card_layout.addWidget(self.manual_ck)

        self.collect_ck = QCheckBox("Collect fail logs", card)
        self.collect_ck.setToolTip(
            "Autosave every rejected localization frame (image + context) to output/debug_fail_*"
        )
        self.collect_ck.setChecked(
            bool(self.cfg.setdefault("debug", {}).get("collect_fail_logs", False))
        )
        self.collect_ck.toggled.connect(self.apply_collect_logs)
        card_layout.addWidget(self.collect_ck)

        # What the fail dumps above explain: the newest reject reason.
        self.reject_title = QLabel("Last reject", card)
        self.reject_title.setStyleSheet(f"color: {BLUE}; font-weight: bold;")
        card_layout.addWidget(self.reject_title)

        reject_row = QHBoxLayout()
        reject_row.setSpacing(6)
        self.dbg_text = QLabel("no rejects yet", card)
        self.dbg_text.setStyleSheet(
            f"background-color: {CONTROL_BG}; color: {YELLOW};"
            f" font-family: '{mono_font_family()}'; font-size: 8pt;"
            " padding: 4px 6px; border-radius: 4px;"
        )
        self.dbg_text.setWordWrap(True)
        compact_label(self.dbg_text)
        reject_row.addWidget(self.dbg_text, stretch=1)

        self.copy_btn = QPushButton("Copy", card)
        self.copy_btn.setToolTip("Copy the last reject reason to the clipboard")
        self.copy_btn.clicked.connect(self.copy_debug)
        reject_row.addWidget(self.copy_btn)
        card_layout.addLayout(reject_row)

        layout.addWidget(card)

    def _build_snapshot_card(self, layout: QVBoxLayout) -> None:
        card, card_layout = self._card("Diagnostic Snapshot")
        card_layout.addWidget(
            self._hint(
                "Saves 3 preview frames, the map crop, and state_log.txt into "
                "a timestamped output/snapshot_* folder (keeps the 20 newest).",
                card,
            )
        )

        btn_row = FlowLayout(h_spacing=8, v_spacing=6)
        self.save_snap_btn = QPushButton("Save frame", card)
        self.save_snap_btn.setIcon(icon("camera"))
        self.save_snap_btn.setToolTip(
            "Save the current capture, map crop, and state log to output/"
        )
        self.save_snap_btn.clicked.connect(self.save_debug_frame)
        btn_row.addWidget(self.save_snap_btn)

        self.open_snap_btn = QPushButton("Open snapshot", card)
        self.open_snap_btn.setIcon(icon("folder"))
        self.open_snap_btn.setToolTip("Open the last saved diagnostic snapshot folder")
        self.open_snap_btn.setVisible(False)
        self.open_snap_btn.clicked.connect(self._open_last_snapshot)
        btn_row.addWidget(self.open_snap_btn)
        card_layout.addLayout(btn_row)

        self.save_status_lbl = QLabel("", card)
        self.save_status_lbl.setStyleSheet(f"color: {GREEN}; font-size: 8pt; font-weight: 500;")
        self.save_status_lbl.setWordWrap(True)
        compact_label(self.save_status_lbl)
        card_layout.addWidget(self.save_status_lbl)

        layout.addWidget(card)

    def _build_bench_card(self, layout: QVBoxLayout) -> None:
        card, card_layout = self._card("Localization Benchmark")
        card_layout.addWidget(
            self._hint(
                "Score sift/orb/xfeat/hybrid on synthetic trajectories with a known "
                "ground truth, sweep one parameter at a time, then export "
                "results.csv + report.md into output/.",
                card,
            )
        )

        btn_row = FlowLayout(h_spacing=8, v_spacing=6)
        self.bench_btn = QPushButton("Open benchmark", card)
        self.bench_btn.setIcon(icon("play"))
        self.bench_btn.setToolTip(
            "Open the benchmark window (headless: python tools/bench_synthetic.py --quick)"
        )
        self.bench_btn.clicked.connect(self.open_bench_window)
        btn_row.addWidget(self.bench_btn)

        self.open_results_btn = QPushButton("Open output", card)
        self.open_results_btn.setIcon(icon("folder"))
        self.open_results_btn.setToolTip("Open output/ - the bench writes results.csv here")
        self.open_results_btn.clicked.connect(self._open_output_dir)
        btn_row.addWidget(self.open_results_btn)
        card_layout.addLayout(btn_row)

        layout.addWidget(card)

    def open_bench_window(self) -> Any:
        """Show (or raise) the benchmark window; created on first use.

        The import is local because the window pulls in QtCharts and the whole
        bench package, which the studio must not pay for at startup.
        """
        window = self._bench_window
        if window is None:
            window = self._bench_factory(self.get_map_name())
            window.destroyed.connect(self._on_bench_destroyed)
            self._bench_window = window
        window.show()
        window.raise_()
        window.activateWindow()
        return window

    def _on_bench_destroyed(self, *_args: object) -> None:
        self._bench_window = None

    def _build_files_card(self, layout: QVBoxLayout) -> None:
        card, card_layout = self._card("Log Files")
        card_layout.addWidget(self._hint("output/ — click a row to open it.", card))

        self._file_rows: list[QLabel] = []
        self._file_buttons: list[QPushButton] = []
        self._file_paths: list[str] = []

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        row = 0
        for label, filename, tip in _STATIC_LOGS:
            grid.addWidget(self._row_label(label, tip, card), row, 0)
            value = self._row_value()
            grid.addWidget(value, row, 1)
            btn = self._row_button(f"Open {filename}", card)
            grid.addWidget(btn, row, 2)
            self._file_rows.append(value)
            self._file_buttons.append(btn)
            self._file_paths.append(os.path.join(self.output_dir, filename))
            row += 1

        for label, _pattern, tip in _COUNTED_LOGS:
            grid.addWidget(self._row_label(label, tip, card), row, 0)
            value = self._row_value()
            grid.addWidget(value, row, 1)
            self._file_rows.append(value)
            row += 1
        card_layout.addLayout(grid)

        bottom = FlowLayout(h_spacing=8, v_spacing=6)
        open_out = QPushButton("Open output", card)
        open_out.setIcon(icon("folder"))
        open_out.setToolTip("Open the output folder in the file explorer")
        open_out.clicked.connect(self._open_output_dir)
        bottom.addWidget(open_out)

        self.refresh_btn = QPushButton("", card)
        self.refresh_btn.setIcon(icon("refresh"))
        self.refresh_btn.setFixedWidth(34)
        self.refresh_btn.setToolTip("Rescan output/")
        self.refresh_btn.clicked.connect(self.refresh_files)
        bottom.addWidget(self.refresh_btn)
        card_layout.addLayout(bottom)

        self.files_status = QLabel("", card)
        self.files_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8pt;")
        self.files_status.setWordWrap(True)
        compact_label(self.files_status)
        card_layout.addWidget(self.files_status)

        layout.addWidget(card)
        self.refresh_files()

    def _row_label(self, text: str, tip: str, parent: QWidget) -> QLabel:
        lbl = QLabel(text, parent)
        lbl.setStyleSheet(f"color: {TEXT};")
        lbl.setToolTip(tip)
        compact_label(lbl)
        return lbl

    def _row_value(self) -> QLabel:
        lbl = QLabel("—", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-family: '{mono_font_family()}'; font-size: 8pt;"
        )
        lbl.setToolTip("")
        compact_label(lbl)
        return lbl

    def _row_button(self, tip: str, parent: QWidget) -> QPushButton:
        btn = QPushButton("", parent)
        btn.setIcon(icon("folder"))
        btn.setFixedWidth(30)
        btn.setToolTip(tip)
        btn.clicked.connect(self._open_row)
        policy = QSizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        btn.setSizePolicy(policy)
        return btn

    # ---------------------------------------------------------- log switches

    def nav_log_enabled(self) -> bool:
        """True when the autopilot flight recorder must write nav JSONL."""
        return bool(self.nav_ck.isChecked())

    def apply_nav_log(self) -> None:
        """Persist the nav-log switch (it is what FollowDriver is built with)."""
        enabled = self.nav_ck.isChecked()
        self.app_cfg.navigator.debug = enabled
        self.cfg.setdefault("navigator", {})["debug"] = enabled
        self.save_cfg()

    def apply_collect_logs(self) -> None:
        enabled = self.collect_ck.isChecked()
        self.cfg.setdefault("debug", {})["collect_fail_logs"] = enabled
        self.app_cfg.debug.collect_fail_logs = enabled
        self.save_cfg()
        loc = self.get_loc()
        if loc is not None and hasattr(loc, "set_collect_fail_logs"):
            loc.set_collect_fail_logs(enabled)
        self.refresh_files()

    def toggle_manual_record(self, enabled: bool) -> None:
        """Start/stop recording the user's own driving (temporary tuning aid)."""
        if not enabled:
            self.stop_manual_record()
            return
        loc = self.get_loc()
        if loc is None:
            self.manual_ck.setChecked(False)
            return
        params = dict(
            source="manual",
            map=self.get_map_name(),
            vehicle=self.app_cfg.navigator.vehicle_profile,
            speed_cap_kmh=self.app_cfg.navigator.speed_cap_kmh,
        )
        self._manual_rec = ManualDriveRecorder(
            loc=loc,
            route=self.get_route(),
            params=params,
            out_dir="output",
        )
        self._manual_rec.start()

    def manual_record_active(self) -> bool:
        rec = self._manual_rec
        return rec is not None and bool(getattr(rec, "running", True))

    def stop_manual_record(self) -> None:
        """Stop manual-driving recording if it is running."""
        rec = self._manual_rec
        if rec is not None:
            rec.stop()
            self._manual_rec = None
        if self.manual_ck.isChecked():
            with QSignalBlocker(self.manual_ck):
                self.manual_ck.setChecked(False)
        self.refresh_files()

    # ------------------------------------------------------------ diagnostics

    def set_last_reject(self, text: str) -> None:
        """Show the newest localization reject reason (from the Map section)."""
        self.dbg_text.setText(text or "no rejects yet")
        self.dbg_text.setToolTip(text)

    def copy_debug(self) -> None:
        QApplication.clipboard().setText(self.dbg_text.toolTip() or self.dbg_text.text())

    def save_debug_frame(self) -> None:
        """Capture a live diagnostic snapshot asynchronously."""
        if self._snap_busy:
            return
        loc = self.get_loc()
        if loc is None:
            self._set_save_status("Locator not active", YELLOW)
            return
        self._snap_busy = True
        self.save_snap_btn.setEnabled(False)
        self.save_snap_btn.setText("Saving...")
        self._set_save_status("Saving snapshot…", YELLOW)

        def worker() -> None:
            try:
                mm_gray, mm_bgr, mask, latest = loc.snapshot_debug()
                if mm_gray is None:
                    self.sig_save_failed.emit("No frame captured yet")
                    return
                pose = latest.get("pose") if isinstance(latest, dict) else None
                raw_diag = latest.get("diag") if isinstance(latest, dict) else None
                diag = dict(raw_diag) if isinstance(raw_diag, dict) else {}
                roi = self.cfg.get("capture", {}).get("mmap_roi")
                _, snap_dir, _ = save_debug_snapshot(
                    self.output_dir,
                    mm_gray,
                    mm_bgr,
                    mask,
                    pose,
                    diag,
                    latest,
                    map_name=self.get_map_name(),
                    roi=roi,
                )
                self.sig_save_done.emit(snap_dir)
            except Exception as exc:
                crashlog.log("save debug snapshot", exc)
                self.sig_save_failed.emit(str(exc))
            finally:
                self._snap_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def _on_save_done(self, snap_dir: str) -> None:
        self._last_snapshot_dir = snap_dir
        self._set_save_status(f"✓ Saved to {os.path.basename(snap_dir)}", GREEN)
        self.open_snap_btn.setVisible(True)
        self.save_snap_btn.setEnabled(True)
        self.save_snap_btn.setText("Save frame")
        self.refresh_files()

    def _on_save_failed(self, err_msg: str) -> None:
        self._set_save_status(f"Save failed: {err_msg}", RED)
        self.save_snap_btn.setEnabled(True)
        self.save_snap_btn.setText("Save frame")

    def _set_save_status(self, text: str, color: str) -> None:
        self.save_status_lbl.setText(text)
        self.save_status_lbl.setStyleSheet(f"color: {color}; font-size: 8pt; font-weight: 500;")

    def _open_last_snapshot(self) -> None:
        if self._last_snapshot_dir and os.path.isdir(self._last_snapshot_dir):
            self._open_path(self._last_snapshot_dir)

    # --------------------------------------------------------- log inventory

    def output_names(self) -> list[str]:
        try:
            return os.listdir(self.output_dir)
        except OSError:
            return []

    def _on_scan_timer(self) -> None:
        """Rescan only while the Logs section is the visible sidebar page."""
        if self.isVisible():
            self.refresh_files()

    def refresh_files(self) -> None:
        """Rescan output/ and update the inventory rows (cheap, 2 s cadence)."""
        names = self.output_names()
        index = 0
        for _label, filename, _tip in _STATIC_LOGS:
            path = os.path.join(self.output_dir, filename)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            present = os.path.isfile(path)
            self._file_rows[index].setText(_human_size(size) if present else "missing")
            self._file_rows[index].setStyleSheet(
                f"color: {TEXT_DIM if present else RED};"
                f" font-family: '{mono_font_family()}'; font-size: 8pt;"
            )
            self._file_rows[index].setToolTip(path)
            self._file_buttons[index].setEnabled(present)
            self._file_buttons[index].setProperty("path", path)
            index += 1

        for _label, pattern, _tip in _COUNTED_LOGS:
            count = _count_matching(names, pattern)
            newest = _last_matching(names, pattern)
            row = self._file_rows[index]
            row.setText(str(count) if count else "0")
            row.setStyleSheet(
                f"color: {TEXT if count else TEXT_DIM};"
                f" font-family: '{mono_font_family()}'; font-size: 8pt;"
            )
            row.setToolTip(
                f"{count} × {pattern}\nnewest: {newest}" if count else f"no {pattern} yet"
            )
            index += 1

        self._update_footer_status()

    def _update_footer_status(self) -> None:
        notes: list[str] = []
        rec = self._manual_rec
        if rec is not None:
            notes.append("● recording your driving")
        if self.nav_ck.isChecked():
            notes.append("● nav log armed")
        if self.collect_ck.isChecked():
            notes.append("● fail logs on")
        self.files_status.setText("  |  ".join(notes))
        self.files_status.setStyleSheet(f"color: {GREEN if notes else TEXT_MUTED}; font-size: 8pt;")

    def _open_row(self) -> None:
        btn = self.sender()
        path = btn.property("path") if isinstance(btn, QPushButton) else None
        if path:
            self._open_path(str(path))

    def _open_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._open_path(self.output_dir)

    def _open_path(self, path: str) -> None:
        """Open a log file/folder with the system default application."""
        if not os.path.exists(path):
            self._set_save_status(f"Not found: {os.path.basename(path)}", YELLOW)
            return
        if QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            return
        try:
            os.startfile(path)  # noqa: S606 - Windows shell open of our own output
        except Exception as exc:
            crashlog.log(f"open {path}", exc)
            self._set_save_status(f"Failed to open {os.path.basename(path)}: {exc}", RED)
