"""Map cache card for the Capture tab: status, asset download, and rebuild."""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ... import PROJECT_ROOT, crashlog
from ...vision import asset_sync, locator
from ..theme import BLUE, GREEN, RED, TEXT_DIM, YELLOW
from .common import RoiTabBase


def map_cache_status(name: str) -> tuple[str, str]:
    """Human-readable cache status of `name` and its display color.

    The distributed map is a derived artifact set (feat.npz, mu.npy, gray.txt
    and the top preview level; smaller pyramid levels are generated locally on
    first open). The source PNG only exists on a maintainer machine, which is
    also the only place a rebuild is possible.
    """
    if not name:
        return "No map selected", f"{TEXT_DIM}"
    data_dir = os.path.join(PROJECT_ROOT, "data", "maps")
    mu_path = os.path.join(data_dir, f"{name}_mu.npy")
    feat_path = os.path.join(data_dir, f"{name}_feat.npz")
    gray_path = os.path.join(data_dir, f"{name}_gray.txt")
    has_png = os.path.exists(os.path.join(data_dir, f"{name}_map.png"))

    missing = [
        label
        for label, path in (("feat.npz", feat_path), ("mu.npy", mu_path), ("gray.txt", gray_path))
        if not os.path.exists(path)
    ]
    artifacts = locator.get_store().catalog_artifacts(name)
    if artifacts:
        absent = [f for f in artifacts if not os.path.exists(os.path.join(data_dir, f))]
        missing_previews = [f for f in absent if "_preview_" in f]
    else:
        absent = []
        missing_previews = [
            f"{name}_preview_{sz}.npy"
            for sz in locator.PREVIEW_SIZES
            if not os.path.exists(os.path.join(data_dir, f"{name}_preview_{sz}.npy"))
        ]

    all_artifacts_gone = bool(artifacts) and len(absent) == len(artifacts)
    no_artifacts_at_all = not artifacts and len(missing_previews) == len(locator.PREVIEW_SIZES)
    if len(missing) == 3 and not has_png and (all_artifacts_gone or no_artifacts_at_all):
        return f"Not downloaded: run python tools/download_map.py {name}", f"{RED}"
    if missing:
        hint = "re-download the map assets"
        if has_png:
            hint = "press Rebuild"
        return f"Cache incomplete: missing {', '.join(missing)} ({hint})", f"{RED}"
    if missing_previews:
        hint = "re-download the map assets"
        if has_png:
            hint = "press Rebuild"
        return f"Cache incomplete: missing {', '.join(missing_previews)} ({hint})", f"{YELLOW}"

    try:
        with np.load(feat_path) as idx:
            sig = str(idx.get("gray_sig", [""])[0])
            n_tiles = int(idx.get("gw", 0)) * int(idx.get("gh", 0))
            if sig != locator.get_store().gray_sig(name):
                hint = "re-download the map assets" if not has_png else "press Rebuild"
                return f"Cache stale: palette or map version changed — {hint}", f"{YELLOW}"
            palette = sig.split("|")[0]
            return (
                f"Ready: mu OK, SIFT index OK ({n_tiles} tiles, {palette}), mipmaps OK",
                f"{GREEN}",
            )
    except Exception:
        return "Ready: mu OK, SIFT index OK, mipmaps OK", f"{GREEN}"


class RoiCacheMixin(RoiTabBase):
    """Cache status card, release-archive download, and maintainer rebuild."""

    def _build_cache_card(self, layout: QVBoxLayout) -> None:
        # Map Cache & SIFT Feature Index Management
        card_cache = QGroupBox("Map Cache & SIFT Feature Index", self)
        cache_layout = QVBoxLayout(card_cache)
        cache_layout.setContentsMargins(12, 14, 12, 12)
        cache_layout.setSpacing(6)

        row_map = QHBoxLayout()
        row_map.setSpacing(8)
        row_map.addWidget(QLabel("Active Map:", card_cache))

        all_maps = locator.get_store().available_maps() or ["zestafona", "bakurani", "ozeti"]
        self._cache_map_sel = QComboBox(card_cache)
        self._cache_map_sel.addItems(all_maps)
        cur_map = self.cfg.get("map", {}).get("name", "zestafona")
        if cur_map in all_maps:
            self._cache_map_sel.setCurrentText(cur_map)
        self._cache_map_sel.currentTextChanged.connect(lambda: self.cache_status_refresh())
        row_map.addWidget(self._cache_map_sel)

        self._cache_download_btn = QPushButton("Download", card_cache)
        self._cache_download_btn.setToolTip(
            "Download the map assets (archive + sha256 verification) from the release"
        )
        self._cache_download_btn.clicked.connect(self._cache_download_click)
        row_map.addWidget(self._cache_download_btn)

        self._cache_rebuild_btn = QPushButton("Rebuild Cache & SIFT Index", card_cache)
        self._cache_rebuild_btn.setToolTip(
            "Rebuild mu/previews/SIFT index from the source PNG (maintainer machines only)"
        )
        self._cache_rebuild_btn.clicked.connect(self._cache_rebuild_click)
        row_map.addWidget(self._cache_rebuild_btn)
        row_map.addStretch()
        cache_layout.addLayout(row_map)

        self._cache_status_lbl = QLabel("", card_cache)
        self._cache_status_lbl.setStyleSheet(f"color: {BLUE};")
        cache_layout.addWidget(self._cache_status_lbl)
        self.cache_status_refresh()
        layout.addWidget(card_cache)

    def cache_status_refresh(self) -> None:
        name = self._cache_map_sel.currentText().strip()
        txt, color = self._get_map_cache_status(name)
        self._cache_status_lbl.setText(txt)
        self._cache_status_lbl.setStyleSheet(f"color: {color};")
        busy = self._cache_rebuild_busy or self._cache_download_busy
        ready = color == f"{GREEN}"
        self._cache_download_btn.setEnabled(bool(name) and not busy and not ready)
        has_png = bool(name) and os.path.exists(
            os.path.join(PROJECT_ROOT, "data", "maps", f"{name}_map.png")
        )
        self._cache_rebuild_btn.setEnabled(not busy and has_png)

    def _get_map_cache_status(self, name: str) -> tuple[str, str]:
        return map_cache_status(name)

    def _cache_rebuild_click(self) -> None:
        name = self._cache_map_sel.currentText().strip()
        if not name or self._cache_rebuild_busy or self._cache_download_busy:
            return
        png_path = os.path.join(PROJECT_ROOT, "data", "maps", f"{name}_map.png")
        if not os.path.exists(png_path):
            QMessageBox.critical(
                self,
                "Map Missing",
                f"Map file '{name}_map.png' not found.\n\n"
                "Rebuilding requires the source PNG (maintainer machines).\n"
                "Use the Download button to fetch the prebuilt assets instead.",
            )
            return

        res = QMessageBox.question(
            self,
            "Map Cache",
            f'Rebuild map cache and SIFT feature index for "{name}"?\n\n'
            "This will regenerate mu.npy, preview mipmaps, and the SIFT descriptor index (~1-2 min).",
        )
        if res != QMessageBox.StandardButton.Yes:
            return

        self._cache_rebuild_busy = True
        self._cache_rebuild_btn.setEnabled(False)
        self._cache_status_lbl.setText("Starting rebuild...")
        self._cache_status_lbl.setStyleSheet(f"color: {YELLOW};")
        threading.Thread(target=self._cache_rebuild_worker, args=(name,), daemon=True).start()

    def _cache_rebuild_worker(self, name: str) -> None:
        try:
            locator.rebuild_map_cache(
                name, progress_cb=lambda msg: self.sig_cache_progress.emit(msg)
            )
            self.sig_cache_done.emit(name)
        except Exception as exc:
            crashlog.log(f"rebuild map cache {name}", exc)
            self.sig_cache_failed.emit(str(exc))

    def _cache_download_click(self) -> None:
        name = self._cache_map_sel.currentText().strip()
        if not name or self._cache_download_busy or self._cache_rebuild_busy:
            return
        catalog = asset_sync.load_catalog(locator.get_store().data_maps_dir)
        archive = (catalog.get("maps", {}).get(name, {}) or {}).get("archive") or {}
        size = int(archive.get("size") or 0)
        size_txt = f"\n\nArchive size: {asset_sync.format_bytes(size)}" if size else ""
        res = QMessageBox.question(
            self,
            "Download Map",
            f'Download map assets for "{name}"?{size_txt}\n\n'
            "The archive is verified against the catalog and extracted into data/maps.",
        )
        if res != QMessageBox.StandardButton.Yes:
            return
        self._cache_download_busy = True
        self.cache_status_refresh()  # disables both buttons while busy
        self._cache_status_lbl.setText("Starting download...")
        self._cache_status_lbl.setStyleSheet(f"color: {YELLOW};")
        threading.Thread(
            target=self._cache_download_worker, args=(name, catalog), daemon=True
        ).start()

    def _cache_download_worker(self, name: str, catalog: dict[str, Any]) -> None:
        try:
            store = locator.get_store()
            repo, tag = asset_sync.resolve_repo_tag(catalog)
            ok = asset_sync.download_map(
                name,
                catalog,
                repo,
                tag,
                store.data_maps_dir,
                progress=lambda msg: self.sig_cache_progress.emit(msg),
            )
            if not ok:
                raise RuntimeError("download failed (see output/autopilot.log)")
            self.sig_download_done.emit(name)
        except Exception as exc:
            crashlog.log(f"download map assets {name}", exc)
            self.sig_download_failed.emit(str(exc))

    def _on_cache_progress(self, msg: str) -> None:
        self._cache_status_lbl.setText(msg)

    def _on_cache_done(self, name: str) -> None:
        self._cache_rebuild_busy = False
        self.cache_status_refresh()
        if self.on_map_rebuilt is not None:
            self.on_map_rebuilt(name)
        QMessageBox.information(
            self, "Map Cache", f'Map cache and SIFT index successfully rebuilt for "{name}"!'
        )

    def _on_cache_failed(self, err_msg: str) -> None:
        self._cache_rebuild_busy = False
        self.cache_status_refresh()
        self._cache_status_lbl.setText(f"Rebuild error: {err_msg}")
        self._cache_status_lbl.setStyleSheet(f"color: {RED};")
        QMessageBox.critical(self, "Map Cache", f"Rebuild failed: {err_msg}")

    def _on_download_done(self, name: str) -> None:
        self._cache_download_busy = False
        self.cache_status_refresh()
        if self.on_map_rebuilt is not None:
            self.on_map_rebuilt(name)
        QMessageBox.information(self, "Map Download", f'Map assets downloaded for "{name}"!')

    def _on_download_failed(self, err_msg: str) -> None:
        self._cache_download_busy = False
        self.cache_status_refresh()
        self._cache_status_lbl.setText(f"Download error: {err_msg}")
        self._cache_status_lbl.setStyleSheet(f"color: {RED};")
        QMessageBox.critical(self, "Map Download", f"Download failed: {err_msg}")
