"""Route preset management for the Map tab (PySide6)."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox

from ..presets import PresetManager
from .common import MapTabBase


class MapPresetsMixin(MapTabBase):
    """Preset list management and route <-> preset persistence."""

    def preset_reload(self) -> None:
        """Reload list of presets for the active map.

        The preset used in the last session is selected again, and its route is
        loaded when no route is set (fresh start), so the studio opens ready to
        drive.
        """
        map_name = self.get_map_name()
        self.preset_mgr = PresetManager(subdir=f"data/presets/{map_name}")
        names = self.preset_mgr.list_presets()
        self.p_sel.clear()
        self.p_sel.addItems(names)
        if names:
            last = self.app_cfg.navigator.last_preset
            if last in names:
                self.p_sel.setCurrentText(last)
            else:
                self.p_sel.setCurrentIndex(0)
            if not self.route_pts:
                self.preset_load_sel(persist=False)

    def _ask_preset_name(self, title: str, ok_text: str, initial: str = "") -> str | None:
        """Ask for a preset name (field + OK/Cancel), or None when cancelled."""
        dialog = QInputDialog(self)
        dialog.setWindowTitle(title)
        dialog.setLabelText("Preset name:")
        dialog.setOkButtonText(ok_text)
        dialog.setCancelButtonText("Cancel")
        dialog.setTextValue(initial)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        name = dialog.textValue().strip()
        if not name:
            QMessageBox.warning(self, title, "Enter a preset name")
            return None
        return name

    def _save_route_to_preset(self, name: str) -> bool:
        """Write the current route points to the preset and select it."""
        try:
            self.preset_mgr.save_preset(name, self.route_pts)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f"Failed to save preset: {exc}")
            return False
        self.app_cfg.navigator.last_preset = name
        self.save_cfg()
        self.preset_reload()
        self.p_sel.setCurrentText(name)
        return True

    def preset_save(self) -> None:
        """Save the current route under a new name (Save As)."""
        current = self.p_sel.currentText().strip()
        name = self._ask_preset_name("Save route as", "Save", current)
        if name is None:
            return
        if name != current and os.path.exists(self.preset_mgr.preset_path(name)):
            res = QMessageBox.question(
                self, "Presets", f'Preset "{name}" already exists. Overwrite?'
            )
            if res != QMessageBox.StandardButton.Yes:
                return
        self._save_route_to_preset(name)

    def preset_load_sel(self, persist: bool = True) -> None:
        """Load the selected preset; `persist` remembers it for the next start."""
        name = self.p_sel.currentText().strip()
        if not name:
            return
        try:
            pts = self.preset_mgr.load_preset(name)
        except Exception as exc:
            QMessageBox.critical(self, "Presets", f'Failed to read preset "{name}": {exc}')
            return
        self.route_pts = pts
        self.app_cfg.navigator.last_preset = name
        if persist:
            self.save_cfg()
        self.routes_refresh()

    def preset_delete(self) -> None:
        name = self.p_sel.currentText().strip()
        if not name:
            return
        res = QMessageBox.question(self, "Presets", f'Delete preset "{name}"?')
        if res != QMessageBox.StandardButton.Yes:
            return
        self.preset_mgr.delete_preset(name)
        self.preset_reload()

    def route_new(self) -> None:
        """Create a new empty route preset after asking for a name."""
        name = self._ask_preset_name("New route", "Create")
        if name is None:
            return
        if os.path.exists(self.preset_mgr.preset_path(name)):
            QMessageBox.information(
                self,
                "New route",
                f'Preset "{name}" already exists. Load it or use "Save As" to overwrite.',
            )
            return
        previous = self.route_pts
        self.route_pts = []
        if not self._save_route_to_preset(name):
            self.route_pts = previous
            return
        self._set_edit_mode(True)
