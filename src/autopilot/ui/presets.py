"""Preset manager for waypoint routes.

Handles loading, saving, listing, and sanitizing route preset files on disk.
Guards against directory traversal attacks.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .. import PROJECT_ROOT


class PresetManager:
    """Manages waypoint route presets in a target directory."""

    def __init__(self, base_dir: str | Path | None = None, subdir: str = "data/presets") -> None:
        self.base_dir = Path(base_dir or PROJECT_ROOT)
        self.subdir = subdir

    @property
    def presets_dir(self) -> Path:
        """Directory path where presets are stored."""
        p = self.base_dir / self.subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list_presets(self) -> list[str]:
        """Return sorted list of available preset names (without .json extension)."""
        d = self.presets_dir
        if not d.exists():
            return []
        return sorted(f.stem for f in d.glob("*.json"))

    def preset_path(self, name: str) -> str:
        """Sanitize name and return absolute/normalized path to the preset file."""
        safe_name = os.path.basename(name.strip().replace("\\", "/"))
        return os.path.normpath(str(self.presets_dir / f"{safe_name}.json"))

    def load_preset(self, name: str) -> list[list[float]]:
        """Load and return waypoint points list from preset file."""
        if not name:
            return []
        path = self.preset_path(name)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [list(p) for p in data.get("points", [])]

    def save_preset(self, name: str, points: list[list[float]]) -> str:
        """Save waypoints to preset file with timestamp and return target path."""
        path = self.preset_path(name)
        data = {
            "name": name,
            "points": points,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return path

    def delete_preset(self, name: str) -> bool:
        """Delete preset file if it exists."""
        if not name:
            return False
        path = self.preset_path(name)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
