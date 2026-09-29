"""Shared helpers and the mixin interface for the studio tabs (PySide6)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import QLabel, QLineEdit, QSizePolicy, QWidget

from ...common.config import AppConfig
from ..presets import PresetManager

if TYPE_CHECKING:
    _BaseQt = QWidget
else:
    # Runtime base stays a plain object: deriving every mixin from QWidget
    # would put several Qt bases into MapTab's MRO (instance layout conflict).
    _BaseQt = object


class StringVarCompat:
    """String holder mirroring the entry widgets for tests."""

    def __init__(self, value: str = "") -> None:
        self._val = str(value)

    def get(self) -> str:
        return self._val

    def set(self, val: str) -> None:
        self._val = str(val)


def compact_label(label: QLabel) -> None:
    """Let dynamic status text clip instead of forcing the window to grow."""
    label.setMinimumWidth(0)
    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    label.setSizePolicy(policy)


class RoiTabBase(_BaseQt):
    """Interface shared by the Capture tab mixins (annotations and stubs only)."""

    cfg: dict[str, Any]
    save_cfg: Callable[[], None]
    get_cap: Callable[[], Any]
    get_loc: Callable[[], Any]
    on_map_rebuilt: Callable[[str], None] | None
    roi_vars: dict[str, StringVarCompat]
    speed_vars: dict[str, StringVarCompat]
    coord_inputs: dict[str, QLineEdit]
    speed_inputs: dict[str, QLineEdit]
    status_lbl: Any
    speed_status_lbl: Any
    preview_lbl: Any
    _cache_map_sel: Any
    _cache_status_lbl: Any
    _cache_download_btn: Any
    _cache_rebuild_btn: Any
    _cache_rebuild_busy: bool
    _cache_download_busy: bool
    capture_title_lbl: Any
    speed_title_lbl: Any
    speed_preview_lbl: Any
    save_status_lbl: Any
    save_snap_btn: Any
    open_snap_btn: Any
    _snap_busy: bool
    _last_snapshot_dir: str | None
    sig_cache_progress: Any
    sig_cache_done: Any
    sig_cache_failed: Any
    sig_download_done: Any
    sig_download_failed: Any
    sig_save_done: Any
    sig_save_failed: Any

    def update_preview(self) -> None: ...


class MapTabBase(_BaseQt):
    """Interface shared by the Map tab mixins (annotations and stubs only).

    The concrete implementations live in the mixins and in MapTab; this class
    exists so mypy can resolve the attributes and the few cross-mixin calls
    (and see `self` as a QWidget for dialog parents).
    """

    cfg: dict[str, Any]
    app_cfg: AppConfig
    save_cfg: Callable[[], None]
    get_loc: Callable[[], Any]
    get_map_name: Callable[[], str]
    get_store: Callable[[], Any]
    route_pts: Any
    driver: Any
    map_widget: Any
    preset_mgr: PresetManager
    p_sel: Any
    tune_vars: dict[str, StringVarCompat]
    tune_inputs: dict[str, QLineEdit]
    tune_status: Any
    _tune_container: Any
    _collect_ck: Any
    dbg_text: Any
    _edit_snapshot: list[list[float]] | None
    _route_locked: list[Any]
    _edit_btn: Any
    _apply_btn: Any
    _cancel_btn: Any
    _hint_lbl: Any
    routes_status: Any

    def _set_edit_mode(self, editing: bool) -> None: ...

    def routes_refresh(self) -> None: ...

    def _save_route_to_preset(self, name: str) -> bool:
        raise NotImplementedError

    def preset_reload(self) -> None: ...
