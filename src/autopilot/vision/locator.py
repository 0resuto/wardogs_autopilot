"""Full-map matcher and player localization.

Estimates player position on the full map from minimap captures using SIFT
descriptors matched against an offline spatial FeatureIndex (featureindex.py).
Delegates map file and cache management to MapStore and image filtering to
preprocessing.py.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal, overload

import cv2
import numpy as np

from ..common.log import get_logger
from .hybrid import HybridLocalizer
from .index_search import (
    DEFAULT_MAX_KP,
    IndexSearchMixin,
)
from .index_search import (
    _subsample_train as _subsample_train,
)
from .map_store import (
    DATA_MAPS,
    FULL_DIR,
    PREVIEW_SIZES,
    ROOT,
    MapStore,
)
from .preprocessing import (
    MASK_PATH,
    bgr_to_gray,
    crop_win,
    fill_norm,
    make_mask,
    norm8,
    shadow_fill_norm,
)

__all__ = [
    "DATA_MAPS",
    "FULL_DIR",
    "MASK_PATH",
    "PREVIEW_SIZES",
    "ROOT",
    "MapLocator",
    "MapStore",
    "available_maps",
    "bgr_to_gray",
    "build_previews",
    "crop_win",
    "engine",
    "ensure_previews",
    "fill_norm",
    "full_map_size",
    "global_pose",
    "heading_deg",
    "load_global_map",
    "load_previews",
    "make_mask",
    "map_name",
    "norm8",
    "rebuild_map_cache",
    "set_engine",
    "set_map",
    "shadow_fill_norm",
]
logger = get_logger("locator")

FAST_BUDGET_FRAC = 0.4


class MapLocator(IndexSearchMixin):
    """SIFT feature-index localization engine (the hybrid's anchor)."""

    def __init__(self, store: MapStore | None = None) -> None:
        self.store = store or MapStore()
        self.kind = "sift"
        self.detector = cv2.SIFT.create(nfeatures=6000, contrastThreshold=0.05, edgeThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = 0.80
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

    def heading_deg(self, pose: dict[str, Any]) -> float:
        """Player's heading on the map: 0 deg = north, 90 deg = east (clockwise)."""
        return float(pose["th"])

    @overload
    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = ...,
        min_inl: int = ...,
        prev_th: float | None = ...,
        debug: Literal[True] = ...,
        budget: float | None = ...,
        progress: Callable[[tuple[Any, ...]], None] | None = ...,
        prev_s: float | None = ...,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]: ...

    @overload
    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = ...,
        min_inl: int = ...,
        prev_th: float | None = ...,
        debug: Literal[False] = ...,
        budget: float | None = ...,
        progress: Callable[[tuple[Any, ...]], None] | None = ...,
        prev_s: float | None = ...,
    ) -> dict[str, Any] | None: ...

    @overload
    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = ...,
        min_inl: int = ...,
        prev_th: float | None = ...,
        debug: bool = ...,
        budget: float | None = ...,
        progress: Callable[[tuple[Any, ...]], None] | None = ...,
        prev_s: float | None = ...,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None: ...

    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = None,
        min_inl: int = 4,
        prev_th: float | None = None,
        debug: bool = False,
        budget: float | None = None,
        progress: Callable[[tuple[Any, ...]], None] | None = None,
        prev_s: float | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None:
        """Estimate player position on WHOLE map in native map px via feature index."""
        t0 = time.time()
        diag = self._base_diag(mm, ui_mask)

        loc = self.store.loc_cfg()
        max_kp = int(loc.get("max_kp_frame", DEFAULT_MAX_KP))
        fast = bool(loc.get("fast_clahe", False))
        passes: list[Callable[[np.ndarray, np.ndarray | None], np.ndarray]] = (
            [self._clahe_preprocess, shadow_fill_norm] if fast else [shadow_fill_norm]
        )

        _ = self.store.load_global_map()
        ms = self.store.mini_scale()

        cx = cy = None
        if prev_xy is not None:
            diag["prev"] = (float(prev_xy[0]), float(prev_xy[1]))
            cx, cy = prev_xy[0] / ms, prev_xy[1] / ms

        idx = self.store.get_index(self.kind)
        if idx is None:
            active_name = self.store.map_name()
            diag["reject"] = "no_index"
            diag["detail"] = (
                f"{self.kind} feature index not built for {active_name} — run "
                f"python -m autopilot.vision.featureindex --build {active_name}"
            )
            return (None, diag) if debug else None

        pose = self._localize_index(
            mm,
            ui_mask,
            idx,
            cx,
            cy,
            min_inl=min_inl,
            budget=budget,
            t0=t0,
            progress=progress,
            prev_s=prev_s,
            ms=ms,
            max_kp=max_kp,
            passes=passes,
            diag=diag,
            prev_th=prev_th,
        )
        return (pose, diag) if debug else pose

    def _base_diag(self, mm: np.ndarray, ui_mask: np.ndarray | None) -> dict[str, Any]:
        """Diagnostics skeleton shared by every localization pass."""
        diag: dict[str, Any] = dict(
            mmi_shape=tuple(mm.shape),
            roi=None,
            win_map=None,
            kp_mm=0,
            kp_chunk=0,
            good1=0,
            inl1=0,
            good2=0,
            inl2=0,
            s1=None,
            th1=None,
            t1=None,
            s2=None,
            th2=None,
            t2=None,
            reject=None,
            detail="",
            mode="index",
            mm_mean=None,
            mm_std=None,
            mm_mask_frac=None,
            prev=None,
            search_discs=None,
            search_global=False,
        )
        diag["mm_mean"] = float(mm.mean())
        diag["mm_std"] = float(mm.std())
        if ui_mask is not None and ui_mask.size:
            diag["mm_mask_frac"] = float(ui_mask.mean())
        return diag

    def _localize_index(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        idx: Any,
        cx: float | None,
        cy: float | None,
        *,
        min_inl: int,
        budget: float | None,
        t0: float,
        progress: Callable[[tuple[Any, ...]], None] | None,
        prev_s: float | None,
        ms: float,
        max_kp: int,
        passes: list[Callable[[np.ndarray, np.ndarray | None], np.ndarray]],
        diag: dict[str, Any],
        prev_th: float | None = None,
    ) -> dict[str, Any] | None:
        """Run the prep passes and the index search; fills diag, returns the pose."""
        _kf: list[cv2.KeyPoint] = []
        _df: np.ndarray | None = None
        nfeat = 0
        fp: Any = None
        for i, prep in enumerate(passes):
            # the fast pass gets only a slice of the budget so a fallback can run
            sub_budget = budget
            if budget is not None and len(passes) > 1 and i < len(passes) - 1:
                sub_budget = budget * FAST_BUDGET_FRAC
            mmf = prep(mm, ui_mask)
            _kf, _df = self._detect(mmf, max_kp)
            nfeat = 0 if _df is None else len(_df)
            if nfeat < 4:
                continue
            fp = self._index_find(
                mm,
                ui_mask,
                idx,
                cx,
                cy,
                min_inl=min_inl,
                budget=sub_budget,
                t0=t0,
                progress=progress,
                feats=(_kf, _df),
                prev_s=prev_s,
                prev_th=prev_th,
            )
            if fp is not None and len(fp) == 5 and fp[0] is not None:
                break

        diag["kp_mm"] = nfeat
        diag["kp_pts"] = [kp.pt for kp in _kf] if _kf else []
        diag["inlier_pts"] = []
        if nfeat < 4:
            diag["reject"] = "no_features_flat"
            mask_pct = int(100 * (diag["mm_mask_frac"] or 0))
            diag["detail"] = (
                f"frame without texture (feat={nfeat}, std={diag['mm_std']:.1f}, "
                f"mask={mask_pct}%) — map search impossible"
            )
            return None
        r = d = None
        win = orig = None
        if fp is not None and len(fp) == 5 and fp[0] is not None:
            ri = fp[0]
            di = fp[1]
            wx0 = float(fp[2])
            wy0 = float(fp[3])
            rad = float(fp[4])
            r = dict(
                s=ri["s"],
                th=ri["th"],
                t=(ri["t"][0] - wx0, ri["t"][1] - wy0),
                inl=ri["inl"],
                n_match=ri["n_match"],
            )
            d = di
            win = np.zeros((int(2 * rad), int(2 * rad)), np.uint8)
            orig = (wy0, wx0)
        elif fp is not None and len(fp) >= 2 and fp[1] is not None:
            d = fp[1]

        if r is None:
            if d is not None:
                diag["reject"] = d.get("reject") or "index_no_match"
                diag["detail"] = d.get("detail") or "index search found no pose"
                for k in ("search_discs", "search_global"):
                    if k in d:
                        diag[k] = d[k]
            return None

        if d is not None:
            for k in (
                "mmi_shape",
                "kp_mm",
                "kp_chunk",
                "good1",
                "inl1",
                "good2",
                "inl2",
                "s1",
                "th1",
                "t1",
                "s2",
                "th2",
                "t2",
                "reject",
                "detail",
                "kp_pts",
                "inlier_pts",
            ):
                if k in d:
                    diag[k] = d[k]
            for k in ("search_discs", "search_global"):
                if k in d:
                    diag[k] = d[k]

        wy0, wx0 = orig  # type: ignore[misc]
        diag["win_map"] = [
            wx0 * ms,
            wy0 * ms,
            (wx0 + win.shape[1]) * ms,  # type: ignore[union-attr]
            (wy0 + win.shape[0]) * ms,  # type: ignore[union-attr]
        ]

        wx, wy = self._mm_center_to_map(r, mm)
        return dict(
            s=r["s"],
            th=r["th"],
            t=r["t"],
            inl=r["inl"],
            n_match=r["n_match"],
            map_x=(wx0 + wx) * ms,
            map_y=(wy0 + wy) * ms,
        )


# ==============================================================================
# Backward Compatibility Module Facade
# ==============================================================================

_DEFAULT_STORE = MapStore()
_DEFAULT_LOCATOR = MapLocator(_DEFAULT_STORE)
_DEFAULT_HYBRID: Any = HybridLocalizer(_DEFAULT_LOCATOR, _DEFAULT_STORE)
_ENGINE = "hybrid"

# Legacy shared dict access (e.g. tools/map_match_debug.py -> locator._G["mu"])
_G = _DEFAULT_STORE._g

# Expose algorithms on the module level
SIFT = _DEFAULT_LOCATOR.detector
BF = _DEFAULT_LOCATOR.bf
RATIO = _DEFAULT_LOCATOR.ratio


def engine() -> str:
    """Active localization engine: only 'hybrid' ships."""
    return _ENGINE


def set_engine(kind: str) -> None:
    """Select the localization engine used by the `global_pose` facade.

    Only 'hybrid' (SIFT anchor + ECC tracking) ships; any other name keeps
    hybrid and only resets its inter-frame track.
    """
    if str(kind) != "hybrid":
        logger.warning("[locator] unknown engine %r; keeping hybrid", kind)
    _DEFAULT_HYBRID.reset()
    logger.info("[locator] engine: %s", _ENGINE)


def _active_engine() -> Any:
    """The locator object the `global_pose` facade dispatches to."""
    return _DEFAULT_HYBRID


def reset_track() -> None:
    """Drop the engine's inter-frame track; the next frame re-anchors.

    The hybrid track absorbs a wrong anchor into its frame->map transform
    before any consumer can reject the pose, so a rejected motion-flip must
    reset it or the next frame keeps tracking from the flipped transform.
    """
    reset = getattr(_active_engine(), "reset", None)
    if callable(reset):
        reset()


def get_store() -> MapStore:
    return _DEFAULT_STORE


def set_map(name: str) -> None:
    _DEFAULT_STORE.set_map(name)


def map_name() -> str:
    return _DEFAULT_STORE.map_name()


def available_maps() -> list[str]:
    return _DEFAULT_STORE.available_maps()


def full_map_size(name: str | None = None) -> tuple[int, int] | None:
    return _DEFAULT_STORE.full_map_size(name)


def load_global_map() -> np.ndarray:
    return _DEFAULT_STORE.load_global_map()


def load_previews() -> dict[int, np.ndarray] | None:
    return _DEFAULT_STORE.load_previews()


def ensure_previews() -> bool:
    return _DEFAULT_STORE.ensure_previews()


def build_previews(full: np.ndarray) -> None:
    _DEFAULT_STORE.build_previews(full)


def rebuild_map_cache(name: str, progress_cb: Callable[[str], None] | None = None) -> bool:
    return _DEFAULT_STORE.rebuild_map_cache(name, progress_cb)


@overload
def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = ...,
    min_inl: int = ...,
    prev_th: float | None = ...,
    debug: Literal[True] = ...,
    budget: float | None = ...,
    progress: Callable[[tuple[Any, ...]], None] | None = ...,
    prev_s: float | None = ...,
) -> tuple[dict[str, Any] | None, dict[str, Any]]: ...


@overload
def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = ...,
    min_inl: int = ...,
    prev_th: float | None = ...,
    debug: Literal[False] = ...,
    budget: float | None = ...,
    progress: Callable[[tuple[Any, ...]], None] | None = ...,
    prev_s: float | None = ...,
) -> dict[str, Any] | None: ...


@overload
def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = ...,
    min_inl: int = ...,
    prev_th: float | None = ...,
    debug: bool = ...,
    budget: float | None = ...,
    progress: Callable[[tuple[Any, ...]], None] | None = ...,
    prev_s: float | None = ...,
) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None: ...


def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = None,
    min_inl: int = 4,
    prev_th: float | None = None,
    debug: bool = False,
    budget: float | None = None,
    progress: Callable[[tuple[Any, ...]], None] | None = None,
    prev_s: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None:
    return _active_engine().global_pose(
        mm,
        ui_mask,
        prev_xy=prev_xy,
        min_inl=min_inl,
        prev_th=prev_th,
        debug=debug,
        budget=budget,
        progress=progress,
        prev_s=prev_s,
    )


def heading_deg(pose: dict[str, Any]) -> float:
    return _DEFAULT_LOCATOR.heading_deg(pose)


# Internal legacy helpers exposed for tests and tools
def _mini_scale(name: str | None = None) -> float:
    """Minimap scale from the map config; `name` is kept for call-site compat."""
    return _DEFAULT_STORE.mini_scale()


def _loc_cfg() -> dict[str, Any]:
    return _DEFAULT_STORE.loc_cfg()


def _gray_sig() -> str:
    return _DEFAULT_STORE.gray_sig()


def _gray_sig_on_disk(name: str | None = None) -> str | None:
    return _DEFAULT_STORE.gray_sig_on_disk(name)


def _get_index() -> Any:
    return _DEFAULT_STORE.get_index()


def _norm8(g: np.ndarray) -> np.ndarray:
    return norm8(g)


def _bgr_to_gray(bgr: np.ndarray, conv: str = "luma", gamma: float = 1.0) -> np.ndarray:
    return bgr_to_gray(bgr, conv=conv, gamma=gamma)


def _fill_norm(mm: np.ndarray, ui_mask: np.ndarray | None, per: tuple[float, float]) -> np.ndarray:
    return fill_norm(mm, ui_mask, per)


def _shadow_fill_norm(mm: np.ndarray, ui_mask: np.ndarray | None) -> np.ndarray:
    return shadow_fill_norm(mm, ui_mask)


def _crop_win(
    mu: np.ndarray, cy: float, cx: float, rh: int, rw: int
) -> tuple[np.ndarray, tuple[int, int]]:
    return crop_win(mu, cy, cx, rh, rw)


if __name__ == "__main__":
    logger.info("loading the global map cache...")
    t0_main = time.time()
    mu_main = load_global_map()
    logger.info("ok: mu=%s (%.1fs)", tuple(mu_main.shape), time.time() - t0_main)
