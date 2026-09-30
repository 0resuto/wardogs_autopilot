"""Frame-to-frame ECC tracker anchored by periodic SIFT localization.

Between anchors the pose is updated from consecutive minimap frames with
cv2.findTransformECC (euclidean: translation + rotation, no scale). At the
minimap ROI this is ~25 ms versus 80-100 ms for a full SIFT search and is
sub-pixel accurate (measured 0.2-0.4 px on synthetic inter-frame motions);
the anchor rate (`locator.hybrid_reanchor_s`) and the correlation floor
(`locator.hybrid_min_cc`) bound the drift. A weak or failed step re-runs the
anchor immediately, and so do capture-void gaps and minimap ROI changes.

Geometry: ECC returns the frame-to-frame warp A with ``p_curr = A @ p_prev``
(same convention as estimateAffinePartial2D). The anchor pose is kept as a
frame->map affine ``M`` in mu coordinates, so tracking is simply
``M_new = M @ A^-1``; the pose fields (s, th, map_x, map_y) are re-extracted
from ``M_new`` with the calibrated player center.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from ..common.log import get_logger
from .preprocessing import shadow_fill_norm

logger = get_logger("hybrid")

#: ECC stopping criterion: 30 iterations is enough for the small per-frame
#: motion (a few px / a few degrees); the accuracy is checked by the returned
#: correlation coefficient. The default Gaussian prefilter (size 5) smooths
#: the minimap's rendering aliasing.
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)


def _inv3(m: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine as a full 3x3 matrix."""
    full = np.vstack([m, [0.0, 0.0, 1.0]]).astype(np.float64)
    return np.linalg.inv(full)


class HybridLocalizer:
    """SIFT anchor + ECC inter-frame tracking with the MapLocator interface."""

    def __init__(self, anchor: Any, store: Any) -> None:
        self.anchor = anchor
        self.store = store
        self._prev_pp: np.ndarray | None = None
        self._M: np.ndarray | None = None
        self._anchor_t = 0.0
        self._anchor_map: str | None = None
        self._anchor_inl = 0
        self._anchor_n = 0

    def reset(self) -> None:
        """Forget the track: the next frame runs a full SIFT anchor."""
        self._prev_pp = None
        self._M = None

    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = None,
        min_inl: int = 4,
        prev_th: float = 0.0,
        debug: bool = False,
        budget: float | None = None,
        progress: Callable[[tuple[Any, ...]], None] | None = None,
        prev_s: float | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None:
        """One hybrid step: ECC update when fresh, else a SIFT anchor."""
        cfg = self.store.loc_cfg()
        period = float(cfg.get("hybrid_reanchor_s", 1.0) or 0.0)
        min_cc = float(cfg.get("hybrid_min_cc", 0.5) or 0.0)
        map_now = self.store.map_name()
        if self._anchor_map != map_now:
            self.reset()
            self._anchor_map = map_now
        pp = shadow_fill_norm(mm, ui_mask)
        now = time.time()
        need_anchor = (
            self._prev_pp is None
            or self._M is None
            or self._prev_pp.shape != pp.shape
            or (period > 0.0 and now - self._anchor_t >= period)
        )

        diag: dict[str, Any] = dict(
            mode="hybrid", engine="hybrid", anchor=False, reject=None, detail="OK (hybrid track)"
        )
        if not need_anchor:
            M_prev = self._M
            assert M_prev is not None
            ok, warp, cc, why = self._track(pp)
            if ok and warp is not None and cc >= min_cc:
                M = M_prev @ _inv3(warp)
                pose = self._pose_from_M(M, mm, cfg)
                pose["inl"] = self._anchor_inl
                pose["n_match"] = self._anchor_n
                self._M = M
                self._prev_pp = pp
                diag["cc"] = round(cc, 4)
                diag["anchor_age"] = round(now - self._anchor_t, 2)
                return (pose, diag) if debug else pose
            diag["reject"] = "hybrid_reanchor"
            diag["detail"] = why or ("low correlation %.2f" % cc)

        if progress is not None:
            progress(("global",))
        pose, adiag = self.anchor.global_pose(
            mm,
            ui_mask,
            prev_xy=prev_xy,
            min_inl=min_inl,
            prev_th=prev_th,
            debug=True,
            budget=budget,
            progress=progress,
            prev_s=prev_s,
        )
        adiag = dict(adiag)
        adiag["engine"] = "hybrid"
        adiag["anchor"] = True
        adiag["anchor_age"] = 0.0
        if diag.get("reject"):
            adiag["reanchor"] = diag.get("detail", "")
        if pose is None:
            # no anchor: do not keep tracking from a stale track
            self._prev_pp = None
            self._M = None
            return (None, adiag) if debug else None
        self._M = self._M_from_pose(pose, mm, cfg)
        self._prev_pp = pp
        self._anchor_t = time.time()
        self._anchor_map = map_now
        self._anchor_inl = int(pose.get("inl", 0))
        self._anchor_n = int(pose.get("n_match", 0))
        return (pose, adiag) if debug else pose

    def _track(self, pp: np.ndarray) -> tuple[bool, np.ndarray | None, float, str]:
        """ECC euclidean step from the previous frame to `pp`."""
        assert self._prev_pp is not None
        initial = np.eye(2, 3, dtype=np.float32)
        try:
            cc, estimate = cv2.findTransformECC(
                self._prev_pp.astype(np.float32),
                pp.astype(np.float32),
                initial,
                cv2.MOTION_EUCLIDEAN,
                ECC_CRITERIA,
            )
        except cv2.error as exc:
            return False, None, 0.0, "ecc failed: %s" % exc
        return True, np.asarray(estimate, np.float32), float(cc), ""

    @staticmethod
    def _center_px(mm: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
        """Calibrated player center in frame coordinates (see locator.center_dx)."""
        return (
            mm.shape[1] / 2.0 + float(cfg.get("center_dx", 0.0) or 0.0),
            mm.shape[0] / 2.0 + float(cfg.get("center_dy", 0.0) or 0.0),
        )

    def _M_from_pose(self, pose: dict[str, Any], mm: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
        """Frame->map affine (mu coords) with the calibrated player center at C."""
        s = float(pose["s"])
        th = math.radians(float(pose["th"]))
        a, b = s * math.cos(th), s * math.sin(th)
        cx, cy = self._center_px(mm, cfg)
        ms = self.store.mini_scale()
        c = np.array([float(pose["map_x"]) / ms, float(pose["map_y"]) / ms], np.float64)
        t = c - np.array([[a, -b], [b, a]], np.float64) @ np.array([cx, cy], np.float64)
        return np.array([[a, -b, t[0]], [b, a, t[1]], [0.0, 0.0, 1.0]], np.float64)

    def _pose_from_M(self, M: np.ndarray, mm: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
        """Pose fields (s, th, map center) from the frame->map affine."""
        s = float(np.hypot(M[0, 0], M[1, 0]))
        th = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        cx, cy = self._center_px(mm, cfg)
        c = M @ np.array([cx, cy, 1.0], np.float64)
        ms = self.store.mini_scale()
        return dict(
            s=s,
            th=th,
            t=(float(M[0, 2]), float(M[1, 2])),
            inl=0,
            n_match=0,
            map_x=float(c[0]) * ms,
            map_y=float(c[1]) * ms,
        )
