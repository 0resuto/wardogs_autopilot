"""Feature-index search internals for MapLocator (detect, match, RANSAC).

Extracted from locator.py: the frame detection, the radius/global candidate
matching against the SIFT index and the per-level scale pruning live here.
`MapLocator` inherits this mixin, so the public surface is unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from ..common.log import get_logger
from .preprocessing import shadow_fill_norm

logger = get_logger("locator")

DEFAULT_MAX_KP = 1200
DEFAULT_EARLY_INL = 40
# OpenCV's BFMatcher packs every train row index into 18 bits (1 << 18), so
# knnMatch aborts once a candidate set reaches 262144 descriptors. A growing
# radius search on a dense map can collect that many: thin oversized sets
# before matching (this cv2.error used to kill the tracker thread).
BF_MAX_TRAIN_DESC = 200_000


def _over(t0: float, budget: float | None) -> bool:
    """Check whether frame time budget has elapsed."""
    return budget is not None and (time.time() - t0) > budget


def _subsample_train(
    pts: np.ndarray, desc: np.ndarray, cap: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly thin a train set, keeping points and descriptors aligned."""
    if cap is None:
        cap = BF_MAX_TRAIN_DESC
    idx = np.unique(np.linspace(0, len(desc) - 1, cap).astype(np.intp))
    return np.asarray(pts)[idx], np.asarray(desc)[idx]


def _mark_search(
    diag: dict[str, Any], discs: list[tuple[float, float, float]], global_pass: bool
) -> None:
    """Publish the search region(s) of this frame into the diagnostics."""
    diag["search_discs"] = [[float(v) for v in d] for d in discs]
    diag["search_global"] = bool(global_pass)


class IndexSearchMixin:
    """Index-based pose search (radius + global) on top of a MapStore."""

    store: Any
    sift: Any
    bf: Any
    clahe: Any
    ratio: float

    def _clahe_preprocess(self, mm: np.ndarray, ui_mask: np.ndarray | None) -> np.ndarray:
        """Fill UI pixels with the background median, then local-contrast (CLAHE).

        Yields fewer, more structural keypoints than the percentile stretch, so
        it is ~2x faster to match; used as an optional first pass with a
        fallback to `shadow_fill_norm` when it finds no pose.
        """
        m = mm.copy()
        if ui_mask is not None and ui_mask.size:
            m[ui_mask] = int(np.median(m[~ui_mask]))
        return self.clahe.apply(m)

    def _detect(self, mmf: np.ndarray, max_kp: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """SIFT keypoints/descriptors of one frame, capped by descending response.

        Tree canopy and other repetitive texture can yield thousands of weak,
        non-distinctive keypoints (measured 1400+ on a forest frame) that inflate
        the BF.knnMatch cost and dilute RANSAC without adding real inliers.
        Ranking by response and keeping `max_kp` retains the structural points
        while bounding the per-frame match cost.
        """
        kp, desc = self.sift.detectAndCompute(mmf, None)
        if desc is not None:
            # SIFT descriptors are integer 0..255; the index stores uint8 and
            # BFMatcher requires both sides to share the descriptor type.
            desc = np.asarray(desc, dtype=np.uint8)
        if desc is None or max_kp <= 0 or len(kp) <= max_kp:
            return list(kp), desc
        order = np.argsort([k.response for k in kp])[::-1][:max_kp]
        return [kp[int(i)] for i in order], desc[order]

    def _mm_center_to_map(self, r: dict[str, Any], mm: np.ndarray) -> tuple[float, float]:
        """Player center in map (mu) coords from a pose with map-space translation.

        The player marker is assumed at the ROI midpoint shifted by the
        calibrated `center_dx`/`center_dy` offset (minimap px). The offset is
        applied in the frame's own coordinates, before the pose rotation, so the
        calibration holds at any heading: an uncalibrated off-center marker
        would otherwise make the reported map position circle the true one as
        the minimap rotates.
        """
        cfg = self.store.loc_cfg()
        px = mm.shape[1] / 2.0 + float(cfg.get("center_dx", 0.0) or 0.0)
        py = mm.shape[0] / 2.0 + float(cfg.get("center_dy", 0.0) or 0.0)
        th_r = np.radians(r["th"])
        a, b = r["s"] * np.cos(th_r), r["s"] * np.sin(th_r)
        return (r["t"][0] + a * px - b * py, r["t"][1] + b * px + a * py)

    def _pose_via_index(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        idx: Any,
        cx: float,
        cy: float,
        r: float | None,
        thr: dict[str, Any] | None = None,
        budget: float | None = None,
        t0: float | None = None,
        feats: tuple[list[cv2.KeyPoint], np.ndarray | None] | None = None,
        cands: list[tuple[Any, np.ndarray, np.ndarray]] | None = None,
        prev_s: float | None = None,
        early_inl: int = 0,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """SIFT match of the minimap against index descriptors within radius r (or globally)."""
        start_t = t0 if t0 is not None else time.time()
        if feats is not None:
            kp1, d1 = feats
        else:
            mmf = shadow_fill_norm(mm, ui_mask)
            max_kp = int(self.store.loc_cfg().get("max_kp_frame", DEFAULT_MAX_KP))
            kp1, d1 = self._detect(mmf, max_kp)
        diag: dict[str, Any] = dict(
            mmi_shape=tuple(mm.shape),
            roi=None,
            kp_mm=0 if d1 is None else len(d1),
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
        )
        if d1 is None or len(d1) < 4:
            diag["reject"] = "no_features_frame"
            diag["detail"] = "index path: frame without SIFT features"
            return None, diag

        if thr is None:
            cfg = self.store.loc_cfg()
            thr = dict(
                ratio=float(cfg.get("ratio", self.ratio)),
                min_inl=int(cfg.get("min_inl", 4)),
                min_inl_rate=float(cfg.get("min_inl_rate", 0.0)),
            )
        min_inl = int(thr["min_inl"])
        ratio = float(thr["ratio"])
        min_rate = float(thr["min_inl_rate"])

        if cands is None:
            if r is None:
                cands = idx.global_candidates(int(self.store.loc_cfg()["global_max_features"]))
            else:
                cands = idx.radius_candidates(cx, cy, r)
        scope = "global" if r is None else f"local(r={r:.0f})"
        # Tracking with a known scale only needs the pyramid levels close to the
        # previous match: the query scale changes smoothly, and matching all
        # three levels tripled the knn cost for nothing.
        if r is not None and prev_s is not None and len(cands) > 1:
            keep_close = [
                c for c in cands if abs(float(idx.levels[int(c[0])]) - prev_s) <= 0.25 * prev_s
            ]
            if keep_close:
                cands = keep_close

        best: dict[str, Any] | None = None
        best_len: int | None = None
        tried = []
        budget_hit = False
        n_tried = 0
        for _li, pts, d2 in cands:
            if _over(start_t, budget):
                budget_hit = True
                break
            n_tried += 1
            if d2 is None or len(d2) < 2:
                continue
            if len(d2) > BF_MAX_TRAIN_DESC:
                pts, d2 = _subsample_train(pts, d2)
            kn = self.bf.knnMatch(d1, d2, k=2)
            good = [g for g, n in kn if g.distance < ratio * n.distance]
            if len(good) < 4:
                continue
            src = np.array([kp1[g.queryIdx].pt for g in good], dtype=np.float32).reshape(-1, 1, 2)
            dst = np.array([pts[g.trainIdx] for g in good], dtype=np.float32).reshape(-1, 1, 2)
            m3, inl_mask = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=6.0,
                maxIters=2000,
                confidence=0.999,
            )
            if m3 is None or inl_mask is None:
                continue
            inl = int(inl_mask.sum())
            if inl < min_inl:
                continue
            if min_rate > 0.0 and inl < min_rate * len(good):
                tried.append((len(good), inl, f"inl_rate {inl / max(len(good), 1):.2f}"))
                continue
            s = float(np.hypot(m3[0, 0], m3[0, 1]))
            if not 0.7 <= s <= 2.6:
                tried.append((len(good), inl, f"scale {s:.2f}"))
                continue
            inliers = [good[i] for i, v in enumerate(inl_mask.flatten()) if v]
            inlier_pts = [kp1[g.queryIdx].pt for g in inliers]
            cand_pose: dict[str, Any] = dict(
                s=s,
                th=float(np.degrees(np.arctan2(m3[1, 0], m3[0, 0]))),
                t=(float(m3[0, 2]), float(m3[1, 2])),
                inl=inl,
                n_match=len(good),
                inlier_pts=inlier_pts,
            )
            if best is None or inl > int(best["inl"]):
                best = cand_pose
                best_len = len(d2)
            if early_inl > 0 and inl >= early_inl:
                # A match this strong will not be beaten by the remaining
                # scale levels: stop paying for their knnMatch.
                break

        diag["kp_pts"] = [kp.pt for kp in kp1]
        if best is None:
            diag["inlier_pts"] = []
            if budget_hit:
                diag["reject"] = "budget_timeout"
                diag["detail"] = (
                    f"frame budget exhausted during {scope} search (candidates tried={n_tried})"
                )
            else:
                diag["reject"] = "index_no_match"
                diag["detail"] = f"index ({scope}) search found no pose (tried={tried or '-'})"
            return None, diag

        diag.update(
            kp_mm=len(d1),
            kp_chunk=best_len,
            good1=best["n_match"],
            inl1=best["inl"],
            s1=best["s"],
            th1=best["th"],
            t1=best["t"],
            good2=best["n_match"],
            inl2=best["inl"],
            s2=best["s"],
            th2=best["th"],
            t2=best["t"],
            inlier_pts=best.get("inlier_pts", []),
            reject=None,
            detail="OK (index)",
        )
        return best, diag

    def _index_find(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        idx: Any,
        cx: float | None,
        cy: float | None,
        min_inl: int = 4,
        budget: float | None = None,
        t0: float | None = None,
        progress: Callable[[tuple[Any, ...]], None] | None = None,
        feats: tuple[list[cv2.KeyPoint], np.ndarray | None] | None = None,
        prev_s: float | None = None,
    ) -> (
        tuple[dict[str, Any] | None, dict[str, Any], float, float, float]
        | tuple[None, dict[str, Any]]
    ):
        """Index-based pose: growing radius around (cx, cy), then whole map."""
        start_t = t0 if t0 is not None else time.time()
        cfg = self.store.loc_cfg()
        rad = float(cfg["local_radius"])
        growth = float(cfg["radius_growth"])
        track_radius = float(cfg.get("track_radius", rad * 2.0))

        base = dict(
            ratio=float(cfg.get("ratio", self.ratio)),
            min_inl=int(cfg.get("min_inl", min_inl)),
            min_inl_rate=float(cfg.get("min_inl_rate", 0.0)),
        )
        local_thr = dict(
            ratio=float(cfg.get("ratio_local", base["ratio"])),
            min_inl=int(cfg.get("min_inl_local", base["min_inl"])),
            min_inl_rate=float(cfg.get("min_inl_rate_local", base["min_inl_rate"])),
        )
        global_thr = dict(
            ratio=float(cfg.get("ratio_global", base["ratio"])),
            min_inl=int(cfg.get("min_inl_global", base["min_inl"])),
            min_inl_rate=float(cfg.get("min_inl_rate_global", base["min_inl_rate"])),
        )
        total = idx.total_features()
        last_diag: dict[str, Any] | None = None
        discs: list[tuple[float, float, float]] = []
        global_pass = False
        # Perf knob with its own key: stop trying further scale-level
        # candidates once a match reaches this inlier count (0 = exhaustive
        # scan). Do not derive it from vote_inl_skip: that is the tracker's
        # trust gate, and tuning it silently changed matcher thoroughness.
        early_inl = int(cfg.get("early_inl", DEFAULT_EARLY_INL) or 0)

        if cx is not None and cy is not None:
            qx, qy = cx, cy
            for _ in range(10):
                if _over(start_t, budget):
                    break
                if progress is not None:
                    progress(("disc", qx, qy, rad))
                discs.append((qx, qy, rad))
                cands = idx.radius_candidates(qx, qy, rad)
                thr = global_thr if rad > track_radius else local_thr
                res, dd = self._pose_via_index(
                    mm,
                    ui_mask,
                    idx,
                    qx,
                    qy,
                    rad,
                    thr=thr,
                    budget=budget,
                    t0=start_t,
                    feats=feats,
                    cands=cands,
                    prev_s=prev_s,
                    early_inl=early_inl,
                )
                last_diag = dd
                if res is not None:
                    mx, my = self._mm_center_to_map(res, mm)
                    if abs(mx - qx) <= rad + 2.0 and abs(my - qy) <= rad + 2.0:
                        _mark_search(dd, discs, False)
                        return res, dd, mx - rad, my - rad, rad
                cov = sum(len(p) for _, p, _ in cands)
                if total and cov >= 0.30 * total:
                    break
                rad *= growth

        if not _over(start_t, budget):
            global_pass = True
            if progress is not None:
                progress(("global",))
            res, dd = self._pose_via_index(
                mm,
                ui_mask,
                idx,
                0.0,
                0.0,
                None,
                thr=global_thr,
                budget=budget,
                t0=start_t,
                feats=feats,
                early_inl=early_inl,
            )
            last_diag = dd
            if res is not None:
                mx, my = self._mm_center_to_map(res, mm)
                rad = max(float(cfg["local_radius"]), 350.0)
                _mark_search(dd, discs, True)
                return res, dd, mx - rad, my - rad, rad

        if last_diag is not None:
            _mark_search(last_diag, discs, global_pass)
        if _over(start_t, budget):
            if last_diag is None:
                last_diag = dict(reject="budget_timeout", detail="frame budget exhausted")
                _mark_search(last_diag, discs, global_pass)
            elif last_diag.get("reject") is None:
                last_diag["reject"] = "budget_timeout"
                last_diag["detail"] = "frame budget exhausted"
        return None, last_diag or {}
