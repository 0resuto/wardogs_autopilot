"""Offline feature index of a map for radius-based localization.

Builds once per map (CLI: python -m autopilot.vision.featureindex --build
zestafona) and stores descriptors of the downscaled map pyramid in
data/maps/<name>_feat.npz. Descriptors are integer 0..255, so they are stored
as uint8 inside a compressed npz (5x smaller than the old float32 npz; the
live query is cast to uint8 in MapLocator._detect because BFMatcher requires
both sides to share the type). The minimap is then matched via BF.knnMatch
against the descriptors of the tiles around the last known position (radius
search), avoiding per-frame SIFT detection over a big window.

The index is a cache only: load_index() returns None when the npz is absent and
the caller falls back to the classic window + coarse search (see locator.py).
"""

import argparse
import math
import os
import time
from typing import Any

import cv2
import numpy as np

from ..common.log import get_logger

logger = get_logger("featureindex")


TILE = 512
# Per-tile feature cap. Forest tiles hold up to ~3400 SIFT points; capping by
# response at 600 kept only the strongest (canopy) points and dropped the
# weaker road/building features, so low-structure forest frames had no matches
# in the index (measured: 3 vs 12 RANSAC inliers on the same tile). Keep nearly
# all of them.
MAX_PER_TILE = 3000
LEVELS = (1.0, 0.8, 0.6)

# Supported descriptor kinds: SIFT only. The tag is kept in the npz and in
# MapStore.get_index so a stale cache of another kind is rejected loudly.
INDEX_KINDS = ("sift",)
_INDEX_KIND = "sift"

# Descriptor build must match the query's preprocessing space. In practice the
# game minimap aligns with the RAW map pixels: descriptors from raw tiles match
# real captured frames far stronger than percentile-normalized ones (measured:
# 60 vs 4 RANSAC inliers on an identical frame), so tiles are built WITHOUT
# pixel normalization. 'raw' is that tag; bump it whenever the build or the
# matching-side contract changes to invalidate stale npz caches.
_INDEX_NORM = "raw"

# Storage format tag: uint8 descriptors inside a compressed npz. Old float32
# indexes lack this field and are rejected by MapStore.get_index (with a clear
# "re-download or rebuild" warning) instead of silently loading 5x bigger.
_INDEX_FMT = "u8z"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MAPS_DIR = os.path.join(ROOT, "data", "maps")


def _feat_path(name, kind=_INDEX_KIND):
    """Cache path of a map's feature index; SIFT keeps the legacy file name."""
    suffix = "_feat.npz" if kind == "sift" else "_feat_%s.npz" % kind
    return os.path.join(_MAPS_DIR, "%s%s" % (name, suffix))


def _make_detector(max_per_tile: int, contrast: float):
    """SIFT detector for an index build; `contrast` tunes its sensitivity."""
    if abs(float(contrast) - 0.05) > 1e-6:
        return cv2.SIFT.create(nfeatures=6000, contrastThreshold=float(contrast), edgeThreshold=12)
    from . import locator  # lazy: locator imports this module at load time

    return locator.SIFT


def _tile_features(
    detector: Any,
    crop: np.ndarray,
    edge_w: int,
    edge_h: int,
    max_per_tile: int,
    norm_fn,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(points, descriptors) for one build tile, or None when empty."""
    if norm_fn is not None:
        crop = norm_fn(crop)
    kp, desc = detector.detectAndCompute(crop, None)
    if desc is None or len(desc) == 0:
        return None
    # drop points reflected by border padding (outside the real tile)
    keep_idx = [i for i, p in enumerate(kp) if p.pt[0] < edge_w and p.pt[1] < edge_h]
    if not keep_idx:
        return None
    resp = [kp[i].response for i in keep_idx]
    order = [
        i for _, i in sorted(zip(resp, keep_idx, strict=True), key=lambda x: x[0], reverse=True)
    ]
    if len(order) > max_per_tile:
        order = order[:max_per_tile]
    pts = np.asarray([[kp[i].pt[0], kp[i].pt[1]] for i in order], np.float32)
    return pts, np.asarray(desc[order], np.uint8)


def _grid_dims(img_w, img_h, tile):
    return max(1, int(math.ceil(img_w / float(tile)))), max(1, int(math.ceil(img_h / float(tile))))


def _pad_to(crop, th, tw):
    """Pad to th x tw with BORDER_REFLECT, stepwise (reflect cannot pad
    by more than the current size in one call)."""
    while crop.shape[0] < th:
        pad = min(crop.shape[0], th - crop.shape[0])
        crop = cv2.copyMakeBorder(crop, 0, pad, 0, 0, cv2.BORDER_REFLECT)
    while crop.shape[1] < tw:
        pad = min(crop.shape[1], tw - crop.shape[1])
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad, cv2.BORDER_REFLECT)
    return crop


def build_index(
    name,
    levels=LEVELS,
    tile=TILE,
    max_per_tile=MAX_PER_TILE,
    contrast=0.05,
    norm=_INDEX_NORM,
    kind=_INDEX_KIND,
    progress=False,
):
    """Build the feature index cache for one map; returns the npz path.

    Reads the already-cached *mu.npy (never the full PNG) and extracts SIFT
    descriptors per pyramid level on a tile grid with reflected borders. With
    norm='raw' (default) the tiles feed the detector exactly as the map pixels
    are; 'perc298' percentile-normalizes each tile (kept for experiments).
    Level coordinates are divided by the level scale into map (mu) pixels.
    Returns None when the map cache is missing or the build failed.
    """
    if kind not in INDEX_KINDS:
        raise ValueError("unknown index kind %r (expected %s)" % (kind, " or ".join(INDEX_KINDS)))
    from . import locator  # lazy: locator imports this module at load time

    locator.set_map(name)
    mu = locator.load_global_map()
    ms = locator._mini_scale(name)
    mu_h, mu_w = mu.shape
    gw, gh = _grid_dims(mu_w, mu_h, tile)
    out: dict[str, Any] = dict(
        ms=float(ms),
        mu_h=int(mu_h),
        mu_w=int(mu_w),
        tile=int(tile),
        gw=int(gw),
        gh=int(gh),
        gray_sig=np.array([locator._gray_sig()], dtype="U128"),
        levels=np.asarray(levels, dtype=np.float32),
        norm=np.asarray([norm], dtype="U32"),
        fmt=np.asarray([_INDEX_FMT], dtype="U16"),
        kind=np.asarray([kind], dtype="U8"),
    )
    detector = _make_detector(max_per_tile, contrast)
    norm_fn = (lambda c: locator._shadow_fill_norm(c, None)) if norm == "perc298" else None
    desc_dim, desc_dtype = 128, np.uint8

    for k, f in enumerate(levels):
        Wimg = max(1, int(round(mu_w * f)))
        Himg = max(1, int(round(mu_h * f)))
        img = mu if f == 1.0 else cv2.resize(mu, (Wimg, Himg), interpolation=cv2.INTER_AREA)
        gw_k, gh_k = _grid_dims(Wimg, Himg, tile)
        pts_all, desc_all, tile_all = [], [], []
        t_proc = time.time()
        n_done = 0
        for ty in range(gh_k):
            for tx in range(gw_k):
                x0, y0 = tx * tile, ty * tile
                edge_w = min(tile, Wimg - x0)
                edge_h = min(tile, Himg - y0)
                crop = img[y0 : y0 + tile, x0 : x0 + tile]
                if crop.shape[0] < tile or crop.shape[1] < tile:
                    crop = _pad_to(crop, tile, tile)
                feats = _tile_features(detector, crop, edge_w, edge_h, max_per_tile, norm_fn)
                n_done += 1
                if progress and n_done % 64 == 0:
                    logger.info(
                        "  level %.2f: tiles %d/%d (%.0fs)",
                        f,
                        n_done,
                        gw_k * gh_k,
                        time.time() - t_proc,
                    )
                if feats is None:
                    continue
                pts, desc = feats
                pts_all.append((pts + np.array([x0, y0], dtype=np.float32)) / f)
                desc_all.append(desc)
                tile_all.append(np.full(len(pts), ty * gw_k + tx, dtype=np.int32))
        if pts_all:
            out["pts_lv%d" % k] = np.concatenate(pts_all, axis=0)
            out["desc_lv%d" % k] = np.concatenate(desc_all, axis=0)
            out["tile_lv%d" % k] = np.concatenate(tile_all, axis=0)
        else:
            out["pts_lv%d" % k] = np.empty((0, 2), np.float32)
            out["desc_lv%d" % k] = np.empty((0, desc_dim), desc_dtype)
            out["tile_lv%d" % k] = np.empty((0,), np.int32)
        if progress:
            logger.info(
                "level %.2f (img %dx%d): %d features in %.0fs",
                f,
                Wimg,
                Himg,
                len(out["pts_lv%d" % k]),
                time.time() - t_proc,
            )

    path = _feat_path(name, kind)
    np.savez_compressed(path, **out)
    return path


def load_index(name, kind=_INDEX_KIND):
    """Load a built feature index of `kind`, or None if the cache does not exist."""
    path = _feat_path(name, kind)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            return FeatureIndex(name, d)
    except Exception:  # noqa: BLE001 — corrupt cache: treat as absent
        return None


class FeatureIndex:
    """In-memory map feature index: per-level descriptor sets + tile grid.

    Points and descriptors of every level lie in one flat array; the label
    tile_lv tells which tile each point came from, so a radius query selects
    the tiles intersecting the circle and then filters points by distance.
    All coordinates are in mu pixels (minimap scale), like locator windows.
    """

    def __init__(self, name, data):
        self.name = name
        self.ms = float(data["ms"])
        self.mu_h = int(data["mu_h"])
        self.mu_w = int(data["mu_w"])
        self.tile = int(data["tile"])
        self.gray_sig = None
        if "gray_sig" in data.files and data["gray_sig"].size:
            self.gray_sig = str(data["gray_sig"][0])
        self.norm = None
        if "norm" in data.files and data["norm"].size:
            self.norm = str(data["norm"][0])
        self.fmt = None
        if "fmt" in data.files and data["fmt"].size:
            self.fmt = str(data["fmt"][0])
        # pre-kind caches are SIFT by definition
        self.kind = _INDEX_KIND
        if "kind" in data.files and data["kind"].size:
            self.kind = str(data["kind"][0])
        desc_dtype = np.uint8
        self.levels = np.asarray(data["levels"], np.float32)
        self._levels = []
        for k, f in enumerate(self.levels):
            Wimg = max(1, int(round(self.mu_w * f)))
            Himg = max(1, int(round(self.mu_h * f)))
            gw_k, gh_k = _grid_dims(Wimg, Himg, self.tile)
            tile = np.asarray(data["tile_lv%d" % k], np.int32)
            self._levels.append(
                dict(
                    level=float(f),
                    pts=np.asarray(data["pts_lv%d" % k], np.float32),
                    desc=np.asarray(data["desc_lv%d" % k], desc_dtype),
                    tile=tile,
                    _tile_cache=None,  # (key, pts, desc) of the last tile gather
                    img_w=Wimg,
                    img_h=Himg,
                    gw=gw_k,
                    gh=gh_k,
                )
            )
        # most detailed level (max features) first — matches are tried on it first
        self._levels.sort(key=lambda lv: len(lv["desc"]), reverse=True)

    def _tile_range(self, cx, cy, r):
        """Level indices whose tiles overlap the circle in mu coords.

        The per-tile point gather is cached per level by tile rectangle: while
        tracking, the radius query covers the same tiles frame after frame, and
        re-gathering ~15k descriptor rows (7 MB) each time was pure waste. The
        exact distance filter still runs on every call because the center moves
        inside the tiles.
        """
        q = []
        for li in self._levels:
            f = li["level"]
            # circle center/radius in the level image pixels
            cfx, cfy = cx * f, cy * f
            cr = r * f
            tx0 = max(0, int((cfx - cr) / self.tile))
            tx1 = min(li["gw"] - 1, int((cfx + cr) / self.tile))
            ty0 = max(0, int((cfy - cr) / self.tile))
            ty1 = min(li["gh"] - 1, int((cfy + cr) / self.tile))
            if tx1 < tx0 or ty1 < ty0:
                continue
            key = (tx0, tx1, ty0, ty1)
            cached = li.get("_tile_cache")
            if cached is not None and cached[0] == key:
                pts_t, desc_t = cached[1], cached[2]
            else:
                ids = [
                    ty * li["gw"] + tx for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)
                ]
                keep = np.isin(li["tile"], ids)
                pts_t = li["pts"][keep]
                desc_t = li["desc"][keep]
                li["_tile_cache"] = (key, pts_t, desc_t)
            if len(pts_t) == 0:
                continue
            d = np.hypot(pts_t[:, 0] - cx, pts_t[:, 1] - cy)
            sel = d <= r
            if np.any(sel):
                q.append((li, pts_t[sel], desc_t[sel]))
        return q

    def total_features(self, level=None):
        """Number of descriptor points (all levels, or one level index)."""
        if level is None:
            return sum(len(lv["desc"]) for lv in self._levels)
        return len(self._levels[level]["desc"])

    def radius_candidates(self, cx, cy, r, level=None):
        """(lvl_idx, pts, desc) for levels overlapping the radius circle.

        Returns a list sorted by descending point count; None filter of the
        grid is already applied (only points within `r` of (cx, cy)).
        """
        cand = self._tile_range(cx, cy, r)
        if level is not None:
            cand = [c for c in cand if c[0] is self._levels[level]]
        cand.sort(key=lambda c: len(c[1]), reverse=True)
        return [(self._levels.index(c[0]), c[1], c[2]) for c in cand]

    def global_candidates(self, cap=60000):
        """Whole-map descriptors of the densest level, uniformly sampled.

        The uniform `cap` sample keeps the knnMatch cost bounded on a cold
        start while the descriptor distribution stays representative.
        """
        li = self._levels[0]
        m = len(li["desc"])
        if m > cap:
            idx = np.linspace(0, m - 1, cap).astype(int)
            idx = np.unique(idx)
            return [(0, li["pts"][idx], li["desc"][idx])]
        return [(0, li["pts"], li["desc"])]


def main():
    ap = argparse.ArgumentParser(
        prog="featureindex",
        description="Build/cache the SIFT feature index for maps (data/maps/<name>_feat.npz).",
    )
    ap.add_argument(
        "--build", nargs="+", metavar="NAME", help="map names to index (zestafona bakurani ozeti)"
    )
    ap.add_argument("--rebuild", action="store_true", help="overwrite an existing cache")
    ap.add_argument(
        "--kind",
        choices=INDEX_KINDS,
        default=_INDEX_KIND,
        help="descriptor kind to build (default: %s)" % _INDEX_KIND,
    )
    ap.add_argument(
        "--levels",
        default=None,
        help="scale levels, comma list e.g. 1.0,0.9,0.8,0.7,0.6 (default: 1.0,0.8,0.6)",
    )
    ap.add_argument(
        "--max-per-tile",
        type=int,
        default=None,
        help="max stored features per tile (default: 3000)",
    )
    ap.add_argument(
        "--contrast",
        type=float,
        default=None,
        help="SIFT contrastThreshold for the build (default: 0.05; lower = denser)",
    )
    ap.add_argument(
        "--norm",
        choices=("raw", "perc298"),
        default=_INDEX_NORM,
        help="tile preprocessing for SIFT (default: %s)" % _INDEX_NORM,
    )
    ap.add_argument("--progress", action="store_true", help="print per-tile progress")
    args = ap.parse_args()

    if not args.build:
        ap.error("nothing to do: pass --build <names>")
    levels: tuple[float, ...] = LEVELS
    if args.levels:
        levels = tuple(sorted((float(x) for x in args.levels.split(",")), reverse=True))
    max_per_tile = args.max_per_tile if args.max_per_tile is not None else MAX_PER_TILE
    contrast = args.contrast if args.contrast is not None else 0.05
    for name in args.build:
        path = _feat_path(name, args.kind)
        if os.path.exists(path) and not args.rebuild:
            logger.info("%s: %s cache exists (use --rebuild to overwrite)", name, args.kind)
            continue
        logger.info("building %s index for %s (norm=%s)...", args.kind, name, args.norm)
        t0 = time.time()
        out = build_index(
            name,
            levels=levels,
            max_per_tile=max_per_tile,
            contrast=contrast,
            norm=args.norm,
            kind=args.kind,
            progress=args.progress,
        )
        if out:
            logger.info("%s: wrote %s (%.1fs)", name, out, time.time() - t0)
        else:
            logger.error("%s: build failed", name)


if __name__ == "__main__":
    main()
