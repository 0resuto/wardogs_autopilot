"""Map storage, cache management, and preview pyramids for WARDOGS.

Manages full-map images in full_maps/, precomputed downscaled caches (mu.npy,
preview mipmaps, color previews) in data/maps/, and cache validation signatures.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from .. import PROJECT_ROOT
from ..common.config import AppConfig, LocatorConfig, MapConfig
from ..common.log import get_logger
from .featureindex import _INDEX_NORM, load_index

logger = get_logger("map_store")

ROOT = PROJECT_ROOT
DATA_MAPS = os.path.join(PROJECT_ROOT, "data", "maps")
FULL_DIR = DATA_MAPS  # Alias for backward compatibility

PREVIEW_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
COLOR_PREVIEW_SIZE = 4096


class MapStore:
    """Thread-safe manager for map files, caches, and active map state."""

    def __init__(
        self,
        full_dir: str = FULL_DIR,
        data_maps_dir: str = DATA_MAPS,
        default_map: str = "zestafona",
    ) -> None:
        self.full_dir = full_dir
        self.data_maps_dir = data_maps_dir
        self._cur_name = default_map
        self._map_lock = threading.Lock()
        self._cache_lock = threading.Lock()

        self._g: dict[str, Any] = {"mu": None, "ms": 2.6544}
        self._color_map_cache: dict[str, Any] = {"name": None, "img": None}
        self._loc_cfg_cache: tuple[float | None, dict[str, Any] | None] = (None, None)
        self._map_cfg_cache: tuple[float | None, dict[str, Any] | None] = (None, None)

    # ---------- Path helpers ----------

    def full_path(self, name: str | None = None) -> str:
        """Absolute path to the full-resolution map PNG."""
        map_name = name or self._cur_name
        return os.path.join(self.data_maps_dir, f"{map_name}_map.png")

    def cache_path(self, name: str, suffix: str) -> str:
        """Path to a per-map cache file in data/maps/."""
        return os.path.join(self.data_maps_dir, f"{name}_{suffix}")

    def preview_path(self, name: str, n: int) -> str:
        """Path to a preview pyramid level npy file."""
        return os.path.join(self.data_maps_dir, f"{name}_preview_{n}.npy")

    # ---------- Map Registry ----------

    def available_maps(self) -> list[str]:
        """List of available map names discovered from data/maps/*_map.png or catalog.json."""
        if not os.path.isdir(self.data_maps_dir):
            return []
        maps: set[str] = {
            f[: -len("_map.png")] for f in os.listdir(self.data_maps_dir) if f.endswith("_map.png")
        }
        if not maps:
            catalog_path = os.path.join(self.data_maps_dir, "catalog.json")
            if os.path.exists(catalog_path):
                try:
                    with open(catalog_path, encoding="utf-8") as f:
                        data = json.load(f)
                    maps.update(data.get("maps", {}).keys())
                except Exception:
                    pass
        return sorted(maps)

    def full_map_size(self, name: str | None = None) -> tuple[int, int] | None:
        """Native (w, h) size of the full map from the PNG header or catalog manifest."""
        target_name = name or self._cur_name
        path = self.full_path(target_name)
        try:
            with open(path, "rb") as f:
                head = f.read(24)
        except OSError:
            head = b""
        if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
            return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))

        catalog_path = os.path.join(self.data_maps_dir, "catalog.json")
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, encoding="utf-8") as f:
                    data = json.load(f)
                size = data.get("maps", {}).get(target_name, {}).get("size")
                if size and len(size) == 2:
                    return (int(size[0]), int(size[1]))
            except Exception:
                pass
        return None

    def set_map(self, name: str) -> None:
        """Switch active map and reset map-specific in-memory caches."""
        name = str(name or "zestafona")
        with self._map_lock:
            if name == self._cur_name:
                return
            self._g.clear()
            self._g.update(mu=None, ms=2.6544)
            self._cur_name = name

    def map_name(self) -> str:
        """Current active map name."""
        return self._cur_name

    # ---------- Config caches ----------

    def loc_cfg(self) -> dict[str, Any]:
        """'locator' block of config.json, cached by file mtime."""
        path = os.path.join(ROOT, "config.json")
        mtime = None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            pass
        if self._loc_cfg_cache[0] == mtime and self._loc_cfg_cache[1] is not None:
            return self._loc_cfg_cache[1]

        try:
            cfg = AppConfig.load(path).locator.model_dump()
        except Exception:
            cfg = LocatorConfig().model_dump()
        self._loc_cfg_cache = (mtime, cfg)
        return cfg

    def map_cfg(self) -> dict[str, Any]:
        """'map' block of config.json, cached by file mtime."""
        path = os.path.join(ROOT, "config.json")
        mtime = None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            pass
        if self._map_cfg_cache[0] == mtime and self._map_cfg_cache[1] is not None:
            return self._map_cfg_cache[1]

        try:
            cfg = AppConfig.load(path).map.model_dump()
        except Exception:
            cfg = MapConfig().model_dump()
        self._map_cfg_cache = (mtime, cfg)
        return cfg

    def gray_sig(self) -> str:
        """Machine-readable ID of current map gray conversion (cache key)."""
        c = self.map_cfg()
        return f"{c['gray_conv']}-{float(c['gray_gamma']):.3f}"

    def gray_sig_on_disk(self, name: str | None = None) -> str | None:
        """Gray signature the on-disk mu cache was built with, or None."""
        try:
            with open(self.cache_path(name or self._cur_name, "gray.txt"), encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def load_chunk2map(self, name: str | None = None) -> dict[str, float] | None:
        """Load chunk px -> full-map pixels mapping if cached."""
        map_name = name or self._cur_name
        path = self.cache_path(map_name, "chunk2map.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path)
        return dict(
            s=float(d["s"]),
            th=float(d["th"]),
            tx=float(d["tx"]),
            ty=float(d["ty"]),
        )

    def mini_scale(self, name: str | None = None) -> float:
        """Pixel scale of minimap window relative to map."""
        c2m = self.load_chunk2map(name)
        if c2m:
            return float(c2m["s"])
        cfg_scale = self.map_cfg().get("mini_scale")
        if cfg_scale is not None:
            return float(cfg_scale)
        return float(self._g.get("ms", 2.6544))

    def get_index(self) -> Any:
        """Load cached SIFT feature index for active map, checking signature matches."""
        idx = self._g.get("idx")
        if idx is not None and getattr(idx, "name", None) == self._cur_name:
            return idx
        with self._cache_lock:
            idx = self._g.get("idx")
            if idx is not None and getattr(idx, "name", None) == self._cur_name:
                return idx
            try:
                idx = load_index(self._cur_name)
            except Exception:
                idx = None
            if idx is not None:
                got = getattr(idx, "gray_sig", None)
                want = self.gray_sig()
                if got is not None and got != want:
                    logger.warning(
                        "[map_store] feature index gray mismatch: built=%s, current=%s "
                        "— rebuild with featureindex --rebuild",
                        got,
                        want,
                    )
                    idx = None
            if idx is not None and getattr(idx, "norm", None) != _INDEX_NORM:
                logger.warning(
                    "[map_store] feature index build mismatch (norm=%s, want=%s) "
                    "— rebuild with featureindex --rebuild",
                    getattr(idx, "norm", None),
                    _INDEX_NORM,
                )
                idx = None
            self._g["idx"] = idx
        return self._g.get("idx")

    # ---------- Previews & Map caches ----------

    def build_previews(self, full: np.ndarray) -> None:
        """Build and cache downscaled levels (512..16384) from the full map."""
        for n in PREVIEW_SIZES:
            img = cv2.resize(full, (n, n), interpolation=cv2.INTER_AREA)
            try:
                np.save(self.preview_path(self._cur_name, n), img)
            except OSError as exc:
                logger.warning("[map_store] failed to save preview %d: %s", n, exc)

    def load_previews(self) -> dict[int, np.ndarray] | None:
        """Preview pyramid of active map from cache, or None."""
        sizes = [n for n in PREVIEW_SIZES if os.path.exists(self.preview_path(self._cur_name, n))]
        if not sizes:
            return None
        d = {}
        for n in sizes:
            try:
                d[n] = np.load(self.preview_path(self._cur_name, n))
            except Exception:
                return None
        return d

    def ensure_previews(self) -> bool:
        """Ensure preview pyramid exists on disk."""
        name = self._cur_name
        if all(os.path.exists(self.preview_path(name, n)) for n in PREVIEW_SIZES):
            return True
        full = cv2.imread(self.full_path(name), cv2.IMREAD_GRAYSCALE)
        if full is None:
            return False
        self.build_previews(full)
        return True

    def color_map(self) -> np.ndarray:
        """Color map of active map at COLOR_PREVIEW_SIZE^2, cached per map."""
        cur = self._color_map_cache.get("name")
        if cur == self._cur_name and self._color_map_cache.get("img") is not None:
            return self._color_map_cache["img"]

        cache = self.cache_path(self._cur_name, f"rgb{COLOR_PREVIEW_SIZE}.npy")
        img = None
        try:
            img = np.load(cache)
        except Exception:
            img = None

        if img is None:
            with self._cache_lock:
                cur = self._color_map_cache.get("img")
                if cur is not None and self._color_map_cache.get("name") == self._cur_name:
                    return cur
                reduced = cv2.imread(self.full_path(self._cur_name), cv2.IMREAD_REDUCED_GRAYSCALE_8)
                if reduced is None:
                    raise OSError(f"map not read: {self.full_path(self._cur_name)}")
                resized = cv2.resize(
                    reduced,
                    (COLOR_PREVIEW_SIZE, COLOR_PREVIEW_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                try:
                    np.save(cache, img)
                except OSError:
                    pass
        self._color_map_cache.update(name=self._cur_name, img=img)
        return img

    def load_global_map(self) -> np.ndarray:
        """Load or build the global mu map at minimap scale."""
        if self._g["mu"] is not None:
            return self._g["mu"]

        name = self._cur_name
        cache_mu = self.cache_path(name, "mu.npy")
        sig = self.gray_sig()
        cached_sig = self.gray_sig_on_disk(name)
        default = "luma-1.000"

        stale_sig = (cached_sig is None and sig != default) or (
            cached_sig is not None and cached_sig != sig
        )
        if os.path.exists(cache_mu) and not stale_sig:
            loaded_mu: np.ndarray = np.load(cache_mu)
            self._g["mu"] = loaded_mu
            self.ensure_previews()
            return loaded_mu

        if stale_sig:
            logger.info(
                "[map_store] map gray changed (%s -> %s): rebuilding caches",
                cached_sig if cached_sig else default,
                sig,
            )

        with self._cache_lock:
            cached_mu = self._g.get("mu")
            if cached_mu is not None and isinstance(cached_mu, np.ndarray):
                return cached_mu
            gray = cv2.imread(self.full_path(name), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise OSError(f"full map not read: {self.full_path(name)}")
            h, _ = gray.shape[:2]
            ms = self.mini_scale()
            side = int(round(h / ms))
            mu = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
            self.build_previews(gray)
            del gray
            try:
                np.save(cache_mu, mu)
                with open(self.cache_path(name, "gray.txt"), "w", encoding="utf-8") as f:
                    f.write(sig)
            except OSError as exc:
                logger.error("[map_store] failed to save map cache: %s", exc)
            self._g.update(mu=mu, ms=ms)
            return mu

    def rebuild_map_cache(
        self, name: str, progress_cb: Callable[[str], None] | None = None
    ) -> bool:
        """Rebuild mu.npy, preview mipmaps, and SIFT feature index for specified map."""
        with self._cache_lock:
            self.set_map(name)
            if progress_cb:
                progress_cb("Step 1/3: Reading full map and generating mu...")
            cache_mu = self.cache_path(name, "mu.npy")
            gray = cv2.imread(self.full_path(name), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise OSError(f"full map not read: {self.full_path(name)}")
            h, _ = gray.shape[:2]

            ms = self.mini_scale(name)
            side = int(round(h / ms))
            mu = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
            np.save(cache_mu, mu)
            sig = self.gray_sig()
            with open(self.cache_path(name, "gray.txt"), "w", encoding="utf-8") as f:
                f.write(sig)

            if progress_cb:
                progress_cb("Step 2/3: Building preview mipmaps (512..16384)...")
            self.build_previews(gray)
            del gray

            self._g.update(mu=mu, ms=ms)

            if progress_cb:
                progress_cb("Step 3/3: Building SIFT feature index...")
            from . import featureindex

            featureindex.build_index(name, progress=False)

            if progress_cb:
                progress_cb("Done: Map cache and SIFT index ready!")
            return True
