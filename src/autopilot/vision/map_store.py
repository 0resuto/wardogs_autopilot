"""Map storage, cache management, and preview pyramids for WARDOGS.

Manages precomputed derived caches (mu.npy, grayscale preview mipmaps) in
data/maps/ and their cache validation signatures. The source PNG is only
needed on a maintainer machine to rebuild the caches; a distributed install
downloads the artifacts together with the catalog that records the build
source.
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
from ..common.config import AppConfig, LocatorConfig, MapConfig, resolve_config_path
from ..common.log import get_logger
from .featureindex import _INDEX_FMT, _INDEX_NORM, load_index
from .preprocessing import bgr_to_gray

logger = get_logger("map_store")

ROOT = PROJECT_ROOT
DATA_MAPS = os.path.join(PROJECT_ROOT, "data", "maps")
FULL_DIR = DATA_MAPS  # Alias for backward compatibility

PREVIEW_SIZES = (512, 1024, 2048, 4096, 8192, 16384)


class MapStore:
    """Thread-safe manager for map files, caches, and active map state."""

    def __init__(
        self,
        full_dir: str = FULL_DIR,
        data_maps_dir: str = DATA_MAPS,
        default_map: str = "zestafona",
        config_path: str | None = None,
    ) -> None:
        self.full_dir = full_dir
        self.data_maps_dir = data_maps_dir
        self._cur_name = default_map
        self._config_path = str(
            resolve_config_path(config_path or os.path.join(ROOT, "config.json"))
        )
        self._map_lock = threading.Lock()
        self._cache_lock = threading.Lock()

        self._g: dict[str, Any] = {"mu": None, "ms": 2.6544}
        self._catalog_cache: tuple[float | None, dict[str, Any] | None] = (None, None)
        self._loc_cfg_cache: tuple[str | None, float | None, dict[str, Any] | None] = (
            None,
            None,
            None,
        )
        self._map_cfg_cache: tuple[str | None, float | None, dict[str, Any] | None] = (
            None,
            None,
            None,
        )

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
        """Map names from catalog.json, or downloaded *_map.png as a fallback."""
        if not os.path.isdir(self.data_maps_dir):
            return []
        maps: set[str] = set(self._catalog().get("maps", {}).keys())
        if not maps:
            maps = {
                f[: -len("_map.png")]
                for f in os.listdir(self.data_maps_dir)
                if f.endswith("_map.png")
            }
        return sorted(maps)

    def full_map_size(self, name: str | None = None) -> tuple[int, int] | None:
        """Native (w, h) size of the full map from the catalog or the PNG header."""
        target_name = name or self._cur_name
        entry = self._catalog().get("maps", {}).get(target_name) or {}
        size = entry.get("size")
        if size and len(size) == 2:
            return (int(size[0]), int(size[1]))

        path = self.full_path(target_name)
        try:
            with open(path, "rb") as f:
                head = f.read(24)
        except OSError:
            return None
        if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
            return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
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

    def set_config_path(self, path: str | None) -> None:
        """Point the store at another config file (e.g. a --config path)."""
        new_path = str(resolve_config_path(path or os.path.join(ROOT, "config.json")))
        if new_path == self._config_path:
            return
        self._config_path = new_path
        self._loc_cfg_cache = (None, None, None)
        self._map_cfg_cache = (None, None, None)

    def config_path(self) -> str:
        """Absolute path of the config file this store reads settings from."""
        return self._config_path

    def loc_cfg(self) -> dict[str, Any]:
        """'locator' block of the active config file, cached by path + mtime."""
        path = self._config_path
        mtime = None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            pass
        cache = self._loc_cfg_cache
        if cache[0] == path and cache[1] == mtime and cache[2] is not None:
            return cache[2]

        try:
            cfg = AppConfig.load(path).locator.model_dump()
        except Exception as exc:
            # Never fail silently here: a silent fallback to defaults looks
            # exactly like "the saved config was ignored" from the UI.
            logger.warning("[map_store] locator config unreadable (%s): using defaults", exc)
            cfg = LocatorConfig().model_dump()
        self._loc_cfg_cache = (path, mtime, cfg)
        return cfg

    def map_cfg(self) -> dict[str, Any]:
        """'map' block of the active config file, cached by path + mtime."""
        path = self._config_path
        mtime = None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            pass
        cache = self._map_cfg_cache
        if cache[0] == path and cache[1] == mtime and cache[2] is not None:
            return cache[2]

        try:
            cfg = AppConfig.load(path).map.model_dump()
        except Exception as exc:
            logger.warning("[map_store] map config unreadable (%s): using defaults", exc)
            cfg = MapConfig().model_dump()
        self._map_cfg_cache = (path, mtime, cfg)
        return cfg

    def map_signature(self, name: str | None = None) -> str:
        """Map identity used in cache checks.

        A machine holding the source PNG tracks its size+mtime (maintainer
        flow); a distributed install without the PNG uses the build source
        recorded in catalog.json, so the shipped caches stay valid.
        """
        try:
            st = os.stat(self.full_path(name))
        except OSError:
            entry = self._catalog().get("maps", {}).get(name or self._cur_name) or {}
            source = entry.get("source")
            return str(source) if source else "missing"
        return f"{st.st_size}-{st.st_mtime_ns}"

    def gray_sig(self, name: str | None = None) -> str:
        """Cache key: gray palette + minimap scale + map file identity.

        Including the map file itself means replacing the PNG invalidates the
        mu/preview/feature-index caches even when the palette did not change.
        """
        c = self.map_cfg()
        return (
            f"{c['gray_conv']}-{float(c['gray_gamma']):.3f}"
            f"|ms={self.mini_scale():.4f}|{self.map_signature(name)}"
        )

    def _read_map_gray(self, name: str | None = None) -> np.ndarray | None:
        """Map PNG -> gray in the configured palette at native/2 resolution.

        Reads the color texture (not IMREAD_GRAYSCALE), so `gray_conv` and
        `gray_gamma` actually take effect: the game's minimap palette is the
        'desat' conversion (R*0.3 + G*0.59 + B*0.11).
        """
        bgr = cv2.imread(self.full_path(name), cv2.IMREAD_REDUCED_COLOR_2)
        if bgr is None:
            return None
        cfg = self.map_cfg()
        return bgr_to_gray(
            bgr,
            conv=str(cfg.get("gray_conv", "luma")),
            gamma=float(cfg.get("gray_gamma", 1.0)),
        )

    def gray_sig_on_disk(self, name: str | None = None) -> str | None:
        """Gray signature the on-disk mu cache was built with, or None."""
        try:
            with open(self.cache_path(name or self._cur_name, "gray.txt"), encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def mini_scale(self) -> float:
        """Pixel scale of the minimap window relative to the map (from config)."""
        cfg_scale = self.map_cfg().get("mini_scale")
        if cfg_scale is not None:
            return float(cfg_scale)
        return float(self._g.get("ms", 2.6544))

    def _catalog(self) -> dict[str, Any]:
        """data/maps/catalog.json, cached by file mtime."""
        path = os.path.join(self.data_maps_dir, "catalog.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        cache = self._catalog_cache
        if cache[0] == mtime and cache[1] is not None:
            return cache[1]
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        self._catalog_cache = (mtime, data)
        return data

    def m_per_px(self, name: str | None = None) -> float:
        """Physical map scale (meters per native map px); 0 when unknown."""
        entry = self._catalog().get("maps", {}).get(name or self._cur_name) or {}
        try:
            value = float(entry.get("m_per_px", 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return value if value > 0.0 else 0.0

    def px_per_m(self, name: str | None = None) -> float:
        """Known map scale in native map px per meter; 0 when unknown."""
        m_per_px = self.m_per_px(name)
        return 1.0 / m_per_px if m_per_px > 0.0 else 0.0

    def catalog_artifacts(self, name: str | None = None) -> dict[str, dict[str, Any]]:
        """Artifact metadata (size, sha256) the catalog lists for a map."""
        entry = self._catalog().get("maps", {}).get(name or self._cur_name) or {}
        arts = entry.get("artifacts")
        return dict(arts) if isinstance(arts, dict) else {}

    def get_index(self, kind: str = "sift") -> Any:
        """Load the cached feature index of `kind` for the active map.

        Each kind is cached separately (SIFT and ORB indexes coexist); every
        cache entry is checked against the palette signature, storage format
        and kind tag before use.
        """
        cache = self._g.get("idx")
        if not isinstance(cache, dict):
            cache = {}
            self._g["idx"] = cache
        idx = cache.get(kind)
        if idx is not None and getattr(idx, "name", None) == self._cur_name:
            return idx
        with self._cache_lock:
            idx = cache.get(kind)
            if idx is not None and getattr(idx, "name", None) == self._cur_name:
                return idx
            try:
                idx = load_index(self._cur_name, kind)
            except Exception:
                idx = None
            if idx is not None:
                got = getattr(idx, "gray_sig", None)
                want = self.gray_sig()
                if got is not None and got != want:
                    logger.warning(
                        "[map_store] feature index gray mismatch: built=%s, current=%s "
                        "— download the map assets or rebuild from the source PNG",
                        got,
                        want,
                    )
                    idx = None
            if idx is not None and getattr(idx, "fmt", None) != _INDEX_FMT:
                logger.warning(
                    "[map_store] feature index format mismatch (fmt=%s, want=%s) "
                    "— download the map assets or rebuild from the source PNG",
                    getattr(idx, "fmt", None),
                    _INDEX_FMT,
                )
                idx = None
            if idx is not None and getattr(idx, "norm", None) != _INDEX_NORM:
                logger.warning(
                    "[map_store] feature index build mismatch (norm=%s, want=%s) "
                    "— download the map assets or rebuild from the source PNG",
                    getattr(idx, "norm", None),
                    _INDEX_NORM,
                )
                idx = None
            if idx is not None and getattr(idx, "kind", "sift") != kind:
                logger.warning(
                    "[map_store] feature index kind mismatch (kind=%s, want=%s) "
                    "— rebuild the index",
                    getattr(idx, "kind", None),
                    kind,
                )
                idx = None
            cache[kind] = idx
        return cache.get(kind)

    # ---------- Previews & Map caches ----------

    def build_previews(self, full: np.ndarray) -> None:
        """Build and cache the level ladder from a full-resolution gray map.

        Levels larger than the source are skipped: the shipped top preview of
        a 16384 map is the native 8192 gray, and upscaling it to 16384 would
        only write an interpolated 268 MB copy per map.
        """
        for n in PREVIEW_SIZES:
            if n > full.shape[0]:
                continue
            img = (
                full
                if n == full.shape[0]
                else cv2.resize(full, (n, n), interpolation=cv2.INTER_AREA)
            )
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
        """Ensure the preview pyramid: derive missing smaller levels locally.

        Only the largest level ships with the map (release size); the rest are
        regenerated from it on first use and cached in data/maps. With no
        preview at all, a maintainer machine rebuilds the ladder from the
        source PNG instead.
        """
        name = self._cur_name
        if all(os.path.exists(self.preview_path(name, n)) for n in PREVIEW_SIZES):
            return True
        with self._cache_lock:
            if all(os.path.exists(self.preview_path(name, n)) for n in PREVIEW_SIZES):
                return True
            levels = [n for n in PREVIEW_SIZES if os.path.exists(self.preview_path(name, n))]
            if not levels:
                full = self._read_map_gray(name)
                if full is None:
                    return False
                self.build_previews(full)
                return True
            top = max(levels)
            try:
                img = np.load(self.preview_path(name, top))
            except Exception:
                return False
            for n in [s for s in PREVIEW_SIZES if s < top][::-1]:
                img = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
                if os.path.exists(self.preview_path(name, n)):
                    continue
                try:
                    np.save(self.preview_path(name, n), img)
                except OSError as exc:
                    logger.warning("[map_store] failed to save preview %d: %s", n, exc)
            return True

    def load_global_map(self) -> np.ndarray:
        """Load or build the global mu map at minimap scale."""
        if self._g["mu"] is not None:
            return self._g["mu"]

        name = self._cur_name
        cache_mu = self.cache_path(name, "mu.npy")
        sig = self.gray_sig()
        cached_sig = self.gray_sig_on_disk(name)

        stale_sig = cached_sig != sig
        if os.path.exists(cache_mu) and not stale_sig:
            loaded_mu: np.ndarray = np.load(cache_mu)
            self._g["mu"] = loaded_mu
            self.ensure_previews()
            return loaded_mu

        if stale_sig:
            logger.info(
                "[map_store] map cache stale (%s -> %s): rebuilding caches",
                cached_sig or "missing",
                sig,
            )

        with self._cache_lock:
            cached_mu = self._g.get("mu")
            if cached_mu is not None and isinstance(cached_mu, np.ndarray):
                return cached_mu
            gray = self._read_map_gray(name)
            if gray is None:
                raise OSError(
                    f"source map not readable: {self.full_path(name)} "
                    "(rebuilding needs the source PNG; "
                    "download the prebuilt assets with tools/download_map.py)"
                )
            h, _ = gray.shape[:2]
            ms = self.mini_scale()
            side = int(round(h * 2 / ms))
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
        """Rebuild mu.npy, preview mipmaps, and SIFT feature index for specified map.

        The active map is restored afterwards (rebuilding a background map must
        not switch the store), and the in-memory feature index of the rebuilt
        map is dropped so the freshly written npz is picked up on the next
        frame instead of surviving until a restart.
        """
        with self._cache_lock:
            previous_name = self._cur_name
            self.set_map(name)
            try:
                if progress_cb:
                    progress_cb("Step 1/3: Reading full map and generating mu...")
                cache_mu = self.cache_path(name, "mu.npy")
                gray = self._read_map_gray(name)
                if gray is None:
                    raise OSError(
                        f"source map not readable: {self.full_path(name)} "
                        "(rebuilding needs the source PNG; "
                        "download the prebuilt assets with tools/download_map.py)"
                    )
                h, _ = gray.shape[:2]

                ms = self.mini_scale()
                side = int(round(h * 2 / ms))
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
                self._g.pop("idx", None)
            finally:
                self.set_map(previous_name)

            if progress_cb:
                progress_cb("Done: Map cache and SIFT index ready!")
            return True
