"""Download and install the per-map release archives (CLI and studio share this).

`data/maps/catalog.json` points at one `<map>.zip` per map in a GitHub release.
This module downloads the archive, verifies its sha256, then extracts every
artifact into data/maps verifying the per-file sha256 from the catalog; members
land in a staging directory first and are moved into place only after all of
them verified. A catalog without archives falls back to downloading the loose
artifacts directly. `WARDOGS_ASSET_REPO` / `WARDOGS_ASSET_TAG` override the
catalog values.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from typing import Any

from ..common.log import get_logger

logger = get_logger("asset_sync")

ProgressCb = Callable[[str], None]


def load_catalog(data_maps_dir: str) -> dict[str, Any]:
    """Read data/maps/catalog.json ({} when absent or malformed)."""
    path = os.path.join(data_maps_dir, "catalog.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_repo_tag(catalog: dict[str, Any]) -> tuple[str, str]:
    """Asset repository and release tag, with environment overrides."""
    repo = os.environ.get("WARDOGS_ASSET_REPO") or str(catalog.get("repo", ""))
    tag = os.environ.get("WARDOGS_ASSET_TAG") or str(catalog.get("tag", ""))
    return repo, tag


def format_bytes(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    dest_path: str,
    expected_size: int | None,
    expected_sha256: str | None,
    progress: ProgressCb | None = None,
) -> bool:
    """Stream `url` to `dest_path` (tmp + replace) verifying the sha256."""
    tmp_path = dest_path + ".tmp"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info("[asset_sync] downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "wardogs-autopilot-downloader"})
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length") or (expected_size or 0))
            downloaded = 0
            start_time = time.time()
            last_print = 0.0
            hasher = hashlib.sha256()
            with open(tmp_path, "wb") as out_file:
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if progress and (now - last_print > 0.25 or downloaded == total):
                        last_print = now
                        speed = downloaded / max(now - start_time, 0.001)
                        if total > 0:
                            progress(
                                f"Downloading {format_bytes(downloaded)} / {format_bytes(total)}"
                                f" ({downloaded / total * 100:.0f}%, {format_bytes(speed)}/s)"
                            )
                        else:
                            progress(f"Downloading {format_bytes(downloaded)}")
            if expected_sha256 and hasher.hexdigest().lower() != str(expected_sha256).lower():
                logger.error("[asset_sync] checksum mismatch for %s", dest_path)
                print(f"Checksum mismatch for {dest_path}", file=sys.stderr)
                os.remove(tmp_path)
                return False
        os.replace(tmp_path, dest_path)
        return True
    except (urllib.error.URLError, OSError) as exc:
        logger.error("[asset_sync] download failed: %s", exc)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def extract_archive(
    name: str,
    info: dict[str, Any],
    zip_path: str,
    data_maps_dir: str,
    progress: ProgressCb | None = None,
) -> bool:
    """Extract the artifact set from the archive, verifying every file.

    Members land in a staging directory first and are moved into data/maps
    only after all of them verified, so a corrupt archive cannot leave a
    half-replaced cache behind.
    """
    artifacts = info.get("artifacts", {})
    staging = os.path.join(data_maps_dir, f".extract_{name}")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            for filename, meta in artifacts.items():
                if os.path.basename(filename) != filename:
                    raise ValueError(f"unsafe artifact name: {filename}")
                if filename not in names:
                    raise ValueError(f"archive is missing {filename}")
                if progress:
                    progress(f"Extracting {filename}")
                out_path = os.path.join(staging, filename)
                hasher = hashlib.sha256()
                with zf.open(filename) as src, open(out_path, "wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        hasher.update(chunk)
                        dst.write(chunk)
                if hasher.hexdigest() != str(meta.get("sha256", "")).lower():
                    raise ValueError(f"checksum mismatch after extraction: {filename}")
        for filename in artifacts:
            os.replace(os.path.join(staging, filename), os.path.join(data_maps_dir, filename))
    except Exception as exc:  # noqa: BLE001 — any bad archive is a failed download
        logger.error("[asset_sync] extract error for %s: %s", name, exc)
        if progress:
            progress(f"Extract error: {exc}")
        return False
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if progress:
        progress(f"Extracted {len(artifacts)} artifacts")
    return True


def download_map(
    name: str,
    catalog: dict[str, Any],
    repo: str,
    tag: str,
    data_maps_dir: str,
    force: bool = False,
    progress: ProgressCb | None = None,
) -> bool:
    """Download and install one map (archive preferred, loose files fallback)."""
    maps = catalog.get("maps", {})
    if name not in maps:
        logger.error("[asset_sync] unknown map '%s'", name)
        return False
    info = maps[name]
    artifacts = info.get("artifacts", {})
    if not artifacts:
        logger.error("[asset_sync] map '%s' has no artifacts in the catalog", name)
        return False
    base_url = f"https://github.com/{repo}/releases/download/{tag}"

    archive = info.get("archive") or {}
    zip_name = str(archive.get("name", ""))
    if zip_name:
        zip_path = os.path.join(data_maps_dir, zip_name)
        present = all(os.path.exists(os.path.join(data_maps_dir, f)) for f in artifacts)
        if present and not force:
            if progress:
                progress(f"All artifacts already present for '{name}'")
            return True
        url = f"{base_url}/{zip_name}"
        if not download_file(url, zip_path, archive.get("size"), archive.get("sha256"), progress):
            return False
        try:
            return extract_archive(name, info, zip_path, data_maps_dir, progress)
        finally:
            try:
                os.remove(zip_path)
            except OSError:
                pass

    # Catalog without archives: download the loose artifacts (custom or old catalogs)
    for filename, meta in artifacts.items():
        dest = os.path.join(data_maps_dir, filename)
        if os.path.exists(dest) and not force:
            if progress:
                progress(f"File already exists: {filename}")
            continue
        url = f"{base_url}/{filename}"
        if not download_file(url, dest, meta.get("size"), meta.get("sha256"), progress):
            return False
    return True


def verify_map(name: str, catalog: dict[str, Any], data_maps_dir: str) -> bool:
    """Check every local artifact of `name` against the catalog sha256."""
    info = catalog.get("maps", {}).get(name)
    if info is None:
        logger.error("[asset_sync] unknown map '%s'", name)
        return False
    artifacts = info.get("artifacts", {})
    missing: list[str] = []
    corrupt: list[str] = []
    for filename, meta in artifacts.items():
        path = os.path.join(data_maps_dir, filename)
        if not os.path.exists(path):
            missing.append(filename)
            continue
        if sha256_file(path) != str(meta.get("sha256", "")).lower():
            corrupt.append(filename)
    if missing:
        logger.error("[asset_sync] '%s' is missing: %s", name, ", ".join(missing))
        return False
    if corrupt:
        logger.error("[asset_sync] '%s' is corrupt: %s", name, ", ".join(corrupt))
        return False
    return True
