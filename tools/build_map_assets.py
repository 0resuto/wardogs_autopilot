#!/usr/bin/env python3
"""Refresh data/maps/catalog.json from the derived map artifacts.

The app never needs the original map PNG at runtime: localization uses
<map>_feat.npz, the global match uses <map>_mu.npy, the Map tab renders the
top <map>_preview_*.npy level (smaller levels are derived locally from it) and
<map>_gray.txt carries the cache signature. This tool records that artifact
set (size + sha256) plus the build source in catalog.json, so
tools/download_map.py can distribute derived files only.

Usage:
    python tools/build_map_assets.py --all
    python tools/build_map_assets.py zestafona --rebuild       # needs <map>_map.png
    python tools/build_map_assets.py --all --export dist-assets

Without --rebuild the catalog is refreshed from the artifacts already on disk
(that is what a release needs). --rebuild first regenerates mu/previews/SIFT
index from the source PNG via MapStore.rebuild_map_cache (maintainer flow).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MAPS = os.path.join(ROOT, "data", "maps")
CATALOG_PATH = os.path.join(DATA_MAPS, "catalog.json")

BASE_ARTIFACT_SUFFIXES = ("_feat.npz", "_mu.npy", "_gray.txt")
PREVIEW_TOP_MAX = 16384


def artifact_suffixes(entry: dict) -> tuple[str, ...]:
    """Artifacts to ship for a map: only the top preview level.

    The smaller pyramid levels are derived locally from it on first use
    (MapStore.ensure_previews), so a 32768 map ships 16384 and a 16384 map
    ships the native 8192 gray.
    """
    size = entry.get("size") or (32768, 32768)
    top = min(PREVIEW_TOP_MAX, int(size[0]) // 2)
    return BASE_ARTIFACT_SUFFIXES + (f"_preview_{top}.npy",)


CATALOG_NOTES = [
    "Derived artifacts only: the original map PNG is not distributed. A machine "
    "holding the source PNG refreshes this catalog with tools/build_map_assets.py "
    "(--rebuild regenerates the caches first).",
    "Each map ships as a single <map>.zip (npz members stored uncompressed, "
    "raw npy/txt members deflated); tools/download_map.py downloads the archive, "
    "checks its sha256, then extracts every artifact and verifies the per-file "
    "sha256 before replacing the local files.",
    "Only the top preview level ships per map (16384 for 32768 maps, the native "
    "8192 gray for 16384 maps); the smaller pyramid levels are derived locally "
    "from it on first use (MapStore.ensure_previews).",
    "m_per_px is the physical map scale from the game's minimap texture spec: "
    "every map is 16320 m across (0.498046875 m/px at 32768, 0.99609375 m/px at "
    "16384).",
    "'source' is the identity (size-mtime) of the PNG the artifacts were built "
    "from; the runtime cache signature is <palette>|<mini_scale>|<source>, so a "
    "different palette or mini_scale makes the caches stale (a rebuild requires "
    "the source PNG).",
    "Validate the scale in game with: python tools/calibrate_vehicle.py "
    "output/nav_dbg_*.jsonl (px_per_m report).",
    "Release assets must be re-uploaded to match these sha256 values before "
    "tools/download_map.py can verify them.",
]


def format_bytes(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def load_catalog() -> dict:
    if not os.path.exists(CATALOG_PATH):
        print(f"Error: catalog manifest not found at {CATALOG_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_source(name: str) -> str | None:
    """Two-part source identity (size-mtime) recorded in <name>_gray.txt."""
    path = os.path.join(DATA_MAPS, f"{name}_gray.txt")
    try:
        with open(path, encoding="utf-8") as f:
            sig = f.read().strip()
    except OSError:
        return None
    parts = sig.split("|")
    if len(parts) < 3 or not parts[2]:
        return None
    return parts[2]


def refresh_entry(catalog: dict, name: str) -> tuple[int, int]:
    """Write the artifact set of `name` into the catalog; returns (count, bytes)."""
    entry = catalog.setdefault("maps", {}).setdefault(name, {})
    entry.pop("files", None)
    source = read_source(name)
    if source is None:
        print(
            f"[{name}] {name}_gray.txt missing or malformed — rebuild the caches first",
            file=sys.stderr,
        )
        sys.exit(1)

    artifacts: dict[str, dict] = {}
    total_bytes = 0
    for suffix in artifact_suffixes(entry):
        fname = f"{name}{suffix}"
        path = os.path.join(DATA_MAPS, fname)
        if not os.path.exists(path):
            print(f"[{name}] missing artifact: {fname}", file=sys.stderr)
            sys.exit(1)
        size = os.path.getsize(path)
        digest = sha256_file(path)
        artifacts[fname] = {"size": size, "sha256": digest}
        total_bytes += size
        print(f"  {fname:<28} {format_bytes(size):>10}  {digest[:12]}...")

    entry["source"] = source
    entry["artifacts"] = artifacts
    return len(artifacts), total_bytes


def rebuild_caches(name: str) -> None:
    """Regenerate mu/previews/SIFT index from the source PNG (maintainer flow)."""
    if os.path.join(ROOT, "src") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "src"))
    from autopilot.vision.map_store import MapStore

    store = MapStore()
    store.rebuild_map_cache(name, progress_cb=lambda msg: print(f"  {msg}"))


def export_artifacts(name: str, entry: dict, dest_root: str) -> dict:
    """Write the release archive <map>.zip under dest_root; returns its metadata.

    Compressed npz members are stored as-is (recompressing them is wasted CPU
    for ~0 gain); the raw .npy/.txt members are deflated.
    """
    os.makedirs(dest_root, exist_ok=True)
    zip_path = os.path.join(dest_root, f"{name}.zip")
    suffixes = artifact_suffixes(entry)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for suffix in suffixes:
            fname = f"{name}{suffix}"
            compression = zipfile.ZIP_STORED if fname.endswith(".npz") else zipfile.ZIP_DEFLATED
            zf.write(os.path.join(DATA_MAPS, fname), arcname=fname, compress_type=compression)
    meta = {
        "name": f"{name}.zip",
        "size": os.path.getsize(zip_path),
        "sha256": sha256_file(zip_path),
    }
    print(f"  exported {len(suffixes)} artifacts to {zip_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh data/maps/catalog.json from derived map artifacts."
    )
    parser.add_argument("maps", nargs="*", help="Map names (default: all from the catalog)")
    parser.add_argument("--all", action="store_true", help="Refresh every catalog map")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate caches from the source PNG before refreshing (needs <map>_map.png)",
    )
    parser.add_argument(
        "--export",
        metavar="DIR",
        help="Also build the release archives DIR/<map>.zip and record their hashes",
    )
    args = parser.parse_args()

    catalog = load_catalog()
    targets = list(catalog.get("maps", {}).keys()) if args.all or not args.maps else args.maps

    grand_total = 0
    for name in targets:
        if name not in catalog.get("maps", {}):
            print(
                f"Error: '{name}' is not in the catalog — add an entry "
                "(display_name, size, m_per_px) first",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"--- {name} ---")
        if args.rebuild:
            rebuild_caches(name)
        _count, total_bytes = refresh_entry(catalog, name)
        grand_total += total_bytes
        if args.export:
            catalog["maps"][name]["archive"] = export_artifacts(
                name, catalog["maps"][name], args.export
            )
        print()

    catalog["notes"] = CATALOG_NOTES
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        f.write("\n")
    print(f"Catalog updated: {CATALOG_PATH}")
    print(f"Total artifact size: {format_bytes(grand_total)}")


if __name__ == "__main__":
    main()
