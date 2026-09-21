#!/usr/bin/env python3
"""Map asset downloader for WARDOGS Autopilot.

Downloads high-resolution game maps from GitHub Releases into data/maps/,
verifies SHA-256 integrity, and optionally generates SIFT index and mipmaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MAPS = os.path.join(ROOT, "data", "maps")
CATALOG_PATH = os.path.join(DATA_MAPS, "catalog.json")


def load_catalog() -> dict:
    if not os.path.exists(CATALOG_PATH):
        print(f"Error: Catalog manifest not found at {CATALOG_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def format_bytes(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def list_maps(catalog: dict) -> None:
    maps = catalog.get("maps", {})
    print(f"\n{'Map Name':<12} {'Display Name':<14} {'Resolution':<14} {'Status':<14} {'File Size'}")
    print("-" * 72)
    for name, info in maps.items():
        disp = info.get("display_name", name)
        size_str = f"{info['size'][0]}x{info['size'][1]}"
        target_file = os.path.join(DATA_MAPS, f"{name}_map.png")
        if os.path.exists(target_file):
            actual_size = os.path.getsize(target_file)
            status = "Downloaded"
            size_disp = format_bytes(actual_size)
        else:
            status = "Missing"
            expected_size = sum(f["size"] for f in info.get("files", {}).values())
            size_disp = format_bytes(expected_size)
        print(f"{name:<12} {disp:<14} {size_str:<14} {status:<14} {size_disp}")
    print()


def download_file(url: str, dest_path: str, expected_size: int | None, expected_sha256: str | None) -> bool:
    tmp_path = dest_path + ".tmp"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"Connecting: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "wardogs-autopilot-downloader"})

    try:
        with urllib.request.urlopen(req) as resp:
            total_bytes = int(resp.headers.get("Content-Length") or (expected_size or 0))
            downloaded = 0
            start_time = time.time()
            last_print = 0.0
            hasher = hashlib.sha256() if expected_sha256 else None

            with open(tmp_path, "wb") as out_file:
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print > 0.25 or downloaded == total_bytes:
                        elapsed = max(now - start_time, 0.001)
                        speed = downloaded / elapsed
                        if total_bytes > 0:
                            pct = (downloaded / total_bytes) * 100.0
                            eta = (total_bytes - downloaded) / speed if speed > 0 else 0
                            bar_len = 30
                            filled = int(bar_len * downloaded // total_bytes)
                            bar = "=" * filled + "-" * (bar_len - filled)
                            msg = (
                                f"\r[{bar}] {pct:5.1f}% | {format_bytes(downloaded)}/{format_bytes(total_bytes)} "
                                f"| {format_bytes(speed)}/s | ETA {eta:4.0f}s"
                            )
                        else:
                            msg = f"\rDownloaded {format_bytes(downloaded)} | {format_bytes(speed)}/s"
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                        last_print = now
            print()

            if expected_sha256 and hasher:
                digest = hasher.hexdigest()
                if digest.lower() != expected_sha256.lower():
                    print(f"Checksum mismatch for {dest_path}!", file=sys.stderr)
                    print(f"  Expected: {expected_sha256}", file=sys.stderr)
                    print(f"  Got:      {digest}", file=sys.stderr)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    return False

            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(tmp_path, dest_path)
            return True

    except (urllib.error.URLError, OSError) as exc:
        print(f"\nDownload error: {exc}", file=sys.stderr)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def download_map(name: str, catalog: dict, repo: str, tag: str, force: bool = False, rebuild: bool = False) -> bool:
    maps = catalog.get("maps", {})
    if name not in maps:
        print(f"Error: Unknown map '{name}'. Available maps: {list(maps.keys())}", file=sys.stderr)
        return False

    info = maps[name]
    files = info.get("files", {})
    base_url = f"https://github.com/{repo}/releases/download/{tag}"

    print(f"\n--- Downloading Map: {info.get('display_name', name)} ({name}) ---")
    all_ok = True
    for filename, meta in files.items():
        dest = os.path.join(DATA_MAPS, filename)
        if os.path.exists(dest) and not force:
            print(f"File already exists: {dest} (use --force to re-download)")
            continue

        url = f"{base_url}/{filename}"
        ok = download_file(url, dest, meta.get("size"), meta.get("sha256"))
        if not ok:
            all_ok = False
            break

    if all_ok and rebuild:
        print(f"\nBuilding local caches & SIFT index for '{name}'...")
        if os.path.join(ROOT, "src") not in sys.path:
            sys.path.insert(0, os.path.join(ROOT, "src"))
        from autopilot.vision.map_store import MapStore

        store = MapStore()
        store.rebuild_map_cache(name, progress_cb=lambda msg: print(f"  {msg}"))
        print(f"Caches for '{name}' successfully built!")

    return all_ok


def main() -> None:
    catalog = load_catalog()
    default_repo = os.environ.get("WARDOGS_ASSET_REPO", catalog.get("repo", "owner/wardogs-autopilot"))
    default_tag = os.environ.get("WARDOGS_ASSET_TAG", catalog.get("tag", "v1.0.0"))

    parser = argparse.ArgumentParser(description="Download game map assets for WARDOGS Autopilot.")
    parser.add_argument("map", nargs="?", help="Map name to download (e.g. zestafona, bakurani, ozeti)")
    parser.add_argument("--all", action="store_true", help="Download all available maps")
    parser.add_argument("--list", action="store_true", help="List maps and their local status")
    parser.add_argument("--repo", default=default_repo, help=f"GitHub repository (default: {default_repo})")
    parser.add_argument("--tag", default=default_tag, help=f"Release tag (default: {default_tag})")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist")
    parser.add_argument("--rebuild", action="store_true", help="Automatically build mipmaps & SIFT index after download")

    args = parser.parse_args()

    if args.list or (not args.map and not args.all):
        list_maps(catalog)
        if not args.list:
            print("Usage example: python tools/download_map.py zestafona")
            print("               python tools/download_map.py --all")
        return

    targets = list(catalog.get("maps", {}).keys()) if args.all else [args.map]

    success_count = 0
    for target in targets:
        if download_map(target, catalog, args.repo, args.tag, force=args.force, rebuild=args.rebuild):
            success_count += 1

    if success_count == len(targets):
        print("\nAll requested maps are ready!")
    else:
        print(f"\nCompleted {success_count}/{len(targets)} maps.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
