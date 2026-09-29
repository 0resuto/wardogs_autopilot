#!/usr/bin/env python3
"""CLI for the WARDOGS map asset archives (see autopilot.vision.asset_sync).

Lists maps and their local status, downloads the per-map release archive
(<map>.zip) from GitHub Releases with archive + per-file sha256 verification,
extracts it into data/maps/, or verifies the local artifact set.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from autopilot.vision.asset_sync import (  # noqa: E402
    download_map,
    format_bytes,
    load_catalog,
    resolve_repo_tag,
    verify_map,
)

DATA_MAPS = os.path.join(ROOT, "data", "maps")


def list_maps(catalog: dict) -> None:
    maps = catalog.get("maps", {})
    print(
        f"\n{'Map Name':<12} {'Display Name':<14} {'Resolution':<14} "
        f"{'Status':<14} {'Download Size'}"
    )
    print("-" * 76)
    for name, info in maps.items():
        disp = info.get("display_name", name)
        size_str = f"{info['size'][0]}x{info['size'][1]}"
        artifacts = info.get("artifacts", {})
        present = [f for f in artifacts if os.path.exists(os.path.join(DATA_MAPS, f))]
        archive = info.get("archive") or {}
        size_disp = format_bytes(
            int(archive.get("size") or 0) or sum(meta["size"] for meta in artifacts.values())
        )
        if artifacts and len(present) == len(artifacts):
            status = "Downloaded"
        elif present:
            status = f"Partial {len(present)}/{len(artifacts)}"
        else:
            status = "Missing"
        print(f"{name:<12} {disp:<14} {size_str:<14} {status:<14} {size_disp}")
    print()


def main() -> None:
    catalog = load_catalog(DATA_MAPS)
    if not catalog:
        catalog_path = os.path.join(DATA_MAPS, "catalog.json")
        print(f"Error: catalog manifest not found at {catalog_path}", file=sys.stderr)
        sys.exit(1)
    default_repo, default_tag = resolve_repo_tag(catalog)

    parser = argparse.ArgumentParser(description="Download game map assets for WARDOGS Autopilot.")
    parser.add_argument(
        "map", nargs="?", help="Map name to download (e.g. zestafona, bakurani, ozeti)"
    )
    parser.add_argument("--all", action="store_true", help="Download all available maps")
    parser.add_argument("--list", action="store_true", help="List maps and their local status")
    parser.add_argument(
        "--repo", default=default_repo, help=f"GitHub repository (default: {default_repo or '-'})"
    )
    parser.add_argument(
        "--tag", default=default_tag, help=f"Release tag (default: {default_tag or '-'})"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files even if they already exist"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify local artifacts against the catalog sha256",
    )

    args = parser.parse_args()

    if args.list or (not args.map and not args.all):
        list_maps(catalog)
        if not args.list:
            print("Usage example: python tools/download_map.py zestafona")
            print("               python tools/download_map.py --all")
        return

    if not args.verify and (not args.repo or not args.tag):
        print(
            "Error: no asset repository/tag in the catalog (set WARDOGS_ASSET_REPO/TAG)",
            file=sys.stderr,
        )
        sys.exit(1)

    targets = list(catalog.get("maps", {}).keys()) if args.all else [args.map]

    success_count = 0
    for target in targets:
        if args.verify:
            ok = verify_map(target, catalog, DATA_MAPS)
            print(f"{'OK' if ok else 'FAILED'}: '{target}'")
        else:
            print(f"\n--- {target} ---")
            ok = download_map(
                target, catalog, args.repo, args.tag, DATA_MAPS, force=args.force, progress=print
            )
            print(f"{'OK' if ok else 'FAILED'}: '{target}'")
        if ok:
            success_count += 1

    if success_count == len(targets):
        print("\nAll requested maps are ready!")
    else:
        print(f"\nCompleted {success_count}/{len(targets)} maps.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
