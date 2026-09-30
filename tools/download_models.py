#!/usr/bin/env python3
"""Download the ONNX models the XFeat engine needs (data/models/).

The model files are not vendored (data/models is git-ignored): this tool
fetches them from the official kornia/xfeat Hugging Face repository, checks
their sha256, and atomically installs them into data/models/.

Usage:
    python tools/download_models.py            # fetch what is missing/stale
    python tools/download_models.py --verify   # check the local files only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "data", "models")

_REPO = "https://huggingface.co/kornia/xfeat/resolve/main"
MODELS: dict[str, str] = {
    "xfeat_backbone.onnx": "86d7d549b380405f208933efb5202e1584d9762f3a72e06e7ed81ca1436972e0",
    "xfeat_backbone.onnx.data": "d4498528d37bf7c737cce9c135f9b0340d828bab7dc808339e50553ac8c1b7d9",
    # tiny MatMul+ArgMax graph built with tools/build_match_graph.py
    "xfeat_match.onnx": "81129ff1f7ba02bcec9032b17337a8ff024d2a4c94d2f36b4d1439041fae1415",
}


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(name: str, digest: str) -> None:
    target = os.path.join(MODELS_DIR, name)
    os.makedirs(MODELS_DIR, exist_ok=True)
    url = f"{_REPO}/{name}"
    print(f"downloading {name} ...")
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=MODELS_DIR, suffix=".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp.name, "wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        got = sha256_file(tmp.name)
        if got != digest:
            raise RuntimeError(f"sha256 mismatch for {name}: got {got}, want {digest}")
        os.replace(tmp.name, target)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    print(f"  ok: {name} ({os.path.getsize(target)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="only verify the local files")
    args = parser.parse_args()

    missing = []
    bad = []
    for name, digest in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        if sha256_file(path) != digest:
            bad.append(name)

    if args.verify:
        for name in missing:
            print(f"missing: {name}")
        for name in bad:
            print(f"checksum mismatch: {name}")
        ok = not missing and not bad
        print("models: OK" if ok else "models: NOT READY")
        sys.exit(0 if ok else 1)

    for name in missing + bad:
        download(name, MODELS[name])
    if not missing and not bad:
        print("models: already present and valid")


if __name__ == "__main__":
    main()
