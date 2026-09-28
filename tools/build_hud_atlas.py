"""Build the Barlow digit atlas for the HUD speed OCR.

Renders digits 0-9 with data/hud/Barlow-Regular.ttf at a high resolution,
trims every glyph to its ink box, fits it into the canonical cell used by
autopilot.vision.hud_speed and writes data/hud/barlow_digits.png plus
data/hud/barlow_atlas.json.

Pillow is needed only to (re)build the atlas:

    uv run --with pillow python tools/build_hud_atlas.py

The atlas is committed, so the application and tests never need Pillow.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.vision.hud_speed import (  # noqa: E402
    ATLAS_LABELS,
    ATLAS_META_PATH,
    ATLAS_PATH,
    GLYPH_H,
    GLYPH_W,
    fit_glyph,
)

FONT_PATH = os.path.join(ROOT, "data", "hud", "Barlow-Regular.ttf")
RENDER_SIZE = 160


def build() -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - dev-only tool
        raise SystemExit(
            "Pillow is required to build the atlas: "
            "uv run --with pillow python tools/build_hud_atlas.py"
        ) from exc

    if not os.path.exists(FONT_PATH):
        raise SystemExit(f"font not found: {FONT_PATH}")

    font = ImageFont.truetype(FONT_PATH, RENDER_SIZE)
    cells: list[np.ndarray] = []
    for label in ATLAS_LABELS:
        canvas = Image.new("L", (RENDER_SIZE * 2, RENDER_SIZE * 2), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((RENDER_SIZE // 2, RENDER_SIZE // 2), label, fill=255, font=font)
        ink = canvas.getbbox()
        if ink is None:
            raise SystemExit(f"no ink rendered for {label!r}")
        glyph = np.asarray(canvas.crop(ink), dtype=np.uint8)
        cells.append(fit_glyph(glyph, GLYPH_W, GLYPH_H))

    strip = np.concatenate(cells, axis=1)
    if strip.shape != (GLYPH_H, GLYPH_W * len(ATLAS_LABELS)):
        raise SystemExit(f"unexpected atlas shape {strip.shape}")

    os.makedirs(os.path.dirname(ATLAS_PATH), exist_ok=True)
    import cv2

    if not cv2.imwrite(ATLAS_PATH, strip):
        raise SystemExit(f"failed to write {ATLAS_PATH}")
    meta = {
        "source": os.path.basename(FONT_PATH),
        "render_size": RENDER_SIZE,
        "glyph_w": GLYPH_W,
        "glyph_h": GLYPH_H,
        "labels": ATLAS_LABELS,
    }
    with open(ATLAS_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return ATLAS_PATH


def main() -> None:
    path = build()
    print(f"wrote {path} and {ATLAS_META_PATH}")


if __name__ == "__main__":
    main()
