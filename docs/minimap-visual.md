# Minimap visual processing - measured spec check

The game build ships a minimap material spec (`wardogs-minimap-visual.md`,
from the cooked assets: `MI_Map_VirtualTexture` / `M_UI_Map`). It describes
desaturate -> levels -> digital grunge -> grid. This note records what the
*actually captured* minimap frames show and what the app uses for the index.

## Measurements (real capture vs the light source `data/maps/*_map.png`)

Method: locate a captured frame, align the source map to it with a RANSAC
similarity (SIFT, 10 inliers), then compare pixel correlation (NCC) and, more
importantly, descriptor matching (SIFT + ratio + RANSAC inliers on the same
window).

| Map processing | Pixel NCC | Matches | RANSAC inliers |
| --- | --- | --- | --- |
| spec desaturate `R*0.3+G*0.59+B*0.11` | 0.896 | 162 | **149** |
| luma (BT.601, old pipeline) | 0.850 | 158 | 147 |
| spec full pipeline incl. levels | -0.706 | 7 | 2 |
| desaturate + measured tone `0.384*x^0.798` | 0.896 | 13 | 3 |
| stale cache from the previous map file | 0.678 | 36 | 22 |

Findings:

1. The captured minimap is a **monotone function of the desaturated source**:
   the bin medians give `capture ~= 0.384 * desat^0.798` with a residual of
   0.010 (IQR <= 0.04), i.e. desaturate + a darkening tone curve.
2. The spec `levels` step inverts the image (low > high); the capture is not
   inverted relative to the source, and descriptors collapse to 2 inliers.
   The levels material most likely belongs to the fullscreen tactical map (M),
   not to the corner minimap we capture.
3. Building the index from the tone-matched (dark) raster is *worse*: SIFT
   finds 217 keypoints instead of 6000 and only 3 inliers. A full-contrast
   index plus a normalized query is the right combination.
4. Rebuilding the caches from the current source raised matching from 22 to
   147 inliers on the reference frame.

## What the app does

- `map.gray_conv = "desat"` (`preprocessing.bgr_to_gray` palette) - the
  minimap material luminance factors.
- No levels / grunge / grid and no tone curve in the index; the query side
  keeps `shadow_fill_norm` (percentile stretch, no inversion).
- Cache keys include the map file identity (size + mtime) and the palette, so
  replacing `data/maps/<name>_map.png` invalidates `mu`, previews and the
  feature index automatically (`gray_sig`,
  `MapStore.map_signature`).
- The `U128` dtype stores the full signature in `*_feat.npz` (a `U32` field
  silently truncated it before).
