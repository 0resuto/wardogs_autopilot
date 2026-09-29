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

## Map scale is data

The physical map scale is a catalog constant, not a runtime guess:

| Map | size | m_per_px | px_per_m |
| --- | --- | --- | --- |
| zestafona, ozeti | 32768 | 0.498046875 (16320 m / 32768) | 2.008 |
| bakurani | 16384 | 0.99609375 (16320 m / 16384) | 1.004 |

`data/maps/catalog.json` carries `m_per_px`; `MapStore.px_per_m(name)` feeds
`SpeedController` (injected by the studio), so corridors, corner radii,
braking distances and the cruise baseline are correct from the first frame -
the old "max observed speed equals the configured cap" estimate survives only
for maps without a catalog scale.

The speedometer OCR ratio (`mv / (kmh/3.6)`) validates the constant: a
persistent deviation beyond 15% is logged and the measured value adopted once,
because it is a direct measurement and the catalog may describe an older map
revision. An OCR estimate is only trusted after 25 samples with a tight
inter-quartile spread (IQR/median <= 15%, which rejects acceleration lag, gear
shifts and localization jitter) and plausible bounds (0.3..10 px/m); outside
those gates it is ignored with a warning. Validate in game with
`python tools/calibrate_vehicle.py output/nav_dbg_*.jsonl`.

## Distributed artifacts and the cache signature

`data/maps/catalog.json` lists the derived artifact set of every map
(`<map>_feat.npz`, `<map>_mu.npy`, the top `<map>_preview_<n>.npy` level,
`<map>_gray.txt`) with sizes and sha256, plus `source` - the identity of the
source PNG the caches were built from - and `archive` (`<map>.zip` with its
own size/sha256). Each map is distributed as that one archive;
`tools/download_map.py` checks the archive hash, extracts the members and
verifies every file against the catalog before replacing the local cache.
Only the top preview ships (16384 for 32768 maps, the native 8192 gray for
16384 maps); the smaller pyramid levels are derived from it locally on first
open (`MapStore.ensure_previews`). The original map PNG is not distributed;
only a maintainer machine holding it can rebuild the caches
(`tools/build_map_assets.py --rebuild`). The runtime signature is
`<palette>|<mini_scale>|<source>`; it is compared against `gray.txt` and the
signature embedded in the feature index, so a different palette, mini scale or
source invalidates the caches. `tools/download_map.py` fetches the artifact set
and verifies the checksums (`--verify`); the GitHub release assets must be
re-uploaded to match the catalog before downloads verify cleanly again.

The SIFT index stores its integer descriptors as uint8 inside a compressed npz
(about 5x smaller than the old float32 files: 3.05 GB -> 588 MB across the
three maps); indexes without the `u8z` format marker are rejected with a
re-download/rebuild hint. The live query is cast to uint8 in
`MapLocator._detect` because BFMatcher requires matching descriptor types.

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
