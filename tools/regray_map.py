"""Change the map color->gray conversion and rebuild every derived cache.

The in-game minimap does not gray-scale with OpenCV's BT.601 luma: yellow
patches come out brighter and neutral grays darker than the game's render.
`locator._bgr_to_gray` supports 'luma' (OpenCV default), 'equal' (channel
average) and 'bt709', plus an optional gamma. This tool:

  1. writes the chosen conversion into config.json (map.gray_conv,
     map.gray_gamma),
  2. rebuilds mu/coarse/preview caches of the affected maps (reads the full
     color PNG once, ~3 GB RAM transiently),
  3. rebuilds the SIFT feature index of each affected map (descriptors come
     from the same mu, so they must match).

Usage:
  python tools/regray_map.py --conv equal zestafona
  python tools/regray_map.py --conv equal --gamma 1.15 zestafona bakurani ozeti
  python tools/regray_map.py                    # rebuild with current config
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

CONFIG = os.path.join(ROOT, 'config.json')

CONVS = ('luma', 'equal', 'bt709')


def load_config():
    with open(CONFIG, encoding='utf-8') as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--conv', choices=CONVS, default=None,
                    help='color->gray conversion (default: keep current)')
    ap.add_argument('--gamma', type=float, default=None,
                    help='gamma curve, <1 brightens midtones (default: keep)')
    ap.add_argument('--no-index', action='store_true',
                    help='rebuild map caches only, not the feature index')
    ap.add_argument('maps', nargs='*',
                    help='map names to rebuild (default: config map.name)')
    args = ap.parse_args()

    cfg = load_config()
    mcfg = cfg.setdefault('map', {})
    old_sig = '%s-%.3f' % (mcfg.get('gray_conv', 'luma'),
                           float(mcfg.get('gray_gamma', 1.0)))
    if args.conv:
        mcfg['gray_conv'] = args.conv
    if args.gamma is not None:
        mcfg['gray_gamma'] = args.gamma
    conv = mcfg.get('gray_conv', 'luma')
    gamma = float(mcfg.get('gray_gamma', 1.0))
    new_sig = '%s-%.3f' % (conv, gamma)
    if old_sig != new_sig:
        save_config(cfg)
        print('config.json: gray %s -> %s' % (old_sig, new_sig))
    else:
        print('gray unchanged: %s' % new_sig)

    names = args.maps or [mcfg.get('name', 'zestafona')]

    from autopilot.vision import locator
    from autopilot.vision.featureindex import build_index

    for name in names:
        print('\n== %s ==' % name)
        t0 = time.time()
        locator.set_map(name)
        mu = locator.load_global_map()
        disk_sig = locator._gray_sig_on_disk(name)
        ok = disk_sig == locator._gray_sig()
        print('  map %dx%d (%.1fs)  on-disk gray=%s %s'
              % (mu.shape[0], mu.shape[1], time.time() - t0, disk_sig,
                 'OK' if ok else 'MISMATCH'))
        if not ok:
            print('  ABORT: mu cache does not match config gray — index '
                  'rebuild skipped, fix the cache first')
            continue
        if not args.no_index:
            t1 = time.time()
            path = build_index(name, progress=False)
            print('  feature index rebuilt: %s (%.1fs)'
                  % (os.path.basename(path), time.time() - t1))

    print('\ndone. Restart the app (or switch maps) so it picks up the new mu.')


if __name__ == '__main__':
    main()
