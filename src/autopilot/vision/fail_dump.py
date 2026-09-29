"""Autosave and pruning of failed localization frames (post-run triage)."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import cv2
import numpy as np

from .. import PROJECT_ROOT
from ..common.config import LocatorConfig
from ..common.log import get_logger

logger = get_logger("tracker")

FAIL_SAVE_PERIOD_S = 1.0
FAIL_KEEP_FILES = 200
FAIL_REASONS = (
    "no_features_flat",
    "no_match_global",
    "budget_timeout",
    "no_index",
    "index_no_match",
    "no_features_frame",
    "vote_reject",
    "locator_error",
)


def fail_payload(
    diag: dict,
    mm: np.ndarray,
    mask: np.ndarray | None,
    *,
    locator_cfg: LocatorConfig,
    prev_xy: tuple[float, float] | None,
    attempt: int,
) -> dict[str, Any]:
    """Structured context of a failed frame for offline replay/triage."""
    loc = locator_cfg.model_dump()
    prev = prev_xy if prev_xy is not None else diag.get("prev")
    return {
        "ts": time.time(),
        "reject": diag.get("reject"),
        "detail": diag.get("detail"),
        "mode": diag.get("mode"),
        "prev": list(prev) if prev is not None else None,
        "roi": diag.get("roi"),
        "attempt": attempt,
        "frame_shape": list(mm.shape),
        "mm_mean": diag.get("mm_mean"),
        "mm_std": diag.get("mm_std"),
        "mm_mask_frac": diag.get("mm_mask_frac"),
        "kp_mm": diag.get("kp_mm"),
        "kp_chunk": diag.get("kp_chunk"),
        "good": diag.get("good1"),
        "inl": diag.get("inl1"),
        "s": diag.get("s1"),
        "th": diag.get("th1"),
        "t": diag.get("t1"),
        "search_discs": diag.get("search_discs"),
        "search_global": diag.get("search_global"),
        "vote": diag.get("vote"),
        "reject_tally": diag.get("reject_tally"),
        "mask_px": int(np.asarray(mask).sum()) if mask is not None and mask.size > 0 else None,
        "thr": {
            k: loc.get(k)
            for k in (
                "max_kp_frame",
                "track_radius",
                "ratio_local",
                "min_inl_local",
                "min_inl_rate_local",
                "ratio_global",
                "min_inl_global",
                "min_inl_rate_global",
                "vote_need",
                "vote_inl_skip",
                "jump_gate_px",
                "heading_gate_deg",
            )
            if loc.get(k) is not None
        },
    }


def save_fail_frame(
    diag: dict,
    mm: np.ndarray,
    bgr: np.ndarray | None,
    mask: np.ndarray | None,
    *,
    locator_cfg: LocatorConfig,
    prev_xy: tuple[float, float] | None,
    attempt: int,
    out_dir: str | None = None,
    keep: int = FAIL_KEEP_FILES,
) -> None:
    """Autosave a failed frame (gray + color + context) for post-run analysis."""
    try:
        target_dir = out_dir or os.path.join(PROJECT_ROOT, "output")
        os.makedirs(target_dir, exist_ok=True)
        ms = int((time.time() % 1.0) * 1000)
        base = "debug_fail_%s_%03d" % (time.strftime("%Y%m%d_%H%M%S"), ms)
        cv2.imwrite(os.path.join(target_dir, base + ".png"), mm)
        if bgr is not None and bgr.size > 0:
            cv2.imwrite(os.path.join(target_dir, base + "_rgb.png"), bgr)
        payload = fail_payload(
            diag, mm, mask, locator_cfg=locator_cfg, prev_xy=prev_xy, attempt=attempt
        )
        with open(os.path.join(target_dir, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(os.path.join(target_dir, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("reject=%s\n" % diag.get("reject"))
            f.write("detail=%s\n" % diag.get("detail"))
            f.write("prev=%s\n" % (payload["prev"],))
            f.write("roi=%s\n" % (diag.get("roi"),))
            f.write("kp_mm=%d kp_chunk=%d\n" % (diag.get("kp_mm", 0), diag.get("kp_chunk", 0)))
            f.write(
                "mm_std=%.1f mask=%d%%\n"
                % (diag.get("mm_std") or 0, int(100 * (diag.get("mm_mask_frac") or 0)))
            )
            tally = diag.get("reject_tally") or {}
            if tally:
                f.write("tally=%s\n" % " ".join("%s=%d" % (k, v) for k, v in tally.items()))
            f.write("thr=%s\n" % json.dumps(payload["thr"], sort_keys=True))
        prune_fail_files(target_dir, keep)
        logger.info(
            "[tracker] saved fail frame %s (reject=%s kp=%s)",
            base,
            diag.get("reject"),
            diag.get("kp_mm"),
        )
    except Exception:  # noqa: BLE001
        pass


def prune_fail_files(out_dir: str, keep: int = FAIL_KEEP_FILES) -> None:
    """Keep only the newest `keep` debug_fail_* files (names sort by time)."""
    files = sorted(p for p in os.listdir(out_dir) if p.startswith("debug_fail_"))
    while len(files) > keep:
        try:
            os.remove(os.path.join(out_dir, files[0]))
        except OSError:
            pass
        files = files[1:]
