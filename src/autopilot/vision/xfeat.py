"""XFeat learned local features on GPU via ONNX Runtime.

XFeat (CVPR 2024) is a lightweight CNN keypoint detector + 64-D descriptor:
one forward pass yields a dense descriptor map, a keypoint heatmap and a
reliability map. This module wraps the exported backbone from
huggingface.co/kornia/xfeat (Apache-2.0) and reimplements the official
post-processing (NMS, top-k, bilinear descriptor sampling, L2 normalisation)
in NumPy, so the whole extractor runs on the DirectML/CUDA execution provider
of ONNX Runtime.

Frames are edge-padded to a multiple of 32 (no resize), so points stay in the
original pixel coordinates and the scale matches the map index 1:1.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np

from .. import PROJECT_ROOT
from ..common.log import get_logger

logger = get_logger("xfeat")

MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
MODEL_FILE = os.path.join(MODEL_DIR, "xfeat_backbone.onnx")
MATCH_MODEL_FILE = os.path.join(MODEL_DIR, "xfeat_match.onnx")

#: GPU providers in preference order; CPU is intentionally not accepted.
_GPU_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider")

_session_lock = threading.Lock()
_session: Any = None
_provider: str | None = None
_matcher_lock = threading.Lock()
_matcher: Any = None


class XFeatError(RuntimeError):
    """Model file or GPU execution provider not available."""


def available() -> bool:
    return os.path.exists(MODEL_FILE)


def provider() -> str | None:
    """Execution provider of the live session (None before the first use)."""
    return _provider


def _get_session() -> Any:
    global _session, _provider
    if _session is not None:
        return _session
    with _session_lock:
        if _session is None:
            _session = _ort_session(MODEL_FILE, "backbone")
            _provider = _session.get_providers()[0]
    return _session


def _ort_session(path: str, what: str) -> Any:
    if not os.path.exists(path):
        raise XFeatError(f"{what} missing — run: python tools/download_models.py")
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    have = set(ort.get_available_providers())
    chosen = [p for p in _GPU_PROVIDERS if p in have]
    if not chosen:
        raise XFeatError(
            "no GPU execution provider (TensorRT/CUDA/DirectML) — install "
            "onnxruntime-directml (see pyproject) and a DirectX 12 capable GPU"
        )
    sess = ort.InferenceSession(path, opts, providers=chosen)
    logger.info("[xfeat] %s on %s", what, sess.get_providers()[0])
    return sess


def _get_matcher_session() -> Any:
    """GPU session of the tiny MatMul+ArgMax matching graph."""
    global _matcher
    if _matcher is not None:
        return _matcher
    with _matcher_lock:
        if _matcher is None:
            _matcher = _ort_session(MATCH_MODEL_FILE, "match graph")
    return _matcher


def match_pairs(q: np.ndarray, t: np.ndarray, min_cos: float, chunk: int = 16384) -> np.ndarray:
    """Mutual-nearest-neighbour cosine matches on the GPU.

    `q` (Nq,64) and `t` (Nt,64) are L2-normalised descriptors (float32 or
    float16). The MatMul+ArgMax graph returns the per-query and per-train
    argmax; train sets are matched in chunks so the similarity matrix stays
    bounded. Returns an (K,2) int64 array of (query, train) index pairs.
    """
    qh = np.ascontiguousarray(q, dtype=np.float16)
    th = np.asarray(t, dtype=np.float16)
    nq, nt = len(qh), len(th)
    if nq == 0 or nt == 0:
        return np.zeros((0, 2), np.int64)
    sess = _get_matcher_session()
    rows = np.arange(nq)
    best_sim = np.full(nq, -1.0, np.float32)
    best_idx = np.zeros(nq, np.int64)
    tb_sim = np.full(nt, -1.0, np.float32)
    tb_q = np.zeros(nt, np.int64)
    for s in range(0, nt, chunk):
        block = np.ascontiguousarray(th[s : s + chunk].T)
        fwd_val, fwd_idx, bwd_val, bwd_idx = sess.run(None, {"A": qh, "B": block})
        fv = fwd_val.astype(np.float32)
        upd = fv > best_sim
        best_sim[upd] = fv[upd]
        best_idx[upd] = s + fwd_idx[upd]
        bv = bwd_val.astype(np.float32)
        sl = slice(s, s + block.shape[1])
        upd_t = bv > tb_sim[sl]
        tb_sim[sl][upd_t] = bv[upd_t]
        tb_q[sl][upd_t] = bwd_idx[upd_t]
    keep = (tb_q[best_idx] == rows) & (best_sim > min_cos)
    return np.stack([np.flatnonzero(keep), best_idx[keep]], axis=1)


def _pad32(img: np.ndarray) -> np.ndarray:
    """Edge-pad to a multiple of 32 (the model's downsample factor)."""
    h, w = img.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)


def _preprocess(gray: np.ndarray) -> np.ndarray:
    padded = _pad32(gray)
    x = padded.astype(np.float32) / 255.0
    return np.repeat(x[None, None], 3, axis=1)


def _bilinear_sample(
    feat: np.ndarray, xs: np.ndarray, ys: np.ndarray, src_w: int, src_h: int
) -> np.ndarray:
    """Sample an (C,H,W) map at (xs,ys); matches torch grid_sample(align_corners=False)."""
    h, w = feat.shape[1], feat.shape[2]
    gx = 2.0 * (xs / (src_w - 1.0)) - 1.0
    gy = 2.0 * (ys / (src_h - 1.0)) - 1.0
    sx = (gx + 1.0) * 0.5 * w - 0.5
    sy = (gy + 1.0) * 0.5 * h - 0.5
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    wx = (sx - x0).astype(np.float32)
    wy = (sy - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    v00 = feat[:, y0c, x0c]
    v01 = feat[:, y0c, x1c]
    v10 = feat[:, y1c, x0c]
    v11 = feat[:, y1c, x1c]
    out = (
        v00 * (1.0 - wx)[None, :] * (1.0 - wy)[None, :]
        + v01 * wx[None, :] * (1.0 - wy)[None, :]
        + v10 * (1.0 - wx)[None, :] * wy[None, :]
        + v11 * wx[None, :] * wy[None, :]
    )
    return out.T  # (N, C)


def extract(
    gray: np.ndarray,
    top_k: int = 2000,
    threshold: float = 0.05,
    nms: int = 5,
    derotate_deg: float = 0.0,
    margin: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect and describe keypoints in a grayscale frame.

    XFeat descriptors are not rotation invariant (measured usable up to
    ~30 deg), while the minimap rotates with the player. With `derotate_deg`
    the frame content is counter-rotated first so the descriptors are sampled
    in the north-up orientation the map index was built in; the returned
    points are transformed back to the ORIGINAL frame coordinates, so the
    estimated affine still maps original-frame pixels to the map.

    Returns (points (N,2) float32, descriptors (N,64) float32 L2-normalised,
    scores (N,) float32).
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    back: np.ndarray | None = None
    if abs(derotate_deg) > 1e-3:
        fwd = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -float(derotate_deg), 1.0)
        back = cv2.invertAffineTransform(fwd)
        gray = cv2.warpAffine(
            gray, fwd, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    x = _preprocess(gray)
    desc_map, heat, rel = _get_session().run(None, {"image": x})
    heat = heat[0, 0]
    rel = rel[0, 0]
    desc_map = desc_map[0]

    dil = cv2.dilate(heat, np.ones((nms, nms), np.uint8))
    mask = (heat > threshold) & (heat >= dil)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (
            np.zeros((0, 2), np.float32),
            np.zeros((0, 64), np.float32),
            np.zeros((0,), np.float32),
        )
    scores = heat[ys, xs] * rel[ys, xs]
    order = np.argsort(-scores)[: max(1, int(top_k))]
    xs, ys, scores = xs[order], ys[order], scores[order]

    # drop points in the edge padding / derotation border band
    inside = (xs >= margin) & (xs < w - margin) & (ys >= margin) & (ys < h - margin)
    xs, ys, scores = xs[inside], ys[inside], scores[inside]
    if len(xs) == 0:
        return (
            np.zeros((0, 2), np.float32),
            np.zeros((0, 64), np.float32),
            np.zeros((0,), np.float32),
        )

    feats = _bilinear_sample(
        desc_map, xs.astype(np.float32), ys.astype(np.float32), x.shape[3], x.shape[2]
    )
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-8)

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if back is not None:
        ones = np.ones((len(pts), 1), np.float32)
        pts = (back @ np.hstack([pts, ones]).T).T.astype(np.float32)
    return pts, feats.astype(np.float32), scores.astype(np.float32)


def extract_tile(
    tile: np.ndarray, top_k: int = 2000, threshold: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """(points, descriptors) for an index build tile (same pipeline as extract)."""
    pts, desc, _scores = extract(tile, top_k=top_k, threshold=threshold)
    return pts, desc
