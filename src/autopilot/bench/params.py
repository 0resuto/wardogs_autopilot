"""Tunable parameter registry, OAT sweep expansion, and the in-memory overlay.

The bench must never touch `config.json`, so every override is applied by
`ConfigOverlay`: it shadows `store.loc_cfg` on the live store instance for the
duration of a `with` block and restores it afterwards, even on exceptions. The
engine therefore reads exactly the values a sweep asks for, and the next run
sees the untouched config again.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Value = int | float


@dataclass(frozen=True)
class ParamSpec:
    """One sweepable locator parameter and the values the bench offers for it."""

    key: str
    values: tuple[Value, ...]
    label: str

    @property
    def value_type(self) -> type:
        """int/float the CLI values are parsed with (from the first value)."""
        return int if self.values and isinstance(self.values[0], int) else float

    def parse(self, raw: str) -> Value:
        return self.value_type(raw)


def _spec(key: str, values: Sequence[Value], label: str) -> ParamSpec:
    return ParamSpec(key=key, values=tuple(values), label=label)


#: Engine-specific sweepables. Values are the curated OAT grid, not a range.
ENGINE_PARAMS: dict[str, list[ParamSpec]] = {
    "sift": [
        _spec("ratio_local", (0.80, 0.85, 0.90), "Lowe ratio for the tracking radius"),
        _spec("min_inl_local", (4, 8, 16), "Minimum inliers inside the tracking radius"),
        _spec("track_radius", (450.0, 900.0), "Tracking search radius (px)"),
        _spec("max_kp_frame", (600, 1200, 2400), "Keypoints kept per frame"),
        _spec("ransac_px", (2.0, 3.0, 6.0), "RANSAC inlier threshold (px)"),
    ],
    "orb": [
        _spec("ratio_local", (0.80, 0.85, 0.90), "Lowe ratio for the tracking radius"),
        _spec("min_inl_local", (4, 8, 16), "Minimum inliers inside the tracking radius"),
        _spec("track_radius", (450.0, 900.0), "Tracking search radius (px)"),
        _spec("max_kp_frame", (600, 1200, 2400), "Keypoints kept per frame"),
        _spec("ransac_px", (2.0, 3.0, 6.0), "RANSAC inlier threshold (px)"),
    ],
    "xfeat": [
        _spec("xfeat_top_k", (500, 1000, 2000, 4000), "Keypoints kept per frame"),
        _spec("xfeat_min_cos", (0.75, 0.82, 0.88), "Minimum cosine similarity of a match"),
        _spec("xfeat_threshold", (0.03, 0.05, 0.10), "Keypoint heatmap detection threshold"),
        _spec("ransac_px", (2.0, 3.0, 6.0), "RANSAC inlier threshold (px)"),
    ],
    "hybrid": [
        _spec("hybrid_reanchor_s", (0.5, 1.0, 2.0), "Seconds between SIFT anchors"),
        _spec("hybrid_min_cc", (0.3, 0.5, 0.7), "Minimum ECC correlation coefficient"),
        _spec("ransac_px", (2.0, 3.0, 6.0), "RANSAC inlier threshold (px)"),
    ],
}

#: Sweepables shared by every engine (the vote gate and the pose smoother).
SHARED_PARAMS: list[ParamSpec] = [
    _spec("smooth_alpha", (0.0, 0.35, 0.5, 0.8), "Alpha-beta smoothing weight (0 = raw)"),
    _spec("vote_need", (1, 3, 5), "Agreeing frames a relocation vote needs"),
]


def param_registry(engine: str | None = None) -> dict[str, ParamSpec]:
    """Every valid sweep key (optionally restricted to one engine + the shared)."""
    registry: dict[str, ParamSpec] = {}
    if engine is None:
        for specs in ENGINE_PARAMS.values():
            for spec in specs:
                registry.setdefault(spec.key, spec)
    else:
        for spec in ENGINE_PARAMS.get(str(engine), []):
            registry.setdefault(spec.key, spec)
    for spec in SHARED_PARAMS:
        registry.setdefault(spec.key, spec)
    return registry


def sweep_keys(engine: str | None = None) -> list[str]:
    """Registry keys in declaration order."""
    return list(param_registry(engine))


def parse_sweep(expr: str) -> tuple[str, list[Value]]:
    """Parse `"key=v1,v2"` into `(key, [values])`.

    Values are parsed to the registry's value type. An unknown key raises
    ValueError listing every valid key.
    """
    text = str(expr or "").strip()
    if "=" not in text:
        raise ValueError(
            f"sweep must look like key=v1,v2 (got {expr!r}); valid keys: {', '.join(sweep_keys())}"
        )
    key, _, raw_values = text.partition("=")
    key = key.strip()
    registry = param_registry()
    if key not in registry:
        raise ValueError(f"unknown sweep key {key!r}; valid keys: {', '.join(registry)}")
    spec = registry[key]
    tokens = [v.strip() for v in raw_values.split(",") if v.strip()]
    if not tokens:
        raise ValueError(f"sweep {key!r} has no values; valid keys: {', '.join(registry)}")
    try:
        return key, [spec.parse(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"bad value for {key!r}: {exc}") from exc


def expand_oat(
    sweeps: Iterable[tuple[str, Sequence[Value]]], base_overrides: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Cartesian expansion of the swept keys; each dict carries one value per key.

    Every returned dict also carries `base_overrides`, so a sweep of two keys
    with three values each yields 3x3 = 9 configurations.
    """
    base = dict(base_overrides or {})
    keys = [(str(key), list(values)) for key, values in sweeps]
    for key, values in keys:
        if not values:
            raise ValueError(f"sweep {key!r} has no values")
    if not keys:
        return [dict(base)]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*(values for _key, values in keys)):
        cfg = dict(base)
        cfg.update(dict(zip((key for key, _v in keys), combo, strict=True)))
        out.append(cfg)
    return out


class ConfigOverlay:
    """Shadow `store.loc_cfg` with extra locator values for the `with` body.

    Only the instance attribute is replaced, so the override never reaches
    `config.json` and every other store instance keeps the real method.
    """

    def __init__(self, store: Any, overrides: Mapping[str, Any] | None = None) -> None:
        self.store = store
        self.overrides = {str(k): v for k, v in dict(overrides or {}).items() if v is not None}
        self._orig: Any = None
        self._shadowed = False

    def __enter__(self) -> ConfigOverlay:
        # Read the flag BEFORE shadowing: after the assignment the attribute is
        # always in vars(store), and a nested overlay must restore the shadow it
        # found instead of deleting it.
        self._shadowed = "loc_cfg" in vars(self.store)
        self._orig = self.store.loc_cfg
        overlay = self.overrides

        def _patched() -> dict[str, Any]:
            return {**self._orig(), **overlay}

        self.store.loc_cfg = _patched
        return self

    def __exit__(self, *_exc: object) -> Literal[False]:
        if self._shadowed:
            # A parent overlay already shadowed loc_cfg: put its shadow back.
            self.store.loc_cfg = self._orig
        else:
            # Deleting the instance attribute uncovers the class method, so the
            # store is left exactly as it was found.
            try:
                delattr(self.store, "loc_cfg")
            except AttributeError:
                self.store.loc_cfg = self._orig
        return False
