"""Map asset preflight and scenario rendering shared by the CLI and the studio.

Everything the bench needs from one map — the `mu` cache, the top preview
level, the physical scale and the UI mask — is loaded once through
`load_inputs`, and every missing piece becomes a `BenchSkip` carrying the
one-line remedy. The CLI prints the message and exits 0; the studio shows it in
the status line. Neither path may crash on a half-downloaded map.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..vision import locator, preprocessing
from .scenario import Scenario, ScenarioSpec, _pick_center, build_scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Mip level the synthetic frames are cut from (native/2). Reading the frames
#: from a preview while the index was built from `mu` is the deliberate gap.
PREVIEW_LEVEL = 16384


class BenchSkip(RuntimeError):
    """A required asset is missing: report it and move on, never crash."""


@dataclass(frozen=True)
class ScenarioInputs:
    """Everything `build_scenario` needs, loaded once per map."""

    map_name: str
    mu: np.ndarray
    preview: np.ndarray
    mini_scale: float
    center_mu: tuple[float, float]
    px_per_m: float
    mask: np.ndarray | None = None
    mask_missing: bool = False

    def render(self, spec: ScenarioSpec) -> Scenario:
        """Render one spec against these inputs."""
        return build_scenario(
            spec,
            self.mu,
            self.preview,
            self.mini_scale,
            self.center_mu,
            self.mask,
            px_per_m=self.px_per_m,
        )

    def render_all(self, specs: Sequence[ScenarioSpec]) -> list[Scenario]:
        """Render every spec; the frames are reused by all engines and configs."""
        return [self.render(spec) for spec in specs]


def load_inputs(map_name: str) -> ScenarioInputs:
    """Load the map assets for `map_name` or raise `BenchSkip` with the fix."""
    name = str(map_name)
    store = locator.get_store()
    locator.set_map(name)
    try:
        mu = locator.load_global_map()
    except Exception as exc:  # noqa: BLE001 - any missing/!unreadable cache is a skip
        raise BenchSkip(
            f"map cache for '{name}' is unavailable: {exc} "
            f"(run: python tools/download_map.py {name})"
        ) from exc

    px_per_m = float(store.px_per_m(name))
    if px_per_m <= 0.0:
        raise BenchSkip(f"no physical map scale for '{name}' in data/maps/catalog.json")

    center = _pick_center(mu)
    if center is None:
        raise BenchSkip(f"no well-textured window found near the center of {name}")

    preview_path = store.preview_path(name, PREVIEW_LEVEL)
    if not os.path.exists(preview_path):
        raise BenchSkip(f"preview level missing: {os.path.relpath(preview_path)}")

    mask_missing = False
    try:
        mask = preprocessing.make_mask()
    except OSError:
        mask, mask_missing = None, True

    return ScenarioInputs(
        map_name=name,
        mu=mu,
        preview=np.load(preview_path),
        mini_scale=store.mini_scale(),
        center_mu=center,
        px_per_m=px_per_m,
        mask=mask,
        mask_missing=mask_missing,
    )


def render_scenarios(specs: Sequence[ScenarioSpec], map_name: str) -> list[Scenario]:
    """Load the map assets and render every spec (the CLI's fast path)."""
    return load_inputs(map_name).render_all(specs)
