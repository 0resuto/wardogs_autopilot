"""Reusable interactive map canvas with smooth pan, zoom, and mipmap rendering."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any

import numpy as np

from .imaging import to_photo
from .map_renderer import crop_map_viewport

DARK_CANVAS = "#1e1e1e"
RGB_CANVAS = (30, 30, 30)


class InteractiveMapCanvas(tk.Frame):
    """Tkinter frame hosting an interactive, zoomable, and pannable map canvas."""

    def __init__(
        self,
        master: tk.Widget,
        on_overlay: Callable[[tk.Canvas, tuple[float, float, float] | None], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.on_overlay = on_overlay
        self.on_status = on_status

        self.canvas = tk.Canvas(
            self,
            bg=DARK_CANVAS,
            highlightthickness=1,
            highlightbackground="#3f3f3f",
        )
        self.canvas.pack(fill="both", expand=True)

        self._map8: np.ndarray | None = None
        self._map_pyr: dict[int, np.ndarray] | None = None
        self._map_size = 32768
        self._thumb = 8

        self._disp: tuple[float, float, float] | None = None  # (scale, offx, offy)
        self._fit_scale = 0.0
        self._pending: tuple[float, float, float] | None = None
        self._pan: tuple[float, float, float, float] | None = None  # (sx, sy, ox0, oy0)
        self._bg_photo: Any = None
        self._bg_state: tuple[float, float, float, float, float] | None = None

        self._resize_job: str | None = None
        self._zoom_job: str | None = None

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<Button-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

    @property
    def disp(self) -> tuple[float, float, float] | None:
        return self._disp

    @property
    def fit_scale(self) -> float:
        return self._fit_scale

    def set_map(
        self,
        map8: np.ndarray | None,
        map_pyr: dict[int, np.ndarray] | None,
        map_size: int = 32768,
        thumb: int = 8,
    ) -> None:
        """Update active map textures and trigger view recalculation."""
        self._map8 = map8
        self._map_pyr = map_pyr
        self._map_size = map_size
        self._thumb = thumb
        self._bg_state = None
        self._bg_photo = None
        self.redraw()

    def to_canvas(self, nx: float, ny: float) -> tuple[float, float]:
        """Convert native map pixel coordinate (32768) to canvas coordinates."""
        if self._disp is None:
            return 0.0, 0.0
        s, ox, oy = self._disp
        return nx / self._thumb * s + ox, ny / self._thumb * s + oy

    def to_native(self, cx: float, cy: float) -> tuple[float, float]:
        """Convert canvas coordinate to native map pixel (32768)."""
        if self._disp is None:
            return 0.0, 0.0
        s, ox, oy = self._disp
        return (cx - ox) / s * self._thumb, (cy - oy) / s * self._thumb

    def redraw(self) -> None:
        """Fit entire map into canvas view."""
        if self._map8 is None:
            self.canvas.delete("all")
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50:
            return
        mw, mh = self._map8.shape[1], self._map8.shape[0]
        scale = min(w / mw, h / mh)
        dw, dh = max(1, int(mw * scale)), max(1, int(mh * scale))
        self._fit_scale = scale
        self.apply_view(scale, (w - dw) / 2.0, (h - dh) / 2.0)

    def apply_view(self, s: float, ox: float, oy: float) -> None:
        """Apply scale and offset to viewport, render background and overlay."""
        self._disp = (s, ox, oy)
        if self._map8 is not None:
            self._update_background(s, ox, oy)
        if self.on_status and self._fit_scale > 0:
            zoom = 100.0 * s / self._fit_scale
            self.on_status(f"zoom: {zoom:.0f}%")
        if self.on_overlay:
            self.on_overlay(self.canvas, self._disp)

    def _update_background(self, s: float, ox: float, oy: float) -> None:
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50 or self._map8 is None:
            return
        vis_u0 = -ox / s
        vis_v0 = -oy / s
        vis_wu = w / s
        vis_hu = h / s
        st = self._bg_state
        if (
            st is not None
            and abs(st[0] - s) < 1e-9
            and st[1] <= vis_u0
            and st[2] <= vis_v0
            and st[1] + st[3] >= vis_u0 + vis_wu
            and st[2] + st[4] >= vis_v0 + vis_hu
        ):
            self.canvas.coords("bg", st[1] * s + ox, st[2] * s + oy)
            return

        m = 0.25
        ru = vis_u0 - m * vis_wu
        rv = vis_v0 - m * vis_hu
        rw = vis_wu * (1 + 2 * m)
        rh = vis_hu * (1 + 2 * m)
        pyr = self._map_pyr or {self._map8.shape[0]: self._map8}
        img = crop_map_viewport(s, ru, rv, rw, rh, pyr, self._map_size, self._thumb, RGB_CANVAS)
        self._bg_photo = to_photo(img)
        self._bg_state = (s, ru, rv, rw, rh)
        self.canvas.delete("bg")
        self.canvas.create_image(
            ru * s + ox, rv * s + oy, anchor="nw", image=self._bg_photo, tags="bg"
        )
        self.canvas.tag_lower("bg")

    def _on_resize(self, e: Any = None) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self.redraw)

    def _on_zoom(self, e: Any) -> None:
        if self._disp is None or self._fit_scale <= 0:
            return
        cur = self._pending if self._pending is not None else self._disp
        s, ox, oy = cur
        factor = 1.2 ** (e.delta / 120.0)
        hi = max(32.0, self._fit_scale)
        ns = max(self._fit_scale, min(s * factor, hi))
        if abs(ns - s) < 1e-9:
            return
        ux = (e.x - ox) / s
        uy = (e.y - oy) / s
        self._pending = (ns, e.x - ux * ns, e.y - uy * ns)
        if self._zoom_job is not None:
            self.after_cancel(self._zoom_job)
        self._zoom_job = self.after(20, self._flush_view)

    def _flush_view(self) -> None:
        self._zoom_job = None
        p = self._pending
        if p is None:
            return
        self._pending = None
        self.apply_view(*p)

    def _on_pan_start(self, e: Any) -> None:
        if self._disp is None:
            return
        _, ox, oy = self._disp
        self._pan = (e.x, e.y, ox, oy)

    def _on_pan_move(self, e: Any) -> None:
        if self._pan is None or self._disp is None:
            return
        sx, sy, ox0, oy0 = self._pan
        s, _, _ = self._disp
        self.apply_view(s, ox0 + (e.x - sx), oy0 + (e.y - sy))

    def _on_double_click(self, e: Any = None) -> None:
        self.redraw()

    def center_on(self, nx: float, ny: float) -> None:
        """Center the viewport on a native map coordinate (32768)."""
        if self._disp is None:
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        s, _, _ = self._disp
        tx, ty = nx / self._thumb, ny / self._thumb
        ox = w / 2.0 - tx * s
        oy = h / 2.0 - ty * s
        self.apply_view(s, ox, oy)
