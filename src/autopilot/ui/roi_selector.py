"""Fullscreen rectangular zone picker for the minimap ROI."""

import tkinter as tk

from .imaging import to_photo


class RoiSelector(tk.Toplevel):
    """Shows the target monitor fullscreen; the user drags a rectangle."""

    def __init__(self, master, monitor_bgr, monitor_geom, on_roi, on_cancel) -> None:
        super().__init__(master)
        self._geom = monitor_geom
        self._on_roi = on_roi
        self._on_cancel = on_cancel
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg="black")
        self._x0: int | None = None
        self._y0: int | None = None
        self._rect: int | None = None
        self._pending: tuple[int, int, int, int] | None = None
        self._img = None
        self._photo = None

        w, h = monitor_bgr.shape[1], monitor_bgr.shape[0]
        self.geometry(f"{w}x{h}+{self._geom['left']}+{self._geom['top']}")
        self.canvas = tk.Canvas(self, width=w, height=h, highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self._photo = to_photo(monitor_bgr)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self.canvas.bind("<Button-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<Escape>", self._on_escape)
        self.bind("<Return>", self._confirm)

    def _on_escape(self, e=None) -> None:
        self.destroy()
        self._on_cancel()

    def _on_down(self, e) -> None:
        self._x0, self._y0 = e.x, e.y

    def _on_drag(self, e) -> None:
        if self._rect is not None:
            self.canvas.delete(self._rect)
        if self._x0 is not None and self._y0 is not None:
            self._rect = self.canvas.create_rectangle(
                self._x0, self._y0, e.x, e.y, outline="red", width=2
            )

    def _on_up(self, e) -> None:
        if self._rect is not None:
            self.canvas.delete(self._rect)
        if self._x0 is not None and self._y0 is not None:
            x, y = min(self._x0, e.x), min(self._y0, e.y)
            w, h = abs(e.x - self._x0), abs(e.y - self._y0)
            if w > 5 and h > 5:
                self._rect = self.canvas.create_rectangle(x, y, x + w, y + h, outline="lime", width=2)
                self._pending = (x, y, w, h)

    def _confirm(self, e=None) -> None:
        roi = self._pending
        if roi:
            self.destroy()
            self._on_roi(roi)
        else:
            self.destroy()
            self._on_cancel()
