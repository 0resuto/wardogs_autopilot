"""Row-wrapping layout for the narrow studio panels.

The sidebar column is 330-470 px wide, so a plain `QHBoxLayout` row of buttons
can ask for more width than the column has and Qt silently clips the tail
(the scroll area has its horizontal scrollbar disabled). `FlowLayout` lays items
out left to right and moves an item to the next row when it no longer fits.

An item whose horizontal size policy is `Expanding` additionally absorbs the
space left over on its row (shared proportionally to the size hints), and an
item with `heightForWidth` is measured against the width it actually got. That
is what lets the capture previews sit side by side on a wide sidebar and stack
vertically on a narrow one without any width arithmetic in the panels.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


def _widget(item: QLayoutItem) -> QWidget | None:
    """The widget an item manages, if any (`QWidgetItem` has no `sizePolicy`)."""
    return item.widget() if hasattr(item, "widget") else None


def _expands_horizontally(item: QLayoutItem) -> bool:
    """Whether an item wants the leftover width of its row (size policy check)."""
    widget = _widget(item)
    policy = widget.sizePolicy() if widget is not None else None
    if policy is None:
        return False
    flags = policy.horizontalPolicy()
    value = flags.value if isinstance(flags, QSizePolicy.Policy) else int(flags)
    return bool(value & QSizePolicy.Policy.Expanding.value)


def _height_for_width(item: QLayoutItem, width: int) -> int | None:
    """The height the item wants at `width`, or `None` when it has no opinion.

    A widget may implement `flow_height_for_width(width)` to state its height
    explicitly; that hook is used first because Qt's `heightForWidth` dispatch
    through a widget's own layout is not reliable (it ignores Python overrides
    once the widget has been laid out).
    """
    if width <= 0:
        return None
    widget = _widget(item)
    if widget is None:
        return item.heightForWidth(width) if item.hasHeightForWidth() else None
    hook = getattr(widget, "flow_height_for_width", None)
    if hook is not None:
        return int(hook(width))
    return widget.heightForWidth(width) if widget.hasHeightForWidth() else None


class FlowLayout(QLayout):
    """Left-to-right layout that wraps an item which no longer fits.

    `parent` must be a widget, never a layout: Qt deletes a `QLayout`
    constructed on a widget that already owns a layout, so ownership is taken by
    `QLayout.addLayout` / `QWidget.setLayout` instead.
    """

    def __init__(
        self,
        *,
        h_spacing: int = 6,
        v_spacing: int = 6,
        margin: int = 0,
        one_row_hint: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.h_spacing = max(0, h_spacing)
        self.v_spacing = max(0, v_spacing)
        self.one_row_hint = one_row_hint
        self.setContentsMargins(margin, margin, margin, margin)

    # --- QLayout interface -------------------------------------------------

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt naming
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt naming
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt naming
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt naming
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt naming
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Preferred size: the narrowest useful one, or one row when asked for.

        By default the width is the narrowest useful one (a single item per row)
        so that a panel column with a small minimum width can actually reach it:
        parents size a container to at least this hint, and a one-row hint would
        pin the sidebar open. The container still stretches the layout to
        whatever width it gets, and `heightForWidth` reports the matching
        height. `one_row_hint` restores the one-row hint for a container that
        should hug its content (a floating toolbar) and only wraps when it has
        to.
        """
        widest = QSize(0, 0)
        for item in self._items:
            widest = widest.expandedTo(item.sizeHint())
        if not self.one_row_hint:
            return self._padded(
                QSize(max(widest.width(), self.minimumSize().width()), widest.height())
            )
        row_width = 0
        for item in self._items:
            row_width += item.sizeHint().width() + self.h_spacing
        return self._padded(QSize(max(row_width - self.h_spacing, widest.width()), widest.height()))

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt naming
        """Smallest usable size: the widest single item, every item on its own row."""
        width = 0
        for item in self._items:
            width = max(width, item.minimumSize().width())
        return self._padded(QSize(width, self.heightForWidth(width)))

    # --- geometry ----------------------------------------------------------

    def _padded(self, size: QSize) -> QSize:
        m = self.contentsMargins()
        return QSize(
            size.width() + m.left() + m.right(),
            size.height() + m.top() + m.bottom(),
        )

    def _item_size(self, item: QLayoutItem, width: int) -> QSize:
        """Item size for the row width it would get, clamped to min/max."""
        hint = item.sizeHint()
        max_size = item.maximumSize()
        max_w = max_size.width() if max_size.width() > 0 else hint.width()
        max_h = max_size.height() if max_size.height() > 0 else hint.height()
        floors = item.minimumSize()
        w = max(floors.width(), min(hint.width(), max_w))
        if width > 0:
            # A lone item may shrink below its hint to fit a narrow column.
            w = min(w, max(width, floors.width()))
        return QSize(w, self._item_height(item, w, hint.height(), max_h))

    def _item_height(self, item: QLayoutItem, width: int, fallback: int, max_h: int = 0) -> int:
        """Height for the given width: `heightForWidth` when the item offers it."""
        floor = item.minimumSize().height()
        ceiling = max_h if max_h > 0 else item.maximumSize().height()
        asked = _height_for_width(item, width)
        height = fallback if asked is None else asked
        return max(floor, min(height, ceiling or height))

    def _arrange(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        avail = max(0, rect.width() - m.left() - m.right())

        rows: list[list[QLayoutItem]] = []
        sizes: dict[int, QSize] = {}
        current: list[QLayoutItem] = []
        used = 0
        for item in self._items:
            size = self._item_size(item, avail)
            gap = self.h_spacing if current else 0
            if current and used + gap + size.width() > avail:
                rows.append(current)
                current = []
                used = 0
                gap = 0
            current.append(item)
            sizes[id(item)] = size
            used += gap + size.width()
        if current:
            rows.append(current)

        # Give the leftover width to the expanding items of each row and let an
        # item with `heightForWidth` re-measure itself for the width it won.
        for row in rows:
            row_total = sum(sizes[id(it)].width() for it in row) + self.h_spacing * (len(row) - 1)
            leftover = avail - row_total
            if leftover <= 0:
                continue
            growing = [it for it in row if _expands_horizontally(it)]
            if not growing:
                continue
            weight = sum(sizes[id(it)].width() for it in growing) or len(growing)
            for item in growing:
                size = sizes[id(item)]
                width = size.width() + int(leftover * size.width() / weight)
                limits = item.maximumSize()
                if 0 < limits.width() < width:
                    width = limits.width()
                height = self._item_height(item, width, size.height())
                sizes[id(item)] = QSize(width, height)

        top = rect.y() + m.top()
        y = top
        for row in rows:
            row_height = max(sizes[id(it)].height() for it in row)
            x = rect.x() + m.left()
            for item in row:
                size = sizes[id(item)]
                if apply:
                    item.setGeometry(QRect(x, y, size.width(), min(size.height(), row_height)))
                x += size.width() + self.h_spacing
            y += row_height + self.v_spacing
        if not rows:
            return m.top() + m.bottom()
        return (y - self.v_spacing) - top + m.bottom()
