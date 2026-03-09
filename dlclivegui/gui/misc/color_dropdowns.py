"""
UI elements for color selection dropdowns (bbox colors, matplotlib colormaps).

- BBox color combo: enum-based with BGR swatch icons.
- Colormap combo: Matplotlib registry-based, optional gradient icons.
- ShrinkCurrentWidePopupComboBox: combobox label shrinks to current selection,
  while the popup widens to the longest item to avoid eliding.
"""
# dlclivegui/gui/misc/color_dropdowns.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
)

BGR = tuple[int, int, int]
TEnum = TypeVar("TEnum")


# -----------------------------------------------------------------------------
# Combo sizing: shrink to current selection + wide popup
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ComboSizing:
    """Sizing policy for ShrinkCurrentWidePopupComboBox."""

    # Combobox (label) sizing:
    min_width: int = 80
    max_width: int | None = None
    extra_padding: int = 10

    # Popup sizing:
    popup_extra_padding: int = 24
    popup_elide_mode: Qt.TextElideMode = Qt.ElideNone


class ShrinkCurrentWidePopupComboBox(QComboBox):
    """
    Combobox whose control (label) shrinks to the current selection, while the popup
    widens to fit the widest item (so long entries are not elided).
    """

    def __init__(self, *args, sizing: ComboSizing | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if sizing is None:
            sizing = ComboSizing()
        self._sizing = sizing

        # We drive width explicitly -> fixed horizontal policy is predictable.
        self.setSizePolicy(QSizePolicy.Fixed, self.sizePolicy().verticalPolicy())

        self.currentIndexChanged.connect(lambda _i: self.update_shrink_width())
        self.currentTextChanged.connect(lambda _t: self.update_shrink_width())

    # --- control width (current selection) ---
    def _width_for_current_text(self) -> int:
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)

        text = self.currentText() or ""
        fm = self.fontMetrics()
        text_px = fm.horizontalAdvance(text)

        icon_px = 0
        idx = self.currentIndex()
        if idx >= 0 and not self.itemIcon(idx).isNull():
            icon_px = self.iconSize().width() + 4

        frame = self.style().pixelMetric(QStyle.PM_ComboBoxFrameWidth, opt, self)
        arrow = self.style().pixelMetric(QStyle.PM_ScrollBarExtent, opt, self)

        return text_px + icon_px + (2 * frame) + arrow + int(self._sizing.extra_padding)

    def update_shrink_width(self) -> None:
        """Update combobox control width to fit current item."""
        w = max(int(self._sizing.min_width), self._width_for_current_text())
        if self._sizing.max_width is not None:
            w = min(int(self._sizing.max_width), w)

        if self.width() != w:
            self.setFixedWidth(w)

    # --- popup width (widest item) ---
    def _max_popup_item_width(self) -> int:
        fm = self.fontMetrics()
        icon_w = self.iconSize().width()

        max_w = 0
        for i in range(self.count()):
            t = self.itemText(i) or ""
            w = fm.horizontalAdvance(t)
            if not self.itemIcon(i).isNull():
                w += icon_w + 6
            max_w = max(max_w, w)

        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        frame = self.style().pixelMetric(QStyle.PM_ComboBoxFrameWidth, opt, self)
        scroll = self.style().pixelMetric(QStyle.PM_ScrollBarExtent, opt, self)

        return max(
            max_w + 2 * frame + scroll + int(self._sizing.popup_extra_padding),
            self.width(),
        )

    def showPopup(self) -> None:
        # Ensure control width is up to date
        self.update_shrink_width()

        view = self.view()
        try:
            view.setTextElideMode(self._sizing.popup_elide_mode)
        except Exception:
            pass

        view.setMinimumWidth(self._max_popup_item_width())
        super().showPopup()


# -----------------------------------------------------------------------------
# BBox color combo helpers (enum-based)
# -----------------------------------------------------------------------------
def _bgr_to_qcolor(bgr: BGR) -> QColor:
    return QColor(bgr[2], bgr[1], bgr[0])


def make_bgr_swatch_icon(bgr: BGR, *, width: int = 40, height: int = 16, border: int = 1) -> QIcon:
    """Create a small BGR swatch icon for use in QComboBox items."""
    pix = QPixmap(width, height)
    pix.fill(Qt.transparent)

    p = QPainter(pix)
    p.fillRect(0, 0, width, height, Qt.black)  # border background
    p.fillRect(border, border, width - 2 * border, height - 2 * border, Qt.white)

    p.fillRect(
        border + 1,
        border + 1,
        width - 2 * (border + 1),
        height - 2 * (border + 1),
        _bgr_to_qcolor(bgr),
    )
    p.end()
    return QIcon(pix)


def populate_bbox_color_combo(
    combo: QComboBox,
    colors_enum: Iterable[TEnum],
    *,
    current_bgr: BGR | None = None,
    include_icons: bool = True,
) -> None:
    """
    Populate a QComboBox with bbox colors from an enum (e.g. BBoxColors).
    Stores the enum item as itemData so you can retrieve .value (BGR).
    """
    combo.blockSignals(True)
    combo.clear()

    for enum_item in colors_enum:
        bgr: BGR = enum_item.value
        name = getattr(enum_item, "name", str(enum_item)).title()
        if include_icons:
            combo.addItem(make_bgr_swatch_icon(bgr), name, enum_item)
        else:
            combo.addItem(name, enum_item)

    if current_bgr is not None:
        set_bbox_combo_from_bgr(combo, current_bgr)

    combo.blockSignals(False)


def make_bbox_color_combo(
    colors_enum: Iterable[TEnum],
    *,
    current_bgr: BGR | None = None,
    include_icons: bool = True,
    tooltip: str = "Select bounding box color",
    sizing: ComboSizing | None = None,
) -> QComboBox:
    """Factory: create and populate a bbox color combobox."""
    combo = ShrinkCurrentWidePopupComboBox(sizing=sizing) if sizing is not None else QComboBox()
    combo.setToolTip(tooltip)
    populate_bbox_color_combo(combo, colors_enum, current_bgr=current_bgr, include_icons=include_icons)
    if isinstance(combo, ShrinkCurrentWidePopupComboBox):
        combo.update_shrink_width()

    return combo


def set_bbox_combo_from_bgr(combo: QComboBox, bgr: BGR) -> None:
    """Select the first item whose enum_item.value == bgr."""
    for i in range(combo.count()):
        enum_item = combo.itemData(i)
        if enum_item is not None and getattr(enum_item, "value", None) == bgr:
            combo.setCurrentIndex(i)
            return


def get_bbox_bgr_from_combo(combo: QComboBox, *, fallback: BGR | None = None) -> BGR | None:
    """Return selected BGR value (enum_item.value)."""
    enum_item = combo.currentData()
    if enum_item is None:
        return fallback
    return getattr(enum_item, "value", fallback)


# -----------------------------------------------------------------------------
# Matplotlib colormap combo helpers
# -----------------------------------------------------------------------------
def _safe_mpl_colormaps_registry():
    """Return matplotlib.colormaps registry, or None if matplotlib isn't available."""
    try:
        from matplotlib import colormaps

        return colormaps
    except Exception:
        return None


def list_colormap_names(
    *,
    exclude_reversed: bool = True,
    favorites_first: Sequence[str] | None = None,
    filters: dict[str, int] | None = None,
) -> list[str]:
    """
    List Matplotlib-registered colormap names.

    Args:
        exclude_reversed: Drop *_r.
        favorites_first: If provided, move these names to the top (if present).
        filters: Prefix-family limits, e.g. {"cet_": 5} keeps only the first 5 names
                 that start with "cet_".
    """
    registry = _safe_mpl_colormaps_registry()
    if registry is None:
        return []

    names = sorted(list(registry))  # registry is iterable of names

    if exclude_reversed:
        names = [n for n in names if not n.endswith("_r")]

    if filters:
        # Apply per-prefix limits deterministically.
        kept: list[str] = []
        used: set[str] = set()

        # For each prefix, take first N matches in sorted order.
        for filtered, limit in filters.items():
            limit_n = max(0, int(limit))
            matches = [n for n in names if filtered in n]
            for n in matches[:limit_n]:
                kept.append(n)
                used.add(n)

        # Keep all names not covered by any filtered prefix, plus the limited ones.
        filtered_prefixes = tuple(filters.keys())
        remainder = [n for n in names if (not any(fp in n for fp in filtered_prefixes)) and (n not in used)]
        names = remainder + kept

    if favorites_first:
        fav = [n for n in favorites_first if n in names]
        rest = [n for n in names if n not in set(fav)]
        names = fav + rest

    return names


def make_cmap_gradient_icon(cmap_name: str, *, width: int = 80, height: int = 14) -> QIcon | None:
    """Create a gradient icon by sampling a Matplotlib colormap."""
    registry = _safe_mpl_colormaps_registry()
    if registry is None:
        return None

    try:
        cmap = registry[cmap_name]
    except Exception:
        return None

    x = np.linspace(0.0, 1.0, width)
    rgba = (cmap(x) * 255).astype(np.uint8)  # (width,4)
    rgb = rgba[:, :3]  # (width,3)
    img = np.repeat(rgb[np.newaxis, :, :], height, axis=0)  # (height,width,3)

    qimg = QImage(img.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    return QIcon(pix)


def populate_colormap_combo(
    combo: QComboBox,
    *,
    current: str | None = None,
    include_icons: bool = True,
    exclude_reversed: bool = True,
    favorites_first: Sequence[str] | None = None,
    filters: dict[str, int] | None = None,
    icon_width: int = 80,
    icon_height: int = 14,
    editable_if_no_mpl: bool = True,
) -> None:
    """
    Populate a QComboBox with Matplotlib colormap names.

    Stores the cmap name string as itemData for each entry.
    """
    names = list_colormap_names(
        exclude_reversed=exclude_reversed,
        favorites_first=favorites_first,
        filters=filters,
    )

    combo.blockSignals(True)
    combo.clear()

    # Matplotlib not available: allow typing a name
    if not names:
        if editable_if_no_mpl:
            combo.setEditable(True)
        if current:
            combo.addItem(current, current)
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        return

    for name in names:
        if include_icons:
            icon = make_cmap_gradient_icon(name, width=icon_width, height=icon_height)
            if icon is not None:
                combo.addItem(icon, name, name)
            else:
                combo.addItem(name, name)
        else:
            combo.addItem(name, name)

    if current:
        set_cmap_combo_from_name(combo, current)

    combo.blockSignals(False)


def make_colormap_combo(
    *,
    current: str = "viridis",
    tooltip: str = "Select colormap to use when displaying keypoints",
    sizing: ComboSizing | None = None,
    include_icons: bool = True,
    exclude_reversed: bool = True,
    favorites_first: Sequence[str] | None = None,
    filters: dict[str, int] | None = None,
    icon_width: int = 80,
    icon_height: int = 14,
    editable_if_no_mpl: bool = True,
    on_changed: Callable[[str], None] | None = None,
) -> ShrinkCurrentWidePopupComboBox:
    """Factory: create + populate + apply sizing behavior for a colormap combo."""
    if sizing is None:
        sizing = ComboSizing()
    combo = ShrinkCurrentWidePopupComboBox(sizing=sizing)
    combo.setToolTip(tooltip)

    populate_colormap_combo(
        combo,
        current=current,
        include_icons=include_icons,
        exclude_reversed=exclude_reversed,
        favorites_first=favorites_first,
        filters=filters,
        icon_width=icon_width,
        icon_height=icon_height,
        editable_if_no_mpl=editable_if_no_mpl,
    )

    combo.update_shrink_width()

    if on_changed is not None:
        combo.currentIndexChanged.connect(lambda _i: on_changed(get_cmap_name_from_combo(combo, fallback=current)))
        combo.currentTextChanged.connect(lambda _t: on_changed(get_cmap_name_from_combo(combo, fallback=current)))

    return combo


def set_cmap_combo_from_name(combo: QComboBox, name: str, *, fallback: str = "viridis") -> None:
    """Select `name` if present, else fallback if present."""
    idx = combo.findData(name)
    if idx >= 0:
        combo.setCurrentIndex(idx)
        return
    idx2 = combo.findData(fallback)
    if idx2 >= 0:
        combo.setCurrentIndex(idx2)


def get_cmap_name_from_combo(combo: QComboBox, *, fallback: str = "viridis") -> str:
    """Return selected colormap name (itemData) or editable text."""
    data = combo.currentData()
    if isinstance(data, str) and data:
        return data
    text = combo.currentText().strip()
    return text or fallback
