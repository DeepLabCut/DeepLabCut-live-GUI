"""UI elements for color selection dropdowns (colors, colormaps)"""

# dlclivegui/gui/misc/color_dropdowns.py
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QComboBox

BGR = tuple[int, int, int]


# ------------------------------
# BBox color combo (enum-based)
# ------------------------------
def make_bgr_swatch_icon(
    bgr: BGR,
    *,
    width: int = 40,
    height: int = 16,
    border: int = 1,
) -> QIcon:
    """Create a small BGR color swatch icon for use in QComboBox items."""
    pix = QPixmap(width, height)
    pix.fill(Qt.transparent)

    p = QPainter(pix)
    # Border/background
    p.fillRect(0, 0, width, height, Qt.black)
    # Inner background
    p.fillRect(border, border, width - 2 * border, height - 2 * border, Qt.white)

    # Convert BGR -> RGB for Qt
    rgb = (bgr[2], bgr[1], bgr[0])
    p.fillRect(border + 1, border + 1, width - 2 * (border + 1), height - 2 * (border + 1), QColor(*rgb))

    p.end()
    return QIcon(pix)


def populate_bbox_color_combo(
    combo: QComboBox,
    colors_enum: Iterable,
    *,
    current_bgr: BGR | None = None,
    include_icons: bool = True,
) -> None:
    """
    Populate a QComboBox with bbox colors from an enum (e.g. BBoxColors).

    The enum items are stored as itemData so you can retrieve .value (BGR).
    """
    combo.blockSignals(True)
    combo.clear()

    for enum_item in colors_enum:
        bgr: BGR = enum_item.value
        name = getattr(enum_item, "name", str(enum_item)).title()

        if include_icons:
            icon = make_bgr_swatch_icon(bgr)
            combo.addItem(icon, name, enum_item)
        else:
            combo.addItem(name, enum_item)

    # Set selection if current_bgr provided
    if current_bgr is not None:
        set_bbox_combo_from_bgr(combo, current_bgr)

    combo.blockSignals(False)


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


# -----------------------------------
# Matplotlib colormap combo (registry)
# -----------------------------------
def _safe_import_matplotlib_colormaps():
    """
    Import matplotlib colormap registry lazily.
    Returns (colormaps_registry, ok_bool).
    """
    try:
        from matplotlib import colormaps

        return colormaps, True
    except Exception:
        return None, False


def list_matplotlib_colormap_names(
    *,
    exclude_reversed: bool = True,
    favorites_first: Sequence[str] | None = None,
) -> list[str]:
    """
    Return a list of registered Matplotlib colormap names.

    Uses `list(matplotlib.colormaps)` (Matplotlib's documented way to list all
    registered colormaps).
    Optionally excludes reversed maps (*_r)
    """
    registry, ok = _safe_import_matplotlib_colormaps()
    if not ok or registry is None:
        return []

    names = sorted(list(registry))
    if exclude_reversed:
        names = [n for n in names if not n.endswith("_r")]

    if favorites_first:
        fav = [n for n in favorites_first if n in names]
        rest = [n for n in names if n not in set(fav)]
        return fav + rest

    return names


def make_cmap_gradient_icon(
    cmap_name: str,
    *,
    width: int = 80,
    height: int = 14,
) -> QIcon | None:
    """
    Create a gradient icon by sampling a Matplotlib colormap.

    Uses the colormap registry lookup: `matplotlib.colormaps[name]`.
    Returns None if Matplotlib isn't available.
    """
    registry, ok = _safe_import_matplotlib_colormaps()
    if not ok or registry is None:
        return None

    try:
        cmap = registry[cmap_name]
    except Exception:
        return None

    x = np.linspace(0.0, 1.0, width)
    rgba = (cmap(x) * 255).astype(np.uint8)  # (width, 4)

    # Convert to RGB row and repeat vertically
    rgb_row = rgba[:, :3]  # (width, 3)
    rgb_img = np.repeat(rgb_row[np.newaxis, :, :], height, axis=0)  # (height, width, 3)

    # QImage referencing numpy memory; copy into QPixmap to own memory safely
    qimg = QImage(rgb_img.data, width, height, 3 * width, QImage.Format.Format_RGB888)
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
) -> None:
    """
    Populate a QComboBox with Matplotlib colormap names.

    - Names come from Matplotlib's colormap registry (`list(colormaps)`).
    - Optionally hides reversed maps (*_r).
    - Stores the name string as itemData.

    Args:
        combo: The QComboBox to populate.
        current: Optional name to select after populating.
        include_icons: If True, adds a gradient icon for each colormap.
        exclude_reversed: If True, excludes colormaps with names ending in "_r".
        favorites_first: Optional list of colormap names to prioritize at the top.
        filters: Optional dict of {substring: min_count} to filter certain colormaps
                 to have a maximum count (e.g. {"cet_": 10},
                 including at most 10 colormaps containing "cet_").
        icon_width: Width of the gradient icons in pixels.
        icon_height: Height of the gradient icons in pixels.
    """
    names = list_matplotlib_colormap_names(
        exclude_reversed=exclude_reversed,
        favorites_first=favorites_first,
    )
    if filters:
        filtered_names = []
        for substr, max_count in filters.items():
            matching = [n for n in names if substr in n]
            filtered_names.extend(matching[:max_count])
        unmatched = [n for n in names if not any(substr in n for substr in filters)]
        filtered_names.extend(unmatched)
        names = filtered_names
    names = sorted(names)

    combo.blockSignals(True)
    combo.clear()

    # If Matplotlib isn't available, still allow typing/selection of a name
    if not names:
        if current:
            combo.addItem(current, current)
            combo.setCurrentIndex(0)
        combo.setEditable(True)
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


def set_cmap_combo_from_name(combo: QComboBox, name: str, *, fallback: str | None = "viridis") -> None:
    """Select `name` if present, else fallback if present."""
    idx = combo.findData(name)
    if idx >= 0:
        combo.setCurrentIndex(idx)
        return
    if fallback:
        idx2 = combo.findData(fallback)
        if idx2 >= 0:
            combo.setCurrentIndex(idx2)


def get_cmap_name_from_combo(combo: QComboBox, *, fallback: str = "viridis") -> str:
    """Return selected colormap name (itemData) or editable text."""
    data = combo.currentData()
    if isinstance(data, str) and data:
        return data
    # If combo is editable
    text = combo.currentText().strip()
    return text or fallback
