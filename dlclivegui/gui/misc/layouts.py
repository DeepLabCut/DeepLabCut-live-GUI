# dlclivegui/gui/misc/layouts.py
from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QSizePolicy, QStyle, QStyleOptionComboBox, QWidget


def _combo_width_for_current_text(combo: QComboBox, extra_padding: int = 10) -> int:
    """
    Compute a width that fits the *current* text + icon + arrow/frame.
    Uses Qt style metrics to be platform/theme-correct.
    """
    opt = QStyleOptionComboBox()
    combo.initStyleOption(opt)

    text = combo.currentText() or ""
    fm = combo.fontMetrics()
    text_px = fm.horizontalAdvance(text)

    icon_px = 0
    # Account for icon shown in the label
    if not combo.itemIcon(combo.currentIndex()).isNull():
        icon_px = combo.iconSize().width() + 4  # 4px spacing fudge

    # Frame + arrow area (common approach: combo frame + scrollbar extent for arrow)
    frame = combo.style().pixelMetric(QStyle.PM_ComboBoxFrameWidth, opt, combo)
    arrow = combo.style().pixelMetric(QStyle.PM_ScrollBarExtent, opt, combo)

    # Total
    return text_px + icon_px + (2 * frame) + arrow + extra_padding


def enable_combo_shrink_to_current(
    combo: QComboBox,
    *,
    min_width: int = 80,
    max_width: int | None = None,
    extra_padding: int = 10,
) -> QObject:
    """
    Make QComboBox width follow the current item width (instead of widest item).

    Returns an object you can keep alive if you want; in practice, the connections
    are owned by `combo`, but returning a QObject is convenient for future expansion.
    """
    # Let the widget be fixed-width (we'll drive it dynamically)
    combo.setSizePolicy(QSizePolicy.Fixed, combo.sizePolicy().verticalPolicy())

    def _update():
        w = _combo_width_for_current_text(combo, extra_padding=extra_padding)
        w = max(min_width, w)
        if max_width is not None:
            w = min(max_width, w)
        combo.setFixedWidth(w)

    # Update when selection changes
    combo.currentIndexChanged.connect(lambda _i: _update())
    combo.currentTextChanged.connect(lambda _t: _update())

    # First update after items are populated
    _update()

    # Return dummy handle (could be used for disconnecting later if needed)
    return combo


def make_two_field_row(
    left_label: str | None,
    left_widget: QWidget | None,
    right_label: str | None,
    right_widget: QWidget | None,
    left_stretch: int = 1,
    right_stretch: int = 1,
    *,
    key_width: int | None = 100,
    gap: int = 10,
    reserve_key_space_if_none: bool = False,
    style_values: bool = True,
    value_style_qss: str | None = None,
    value_style_types: Sequence[type[QWidget]] = (QLabel,),  # extend if you want
    value_style_classnames: Sequence[str] = ("ElidingPathLabel",),
) -> QWidget:
    """
    Two pairs in one row: [key][value]  [key][value], but dynamically built.

    Key behavior:
      - If a label is None:
          - if reserve_key_space_if_none=False: no key widget is created (no space used).
          - if reserve_key_space_if_none=True: an empty QLabel is created to keep alignment.
      - If a widget is None: that side is omitted entirely.
      - The gap is only inserted if BOTH sides exist.
      - Column stretching applies only to the value columns that actually exist.

    Args:
        left_label: Text for the left key label, or None for no label.
        left_widget: Widget for the left value, or None for no widget.
        right_label: Text for the right key label, or None for no label.
        right_widget: Widget for the right value, or None for no widget.
        left_stretch: Stretch factor for the left value column (default 1).
        right_stretch: Stretch factor for the right value column (default 1).
        key_width: If not None, fixed width for key labels; if 0, minimal width.
        gap: Horizontal gap in pixels between the two pairs (default 10).
        reserve_key_space_if_none: If True, reserves space for key label even if text is None.
        style_values: If True, applies value_style_qss to value widgets of specified types.
        value_style_qss: Custom QSS string for styling value widgets; if None, uses default.
        value_style_types: Widget types to apply value styling to (default: QLabel).
        value_style_classnames: Widget class names to apply value styling to
                                (for custom widgets without importing their classes).

    Returns:
        QWidget: A QWidget containing the two-field row layout.
    """

    # ---------- sanitize args ----------
    if left_widget is None and right_widget is None:
        # Nothing to show; return an empty widget with no margins
        empty = QWidget()
        empty.setContentsMargins(0, 0, 0, 0)
        return empty

    gap = max(0, int(gap))
    # Negative stretch doesn’t make sense; treat as 0
    left_stretch = max(0, int(left_stretch))
    right_stretch = max(0, int(right_stretch))

    row = QWidget()
    grid = QGridLayout(row)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(0)

    # Default value styling
    if value_style_qss is None:
        value_style_qss = """
        QLabel, ElidingPathLabel {
            font-weight: 700;
            color: palette(text);
            background-color: rgba(127,127,127,0.12);
            border: 1px solid rgba(127,127,127,0.18);
            border-radius: 6px;
            padding: 2px 6px;
        }
        """

    def _should_style(w: QWidget) -> bool:
        if isinstance(w, tuple(value_style_types)):
            return True
        # Avoid importing custom widgets just for isinstance; match by class name
        return w.__class__.__name__ in set(value_style_classnames)

    def _style_value(w: QWidget) -> None:
        if not style_values:
            return
        if _should_style(w):
            w.setStyleSheet(value_style_qss)
            if isinstance(w, QLabel):
                w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Keep widgets from stretching weirdly
        sp = w.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Preferred)
        w.setSizePolicy(sp)

    def _make_key_label(text: str | None) -> QLabel | None:
        """
        Create a key QLabel or None.
        If reserve_key_space_if_none=True and text is None, create empty QLabel.
        """
        if text is None and not reserve_key_space_if_none:
            return None

        lbl = QLabel("" if text is None else text)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        if key_width is not None:
            # If key_width is 0, make it truly minimal; else fixed width.
            if key_width <= 0:
                lbl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
            else:
                lbl.setFixedWidth(key_width)

        return lbl

    # ---------- dynamic column builder ----------
    col = 0
    value_cols: list[tuple[int, int]] = []  # (column_index, stretch)

    def _add_pair(label_text: str | None, widget: QWidget | None, stretch: int) -> None:
        nonlocal col
        if widget is None:
            return

        key_lbl = _make_key_label(label_text)
        if key_lbl is not None:
            grid.addWidget(key_lbl, 0, col)
            grid.setColumnStretch(col, 0)
            col += 1

        _style_value(widget)
        grid.addWidget(widget, 0, col)
        value_cols.append((col, stretch))
        col += 1

    left_present = left_widget is not None
    right_present = right_widget is not None

    # Left pair
    _add_pair(left_label, left_widget, left_stretch)

    # Gap only if both exist and gap > 0
    if left_present and right_present and gap > 0:
        spacer = QWidget()
        spacer.setFixedWidth(gap)
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        grid.addWidget(spacer, 0, col)
        grid.setColumnStretch(col, 0)
        col += 1

    # Right pair
    _add_pair(right_label, right_widget, right_stretch)

    # Apply stretch only to value columns that exist
    for c, s in value_cols:
        grid.setColumnStretch(c, s)

    return row
