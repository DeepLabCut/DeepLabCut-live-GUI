"""Utility functions to create common layouts."""

# dlclivegui/gui/misc/layouts.py
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget


def _make_two_field_row(
    left_label: str,
    left_widget: QWidget,
    right_label: str,
    right_widget: QWidget,
    left_stretch: int = 1,
    right_stretch: int = 1,
    *,
    key_width: int = 100,  # width for the "label" columns (Index:, Backend:, etc.)
    gap: int = 10,  # space between the two pairs
) -> QWidget:
    """Two pairs in one row with aligned columns: [key][value]  [key][value]."""

    row = QWidget()
    grid = QGridLayout(row)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(0)

    # Key labels
    l1 = QLabel(left_label)
    l2 = QLabel(right_label)

    for lbl in (l1, l2):
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lbl.setFixedWidth(key_width)
        # lbl.setStyleSheet("QLabel { color: palette(mid); font-weight: 500; }")

    def style_value(w: QWidget):
        w.setStyleSheet(
            """
            QLabel, ElidingPathLabel {
                font-weight: 700;
                color: palette(text);
                background-color: rgba(127,127,127,0.12);
                border: 1px solid rgba(127,127,127,0.18);
                border-radius: 6px;
                padding: 2px 6px;
            }
            """
        )
        if isinstance(w, QLabel):
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        sp = w.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Preferred)
        w.setSizePolicy(sp)

    style_value(left_widget)
    style_value(right_widget)

    # Layout columns: 0=key1, 1=val1, 2=gap spacer, 3=key2, 4=val2
    grid.addWidget(l1, 0, 0)
    grid.addWidget(left_widget, 0, 1)

    spacer = QWidget()
    spacer.setFixedWidth(gap)
    grid.addWidget(spacer, 0, 2)

    grid.addWidget(l2, 0, 3)
    grid.addWidget(right_widget, 0, 4)

    # Stretch values, not keys
    grid.setColumnStretch(1, left_stretch)
    grid.setColumnStretch(4, right_stretch)

    # Prevent keys from stretching
    grid.setColumnStretch(0, 0)
    grid.setColumnStretch(3, 0)
    grid.setColumnStretch(2, 0)

    return row
