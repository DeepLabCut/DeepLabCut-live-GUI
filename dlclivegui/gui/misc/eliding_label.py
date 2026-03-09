from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QLabel, QSizePolicy, QToolTip


class ElidingPathLabel(QLabel):
    """
    QLabel that:
      - keeps the full text internally,
      - shows an elided version (middle-ellipsis by default) based on current width,
      - always shows the full text in a tooltip,
      - copies the full text to clipboard on left click,
      - treats text as PlainText (so '<' and '>' render literally).
    """

    def __init__(self, text: str = "", parent=None, elide_mode=Qt.ElideLeft):
        super().__init__(parent)
        self._full_text = text or ""
        self._elide_mode = elide_mode

        # Important defaults for a stable form layout item
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setCursor(Qt.PointingHandCursor)
        self.setTextFormat(Qt.PlainText)  # ensure '<' and '>' display literally

        self._apply_elision()
        self._sync_tooltip()

    @property
    def full_text(self) -> str:
        return self._full_text

    # --- Public API: call this whenever you want to change the full text ---
    def set_full_text(self, text: str) -> None:
        self._full_text = text or ""
        self._apply_elision()
        self._sync_tooltip()

    # Optional: if other code calls setText(), treat it as setting the full text.
    def setText(self, text: str) -> None:  # type: ignore[override]
        self.set_full_text(text)

    # --- Events ---
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_elision()

    def enterEvent(self, event) -> None:
        # Keep tooltip synced (future-proof if something else touches the text)
        self._sync_tooltip()
        super().enterEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            QGuiApplication.clipboard().setText(self._full_text)
            QToolTip.showText(QCursor.pos(), "Copied path", self)
        super().mouseReleaseEvent(event)

    # --- Internals ---
    def _apply_elision(self) -> None:
        fm = self.fontMetrics()
        # A tiny padding so the ellipsis doesn't jitter against the edge
        available = max(0, self.width() - 4)
        elided = fm.elidedText(self._full_text, self._elide_mode, available)
        super().setText(elided)  # bypass our overridden setText()

    def _sync_tooltip(self) -> None:
        self.setToolTip(self._full_text)
