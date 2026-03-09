from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


class _ScrubMixin:
    """
    Shared scrubbing behavior for spinboxes.

    Requires the subclass to implement:
      - _scrub_get_value() -> float
      - _scrub_set_value(v: float) -> None
      - _scrub_get_step() -> float
      - _scrub_coerce_step(step: float) -> float
    """

    def _scrub_init(self, *, scrub_button=Qt.LeftButton, pixels_per_step: int = 6) -> None:
        self._scrub_button = scrub_button
        self._pixels_per_step = max(1, int(pixels_per_step))
        self._dragging = False
        self._press_pos: QPoint | None = None
        self._press_value: float = 0.0
        self._accum_dx: int = 0

        # Nice UX: don’t immediately rewrite value while typing
        self.setKeyboardTracking(False)

        # Optional: give a hint cursor
        self.setCursor(Qt.SizeHorCursor)

        # Initialize tooltip (keeps your pattern)
        self.setToolTip("")

    def setToolTip(self, text: str, disable_instructions: bool = False) -> None:
        """Override to optionally include scrubbing instructions in the tooltip.

        Args:
            text: The main tooltip text to show (can be empty).
            disable_instructions: If True, do not include scrubbing instructions in the tooltip.
        """
        if disable_instructions:
            super().setToolTip(text)
            return

        # Add usage instructions to the tooltip (real HTML, not escaped entities)
        scrub_hint = "<i>Drag to adjust &nbsp; Shift=slow &nbsp; Ctrl=fast</i>"

        if not text:
            super().setToolTip(f"<qt>{scrub_hint}</qt>")
        else:
            super().setToolTip(f"<qt>{text}<br>{scrub_hint}</qt>")

    def mousePressEvent(self, event):
        if event.button() == self._scrub_button:
            self._press_pos = event.pos()
            self._press_value = float(self._scrub_get_value())
            self._accum_dx = 0
            self._dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is None:
            super().mouseMoveEvent(event)
            return

        dx = event.pos().x() - self._press_pos.x()

        # Only start scrubbing after a small threshold to preserve normal click-to-edit behavior
        if not self._dragging and abs(dx) < 3:
            super().mouseMoveEvent(event)
            return

        self._dragging = True

        # Convert pixel movement into steps (with accumulation so it feels smooth)
        self._accum_dx += dx
        self._press_pos = event.pos()

        steps = int(self._accum_dx / self._pixels_per_step)
        if steps == 0:
            event.accept()
            return

        # Consume used delta
        self._accum_dx -= steps * self._pixels_per_step

        # Base step size comes from singleStep()
        step = float(self._scrub_get_step())

        # Modifiers for fine/coarse control
        mods = event.modifiers()
        if mods & Qt.ShiftModifier:
            step *= 0.1
        if mods & Qt.ControlModifier:
            step *= 10.0

        # Type-specific step constraints (e.g., int step must stay >= 1)
        step = float(self._scrub_coerce_step(step))

        new_value = float(self._scrub_get_value()) + steps * step
        self._scrub_set_value(new_value)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._press_pos is not None and event.button() == self._scrub_button:
            # If we were dragging, prevent the release from selecting text/clicking arrows etc.
            if self._dragging:
                event.accept()
            self._press_pos = None
            self._dragging = False
            return
        super().mouseReleaseEvent(event)


class ScrubSpinBox(_ScrubMixin, QSpinBox):
    """
    QSpinBox with click-drag scrubbing:
    - Drag horizontally to adjust the value
    - Shift: fine control
    - Ctrl: coarse control
    """

    def __init__(self, *args, scrub_button=Qt.LeftButton, pixels_per_step=6, **kwargs):
        super().__init__(*args, **kwargs)
        self._scrub_init(scrub_button=scrub_button, pixels_per_step=pixels_per_step)

    # ---- type-specific hooks ----
    def _scrub_get_value(self) -> float:
        return float(int(self.value()))

    def _scrub_set_value(self, v: float) -> None:
        self.setValue(int(round(v)))

    def _scrub_get_step(self) -> float:
        # QSpinBox.singleStep() is int
        return float(int(self.singleStep()))

    def _scrub_coerce_step(self, step: float) -> float:
        # For integers, ensure at least 1 step
        s = int(round(step))
        return float(max(1, s))


class ScrubDoubleSpinBox(_ScrubMixin, QDoubleSpinBox):
    """
    QDoubleSpinBox with click-drag scrubbing:
    - Drag horizontally to adjust the value
    - Shift: fine control
    - Ctrl: coarse control
    """

    def __init__(self, *args, scrub_button=Qt.LeftButton, pixels_per_step=6, **kwargs):
        super().__init__(*args, **kwargs)
        self._scrub_init(scrub_button=scrub_button, pixels_per_step=pixels_per_step)

    # ---- type-specific hooks ----
    def _scrub_get_value(self) -> float:
        return float(self.value())

    def _scrub_set_value(self, v: float) -> None:
        self.setValue(float(v))

    def _scrub_get_step(self) -> float:
        return float(self.singleStep())

    def _scrub_coerce_step(self, step: float) -> float:
        # For doubles, allow fractional steps (but avoid zero)
        return step if abs(step) > 1e-12 else float(self.singleStep())
