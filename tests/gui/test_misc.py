# tests/gui/test_misc.py
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QGuiApplication, QMouseEvent

from dlclivegui.gui.misc.drag_spinbox import ScrubDoubleSpinBox, ScrubSpinBox
from dlclivegui.gui.misc.eliding_label import ElidingPathLabel

# ------------------------------
#  Splash pixmap tests
# ------------------------------


def test_build_splash_pixmap_valid(monkeypatch):
    splashmod = importlib.import_module("dlclivegui.gui.misc.splash")
    cfg = splashmod.SplashConfig(image="ignored.png", width=600, height=None, keep_aspect=True)

    raw = MagicMock()
    raw.isNull.return_value = False
    raw.width.return_value = 800
    raw.height.return_value = 400
    raw.scaled.return_value = raw

    QPixmap = MagicMock(return_value=raw)
    monkeypatch.setattr(splashmod, "QPixmap", QPixmap)

    pm = splashmod.build_splash_pixmap(cfg)
    assert pm is raw
    raw.scaled.assert_called_once()


def test_build_splash_pixmap_fallback(monkeypatch):
    splashmod = importlib.import_module("dlclivegui.gui.misc.splash")
    splashmod.SplashConfig(image="missing.png", width=600, height=None, keep_aspect=True)

    raw = MagicMock()
    raw.isNull.return_value = True

    empty = MagicMock()
    QPixmap = MagicMock(side_effect=[raw, empty])
    monkeypatch.setattr(splashmod, "QPixmap", QPixmap)


# -------------------------------
#  ElidingPathLabel tests
# -------------------------------


@pytest.fixture
def label(qtbot):
    lbl = ElidingPathLabel("")
    qtbot.addWidget(lbl)
    lbl.show()
    qtbot.waitExposed(lbl)
    return lbl


def set_label_width_to_fit_full_text(lbl: ElidingPathLabel, margin: int = 12):
    """Set the label wide enough so it can display the full text without elision."""
    fm = lbl.fontMetrics()
    full = lbl.toolTip()  # always the full text by design
    needed = fm.horizontalAdvance(full) + margin
    lbl.setFixedWidth(needed)


def set_label_to_narrow(lbl: ElidingPathLabel, width: int = 60):
    """Make the label narrow to force elision."""
    lbl.setFixedWidth(width)


def test_tooltip_is_full_text_and_plain_text(label, qtbot):
    # Contains literal '<' and '>' and should be treated as plain text
    txt = r"/very/long/path/run_<timestamp>/trial_<camera>.avi"
    label.set_full_text(txt)

    # Tooltip always equals the full text
    assert label.toolTip() == txt

    # The label uses PlainText format (so '<' and '>' are not interpreted as HTML)
    assert label.textFormat() == Qt.PlainText


def test_settext_aliases_set_full_text(label, qtbot):
    txt = "C:/data/session/run_<next>/file.mp4"
    label.setText(txt)  # overridden to call set_full_text
    assert label.toolTip() == txt

    # Widen sufficiently: no elision -> text() == full
    set_label_width_to_fit_full_text(label)
    qtbot.wait(10)
    assert label.text() == txt


def test_elides_when_narrow_and_restores_when_wide(label, qtbot):
    full = "C:/a/very/very/long/path/that/should/elide/in/the/middle/file.avi"
    label.set_full_text(full)

    # Narrow -> should contain an ellipsis and be different from full
    set_label_to_narrow(label, width=80)
    qtbot.wait(10)
    narrow_text = label.text()
    assert "…" in narrow_text  # U+2026
    assert narrow_text != full

    # Wide -> no elision, should match full
    set_label_width_to_fit_full_text(label)
    qtbot.wait(10)
    assert label.text() == full


@pytest.mark.parametrize(
    "mode, assert_fn",
    [
        (Qt.ElideLeft, lambda s: s.startswith("…")),
        (Qt.ElideRight, lambda s: s.endswith("…")),
        (Qt.ElideMiddle, lambda s: "…" in s and not s.startswith("…") and not s.endswith("…")),
    ],
)
def test_elide_modes_affect_ellipsis_position(qtbot, mode, assert_fn):
    lbl = ElidingPathLabel(elide_mode=mode)
    qtbot.addWidget(lbl)
    lbl.show()

    txt = "ABCDEFGHIJKLmnopqrstuvwxyz0123456789"
    lbl.set_full_text(txt)

    # Force elision by making it very narrow
    lbl.setFixedWidth(70)
    qtbot.wait(10)

    elided = lbl.text()
    assert "…" in elided
    assert assert_fn(elided)


def test_resize_event_reelides(label, qtbot):
    """Shrinking then expanding should re-elide and then restore the full text."""
    full = "C:/some/pretty/long/path/to/something/useful.bin"
    label.set_full_text(full)

    # Wide first
    set_label_width_to_fit_full_text(label)
    qtbot.wait(10)
    assert label.text() == full

    # Shrink -> elided
    set_label_to_narrow(label, width=80)
    qtbot.wait(10)
    assert "…" in label.text()

    # Expand -> restored
    set_label_width_to_fit_full_text(label)
    qtbot.wait(10)
    assert label.text() == full


def test_click_copies_full_text_to_clipboard(label, qtbot):
    txt = "/data/session/run_<timestamp>/trial_<camera>.mp4"
    label.set_full_text(txt)
    set_label_to_narrow(label, 80)  # ensure elided visually
    qtbot.wait(10)

    qtbot.mouseClick(label, Qt.LeftButton)
    copied = QGuiApplication.clipboard().text()
    assert copied == txt


def test_cursor_and_defaults(label):
    """Sanity checks on usability defaults set in __init__."""
    # pointing-hand cursor for click-to-copy
    assert label.cursor().shape() == Qt.PointingHandCursor
    # no wrapping -> avoids squashing vertically
    assert not label.wordWrap()
    # selectable by mouse -> allows Ctrl+C too
    assert label.textInteractionFlags() & Qt.TextSelectableByMouse


# -------------------------------
#  ScrubSpinBox and ScrubDoubleSpinBox tests
# -------------------------------


def _send_mouse(widget, etype, pos: QPoint, *, button=Qt.LeftButton, buttons=Qt.LeftButton, modifiers=Qt.NoModifier):
    """
    Send a QMouseEvent directly to widget.

    This avoids flakiness where test helpers send move events with no pressed buttons.
    """
    local = QPointF(pos)
    global_pos = widget.mapToGlobal(pos)
    global_f = QPointF(global_pos)

    ev = QMouseEvent(
        etype,
        local,  # localPos
        local,  # scenePos (Qt6 signature wants both; using local is fine for widgets)
        global_f,  # globalPos
        button,
        buttons,
        modifiers,
    )
    # send the event synchronously
    QApplication = widget.window().windowHandle().screen().context().application() if False else None  # noqa: F841
    # We can just call widget.event()/QCoreApplication.sendEvent; widget.event is enough.
    widget.event(ev)
    return ev


def _press(widget, pos: QPoint, *, modifiers=Qt.NoModifier):
    return _send_mouse(
        widget, QEvent.MouseButtonPress, pos, button=Qt.LeftButton, buttons=Qt.LeftButton, modifiers=modifiers
    )


def _move(widget, pos: QPoint, *, modifiers=Qt.NoModifier):
    # For move, button is NoButton but buttons indicates what's currently pressed
    return _send_mouse(widget, QEvent.MouseMove, pos, button=Qt.NoButton, buttons=Qt.LeftButton, modifiers=modifiers)


def _release(widget, pos: QPoint, *, modifiers=Qt.NoModifier):
    return _send_mouse(
        widget, QEvent.MouseButtonRelease, pos, button=Qt.LeftButton, buttons=Qt.NoButton, modifiers=modifiers
    )


@pytest.fixture
def spin(qtbot):
    w = ScrubSpinBox(pixels_per_step=6)
    w.setRange(-10_000, 10_000)
    w.setSingleStep(2)
    w.setValue(10)
    w.resize(120, 30)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    return w


@pytest.fixture
def dspin(qtbot):
    w = ScrubDoubleSpinBox(pixels_per_step=6)
    w.setRange(-10_000.0, 10_000.0)
    w.setSingleStep(0.5)
    w.setDecimals(4)
    w.setValue(10.0)
    w.resize(120, 30)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    return w


def test_init_defaults_spinbox(spin):
    # Nice UX: don’t immediately rewrite value while typing
    assert spin.keyboardTracking() is False

    # Cursor is set to horizontal sizing cursor
    assert spin.cursor().shape() == Qt.SizeHorCursor

    # Tooltip gets initialized (should include scrub hint when empty)
    tip = spin.toolTip()
    assert "Drag to adjust" in tip
    assert "Shift=slow" in tip
    assert "Ctrl=fast" in tip


def test_settooltip_instructions_enabled(spin):
    spin.setToolTip("Hello")
    tip = spin.toolTip()
    assert "Hello" in tip
    assert "Drag to adjust" in tip
    assert "<br" in tip or "br" in tip  # tolerant to formatting


def test_settooltip_disable_instructions(spin):
    spin.setToolTip("Hello", disable_instructions=True)
    assert spin.toolTip() == "Hello"

    spin.setToolTip("", disable_instructions=True)
    assert spin.toolTip() == ""


def test_no_scrub_until_threshold(spin):
    start = QPoint(10, 10)
    _press(spin, start)

    # Move less than threshold (< 3 px): should NOT start scrubbing
    _move(spin, QPoint(12, 10))  # dx = +2
    assert spin.value() == 10

    # Release
    _release(spin, QPoint(12, 10))
    assert spin.value() == 10


def test_accumulation_requires_pixels_per_step(spin):
    # pixels_per_step = 6, singleStep = 2
    start = QPoint(10, 10)
    _press(spin, start)

    # Cross threshold but not enough for one step: dx=4 => dragging true, steps=0
    _move(spin, QPoint(14, 10))
    assert spin.value() == 10

    # Another +2 => accumulated 6 => one step => +2
    _move(spin, QPoint(16, 10))
    assert spin.value() == 12

    _release(spin, QPoint(16, 10))


def test_spinbox_shift_modifier_is_slow_but_coerces_to_min_1(spin):
    # singleStep = 2
    start = QPoint(10, 10)
    _press(spin, start)

    # +6px => 1 step; Shift makes step=0.2, coerced to 1 for int spinbox
    _move(spin, QPoint(16, 10), modifiers=Qt.ShiftModifier)
    assert spin.value() == 11

    _release(spin, QPoint(16, 10))


def test_spinbox_ctrl_modifier_is_fast(spin):
    # singleStep = 2, ctrl => step * 10 => 20
    start = QPoint(10, 10)
    _press(spin, start)

    _move(spin, QPoint(16, 10), modifiers=Qt.ControlModifier)
    assert spin.value() == 30  # 10 + 1*20

    _release(spin, QPoint(16, 10))


def test_spinbox_shift_and_ctrl_cancel_to_base_step(spin):
    # step 2 *0.1 *10 => 2, coerced to 2
    start = QPoint(10, 10)
    _press(spin, start)

    _move(spin, QPoint(16, 10), modifiers=Qt.ShiftModifier | Qt.ControlModifier)
    assert spin.value() == 12  # 10 + 2

    _release(spin, QPoint(16, 10))


def test_spinbox_rounding_in_set_value(spin):
    # Directly test the type-specific hook behavior via public effect
    spin.setValue(0)
    spin._scrub_set_value(1.6)  # rounds to 2
    assert spin.value() == 2

    spin._scrub_set_value(1.4)  # rounds to 1
    assert spin.value() == 1


def test_double_spinbox_basic_step(dspin):
    # singleStep = 0.5
    start = QPoint(10, 10)
    _press(dspin, start)

    _move(dspin, QPoint(16, 10))
    assert dspin.value() == pytest.approx(10.5)

    _release(dspin, QPoint(16, 10))


def test_double_spinbox_shift_modifier_fractional(dspin):
    # shift => 0.5 * 0.1 = 0.05
    start = QPoint(10, 10)
    _press(dspin, start)

    _move(dspin, QPoint(16, 10), modifiers=Qt.ShiftModifier)
    assert dspin.value() == pytest.approx(10.05)

    _release(dspin, QPoint(16, 10))


def test_double_spinbox_ctrl_modifier_fast(dspin):
    # ctrl => 0.5 * 10 = 5.0
    start = QPoint(10, 10)
    _press(dspin, start)

    _move(dspin, QPoint(16, 10), modifiers=Qt.ControlModifier)
    assert dspin.value() == pytest.approx(15.0)

    _release(dspin, QPoint(16, 10))


def test_double_spinbox_coerce_step_avoids_zero(dspin):
    dspin.setSingleStep(0.5)
    # If step becomes ~0, _scrub_coerce_step should revert to singleStep.
    assert dspin._scrub_coerce_step(0.0) == pytest.approx(0.5)
    assert dspin._scrub_coerce_step(1e-13) == pytest.approx(0.5)  # < 1e-12 threshold
