# tests/gui/test_misc.py
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

from dlclivegui.gui.misc.eliding_label import ElidingPathLabel


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
        (Qt.ElideMiddle, lambda s: ("…" in s and not s.startswith("…") and not s.endswith("…"))),
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
