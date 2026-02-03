from __future__ import annotations

import importlib
from unittest.mock import MagicMock


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
    cfg = splashmod.SplashConfig(image="missing.png", width=600, height=None, keep_aspect=True)

    raw = MagicMock()
    raw.isNull.return_value = True

    empty = MagicMock()
    QPixmap = MagicMock(side_effect=[raw, empty])
    monkeypatch.setattr(splashmod, "QPixmap", QPixmap)

    pm = splashmod.build_splash_pixmap(cfg)
    assert pm is empty
    empty.fill.assert_called_once_with(splashmod.Qt.black)
