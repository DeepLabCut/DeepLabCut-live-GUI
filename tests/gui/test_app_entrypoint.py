# tests/gui/test_app_entrypoint_magicmock.py
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

MODULE_UNDER_TEST = "dlclivegui.main"


def _import_fresh():
    if MODULE_UNDER_TEST in sys.modules:
        del sys.modules[MODULE_UNDER_TEST]
    return importlib.import_module(MODULE_UNDER_TEST)


def test_main_valid_splash(monkeypatch):
    appmod = _import_fresh()

    # ---- Patch Qt classes with MagicMocks ----
    QApplication_cls = MagicMock(name="QApplication")
    app_instance = MagicMock(name="QApplication.instance")
    QApplication_cls.return_value = app_instance
    monkeypatch.setattr(appmod, "QApplication", QApplication_cls)

    # QIcon ctor
    QIcon_cls = MagicMock(name="QIcon")
    monkeypatch.setattr(appmod, "QIcon", QIcon_cls)

    # QPixmap ctor → return a pixmap mock representing a VALID image (isNull=False)
    raw_pixmap = MagicMock(name="QPixmap.raw")
    raw_pixmap.isNull.return_value = False
    raw_pixmap.width.return_value = 800
    raw_pixmap.height.return_value = 400
    # scaled should return another pixmap-like object; returning self is fine
    raw_pixmap.scaled.return_value = raw_pixmap

    QPixmap_cls = MagicMock(name="QPixmap")
    QPixmap_cls.return_value = raw_pixmap
    monkeypatch.setattr(appmod, "QPixmap", QPixmap_cls)

    # QSplashScreen ctor → return splash mock
    splash_instance = MagicMock(name="QSplashScreen.instance")
    QSplashScreen_cls = MagicMock(name="QSplashScreen")
    QSplashScreen_cls.return_value = splash_instance
    monkeypatch.setattr(appmod, "QSplashScreen", QSplashScreen_cls)

    # QTimer.singleShot → call callback immediately (don’t wait 1000ms)
    monkeypatch.setattr(appmod.QTimer, "singleShot", lambda ms, fn: fn())

    # Prevent pytest from exiting when sys.exit is called
    captured_exit = {}
    monkeypatch.setattr(appmod.sys, "exit", lambda code: captured_exit.setdefault("code", code))

    # DLCLiveMainWindow ctor → return window mock with show()
    win_instance = MagicMock(name="DLCLiveMainWindow.instance")
    DLCLiveMainWindow_cls = MagicMock(name="DLCLiveMainWindow", return_value=win_instance)
    monkeypatch.setattr(appmod, "DLCLiveMainWindow", DLCLiveMainWindow_cls)

    # ---- Run ----
    appmod.main()

    # ---- Assertions ----
    # Classmethod used
    QApplication_cls.setAttribute.assert_called_once()
    # App created with argv
    QApplication_cls.assert_called_once_with(sys.argv)
    # Window icon set
    app_instance.setWindowIcon.assert_called_once()
    QIcon_cls.assert_called_once_with(appmod.LOGO)

    # Valid pixmap branch hit
    QPixmap_cls.assert_called_once_with(appmod.SPLASH_SCREEN)
    assert raw_pixmap.isNull.called
    raw_pixmap.scaled.assert_called_once()  # used scaled path
    QSplashScreen_cls.assert_called_once_with(raw_pixmap)
    splash_instance.show.assert_called_once()
    splash_instance.close.assert_called_once()

    # Window constructed and shown
    DLCLiveMainWindow_cls.assert_called_once_with()
    win_instance.show.assert_called_once()

    # sys.exit called with app.exec() result
    app_instance.exec.assert_called_once()
    assert captured_exit["code"] == app_instance.exec.return_value


def test_main_fallback_splash(monkeypatch):
    appmod = _import_fresh()

    # QApplication
    QApplication_cls = MagicMock(name="QApplication")
    app_instance = MagicMock(name="QApplication.instance")
    QApplication_cls.return_value = app_instance
    monkeypatch.setattr(appmod, "QApplication", QApplication_cls)

    # QIcon simple patch
    monkeypatch.setattr(appmod, "QIcon", MagicMock(name="QIcon"))

    # QPixmap needs two different instances:
    #   1) raw (isNull=True) → triggers fallback
    #   2) empty pixmap created with (width, height) → will get fill(Qt.black)
    raw_pixmap = MagicMock(name="QPixmap.raw")
    raw_pixmap.isNull.return_value = True

    empty_pixmap = MagicMock(name="QPixmap.empty")
    # When code calls QPixmap(splash_width, splash_height) and then fill(Qt.black)
    # we want to observe that fill was invoked
    empty_pixmap.fill = MagicMock(name="fill")

    QPixmap_cls = MagicMock(name="QPixmap")
    QPixmap_cls.side_effect = [raw_pixmap, empty_pixmap]  # first call: raw, second call: fallback
    monkeypatch.setattr(appmod, "QPixmap", QPixmap_cls)

    # QSplashScreen
    splash_instance = MagicMock(name="QSplashScreen.instance")
    QSplashScreen_cls = MagicMock(name="QSplashScreen", return_value=splash_instance)
    monkeypatch.setattr(appmod, "QSplashScreen", QSplashScreen_cls)

    # Timer immediate
    monkeypatch.setattr(appmod.QTimer, "singleShot", lambda ms, fn: fn())
    # No-op exit
    monkeypatch.setattr(appmod.sys, "exit", lambda code: None)
    # Dummy window
    win_instance = MagicMock(name="DLCLiveMainWindow.instance")
    monkeypatch.setattr(appmod, "DLCLiveMainWindow", MagicMock(return_value=win_instance))

    # Run
    appmod.main()

    # First QPixmap call with SPLASH_SCREEN
    QPixmap_cls.assert_any_call(appmod.SPLASH_SCREEN)
    # Second QPixmap call with (width, height) fallback constructor
    # The code computes height dynamically; we can at least assert it was called twice:
    assert QPixmap_cls.call_count == 2

    # Fallback branch: fill(Qt.black) was called on the empty pixmap
    empty_pixmap.fill.assert_called_once_with(appmod.Qt.black)

    # Splash created with the empty pixmap
    QSplashScreen_cls.assert_called_once_with(empty_pixmap)
    splash_instance.show.assert_called_once()
    splash_instance.close.assert_called_once()

    # Window shown
    win_instance.show.assert_called_once()
