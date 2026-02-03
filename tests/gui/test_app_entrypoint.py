# tests/gui/test_app_entrypoint.py
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

MODULE_UNDER_TEST = "dlclivegui.main"


def _import_fresh():
    if MODULE_UNDER_TEST in sys.modules:
        del sys.modules[MODULE_UNDER_TEST]
    return importlib.import_module(MODULE_UNDER_TEST)


def test_main_with_splash(monkeypatch):
    appmod = _import_fresh()

    # --- Patch Qt app & icon in the entry module's namespace ---
    QApplication_cls = MagicMock(name="QApplication")
    app_instance = MagicMock(name="QApplication.instance")
    QApplication_cls.return_value = app_instance
    monkeypatch.setattr(appmod, "QApplication", QApplication_cls)

    QIcon_cls = MagicMock(name="QIcon")
    monkeypatch.setattr(appmod, "QIcon", QIcon_cls)

    # --- Patch theme flags/constants as they are imported into main.py ---
    appmod.SHOW_SPLASH = True
    appmod.SPLASH_SCREEN = "path/to/splash.png"
    appmod.SPLASH_SCREEN_WIDTH = 640
    appmod.SPLASH_SCREEN_HEIGHT = None
    appmod.SPLASH_KEEP_ASPECT = True
    appmod.SPLASH_SCREEN_DURATION_MS = 1234

    # --- Patch the centralized splash API used by main.py ---
    splash_obj = MagicMock(name="QSplashScreen.mock")
    show_splash_mock = MagicMock(name="show_splash", return_value=splash_obj)
    monkeypatch.setattr(appmod, "show_splash", show_splash_mock)

    # Fire the timer immediately (don’t wait real time)
    captured_ms = {}

    def immediate_single_shot(ms, fn):
        captured_ms["ms"] = ms
        fn()

    monkeypatch.setattr(appmod.QTimer, "singleShot", lambda ms, fn: immediate_single_shot(ms, fn))

    # Prevent pytest from exiting
    captured_exit = {}
    monkeypatch.setattr(appmod.sys, "exit", lambda code: captured_exit.setdefault("code", code))

    # Mock the main window construction & show()
    win_instance = MagicMock(name="DLCLiveMainWindow.instance")
    DLCLiveMainWindow_cls = MagicMock(name="DLCLiveMainWindow", return_value=win_instance)
    monkeypatch.setattr(appmod, "DLCLiveMainWindow", DLCLiveMainWindow_cls)

    # --- Run ---
    appmod.main()

    # --- Assertions ---
    QApplication_cls.assert_called_once_with(sys.argv)
    app_instance.setWindowIcon.assert_called_once()
    QIcon_cls.assert_called_once_with(appmod.LOGO)

    show_splash_mock.assert_called_once()
    cfg = show_splash_mock.call_args[0][0]  # SplashConfig passed to show_splash
    assert cfg.image == appmod.SPLASH_SCREEN
    assert cfg.width == appmod.SPLASH_SCREEN_WIDTH
    assert cfg.height == appmod.SPLASH_SCREEN_HEIGHT
    assert cfg.keep_aspect == appmod.SPLASH_KEEP_ASPECT
    assert cfg.duration_ms == appmod.SPLASH_SCREEN_DURATION_MS

    assert captured_ms["ms"] == appmod.SPLASH_SCREEN_DURATION_MS
    splash_obj.close.assert_called_once()

    DLCLiveMainWindow_cls.assert_called_once_with()
    win_instance.show.assert_called_once()

    app_instance.exec.assert_called_once()
    assert captured_exit["code"] == app_instance.exec.return_value


def test_main_without_splash(monkeypatch):
    appmod = _import_fresh()

    # Patch Qt app creation & window icon
    QApplication_cls = MagicMock(name="QApplication")
    app_instance = MagicMock(name="QApplication.instance")
    QApplication_cls.return_value = app_instance
    monkeypatch.setattr(appmod, "QApplication", QApplication_cls)
    monkeypatch.setattr(appmod, "QIcon", MagicMock(name="QIcon"))

    # Force the no-splash branch
    appmod.SHOW_SPLASH = False

    # show_splash should not be called
    show_splash_mock = MagicMock(name="show_splash")
    monkeypatch.setattr(appmod, "show_splash", show_splash_mock)

    # Timer should not be used when there is no splash
    calls = {"count": 0}
    monkeypatch.setattr(appmod.QTimer, "singleShot", lambda *_a, **_k: calls.__setitem__("count", calls["count"] + 1))

    # Prevent sys.exit from stopping the test process
    monkeypatch.setattr(appmod.sys, "exit", lambda _code: None)

    # Mock the main window
    win_instance = MagicMock(name="DLCLiveMainWindow.instance")
    monkeypatch.setattr(appmod, "DLCLiveMainWindow", MagicMock(return_value=win_instance))

    # Run
    appmod.main()

    # Validate branch behavior
    show_splash_mock.assert_not_called()
    assert calls["count"] == 0
    win_instance.show.assert_called_once()
