# dlclivegui/gui/main.py
from __future__ import annotations

import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from dlclivegui.gui.main_window import DLCLiveMainWindow
from dlclivegui.gui.misc.splash import SplashConfig, show_splash
from dlclivegui.gui.theme import (
    LOGO,
    SHOW_SPLASH,
    SPLASH_KEEP_ASPECT,
    SPLASH_SCREEN,
    SPLASH_SCREEN_DURATION_MS,
    SPLASH_SCREEN_HEIGHT,
    SPLASH_SCREEN_WIDTH,
)


def _maybe_allow_keyboard_interrupt(app: QApplication) -> None:
    """
    Gracefully handle Ctrl+C (SIGINT) by closing the main window and quitting Qt.
    """

    def _request_quit() -> None:
        win = getattr(app, "_main_window", None)
        if win is not None:
            # Trigger your existing closeEvent cleanup (camera stop, threads, timers, etc.)
            win.close()
        else:
            app.quit()

    def _sigint_handler(_signum, _frame) -> None:
        QTimer.singleShot(0, _request_quit)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Keepalive timer to allow Python to handle signals while Qt is running.
    sig_timer = QTimer(app)
    sig_timer.setInterval(100)  # 50–200ms typical; keep low overhead
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start()

    if not hasattr(app, "_sig_timer"):
        app._sig_timer = sig_timer
    else:
        raise RuntimeError("QApplication already has _sig_timer attribute, which is reserved for SIGINT handling.")


def main() -> None:
    # signal.signal(signal.SIGINT, signal.SIG_DFL)

    # HiDPI pixmaps - always enabled in Qt 6 so no need to set it explicitly
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO))
    _maybe_allow_keyboard_interrupt(app)

    if SHOW_SPLASH:
        cfg = SplashConfig(
            enabled=True,
            image=SPLASH_SCREEN,
            width=SPLASH_SCREEN_WIDTH,
            height=SPLASH_SCREEN_HEIGHT,
            duration_ms=SPLASH_SCREEN_DURATION_MS,
            keep_aspect=SPLASH_KEEP_ASPECT,
        )
        splash = show_splash(cfg)

        def show_main():
            if splash is not None:
                splash.close()
            # Keep a reference to avoid premature GC
            app._main_window = DLCLiveMainWindow()
            app._main_window.show()

        QTimer.singleShot(cfg.duration_ms, show_main)
    else:
        app._main_window = DLCLiveMainWindow()
        app._main_window.show()

    def _cleanup():
        t = getattr(app, "_sig_timer", None)
        if t is not None:
            t.stop()

    app.aboutToQuit.connect(_cleanup)
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - manual start
    main()
