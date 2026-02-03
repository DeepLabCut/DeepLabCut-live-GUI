# dlclivegui/gui/app.py (or your launcher)
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


def main() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # HiDPI pixmaps - always enabled in Qt 6 so no need to set it explicitly
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO))

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
            splash.close()
            # Keep a reference to avoid premature GC
            app._main_window = DLCLiveMainWindow()
            app._main_window.show()

        QTimer.singleShot(cfg.duration_ms, show_main)
    else:
        app._main_window = DLCLiveMainWindow()
        app._main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - manual start
    main()
