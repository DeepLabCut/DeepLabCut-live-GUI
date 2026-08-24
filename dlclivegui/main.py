# dlclivegui/gui/main.py
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from dlclivegui.assets import ascii_art as art
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
    Gracefully handle Ctrl+C/SIGTERM by closing the main window and quitting Qt.

    Notes:
    - The small timer keeps Python signal handling responsive while Qt owns the event loop.
    - First Ctrl+C tries graceful cleanup via closeEvent().
    - Second Ctrl+C exits immediately with code 130.
    """
    quitting = {"requested": False}

    def _request_quit() -> None:
        if quitting["requested"]:
            return

        quitting["requested"] = True
        logging.info("Keyboard interrupt received, closing application...")

        win = getattr(app, "_main_window", None)

        if win is not None:
            try:
                # Trigger existing closeEvent cleanup:
                # camera stop, controller shutdown, timers, DLC shutdown, etc.
                win.close()
            except Exception:
                logging.exception("Error while closing main window after Ctrl+C")

        # Explicitly ask Qt to leave app.exec().
        # Do this even after win.close(), because closeEvent cleanup can be async
        # and relying only on quitOnLastWindowClosed can be fragile.
        QTimer.singleShot(0, app.quit)

    def _force_exit() -> None:
        logging.warning("Second interrupt received, forcing process exit.")
        os._exit(130)

    def _sigint_handler(_signum, _frame) -> None:
        if quitting["requested"]:
            _force_exit()
        QTimer.singleShot(0, _request_quit)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Ctrl+Break on Windows.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sigint_handler)

    # Useful when process is terminated from shells/process managers.
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sigint_handler)

    # Parent the timer to app so Qt owns its lifetime.
    sig_timer = QTimer(app)
    sig_timer.setInterval(100)
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start()

    old_timer = getattr(app, "_sig_timer", None)
    if old_timer is not None:
        try:
            old_timer.stop()
        except Exception:
            pass

    app._sig_timer = sig_timer


def configure_logging(debug: bool = False) -> None:
    """Configure local application logging."""
    env_debug = os.environ.get("DLCLIVEGUI_DEBUG_LOGGING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
        "debug",
    )

    enabled = bool(debug or env_debug)
    level = logging.DEBUG if enabled else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    logging.getLogger("dlclivegui").setLevel(level)

    if enabled:
        logging.debug("Debug logging enabled.")


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    default_desc = "Welcome to DeepLabCut-Live GUI!"
    no_art_flag = "--no-art" in argv
    wants_help = any(a in ("-h", "--help") for a in argv)

    # Only build banner description if we're about to print help
    if wants_help and not no_art_flag:
        try:
            desc = art.build_help_description()
        except Exception as e:
            logging.warning(f"Failed to build ASCII art for help description: {e}")
            desc = default_desc
    else:
        desc = default_desc

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-art", action="store_true", help="Disable ASCII art in help and when launching.")
    parser.add_argument("--debug-log", action="store_true", help="Enable debug logging.")
    return parser.parse_known_args(argv)


def main() -> None:
    args, _unknown = parse_args()
    configure_logging(debug=args.debug_log)
    logging.info("Starting DeepLabCut-Live GUI...")

    # If you want a startup banner, PRINT it (not log), and only in TTY contexts.
    if not args.no_art and sys.stdout.isatty() and art.terminal_is_wide_enough():
        try:
            print(art.build_help_description(desc="Welcome to DeepLabCut-Live GUI!"))
        except Exception:
            # Keep startup robust; don't fail if banner fails
            pass

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
