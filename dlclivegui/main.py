import signal
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from dlclivegui.gui.main_window import DLCLiveMainWindow
from dlclivegui.gui.theme import LOGO, SPLASH_SCREEN


def main() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Enable HiDPI pixmaps (optional but recommended)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO))

    # Load and scale splash pixmap
    raw_pixmap = QPixmap(SPLASH_SCREEN)
    splash_width = 600

    if not raw_pixmap.isNull():
        aspect_ratio = raw_pixmap.width() / raw_pixmap.height()
        splash_height = int(splash_width / aspect_ratio)
        scaled_pixmap = raw_pixmap.scaled(
            splash_width,
            splash_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    else:
        # Fallback: empty pixmap; you can also use a color fill if desired
        splash_height = 400
        scaled_pixmap = QPixmap(splash_width, splash_height)
        scaled_pixmap.fill(Qt.black)

    # Create splash with the *scaled* pixmap
    splash = QSplashScreen(scaled_pixmap)
    splash.show()

    # Let the splash breathe without blocking the event loop
    def show_main():
        splash.close()
        window = DLCLiveMainWindow()
        window.show()

    # Show main window after 1500 ms
    QTimer.singleShot(1000, show_main)

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - manual start
    main()
