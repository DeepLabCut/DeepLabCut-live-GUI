# dlclivegui/utils/theme.py
from __future__ import annotations

import enum
from pathlib import Path

import qdarkstyle
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

ASSETS = Path(__file__).parent.parent / "assets"
LOGO = str(ASSETS / "logo.png")
LOGO_ALPHA = str(ASSETS / "logo_transparent.png")
SPLASH_SCREEN = str(ASSETS / "welcome.png")
#### Splash screen config
SHOW_SPLASH = True
SPLASH_SCREEN_WIDTH = 600
SPLASH_SCREEN_HEIGHT = 400
SPLASH_SCREEN_DURATION_MS = 1000
SPLASH_KEEP_ASPECT = True


class AppStyle(enum.Enum):
    SYS_DEFAULT = "system"
    DARK = "dark"


def apply_theme(mode: AppStyle, action_dark: QAction, action_light: QAction) -> None:
    app = QApplication.instance()
    if mode == AppStyle.DARK:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
        action_dark.setChecked(True)
        action_light.setChecked(False)
    else:
        app.setStyleSheet("")
        action_dark.setChecked(False)
        action_light.setChecked(True)
