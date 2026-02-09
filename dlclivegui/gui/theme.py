# dlclivegui/utils/theme.py
from __future__ import annotations

import enum
from contextlib import ExitStack
from importlib import resources

import qdarkstyle
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

# ---- Splash screen config ----
SHOW_SPLASH = True
SPLASH_SCREEN_WIDTH = 600
SPLASH_SCREEN_HEIGHT = 400
SPLASH_SCREEN_DURATION_MS = 1000
SPLASH_KEEP_ASPECT = True


# Keep a global ExitStack to keep temp files alive as long as needed (e.g., app lifetime)
_resource_stack = ExitStack()


def asset_path(name: str) -> str:
    """
    Return a real filesystem path to a packaged asset using importlib.resources.
    The path remains valid while the process runs (managed by _resource_stack).
    """
    # Point to the *package* that contains assets (dlclivegui/assets)
    files = resources.files("dlclivegui.assets").joinpath(name)

    # as_file() yields a context manager that provides a concrete path even
    # for zipped resources; keep it open via a global ExitStack.
    path_ctx = resources.as_file(files)
    real_path = _resource_stack.enter_context(path_ctx)
    return str(real_path)


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


LOGO = asset_path("logo.png")
LOGO_ALPHA = asset_path("logo_transparent.png")
SPLASH_SCREEN = asset_path("welcome.png")
