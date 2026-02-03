# dlclivegui/gui/misc/splash.py
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QSplashScreen


@dataclass(frozen=True)
class SplashConfig:
    enabled: bool = True
    image: str = ""  # Path to the splash image
    width: int = 600  # Target width (px)
    height: int | None = None  # If None, height is computed from aspect ratio or fallback
    duration_ms: int = 1000  # How long to show the splash
    keep_aspect: bool = True  # Keep aspect ratio when scaling
    bg_color = Qt.black  # Fallback background color when image is missing


def build_splash_pixmap(cfg: SplashConfig) -> QPixmap:
    """
    Build a splash pixmap from config. If the image is invalid, returns a filled pixmap
    with fallback size and background color.
    """
    raw = QPixmap(cfg.image)
    if not raw.isNull():
        target_h = int(cfg.width / (raw.width() / raw.height())) if cfg.height is None else cfg.height
        mode = Qt.KeepAspectRatio if cfg.keep_aspect else Qt.IgnoreAspectRatio
        return raw.scaled(cfg.width, target_h, mode, Qt.SmoothTransformation)

    # Fallback when the image file is invalid/missing
    target_h = cfg.height or 400
    pm = QPixmap(cfg.width, target_h)
    pm.fill(cfg.bg_color)
    return pm


def show_splash(cfg: SplashConfig) -> QSplashScreen:
    """
    Create and show the splash screen from config. Returns the QSplashScreen instance.
    The caller is responsible for closing it.
    """
    pm = build_splash_pixmap(cfg)
    splash = QSplashScreen(pm)
    splash.show()
    return splash
