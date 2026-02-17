# tests/gui/test_app_entrypoint.py
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import cv2 as cv
except Exception:
    pytest.skip("OpenCV (opencv-python) is required for these tests.", allow_module_level=True)

import dlclivegui.assets.ascii_art as ascii_mod

MODULE_UNDER_TEST = "dlclivegui.main"


def _import_fresh():
    if MODULE_UNDER_TEST in sys.modules:
        del sys.modules[MODULE_UNDER_TEST]
    return importlib.import_module(MODULE_UNDER_TEST)


@pytest.fixture
def set_use_splash_true(monkeypatch):
    #  Ensure theme.py SHOW_SPLASH is True for tests that rely on it, without affecting other tests
    monkeypatch.setattr("dlclivegui.gui.theme.SHOW_SPLASH", True)


@pytest.fixture
def set_use_splash_false(monkeypatch):
    #  Ensure theme.py SHOW_SPLASH is False for tests that rely on it, without affecting other tests
    monkeypatch.setattr("dlclivegui.gui.theme.SHOW_SPLASH", False)


@pytest.mark.gui
def test_main_with_splash(monkeypatch, set_use_splash_true):
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


@pytest.mark.gui
def test_main_without_splash(monkeypatch, set_use_splash_false):
    appmod = _import_fresh()

    # Patch Qt app creation & window icon
    QApplication_cls = MagicMock(name="QApplication")
    app_instance = MagicMock(name="QApplication.instance")
    QApplication_cls.return_value = app_instance
    monkeypatch.setattr(appmod, "QApplication", QApplication_cls)
    monkeypatch.setattr(appmod, "QIcon", MagicMock(name="QIcon"))

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


# -------------------------
# Fixtures & small helpers
# -------------------------


@pytest.fixture
def tmp_png_gray(tmp_path: Path):
    """Create a simple 16x8 gray gradient PNG without alpha."""
    h, w = 8, 16
    # Horizontal gradient from black to white in BGR
    x = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.tile(x, (h, 1))
    bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    p = tmp_path / "gray.png"
    assert cv.imwrite(str(p), bgr)
    return p


@pytest.fixture
def tmp_png_bgra_logo(tmp_path: Path):
    """Create a small BGRA image with a transparent border and opaque center."""
    h, w = 10, 20
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    # Opaque blue rectangle in center
    bgra[2:8, 5:15, 0] = 255  # B
    bgra[2:8, 5:15, 3] = 255  # A
    p = tmp_path / "logo_bgra.png"
    assert cv.imwrite(str(p), bgra)
    return p


def _force_isatty(monkeypatch, obj, value: bool):
    """
    Ensure obj.isatty() returns value.
    Try instance patch first; if the object disallows attribute assignment,
    patch the method on its class.
    """
    try:
        # Try patching the instance directly
        monkeypatch.setattr(obj, "isatty", lambda: value, raising=False)
    except Exception:
        # Fallback: patch the class method
        cls = type(obj)
        monkeypatch.setattr(cls, "isatty", lambda self: value, raising=True)


@pytest.fixture
def force_tty(monkeypatch):
    """
    Pretend stdout is a TTY and provide a stable terminal size inside the
    module-under-test namespace (and the actual sys).
    """
    # NO_COLOR must be unset for should_use_color("auto")
    monkeypatch.delenv("NO_COLOR", raising=False)

    # Make whatever stdout object exists report TTY=True
    _force_isatty(monkeypatch, sys.stdout, True)
    _force_isatty(monkeypatch, ascii_mod.sys.stdout, True)

    # Ensure terminal size used by the module is deterministic
    monkeypatch.setattr(
        ascii_mod.shutil,
        "get_terminal_size",
        lambda fallback=None: os.terminal_size((80, 24)),
        raising=True,
    )
    return sys.stdout  # not used directly, but handy


@pytest.fixture
def force_notty(monkeypatch):
    """
    Pretend stdout is not a TTY.
    """
    _force_isatty(monkeypatch, sys.stdout, False)
    _force_isatty(monkeypatch, ascii_mod.sys.stdout, False)
    return sys.stdout


# -------------------------
# Terminal / ANSI behavior
# -------------------------


def test_get_terminal_width_tty(force_tty):
    width = ascii_mod.get_terminal_width(default=123)
    assert width == 80


def test_get_terminal_width_notty(force_notty):
    width = ascii_mod.get_terminal_width(default=123)
    assert width == 123


def test_should_use_color_auto_tty(force_tty, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert ascii_mod.should_use_color("auto") is True


def test_should_use_color_auto_no_color(force_tty, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert ascii_mod.should_use_color("auto") is False


def test_should_use_color_modes(force_notty):
    assert ascii_mod.should_use_color("never") is False
    assert ascii_mod.should_use_color("always") is True


def test_terminal_is_wide_enough(force_tty):
    assert ascii_mod.terminal_is_wide_enough(60) is True
    assert ascii_mod.terminal_is_wide_enough(100) is False


# -------------------------
# Image helpers
# -------------------------


def test__to_bgr_converts_gray():
    gray = np.zeros((5, 7), dtype=np.uint8)
    bgr = ascii_mod._to_bgr(gray)
    assert bgr.shape == (5, 7, 3)
    assert bgr.dtype == np.uint8


def test_composite_over_color_bgra(tmp_png_bgra_logo):
    img = cv.imread(str(tmp_png_bgra_logo), cv.IMREAD_UNCHANGED)
    assert img.shape[2] == 4
    bgr = ascii_mod.composite_over_color(img, bg_bgr=(10, 20, 30))
    assert bgr.shape[2] == 3
    # Transparent border should become the bg color
    assert tuple(bgr[0, 0]) == (10, 20, 30)
    # Opaque center should keep blue channel high
    assert bgr[5, 10, 0] == 255


def test_crop_to_content_alpha(tmp_png_bgra_logo):
    img = cv.imread(str(tmp_png_bgra_logo), cv.IMREAD_UNCHANGED)
    cropped = ascii_mod.crop_to_content_alpha(img, alpha_thresh=1, pad=0)
    h, w = cropped.shape[:2]
    assert h == 6  # 2..7 -> 6 rows
    assert w == 10  # 5..14 -> 10 cols
    assert cropped[..., 3].min() == 255


def test_crop_to_content_bg_white(tmp_path):
    # Create white background with a black rectangle
    h, w = 12, 20
    bgr = np.full((h, w, 3), 255, dtype=np.uint8)
    bgr[3:10, 4:15] = 0
    p = tmp_path / "white_bg.png"
    assert cv.imwrite(str(p), bgr)
    cropped = ascii_mod.crop_to_content_bg(bgr, bg="white", tol=10, pad=0)
    assert cropped.shape[0] == 7  # 3..9 -> 7 rows
    assert cropped.shape[1] == 11  # 4..14 -> 11 cols


def test_resize_for_terminal_aspect_env(monkeypatch):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    monkeypatch.setenv("DLCLIVE_ASCII_ASPECT", "0.25")
    resized = ascii_mod.resize_for_terminal(img, width=80, aspect=None)
    # new_h = (h/w) * width * aspect = (100/200)*80*0.25 = 10
    assert resized.shape[:2] == (10, 80)


# -------------------------
# Rendering
# -------------------------


def test_map_luminance_to_chars_simple():
    gray = np.array([[0, 127, 255]], dtype=np.uint8)
    lines = list(ascii_mod._map_luminance_to_chars(gray, fine=False))
    assert len(lines) == 1
    # First char should be the densest in the simple ramp '@', last should be space
    assert lines[0][0] == ascii_mod.ASCII_RAMP_SIMPLE[0]
    assert lines[0][-1] == ascii_mod.ASCII_RAMP_SIMPLE[-1]


def test_color_ascii_lines_basic():
    # Small 2x3 color blocks
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    lines = list(ascii_mod._color_ascii_lines(img, fine=False, invert=False))
    assert len(lines) == 2
    # Expect ANSI 24-bit color sequence present
    assert "\x1b[38;2;" in lines[0]
    # Reset code present
    assert lines[0].endswith("\x1b[0m" * 3) is False  # individual chars have resets, but line won't end with triple


# -------------------------
# Public API: generate & print
# -------------------------


@pytest.mark.parametrize("use_color", ["never", "always"])
def test_generate_ascii_lines_gray(tmp_png_gray, use_color, force_tty):
    lines = list(
        ascii_mod.generate_ascii_lines(
            str(tmp_png_gray),
            width=40,
            aspect=0.5,
            color=use_color,
            fine=False,
            invert=False,
            crop_content=False,
            crop_bg="none",
        )
    )
    assert len(lines) > 0
    # Width equals number of characters per line
    assert all(len(line) == 40 or ("\x1b[38;2;" in line and len(_strip_ansi(line)) == 40) for line in lines)


def _strip_ansi(s: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def test_generate_ascii_lines_crop_alpha(tmp_png_bgra_logo, force_tty):
    lines_no_crop = list(
        ascii_mod.generate_ascii_lines(str(tmp_png_bgra_logo), width=40, aspect=0.5, color="never", crop_content=False)
    )
    lines_crop = list(
        ascii_mod.generate_ascii_lines(str(tmp_png_bgra_logo), width=40, aspect=0.5, color="never", crop_content=True)
    )
    # Both are non-empty; height may change either way depending on aspect ratio
    assert len(lines_no_crop) > 0 and len(lines_crop) > 0
    # Cropping should affect the generated ASCII content
    assert lines_crop != lines_no_crop


def test_print_ascii_writes_file(tmp_png_gray, force_tty, tmp_path):
    out_path = tmp_path / "out.txt"
    ascii_mod.print_ascii(
        str(tmp_png_gray),
        width=30,
        aspect=0.5,
        color="never",
        output=str(out_path),
    )
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    # Expect multiple lines of length 30
    lines = [ln for ln in text.splitlines() if ln]
    assert len(lines) > 0
    assert all(len(ln) == 30 for ln in lines)


def test_build_help_description_tty(tmp_png_bgra_logo, monkeypatch, force_tty):
    monkeypatch.setattr(ascii_mod, "ASCII_IMAGE_PATH", Path(tmp_png_bgra_logo))
    desc = ascii_mod.build_help_description(static_banner=None, color="always", min_width=60)
    assert "DeepLabCut-Live GUI" in desc
    assert "\x1b[36m" in desc  # cyan wrapper now present since TTY is mocked correctly


def test_build_help_description_notty(tmp_png_bgra_logo, monkeypatch, force_notty):
    monkeypatch.setattr(ascii_mod, "ASCII_IMAGE_PATH", Path(tmp_png_bgra_logo))
    desc = ascii_mod.build_help_description(static_banner=None, color="always", min_width=60)
    # Not a TTY -> no banner, just the plain description
    assert desc.strip() == "DeepLabCut-Live GUI — launch the graphical interface."
