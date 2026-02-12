# tests/gui/conftest.py
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from dlclivegui.cameras.factory import CameraFactory
from dlclivegui.gui.main_window import DLCLiveMainWindow


# ---------- Autouse patches to keep GUI tests fast and side-effect-free ----------
@pytest.fixture(autouse=True)
def _patch_camera_factory(monkeypatch, request, fake_backend_factory):
    if request.node.get_closest_marker("gui") is None:
        yield
        return

    monkeypatch.setattr(CameraFactory, "create", staticmethod(fake_backend_factory))
    yield


@pytest.fixture(autouse=True)
def _patch_camera_validation(monkeypatch):
    """
    Accept all cameras regardless of backend and silence warning/error dialogs in the window.
    """
    # 1) Pretend all cameras are available
    monkeypatch.setattr(
        CameraFactory,
        "check_camera_available",
        staticmethod(lambda cam: (True, "")),
    )

    # 2) Silence GUI dialogs during tests
    monkeypatch.setattr(DLCLiveMainWindow, "_show_warning", lambda self, msg: None)
    monkeypatch.setattr(DLCLiveMainWindow, "_show_error", lambda self, msg: None)


@pytest.fixture(autouse=True)
def _patch_dlclive_to_fake(monkeypatch, FakeDLCLiveClass):
    from dlclivegui.services import dlc_processor as dlcp_mod

    monkeypatch.setattr(dlcp_mod, "DLCLive", FakeDLCLiveClass)


@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path):
    """
    Redirect QSettings to a temp directory so persistence tests are deterministic
    and do not touch real user settings.
    """
    from PySide6.QtCore import QSettings

    # Use INI backend for easy temp path redirection
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))

    # Clear keys for this app/org to avoid leakage between tests
    s = QSettings("DeepLabCut", "DLCLiveGUI")
    s.clear()
    s.sync()

    yield


@pytest.fixture(autouse=True)
def no_modal_messageboxes(monkeypatch):
    """
    Fail fast if a QMessageBox is shown unexpectedly.
    This prevents teardown hangs caused by modal dialogs.
    """

    def _report(*args, **kwargs):
        # args often: (parent, title, text, ...)
        title = args[1] if len(args) > 1 else "<no-title>"
        text = args[2] if len(args) > 2 else "<no-text>"
        raise AssertionError(f"Unexpected QMessageBox: {title}\n{text}")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_report))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_report))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(_report))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(_report))
