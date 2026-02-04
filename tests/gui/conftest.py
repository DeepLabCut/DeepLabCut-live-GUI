# tests/gui/conftest.py
from __future__ import annotations

import pytest

from dlclivegui.cameras.factory import CameraFactory
from dlclivegui.config import CameraSettings
from dlclivegui.gui.main_window import DLCLiveMainWindow


# ---------- Autouse patches to keep GUI tests fast and side-effect-free ----------
@pytest.fixture(autouse=True)
def _patch_camera_factory(monkeypatch, request, fake_backend_factory):
    """
    Replace hardware backends with FakeBackend globally for GUI tests.
    We patch at the central creation point used by the controller.
    """
    if request.node.get_closest_marker("gui") is None:
        yield
        return

    def _create_stub(settings: CameraSettings):
        # FakeBackend ignores 'backend' and produces deterministic frames
        return fake_backend_factory(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create_stub))
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
def _patch_dlclive_to_fake(monkeypatch, fake_dlclive_factory):
    """
    Ensure dlclive is replaced by the test double in the DLCLiveProcessor module.
    (The window will instantiate DLCLiveProcessor internally, which imports DLCLive.)
    """
    from dlclivegui.services import dlc_processor as dlcp_mod

    monkeypatch.setattr(dlcp_mod, "DLCLive", fake_dlclive_factory)


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
