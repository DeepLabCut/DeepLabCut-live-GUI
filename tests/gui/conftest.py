# tests/services/gui/conftest.py
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from dlclivegui.cameras import CameraFactory
from dlclivegui.gui.main_window import DLCLiveMainWindow

# from dlclivegui.config import (
#     DEFAULT_CONFIG,
#     ApplicationSettings,
#     CameraSettings,
#     MultiCameraSettings,
# )
from dlclivegui.utils.config_models import (
    DEFAULT_CONFIG,
    ApplicationSettingsModel,
    CameraSettingsModel,
    MultiCameraSettingsModel,
)
from tests.conftest import FakeBackend, FakeDLCLive  # noqa: F401

# ---------- Test helpers: application configuration with two fake cameras ----------


@pytest.fixture
def app_config_two_cams(tmp_path) -> ApplicationSettingsModel:
    """An app config with two enabled cameras (fake backend) and writable recording dir."""
    cfg = ApplicationSettingsModel.from_dict(DEFAULT_CONFIG.to_dict())

    cam_a = CameraSettingsModel(name="CamA", backend="fake", index=0, enabled=True, fps=30.0)
    cam_b = CameraSettingsModel(name="CamB", backend="fake", index=1, enabled=True, fps=30.0)

    cfg.multi_camera = MultiCameraSettingsModel(cameras=[cam_a, cam_b], max_cameras=4, tile_layout="auto")
    cfg.camera = cam_a  # kept for backward-compat single-camera access in UI

    cfg.recording.directory = str(tmp_path / "videos")
    cfg.recording.enabled = True
    return cfg


# ---------- Autouse patches to keep GUI tests fast and side-effect-free ----------


@pytest.fixture(autouse=True)
def _patch_camera_factory(monkeypatch):
    """
    Replace hardware backends with FakeBackend globally for GUI tests.
    We patch at the central creation point used by the controller.
    """

    def _create_stub(settings: CameraSettingsModel):
        # FakeBackend ignores 'backend' and produces deterministic frames
        return FakeBackend(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create_stub))


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
def _patch_dlclive_to_fake(monkeypatch):
    """
    Ensure dlclive is replaced by the test double in the DLCLiveProcessor module.
    (The window will instantiate DLCLiveProcessor internally, which imports DLCLive.)
    """
    from dlclivegui.services import dlc_processor as dlcp_mod

    monkeypatch.setattr(dlcp_mod, "DLCLive", FakeDLCLive)


# ---------- The main window fixture (focus) ----------


@pytest.fixture
def window(qtbot, app_config_two_cams) -> DLCLiveMainWindow:
    """
    Construct the real DLCLiveMainWindow with a valid two-camera config,
    make it headless, show it, and yield it. Threads and timers are managed by close().
    """
    w = DLCLiveMainWindow(config=app_config_two_cams)
    qtbot.addWidget(w)
    # Don't pop windows in CI:
    w.setAttribute(Qt.WA_DontShowOnScreen, True)
    w.show()

    try:
        yield w
    finally:
        # The window's closeEvent stops controllers, recorders, timers, etc.
        # Use .close() to trigger the standard shutdown path.
        try:
            w.close()
        except Exception:
            pass


# ---------- Convenience fixtures that expose controller/processor from the window ----------


@pytest.fixture
def multi_camera_controller(window):
    """
    Return the *controller used by the window* so tests can wait on all_started/all_stopped.
    """
    return window.multi_camera_controller


@pytest.fixture
def dlc_processor(window):
    """
    Return the *processor used by the window* so tests can connect to pose/initialized.
    """
    return window._dlc
