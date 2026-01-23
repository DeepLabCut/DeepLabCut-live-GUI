# dlclivegui_tests/conftest.py
import numpy as np
import pytest
from PySide6.QtCore import Qt

from dlclivegui import gui
from dlclivegui.config import (
    DEFAULT_CONFIG,
    ApplicationSettings,
    CameraSettings,
    MultiCameraSettings,
)
from dlclivegui.dlc_processor import DLCLiveProcessor
from dlclivegui.multi_camera_controller import MultiCameraController
from dlclivegui_tests.fixtures.fake_cameras_backend import StubCameraBackend, StubCameraBehavior
from dlclivegui_tests.fixtures.fake_DLClive import StubDLCLive


@pytest.fixture
def app_config_two_cams(tmp_path) -> ApplicationSettings:
    cfg = ApplicationSettings.from_dict(DEFAULT_CONFIG.to_dict())

    cam_a = CameraSettings(name="CamA", backend="fake", index=0, enabled=True, fps=30.0)
    cam_b = CameraSettings(name="CamB", backend="fake", index=1, enabled=True, fps=30.0)

    cfg.multi_camera = MultiCameraSettings(
        cameras=[cam_a, cam_b], max_cameras=4, tile_layout="auto"
    )
    cfg.camera = cam_a

    cfg.recording.directory = str(tmp_path / "videos")
    cfg.recording.enabled = True
    return cfg


@pytest.fixture(autouse=True)
def _patch_dlclive(monkeypatch):
    """
    Patch dlclive dependency used by dlclivegui.dlc_processor.DLCLiveProcessor
    so it never imports/loads a real model.
    """
    import dlclivegui.dlc_processor as dlcp

    monkeypatch.setattr(dlcp, "DLCLive", StubDLCLive)


@pytest.fixture(autouse=True)
def _patch_camera_factory_create(monkeypatch):
    """
    Patch camera backend creation at the *controller usage site* so real
    MultiCameraController/SingleCameraWorker run unchanged, but hardware is replaced.
    """
    import dlclivegui.multi_camera_controller as mcc

    palette = [(30, 30, 200), (30, 200, 30), (200, 30, 30), (160, 160, 30)]

    def _create_stub(settings: CameraSettings):
        cam_id = f"{settings.backend}:{settings.index}"
        color = palette[int(settings.index) % len(palette)]

        # Optional: tailor per camera to hit conversion code paths
        # emit_mode = "bgr"
        emit_mode = "bgr"

        return StubCameraBackend(
            camera_id=cam_id,
            shape=(240, 320, 3),
            base_color_bgr=color,
            behavior=StubCameraBehavior(),  # configure later for error tests
            emit_mode=emit_mode,
            fake_fps=float(settings.fps) if settings.fps else 30.0,
        )

    monkeypatch.setattr(mcc.CameraFactory, "create", staticmethod(_create_stub))


@pytest.fixture(autouse=True)
def _patch_camera_validation(monkeypatch):
    """
    Test-only: accept all cameras regardless of backend and suppress warning dialogs.
    (Validation will be unit-tested separately.)
    """
    monkeypatch.setattr(
        gui.CameraFactory,
        "check_camera_available",
        staticmethod(lambda cam: (True, "")),
    )
    monkeypatch.setattr(gui.MainWindow, "_show_warning", lambda self, msg: None)
    monkeypatch.setattr(gui.MainWindow, "_show_error", lambda self, msg: None)


@pytest.fixture
def multi_camera_controller():
    """
    Real MultiCameraController. (Cameras are stubbed by _patch_camera_factory_create.)
    """
    ctrl = MultiCameraController()
    try:
        yield ctrl
    finally:
        # Ensure clean shutdown of threads between tests
        if ctrl.is_running():
            ctrl.stop(wait=True)
        ctrl.deleteLater()


@pytest.fixture
def dlc_processor():
    """
    Real DLCLiveProcessor backed by StubDLCLive (patched via _patch_dlclive).
    """
    proc = DLCLiveProcessor()
    try:
        yield proc
    finally:
        proc.shutdown()


@pytest.fixture
def window(qtbot, app_config_two_cams, multi_camera_controller, dlc_processor):
    w = gui.MainWindow(
        config=app_config_two_cams,
        multi_camera_controller=multi_camera_controller,
        dlc_processor=dlc_processor,
    )
    qtbot.addWidget(w)

    w.setAttribute(Qt.WA_DontShowOnScreen, True)
    w.show()
    return w
