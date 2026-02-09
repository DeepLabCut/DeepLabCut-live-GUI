# tests/gui/camera_config/test_cam_dialog_unit.py
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.config import CameraSettings, MultiCameraSettings
from dlclivegui.gui.camera_config_dialog import CameraConfigDialog


@pytest.fixture
def dialog(qtbot, monkeypatch):
    # Patch detect_cameras to avoid hardware access
    monkeypatch.setattr(
        "dlclivegui.cameras.CameraFactory.detect_cameras",
        lambda backend, max_devices=10, **kw: [
            DetectedCamera(index=0, label=f"{backend}-X"),
            DetectedCamera(index=1, label=f"{backend}-Y"),
        ],
    )

    s = MultiCameraSettings(
        cameras=[
            CameraSettings(name="CamA", backend="opencv", index=0, enabled=True),
            CameraSettings(name="CamB", backend="opencv", index=1, enabled=False),
        ]
    )
    d = CameraConfigDialog(None, s)
    qtbot.addWidget(d)
    return d


# ---------------------- UNIT TESTS ----------------------
@pytest.mark.gui
def test_add_camera_populates_working_settings(dialog, qtbot):
    dialog._on_scan_result([DetectedCamera(index=2, label="ExtraCam2")])
    dialog.available_cameras_list.setCurrentRow(0)

    qtbot.mouseClick(dialog.add_camera_btn, Qt.LeftButton)

    added = dialog._working_settings.cameras[-1]
    assert added.index == 2
    assert added.name == "ExtraCam2"


@pytest.mark.gui
def test_remove_camera(dialog, qtbot):
    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.remove_camera_btn, Qt.LeftButton)

    assert len(dialog._working_settings.cameras) == 1
    assert dialog._working_settings.cameras[0].name == "CamB"


@pytest.mark.gui
def test_apply_settings_updates_model(dialog, qtbot):
    dialog.active_cameras_list.setCurrentRow(0)

    dialog.cam_fps.setValue(55.0)
    dialog.cam_gain.setValue(12.0)

    qtbot.mouseClick(dialog.apply_settings_btn, Qt.LeftButton)

    updated = dialog._working_settings.cameras[0]
    assert updated.fps == 55.0
    assert updated.gain == 12.0


@pytest.mark.gui
def test_backend_control_disables_exposure_gain_for_opencv(dialog, monkeypatch):
    from dlclivegui.cameras.base import SupportLevel

    def fake_caps(name: str):
        if name == "opencv":
            return {
                "set_exposure": SupportLevel.UNSUPPORTED,  # or UNSUPPORTED if you prefer
                "set_gain": SupportLevel.UNSUPPORTED,
                "set_resolution": SupportLevel.SUPPORTED,
                "set_fps": SupportLevel.BEST_EFFORT,
                "device_discovery": SupportLevel.SUPPORTED,
                "stable_identity": SupportLevel.SUPPORTED,
            }
        if name == "basler":
            return {
                "set_exposure": SupportLevel.SUPPORTED,
                "set_gain": SupportLevel.SUPPORTED,
                "set_resolution": SupportLevel.SUPPORTED,
                "set_fps": SupportLevel.SUPPORTED,
                "device_discovery": SupportLevel.BEST_EFFORT,
                "stable_identity": SupportLevel.SUPPORTED,
            }
        return {}

    monkeypatch.setattr(
        "dlclivegui.cameras.CameraFactory.backend_capabilities",
        lambda backend_name: fake_caps(backend_name),
        raising=False,
    )

    dialog._update_controls_for_backend("opencv")
    assert not dialog.cam_exposure.isEnabled()
    assert not dialog.cam_gain.isEnabled()

    dialog._update_controls_for_backend("basler")
    assert dialog.cam_exposure.isEnabled()
    assert dialog.cam_gain.isEnabled()
