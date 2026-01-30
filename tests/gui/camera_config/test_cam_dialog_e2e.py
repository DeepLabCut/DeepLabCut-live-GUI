# tests/gui/camera_config/test_cam_dialog_e2e.py
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend
from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.gui.camera_config_dialog import CameraConfigDialog
from dlclivegui.utils.config_models import CameraSettingsModel, MultiCameraSettingsModel

# ---------------- Fake backend ----------------


class FakeBackend(CameraBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self._opened = False

    def open(self):
        self._opened = True

    def close(self):
        self._opened = False

    def read(self):
        return np.zeros((30, 40, 3), dtype=np.uint8), 0.1


# ---------------- Fixtures ----------------


@pytest.fixture
def patch_factory(monkeypatch):
    monkeypatch.setattr(CameraFactory, "create", lambda s: FakeBackend(s))
    monkeypatch.setattr(
        CameraFactory,
        "detect_cameras",
        lambda backend, max_devices=10, **kw: [
            DetectedCamera(index=0, label=f"{backend}-X"),
            DetectedCamera(index=1, label=f"{backend}-Y"),
        ],
    )


@pytest.fixture
def dialog(qtbot, patch_factory):
    s = MultiCameraSettingsModel(
        cameras=[
            CameraSettingsModel(name="A", backend="opencv", index=0, enabled=True),
        ]
    )
    d = CameraConfigDialog(None, s)
    qtbot.addWidget(d)
    return d


# ---------------- End‑to‑End tests ----------------


def test_e2e_async_camera_scan(dialog, qtbot):
    qtbot.mouseClick(dialog.refresh_btn, Qt.LeftButton)

    with qtbot.waitSignal(dialog.scan_finished, timeout=2000):
        pass

    assert dialog.available_cameras_list.count() == 2


def test_e2e_preview_start_stop(dialog, qtbot):
    dialog.active_cameras_list.setCurrentRow(0)

    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)

    # loader thread finishes → preview becomes active
    qtbot.waitUntil(lambda: dialog._loader is None and dialog._preview_active, timeout=2000)

    assert dialog._preview_active

    # preview running → pixmap must update
    qtbot.waitUntil(lambda: dialog.preview_label.pixmap() is not None, timeout=2000)

    # stop preview
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)

    assert dialog._preview_active is False
    assert dialog._preview_backend is None


def test_e2e_apply_settings_reopens_preview(dialog, qtbot):
    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)

    # Wait for preview start
    qtbot.waitUntil(lambda: dialog._loader is None and dialog._preview_active, timeout=2000)

    dialog.cam_fps.setValue(99.0)
    qtbot.mouseClick(dialog.apply_settings_btn, Qt.LeftButton)

    # Should still be active → restarted backend
    qtbot.waitUntil(lambda: dialog._preview_active and dialog._preview_backend is not None, timeout=2000)
