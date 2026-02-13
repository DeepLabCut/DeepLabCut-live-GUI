# tests/gui/camera_config/test_cam_dialog_e2e.py
from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend
from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.config import CameraSettings, MultiCameraSettings
from dlclivegui.gui.camera_config.camera_config_dialog import CameraConfigDialog, CameraLoadWorker
from dlclivegui.gui.camera_config.preview import PreviewState

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _select_backend_for_active_cam(dialog: CameraConfigDialog, cam_row: int = 0) -> str:
    """
    Ensure backend combo is set to the backend of the active camera at cam_row.
    If that backend is not present in the combo, fall back to the current combo backend
    and update the camera setting backend to match (so identity/dup logic stays coherent).
    Returns the backend key actually selected (lowercase).
    """
    # backend requested by the camera settings
    backend = (dialog._working_settings.cameras[cam_row].backend or "").lower()

    idx = dialog.backend_combo.findData(backend)
    if idx >= 0:
        dialog.backend_combo.setCurrentIndex(idx)
        return backend

    # Fallback: use current combo backend (or first item) and update the camera backend to match
    fallback = dialog.backend_combo.currentData()
    if not fallback and dialog.backend_combo.count() > 0:
        fallback = dialog.backend_combo.itemData(0)
        dialog.backend_combo.setCurrentIndex(0)

    fallback = (fallback or "").lower()
    assert fallback, "No backend available in combo"

    # Ensure camera backend matches combo so duplicate logic compares apples-to-apples
    dialog._working_settings.cameras[cam_row].backend = fallback
    # Also update the list item UserRole object (so UI selection holds the updated backend)
    try:
        item = dialog.active_cameras_list.item(cam_row)
        if item is not None:
            cam = item.data(Qt.ItemDataRole.UserRole)
            if cam is not None:
                cam.backend = fallback
                item.setData(Qt.ItemDataRole.UserRole, cam)
    except Exception:
        pass

    # Update labels/UI for consistency
    try:
        dialog._update_active_list_item(cam_row, dialog._working_settings.cameras[cam_row])
        dialog._update_controls_for_backend(fallback)
    except Exception:
        pass

    return fallback


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def patch_detect_cameras(monkeypatch):
    """
    Make discovery deterministic for these tests.
    (GUI conftest patches create(), but not necessarily detect_cameras().)
    """
    monkeypatch.setattr(
        CameraFactory,
        "detect_cameras",
        staticmethod(
            lambda backend, max_devices=10, **kw: [
                DetectedCamera(index=0, label=f"{backend}-X"),
                DetectedCamera(index=1, label=f"{backend}-Y"),
            ]
        ),
    )


@pytest.fixture
def dialog(qtbot, patch_detect_cameras):
    """
    E2E fixture: dialog with scan worker + loader + preview timer enabled.
    Uses a backend that is guaranteed to exist in test registry: 'fake'.
    """
    s = MultiCameraSettings(
        cameras=[
            CameraSettings(name="A", backend="fake", index=0, enabled=True),
        ]
    )
    d = CameraConfigDialog(None, s)
    qtbot.addWidget(d)
    d.show()
    qtbot.waitExposed(d)

    yield d

    # ----- robust teardown -----
    try:
        d._stop_preview()
    except Exception:
        pass

    try:
        d.reject()
    except Exception:
        d.close()

    qtbot.waitUntil(lambda: getattr(d, "_loader", None) is None, timeout=2000)
    qtbot.waitUntil(lambda: getattr(d, "_scan_worker", None) is None, timeout=2000)
    qtbot.waitUntil(lambda: not getattr(d, "_preview_active", False), timeout=2000)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.gui
def test_e2e_async_camera_scan(dialog, qtbot):
    qtbot.mouseClick(dialog.refresh_btn, Qt.LeftButton)
    with qtbot.waitSignal(dialog.scan_finished, timeout=2000):
        pass
    assert dialog.available_cameras_list.count() == 2


@pytest.mark.gui
def test_e2e_preview_start_stop(dialog, qtbot):
    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: dialog._preview.loader is None and dialog._preview.state == PreviewState.ACTIVE, timeout=2000
    )
    assert dialog._preview.backend is not None

    qtbot.waitUntil(lambda: dialog.preview_label.pixmap() is not None, timeout=2000)

    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog._preview.state == PreviewState.IDLE, timeout=2000)

    assert dialog._preview.backend is None
    assert dialog._preview.timer is None


@pytest.mark.gui
def test_e2e_apply_settings_restarts_preview_on_restart_fields(dialog, qtbot, monkeypatch):
    """
    Change a restart-relevant field (fps) and verify preview actually restarts
    by observing open() being called again.
    """

    class CountingBackend(CameraBackend):
        opens = 0

        def __init__(self, settings):
            super().__init__(settings)
            self._opened = False

        def open(self):
            type(self).opens += 1
            self._opened = True

        def close(self):
            self._opened = False

        def read(self):
            return np.zeros((30, 40, 3), dtype=np.uint8), 0.1

    CountingBackend.opens = 0
    monkeypatch.setattr(CameraFactory, "create", staticmethod(lambda s: CountingBackend(s)))

    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: dialog._preview.loader is None and dialog._preview.state == PreviewState.ACTIVE, timeout=2000
    )

    before = CountingBackend.opens
    assert before >= 1

    dialog.cam_fps.setValue(99.0)
    qtbot.mouseClick(dialog.apply_settings_btn, Qt.LeftButton)

    qtbot.waitUntil(lambda: CountingBackend.opens >= before + 1, timeout=2000)
    assert dialog._preview.state == PreviewState.ACTIVE
    assert dialog._preview.backend is not None


@pytest.mark.gui
def test_e2e_apply_settings_does_not_restart_on_crop_or_rotation(dialog, qtbot, monkeypatch):
    """
    Crop/rotation are applied live in preview; Apply should not restart backend.
    We validate by ensuring open() count does not increase.
    """

    class CountingBackend(CameraBackend):
        opens = 0

        def __init__(self, settings):
            super().__init__(settings)
            self._opened = False

        def open(self):
            type(self).opens += 1
            self._opened = True

        def close(self):
            self._opened = False

        def read(self):
            return np.zeros((30, 40, 3), dtype=np.uint8), 0.1

    CountingBackend.opens = 0
    monkeypatch.setattr(CameraFactory, "create", staticmethod(lambda s: CountingBackend(s)))

    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: dialog._preview.loader is None and dialog._preview.state == PreviewState.ACTIVE, timeout=2000
    )

    before = CountingBackend.opens
    assert before >= 1

    dialog.cam_crop_x0.setValue(5)
    dialog.cam_rotation.setCurrentIndex(1)
    qtbot.mouseClick(dialog.apply_settings_btn, Qt.LeftButton)

    qtbot.wait(200)
    assert CountingBackend.opens == before
    assert dialog._preview.state == PreviewState.ACTIVE


@pytest.mark.gui
def test_e2e_selection_change_auto_commits(dialog, qtbot):
    """
    Guard contract: switching selection commits pending edits.
    Use FPS (supported) rather than gain (OpenCV gain is intentionally disabled).
    """
    # Ensure backend combo matches active cam (important for add/dup logic)
    _select_backend_for_active_cam(dialog, cam_row=0)

    # Add second camera deterministically
    dialog._on_scan_result([DetectedCamera(index=1, label="ExtraCam")])
    dialog.available_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.add_camera_btn, Qt.LeftButton)

    assert len(dialog._working_settings.cameras) >= 2

    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog._current_edit_index == 0, timeout=1000)

    dialog.cam_fps.setValue(33.0)
    assert dialog.apply_settings_btn.isEnabled()

    dialog.active_cameras_list.setCurrentRow(1)
    qtbot.waitUntil(lambda: dialog._current_edit_index == 1, timeout=1000)

    assert dialog._working_settings.cameras[0].fps == 33.0


@pytest.mark.gui
def test_cancel_scan(dialog, qtbot, monkeypatch):
    def slow_detect(backend, max_devices=10, should_cancel=None, progress_cb=None, **kwargs):
        for i in range(50):
            if should_cancel and should_cancel():
                break
            if progress_cb:
                progress_cb(f"Scanning… {i}")
            time.sleep(0.02)
        return [DetectedCamera(index=0, label=f"{backend}-X")]

    monkeypatch.setattr(CameraFactory, "detect_cameras", staticmethod(slow_detect))

    qtbot.mouseClick(dialog.refresh_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog.scan_cancel_btn.isVisible(), timeout=1000)

    qtbot.mouseClick(dialog.scan_cancel_btn, Qt.LeftButton)

    with qtbot.waitSignal(dialog.scan_finished, timeout=3000):
        pass

    assert dialog.refresh_btn.isEnabled()
    assert dialog.backend_combo.isEnabled()


@pytest.mark.gui
def test_duplicate_camera_prevented(dialog, qtbot, monkeypatch):
    """
    Duplicate detection compares identity keys including backend.
    Ensure backend combo is set to match existing active camera backend.
    """
    calls = {"n": 0}

    def _warn(parent, title, text, *args, **kwargs):
        calls["n"] += 1
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warn))

    backend = _select_backend_for_active_cam(dialog, cam_row=0)

    initial_count = dialog.active_cameras_list.count()

    # Same backend + same index -> duplicate
    dialog._on_scan_result([DetectedCamera(index=0, label=f"{backend}-X")])
    dialog.available_cameras_list.setCurrentRow(0)

    qtbot.mouseClick(dialog.add_camera_btn, Qt.LeftButton)

    assert dialog.active_cameras_list.count() == initial_count
    assert calls["n"] >= 1


@pytest.mark.gui
def test_max_cameras_prevented(qtbot, monkeypatch, patch_detect_cameras):
    """
    Dialog enforces MAX_CAMERAS enabled cameras. Use backend='fake' for stability.
    """
    calls = {"n": 0}

    def _warn(parent, title, text, *args, **kwargs):
        calls["n"] += 1
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warn))

    s = MultiCameraSettings(
        cameras=[
            CameraSettings(name="C0", backend="fake", index=0, enabled=True),
            CameraSettings(name="C1", backend="fake", index=1, enabled=True),
            CameraSettings(name="C2", backend="fake", index=2, enabled=True),
            CameraSettings(name="C3", backend="fake", index=3, enabled=True),
        ]
    )
    d = CameraConfigDialog(None, s)
    qtbot.addWidget(d)
    d.show()
    qtbot.waitExposed(d)

    try:
        _select_backend_for_active_cam(d, cam_row=0)

        initial_count = d.active_cameras_list.count()

        d._on_scan_result([DetectedCamera(index=4, label="Extra")])
        d.available_cameras_list.setCurrentRow(0)

        qtbot.mouseClick(d.add_camera_btn, Qt.LeftButton)

        assert d.active_cameras_list.count() == initial_count
        assert calls["n"] >= 1
    finally:
        d.reject()


@pytest.mark.gui
def test_ok_auto_applies_pending_edits(dialog, qtbot):
    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog._current_edit_index == 0, timeout=1000)

    dialog.cam_fps.setValue(77.0)
    assert dialog.apply_settings_btn.isEnabled()

    with qtbot.waitSignal(dialog.settings_changed, timeout=2000) as sig:
        qtbot.mouseClick(dialog.ok_btn, Qt.LeftButton)

    emitted = sig.args[0]
    assert emitted.cameras[0].fps == 77.0


@pytest.mark.gui
def test_cancel_loading_preview_button(dialog, qtbot, monkeypatch):
    """
    Deterministic cancel-loading test: slow down worker so Cancel Loading can interrupt.
    """

    def slow_run(self):
        self.progress.emit("Creating backend…")
        time.sleep(0.2)
        if getattr(self, "_cancel", False):
            self.canceled.emit()
            return
        self.progress.emit("Opening device…")
        time.sleep(0.2)
        if getattr(self, "_cancel", False):
            self.canceled.emit()
            return
        self.success.emit(self._cam)

    monkeypatch.setattr(CameraLoadWorker, "run", slow_run)

    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)  # Start Preview => loading active

    qtbot.waitUntil(lambda: dialog._preview.state == PreviewState.LOADING, timeout=1000)

    # Click again => Cancel Loading
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)

    qtbot.waitUntil(lambda: dialog._preview.loader is None and dialog._preview.state == PreviewState.IDLE, timeout=2000)
    assert dialog._preview.backend is None
