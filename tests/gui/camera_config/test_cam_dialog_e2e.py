# tests/gui/camera_config/test_cam_dialog_e2e.py
from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from dlclivegui.cameras.base import CameraBackend
from dlclivegui.cameras.factory import CameraFactory, DetectedCamera
from dlclivegui.config import CameraSettings, MultiCameraSettings
from dlclivegui.gui.camera_config.camera_config_dialog import CameraConfigDialog
from dlclivegui.gui.camera_config.loaders import CameraLoadWorker
from dlclivegui.gui.camera_config.preview import PreviewState

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _run_scan_and_wait(dialog: CameraConfigDialog, qtbot, timeout: int = 2000) -> None:
    """
    Trigger a scan via UI and wait for the dialog's scan_finished,
    which now means: UI is stable and available list is populated (or placeholder).
    """
    qtbot.waitUntil(lambda: not dialog._is_scan_running(), timeout=timeout)
    qtbot.wait(50)

    # Wait for the scan started by *this click* to both start and finish
    with qtbot.waitSignals([dialog.scan_started, dialog.scan_finished], timeout=timeout, order="strict"):
        qtbot.mouseClick(dialog.refresh_btn, Qt.LeftButton)

    # Now the list should be stable
    qtbot.waitUntil(lambda: dialog.available_cameras_list.count() > 0, timeout=timeout)


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

    qtbot.waitUntil(lambda: d._preview.loader is None, timeout=2000)
    qtbot.waitUntil(lambda: not d._is_scan_running(), timeout=5000)
    qtbot.wait(50)
    qtbot.waitUntil(lambda: d._preview.state == PreviewState.IDLE, timeout=2000)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.gui
def test_e2e_async_camera_scan(dialog, qtbot):
    _run_scan_and_wait(dialog, qtbot, timeout=2000)
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
    _select_backend_for_active_cam(dialog, cam_row=0)

    # Discover cameras via UI
    _run_scan_and_wait(dialog, qtbot, timeout=2000)
    assert dialog.available_cameras_list.count() == 2

    # Select the second detected camera to avoid duplicate (index 1)
    dialog.available_cameras_list.setCurrentRow(1)
    qtbot.mouseClick(dialog.add_camera_btn, Qt.LeftButton)

    qtbot.waitUntil(lambda: len(dialog._working_settings.cameras) >= 2, timeout=1000)

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

    # scan_finished = UI stable, not necessarily worker fully stopped / controls unlocked
    with qtbot.waitSignal(dialog.scan_finished, timeout=3000):
        pass

    # Wait until scan controls are unlocked (worker finished)
    qtbot.waitUntil(lambda: dialog.refresh_btn.isEnabled(), timeout=3000)
    qtbot.waitUntil(lambda: dialog.backend_combo.isEnabled(), timeout=3000)


@pytest.mark.gui
def test_duplicate_camera_prevented(dialog, qtbot, monkeypatch):
    calls = {"n": 0}

    def _warn(parent, title, text, *args, **kwargs):
        calls["n"] += 1
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warn))

    _select_backend_for_active_cam(dialog, cam_row=0)
    initial_count = dialog.active_cameras_list.count()

    # Scan normally
    _run_scan_and_wait(dialog, qtbot, timeout=2000)
    assert dialog.available_cameras_list.count() == 2

    # Choose the entry that matches index 0 (duplicate)
    dialog.available_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.add_camera_btn, Qt.LeftButton)

    assert dialog.active_cameras_list.count() == initial_count
    assert calls["n"] >= 1


@pytest.mark.gui
def test_max_cameras_prevented(qtbot, monkeypatch, patch_detect_cameras):
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

        _run_scan_and_wait(d, qtbot, timeout=2000)
        assert d.available_cameras_list.count() == 2

        # Try to add any detected camera (should hit MAX_CAMERAS guard)
        d.available_cameras_list.setCurrentRow(1)
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


@pytest.mark.gui
def test_remove_active_camera_works_while_scan_running(dialog, qtbot, monkeypatch):
    """
    Regression test for:
    - 'When coming back to camera config after choosing a camera, it cannot be removed'
    Root cause: scan_running disabled structure edits (Remove/Move).
    Expected: Remove works even while discovery scan is running.
    """

    # Slow down camera detection so scan stays RUNNING long enough for interaction
    def slow_detect(backend, max_devices=10, should_cancel=None, progress_cb=None, **kwargs):
        for i in range(50):
            if should_cancel and should_cancel():
                break
            if progress_cb:
                progress_cb(f"Scanning… {i}")
            time.sleep(0.02)
        return [
            DetectedCamera(index=0, label=f"{backend}-X"),
            DetectedCamera(index=1, label=f"{backend}-Y"),
        ]

    monkeypatch.setattr(CameraFactory, "detect_cameras", staticmethod(slow_detect))

    # Ensure an active row is selected
    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog.active_cameras_list.currentRow() == 0, timeout=1000)

    initial_active = dialog.active_cameras_list.count()
    initial_model = len(dialog._working_settings.cameras)
    assert initial_active == initial_model == 1

    # Trigger scan; wait until scan controls indicate it's running
    qtbot.mouseClick(dialog.refresh_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog._is_scan_running(), timeout=1000)
    qtbot.waitUntil(lambda: dialog.scan_cancel_btn.isVisible(), timeout=1000)

    # Remove button should be enabled even during scan
    qtbot.waitUntil(lambda: dialog.remove_camera_btn.isEnabled(), timeout=1000)

    # Remove the selected active camera during scan
    qtbot.mouseClick(dialog.remove_camera_btn, Qt.LeftButton)

    assert dialog.active_cameras_list.count() == initial_active - 1
    assert len(dialog._working_settings.cameras) == initial_model - 1

    # Clean up: cancel scan so teardown doesn't hang waiting for scan completion
    if dialog.scan_cancel_btn.isVisible() and dialog.scan_cancel_btn.isEnabled():
        qtbot.mouseClick(dialog.scan_cancel_btn, Qt.LeftButton)

    qtbot.waitUntil(lambda: not dialog._is_scan_running(), timeout=3000)


@pytest.mark.gui
def test_ok_updates_internal_multicamera_settings(dialog, qtbot):
    """
    Regression test for:
    - 'adding another camera and hitting OK does not add the new extra camera'
    when caller reads dialog._multi_camera_settings after closing.

    Expected:
    - OK emits updated settings
    - dialog._multi_camera_settings is updated to match accepted settings
    """

    # Ensure backend combo matches the active camera backend, so duplicate logic behaves consistently
    _select_backend_for_active_cam(dialog, cam_row=0)

    # Scan and add a non-duplicate camera (index 1)
    _run_scan_and_wait(dialog, qtbot, timeout=2000)
    dialog.available_cameras_list.setCurrentRow(1)
    qtbot.mouseClick(dialog.add_camera_btn, Qt.LeftButton)

    qtbot.waitUntil(lambda: dialog.active_cameras_list.count() == 2, timeout=1000)
    assert len(dialog._working_settings.cameras) == 2

    # Click OK and capture emitted settings
    with qtbot.waitSignal(dialog.settings_changed, timeout=2000) as sig:
        qtbot.mouseClick(dialog.ok_btn, Qt.LeftButton)

    emitted = sig.args[0]
    assert isinstance(emitted, MultiCameraSettings)
    assert len(emitted.cameras) == 2

    # Check: internal source-of-truth must match accepted state
    assert dialog._multi_camera_settings is not None
    assert len(dialog._multi_camera_settings.cameras) == 2

    # Optional: ensure camera identities match (names/index/backend)
    assert [(c.backend, int(c.index)) for c in dialog._multi_camera_settings.cameras] == [
        (c.backend, int(c.index)) for c in emitted.cameras
    ]
