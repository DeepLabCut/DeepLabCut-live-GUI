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
from dlclivegui.gui.camera_config_dialog import CameraConfigDialog, CameraLoadWorker

# ---------------- Fake backends ----------------


class FakeBackend(CameraBackend):
    """Simple preview backend that always returns an RGB frame."""

    def __init__(self, settings):
        super().__init__(settings)
        self._opened = False

    def open(self):
        self._opened = True

    def close(self):
        self._opened = False

    def read(self):
        return np.zeros((30, 40, 3), dtype=np.uint8), 0.1


class CountingBackend(CameraBackend):
    """Backend that counts opens (used to validate restart behavior)."""

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


# ---------------- Fixtures ----------------


@pytest.fixture
def patch_factory(monkeypatch):
    """
    Patch camera factory so no hardware access occurs, and scan is deterministic.
    Default backend is FakeBackend unless overridden per-test.
    """
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
    """
    E2E fixture: allow scan thread + preview loader + timer to run.
    Includes robust teardown to avoid leaked threads/timers.
    """
    s = MultiCameraSettings(
        cameras=[
            CameraSettings(name="A", backend="opencv", index=0, enabled=True),
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


# ---------------- E2E tests ----------------


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

    qtbot.waitUntil(lambda: dialog._loader is None and dialog._preview_active, timeout=2000)
    assert dialog._preview_active

    qtbot.waitUntil(lambda: dialog.preview_label.pixmap() is not None, timeout=2000)

    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: not dialog._preview_active, timeout=2000)

    assert dialog._preview_backend is None
    assert dialog._preview_timer is None


@pytest.mark.gui
def test_e2e_apply_settings_restarts_preview_on_restart_fields(dialog, qtbot, monkeypatch):
    """
    Change a restart-relevant field (fps) and verify preview actually restarts
    (open() called again) while staying active.
    """
    CountingBackend.opens = 0
    monkeypatch.setattr(CameraFactory, "create", lambda s: CountingBackend(s))

    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog._loader is None and dialog._preview_active, timeout=2000)

    before = CountingBackend.opens
    assert before >= 1

    dialog.cam_fps.setValue(99.0)
    qtbot.mouseClick(dialog.apply_settings_btn, Qt.LeftButton)

    qtbot.waitUntil(lambda: CountingBackend.opens >= before + 1, timeout=2000)
    assert dialog._preview_active
    assert dialog._preview_backend is not None


@pytest.mark.gui
def test_e2e_apply_settings_does_not_restart_on_crop_or_rotation(dialog, qtbot, monkeypatch):
    """
    Crop/rotation are applied live in preview; Apply should not restart backend.
    We validate by ensuring backend open count does not increase.
    """
    CountingBackend.opens = 0
    monkeypatch.setattr(CameraFactory, "create", lambda s: CountingBackend(s))

    dialog.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog._loader is None and dialog._preview_active, timeout=2000)

    before = CountingBackend.opens
    assert before >= 1

    dialog.cam_crop_x0.setValue(5)
    dialog.cam_rotation.setCurrentIndex(1)
    qtbot.mouseClick(dialog.apply_settings_btn, Qt.LeftButton)

    qtbot.wait(200)
    assert CountingBackend.opens == before
    assert dialog._preview_active


@pytest.mark.gui
def test_e2e_selection_change_auto_commits(dialog, qtbot):
    """
    Guard contract in E2E mode: switching selection commits pending edits.
    We add a second camera deterministically via the available list.
    """
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
        # simulate long scan that can be interrupted
        for i in range(50):
            if should_cancel and should_cancel():
                break
            if progress_cb:
                progress_cb(f"Scanning… {i}")
            time.sleep(0.02)
        # Return something (could be empty if canceled early)
        return [DetectedCamera(index=0, label=f"{backend}-X")]

    monkeypatch.setattr(CameraFactory, "detect_cameras", slow_detect)

    qtbot.mouseClick(dialog.refresh_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog.scan_cancel_btn.isVisible(), timeout=1000)

    qtbot.mouseClick(dialog.scan_cancel_btn, Qt.LeftButton)

    with qtbot.waitSignal(dialog.scan_finished, timeout=3000):
        pass

    # UI should be re-enabled after finish
    assert dialog.refresh_btn.isEnabled()
    assert dialog.backend_combo.isEnabled()


def _select_backend(dialog, backend_name: str):
    idx = dialog.backend_combo.findData(backend_name)
    assert idx >= 0, f"Backend {backend_name} not present"
    dialog.backend_combo.setCurrentIndex(idx)


@pytest.mark.gui
def test_duplicate_camera_prevented(dialog, qtbot, monkeypatch, temp_backend):
    calls = {"n": 0}

    def _warn(parent, title, text, *args, **kwargs):
        calls["n"] += 1
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warn))

    # Ensure the available list is interpreted as "opencv" (identity key uses backend)
    _select_backend(dialog, "opencv")

    initial_count = dialog.active_cameras_list.count()

    dialog._on_scan_result([DetectedCamera(index=0, label="opencv-X")])
    dialog.available_cameras_list.setCurrentRow(0)

    qtbot.mouseClick(dialog.add_camera_btn, Qt.LeftButton)

    assert dialog.active_cameras_list.count() == initial_count
    assert calls["n"] >= 1


@pytest.mark.gui
def test_max_cameras_prevented(qtbot, monkeypatch, patch_factory):
    calls = {"n": 0}

    def _warn(parent, title, text, *args, **kwargs):
        calls["n"] += 1
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warn))

    s = MultiCameraSettings(
        cameras=[
            CameraSettings(name="C0", backend="opencv", index=0, enabled=True),
            CameraSettings(name="C1", backend="opencv", index=1, enabled=True),
            CameraSettings(name="C2", backend="opencv", index=2, enabled=True),
            CameraSettings(name="C3", backend="opencv", index=3, enabled=True),
        ]
    )
    d = CameraConfigDialog(None, s)
    qtbot.addWidget(d)
    d.show()
    qtbot.waitExposed(d)

    initial_count = d.active_cameras_list.count()

    d._on_scan_result([DetectedCamera(index=4, label="Extra")])
    d.available_cameras_list.setCurrentRow(0)

    qtbot.mouseClick(d.add_camera_btn, Qt.LeftButton)

    assert d.active_cameras_list.count() == initial_count
    assert calls["n"] >= 1

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
    # Make loading slow so Cancel Loading has time to work deterministically

    def slow_run(self):
        self.progress.emit("Creating backend…")
        time.sleep(0.2)  # give test time to click cancel
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

    qtbot.waitUntil(lambda: dialog._loading_active, timeout=1000)

    # Click again quickly => Cancel Loading
    qtbot.mouseClick(dialog.preview_btn, Qt.LeftButton)

    # Ensure loader goes away and preview doesn't become active
    qtbot.waitUntil(lambda: dialog._loader is None and not dialog._loading_active, timeout=2000)
    assert dialog._preview_active is False
    assert dialog._preview_backend is None
