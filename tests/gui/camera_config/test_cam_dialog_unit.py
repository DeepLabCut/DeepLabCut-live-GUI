# tests/gui/camera_config/test_cam_dialog_unit.py
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.config import CameraSettings, MultiCameraSettings
from dlclivegui.gui.camera_config_dialog import CameraConfigDialog

# ----------------------------
# Unit dialog fixture (deterministic, no threads)
# ----------------------------


@pytest.fixture
def dialog_unit(qtbot, monkeypatch):
    """
    Unit fixture: disable async scan + probe to keep tests deterministic.

    We want to test:
      - dirty state logic
      - auto-commit guards (commit before selection change, list ops, OK)
      - apply persistence into _working_settings
      - UI list mutations (add/remove/move)

    We do NOT want:
      - background scan threads
      - probe open/close hardware
      - timer-driven preview activity
    """
    # Prevent async scan on init (dialog calls _populate_from_settings -> _refresh_available_cameras)
    monkeypatch.setattr(CameraConfigDialog, "_refresh_available_cameras", lambda self: None)

    # Prevent probe worker from opening backends (selection triggers probe in current dialog)
    monkeypatch.setattr(CameraConfigDialog, "_start_probe_for_camera", lambda *a, **k: None)

    # Ensure capability-driven enable/disable states are deterministic for these unit tests.
    # We intentionally disable gain/exposure for OpenCV by choice, but keep FPS/resolution enabled for UX tests.
    from dlclivegui.cameras import CameraFactory
    from dlclivegui.cameras.base import SupportLevel

    def _caps(backend_name: str):
        key = (backend_name or "").lower()
        if key == "opencv":
            return {
                "set_resolution": SupportLevel.SUPPORTED,
                "set_fps": SupportLevel.SUPPORTED,
                "set_exposure": SupportLevel.UNSUPPORTED,  # by choice
                "set_gain": SupportLevel.UNSUPPORTED,  # by choice
                "device_discovery": SupportLevel.SUPPORTED,
                "stable_identity": SupportLevel.SUPPORTED,
            }
        # Default for tests: allow everything (useful if temp_backend enables gain, etc.)
        return {
            "set_resolution": SupportLevel.SUPPORTED,
            "set_fps": SupportLevel.SUPPORTED,
            "set_exposure": SupportLevel.SUPPORTED,
            "set_gain": SupportLevel.SUPPORTED,
            "device_discovery": SupportLevel.SUPPORTED,
            "stable_identity": SupportLevel.SUPPORTED,
        }

    monkeypatch.setattr(CameraFactory, "backend_capabilities", staticmethod(_caps), raising=False)

    s = MultiCameraSettings(
        cameras=[
            CameraSettings(name="CamA", backend="opencv", index=0, enabled=True),
            CameraSettings(name="CamB", backend="opencv", index=1, enabled=True),
        ]
    )
    d = CameraConfigDialog(None, s)
    qtbot.addWidget(d)
    d.show()
    qtbot.waitExposed(d)
    return d


# ----------------------
# UNIT TESTS
# ----------------------


@pytest.mark.gui
def test_load_form_is_clean(dialog_unit, qtbot):
    """Selecting a camera loads values but should not mark the form dirty."""
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    assert not dialog_unit.apply_settings_btn.isEnabled()
    assert "*" not in dialog_unit.apply_settings_btn.text()


@pytest.mark.gui
@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d.cam_width.setValue(320),
        lambda d: d.cam_height.setValue(240),
        lambda d: d.cam_fps.setValue(55.0),
        lambda d: d.cam_exposure.setValue(1000),
        lambda d: d.cam_gain.setValue(12.0),
        lambda d: d.cam_rotation.setCurrentIndex(1),
        lambda d: d.cam_crop_x0.setValue(10),
        lambda d: d.cam_crop_y0.setValue(10),
        lambda d: d.cam_crop_x1.setValue(50),
        lambda d: d.cam_crop_y1.setValue(50),
        lambda d: d.cam_enabled_checkbox.setChecked(False),
    ],
)
def test_any_change_marks_dirty(dialog_unit, qtbot, mutator):
    """Any user change to an editable field should enable Apply and mark it dirty."""
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    mutator(dialog_unit)

    assert dialog_unit.apply_settings_btn.isEnabled()
    assert "*" in dialog_unit.apply_settings_btn.text()


@pytest.mark.gui
def test_apply_clears_dirty_and_updates_model(dialog_unit, qtbot):
    """Apply persists edits into working settings and clears dirty state."""
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    dialog_unit.cam_fps.setValue(55.0)
    dialog_unit.cam_gain.setValue(12.0)

    assert dialog_unit.apply_settings_btn.isEnabled()
    qtbot.mouseClick(dialog_unit.apply_settings_btn, Qt.LeftButton)

    assert not dialog_unit.apply_settings_btn.isEnabled()
    assert "*" not in dialog_unit.apply_settings_btn.text()

    updated = dialog_unit._working_settings.cameras[0]
    assert updated.fps == 55.0
    assert updated.gain == 0.0  # gain should not update for OpenCV backend (disabled in UI)


@pytest.mark.gui
def test_switch_selection_auto_commits_pending_edits(dialog_unit, qtbot):
    """Guard contract: switching cameras should auto-apply pending edits."""
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    dialog_unit.cam_fps.setValue(42.0)
    assert dialog_unit.apply_settings_btn.isEnabled()

    dialog_unit.active_cameras_list.setCurrentRow(1)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 1, timeout=1000)

    assert dialog_unit._working_settings.cameras[0].fps == 42.0


@pytest.mark.gui
def test_invalid_crop_blocks_switch_and_reverts_selection(dialog_unit, qtbot, monkeypatch):
    """
    Real validation: CameraSettings enforces that if any crop is set,
    we require x1 > x0 and y1 > y0.

    This test intentionally creates an invalid crop and verifies that:
      - selection change is blocked
      - selection reverts to the original row
      - a warning dialog would have been shown (we stub it here)
    """
    calls = {"n": 0, "title": None, "text": None}

    def _warn(parent, title, text, *args, **kwargs):
        calls["n"] += 1
        calls["title"] = title
        calls["text"] = text
        return QMessageBox.Ok

    # Override the global "fail fast" fixture for this test only
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_warn))

    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    # Enable cropping but make it invalid: x1 <= x0 and y1 <= y0
    dialog_unit.cam_crop_x0.setValue(100)
    dialog_unit.cam_crop_y0.setValue(100)
    dialog_unit.cam_crop_x1.setValue(50)  # invalid (x1 <= x0)
    dialog_unit.cam_crop_y1.setValue(80)  # invalid (y1 <= y0)

    assert dialog_unit.apply_settings_btn.isEnabled()

    # Attempt to switch: should be blocked and revert to row 0
    dialog_unit.active_cameras_list.setCurrentRow(1)
    qtbot.waitUntil(lambda: dialog_unit.active_cameras_list.currentRow() == 0, timeout=1000)

    assert calls["n"] >= 1
    assert "Invalid crop rectangle" in (calls["text"] or "")


@pytest.mark.gui
def test_ok_auto_applies_pending_edits_before_emitting(dialog_unit, qtbot):
    """OK should auto-apply pending edits before emitting settings_changed."""
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    # Use FPS here (supported) to ensure the test validates meaningful auto-apply.
    dialog_unit.cam_fps.setValue(77.0)
    assert dialog_unit.apply_settings_btn.isEnabled()

    with qtbot.waitSignal(dialog_unit.settings_changed, timeout=1000) as sig:
        qtbot.mouseClick(dialog_unit.ok_btn, Qt.LeftButton)

    emitted = sig.args[0]
    assert emitted.cameras[0].fps == 77.0


@pytest.mark.gui
def test_add_camera_populates_working_settings(dialog_unit, qtbot):
    """
    Add camera should append a new CameraSettings into _working_settings.
    We directly call _on_scan_result to populate available list deterministically.
    """
    dialog_unit._on_scan_result([DetectedCamera(index=2, label="ExtraCam2")])
    dialog_unit.available_cameras_list.setCurrentRow(0)

    qtbot.mouseClick(dialog_unit.add_camera_btn, Qt.LeftButton)

    added = dialog_unit._working_settings.cameras[-1]
    assert added.index == 2
    assert added.name == "ExtraCam2"


@pytest.mark.gui
def test_remove_camera_shrinks_working_settings(dialog_unit, qtbot):
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.mouseClick(dialog_unit.remove_camera_btn, Qt.LeftButton)

    assert len(dialog_unit._working_settings.cameras) == 1
    assert dialog_unit._working_settings.cameras[0].name == "CamB"


@pytest.mark.gui
def test_move_camera_up_commits_then_reorders(dialog_unit, qtbot):
    """
    Move actions should auto-commit pending edits before reordering.
    """
    dialog_unit.active_cameras_list.setCurrentRow(1)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 1, timeout=1000)

    dialog_unit.cam_fps.setValue(88.0)
    assert dialog_unit.apply_settings_btn.isEnabled()

    qtbot.mouseClick(dialog_unit.move_up_btn, Qt.LeftButton)

    # CamB moved to index 0 after move; commit should persist to that camera
    assert dialog_unit._working_settings.cameras[0].fps == 88.0


@pytest.mark.gui
def test_opencv_gain_is_disabled_and_does_not_change_model(dialog_unit, qtbot):
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    # OpenCV: gain control should be disabled by design
    assert not dialog_unit.cam_gain.isEnabled()

    # Model remains Auto (0.0)
    assert dialog_unit._working_settings.cameras[0].gain == 0.0


@pytest.mark.gui
def test_enter_in_fps_commits_and_does_not_close(dialog_unit, qtbot):
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    le = dialog_unit.cam_fps.lineEdit()
    assert le is not None

    le.setFocus()
    le.selectAll()
    qtbot.keyClicks(le, "55")
    qtbot.keyClick(le, Qt.Key_Return)

    assert dialog_unit.isVisible()
    assert dialog_unit._working_settings.cameras[0].fps == 55.0


@pytest.mark.gui
def test_enter_in_gain_commits_for_gain_capable_backend(dialog_unit, qtbot, temp_backend):
    from dlclivegui.cameras.base import SupportLevel

    with temp_backend(
        "test_gain",
        caps={
            "set_gain": SupportLevel.SUPPORTED,
            "set_exposure": SupportLevel.SUPPORTED,
            "set_resolution": SupportLevel.SUPPORTED,
            "set_fps": SupportLevel.SUPPORTED,
            "device_discovery": SupportLevel.SUPPORTED,
            "stable_identity": SupportLevel.SUPPORTED,
        },
    ):
        # switch camera backend to the temp backend and reload form
        dialog_unit._working_settings.cameras[0].backend = "test_gain"
        dialog_unit._load_camera_to_form(dialog_unit._working_settings.cameras[0])

        assert dialog_unit.cam_gain.isEnabled()

        le = dialog_unit.cam_gain.lineEdit()
        assert le is not None

        le.setFocus()
        le.selectAll()
        qtbot.keyClicks(le, "12.0")
        qtbot.keyClick(le, Qt.Key_Return)

        assert dialog_unit.isVisible()
        assert dialog_unit._working_settings.cameras[0].gain == 12.0


@pytest.mark.gui
def test_disabled_gain_exposure_do_not_affect_model_for_opencv(dialog_unit, qtbot):
    dialog_unit.active_cameras_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog_unit._current_edit_index == 0, timeout=1000)

    assert not dialog_unit.cam_gain.isEnabled()
    assert not dialog_unit.cam_exposure.isEnabled()

    # programmatic setValue shouldn't persist when disabled
    dialog_unit.cam_gain.setValue(12.0)
    dialog_unit.cam_exposure.setValue(1234)

    dialog_unit.cam_fps.setValue(55.0)  # make dirty
    qtbot.mouseClick(dialog_unit.apply_settings_btn, Qt.LeftButton)

    cam = dialog_unit._working_settings.cameras[0]
    assert cam.fps == 55.0
    assert cam.gain == 0.0
    assert cam.exposure == 0
