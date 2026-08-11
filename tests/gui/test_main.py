import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage

from dlclivegui.services.dlc_processor import DLCLiveProcessor, Engine
from dlclivegui.services.multi_camera_controller import MultiFrameData


def pixmap_bytes(label) -> bytes:
    pm = label.pixmap()
    assert pm is not None and not pm.isNull()
    img = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    return bytes(ptr)


@pytest.mark.gui
@pytest.mark.functional
def test_preview_renders_frames(
    qtbot,
    window,
    multi_camera_controller,
    monkeypatch,
):
    """Verify preview controls and frame rendering without camera timing."""
    w = window
    ctrl = multi_camera_controller
    running = False

    camera_id = "fake:index:0"
    frame = np.full((120, 160, 3), 127, dtype=np.uint8)
    frame_data = MultiFrameData(
        frames={camera_id: frame},
        timestamps={camera_id: 123.0},
        source_camera_id=camera_id,
        display_ids={camera_id: "Test camera"},
    )

    initial_pixmap = w.video_label.pixmap()
    initial_cache_key = initial_pixmap.cacheKey() if initial_pixmap is not None else None

    def fake_is_running():
        return running

    def fake_get_active_count():
        return 1 if running else 0

    def fake_start(_camera_settings):
        nonlocal running
        running = True

        def finish_start():
            ctrl.all_started.emit()
            ctrl.frame_ready.emit(frame_data)
            ctrl.display_ready.emit(frame_data)

        QTimer.singleShot(0, finish_start)

    def fake_stop(*, wait=True):
        nonlocal running
        running = False
        QTimer.singleShot(0, ctrl.all_stopped.emit)

    monkeypatch.setattr(ctrl, "is_running", fake_is_running)
    monkeypatch.setattr(
        ctrl,
        "get_active_count",
        fake_get_active_count,
    )
    monkeypatch.setattr(ctrl, "start", fake_start)
    monkeypatch.setattr(ctrl, "stop", fake_stop)

    with qtbot.waitSignal(ctrl.all_started, timeout=1000):
        qtbot.mouseClick(w.preview_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: (
            w._current_frame is not None
            and w.video_label.pixmap() is not None
            and not w.video_label.pixmap().isNull()
            and w.video_label.pixmap().cacheKey() != initial_cache_key
        ),
        timeout=1000,
    )

    assert w._current_frame is not None
    assert w.stop_preview_button.isEnabled()

    with qtbot.waitSignal(ctrl.all_stopped, timeout=1000):
        qtbot.mouseClick(w.stop_preview_button, Qt.LeftButton)

    assert not ctrl.is_running()
    assert w._current_frame is None


@pytest.mark.gui
@pytest.mark.functional
def test_start_inference_emits_pose(qtbot, window, multi_camera_controller, dlc_processor, tmp_path):
    """
    Validate that:
      - Preview is running
      - GUI sets a valid model path
      - Start Inference triggers DLCLiveProcessor initialization
      - initialized(True) fires
      - pose_ready fires at least once
      - Preview can be stopped cleanly
    """

    w = window
    ctrl = multi_camera_controller
    dlc = dlc_processor

    # Start preview first
    with qtbot.waitSignal(ctrl.all_started, timeout=4000):
        qtbot.mouseClick(w.preview_button, Qt.LeftButton)

    # Ensure preview is producing actual GUI frames
    qtbot.waitUntil(
        lambda: w.video_label.pixmap() is not None and not w.video_label.pixmap().isNull(),
        timeout=6000,
    )

    model_weights = tmp_path / "dummy_model.pt"
    model_weights.touch()  # create an empty file to satisfy existence check
    w.model_path_edit.setText(str(model_weights))
    pose_count = [0]

    def _on_pose(result):
        pose_count[0] += 1

    dlc.pose_ready.connect(_on_pose)

    try:
        # Click "Start Inference" and wait for DLCLiveProcessor.initialized(True)
        with qtbot.waitSignal(dlc.initialized, timeout=7000) as init_blocker:
            qtbot.mouseClick(w.start_inference_button, Qt.LeftButton)

        # Validate initialized==True
        assert init_blocker.args[0] is True

        # Wait until at least one pose is emitted
        qtbot.waitUntil(lambda: pose_count[0] >= 1, timeout=7000)

    finally:
        # Avoid leaking connections across tests
        try:
            dlc.pose_ready.disconnect(_on_pose)
        except Exception:
            pass

    with qtbot.waitSignal(ctrl.all_stopped, timeout=4000):
        qtbot.mouseClick(w.stop_preview_button, Qt.LeftButton)

    assert not ctrl.is_running()


def test_dlc_settings_from_ui_validates_detected_model_type(
    monkeypatch,
    window,
    tmp_path,
):
    model_path = tmp_path / "model.pt"
    model_path.touch()
    window.model_path_edit.setText(str(model_path))

    monkeypatch.setattr(
        DLCLiveProcessor,
        "get_model_backend",
        lambda _path: Engine.PYTORCH,
    )

    settings = window._dlc_settings_from_ui()

    assert settings.model_type == "pytorch"
    assert isinstance(settings.model_type, str)
