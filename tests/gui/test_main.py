import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage


def pixmap_bytes(label) -> bytes:
    pm = label.pixmap()
    assert pm is not None and not pm.isNull()
    img = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    return bytes(ptr)


@pytest.mark.gui
@pytest.mark.functional
def test_preview_renders_frames(qtbot, window, multi_camera_controller):
    """
    Validate that:
      - Preview starts (`preview_button` clicked)
      - Camera controller emits all_started
      - GUI receives and renders frames to video_label.pixmap()
      - Preview stops cleanly
    """

    w = window
    ctrl = multi_camera_controller

    with qtbot.waitSignal(ctrl.all_started, timeout=4000):
        qtbot.mouseClick(w.preview_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: w.video_label.pixmap() is not None and not w.video_label.pixmap().isNull(),
        timeout=6000,
    )

    with qtbot.waitSignal(ctrl.all_stopped, timeout=4000):
        qtbot.mouseClick(w.stop_preview_button, Qt.LeftButton)

    assert not ctrl.is_running()


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
