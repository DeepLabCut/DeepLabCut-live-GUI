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
    w = window
    ctrl = multi_camera_controller  # real controller

    # Start preview and wait for controller to report started
    with qtbot.waitSignal(ctrl.all_started, timeout=3000):
        qtbot.mouseClick(w.preview_button, Qt.LeftButton)

    # Wait until the GUI has a pixmap (real controller will emit frames via stub cameras)
    qtbot.waitUntil(
        lambda: w.video_label.pixmap() is not None and not w.video_label.pixmap().isNull(),
        timeout=5000,
    )

    # Stop preview and wait for stopped signal
    with qtbot.waitSignal(ctrl.all_stopped, timeout=3000):
        qtbot.mouseClick(w.stop_preview_button, Qt.LeftButton)

    assert not ctrl.is_running()


@pytest.mark.gui
@pytest.mark.functional
def test_start_inference_emits_pose(qtbot, window, multi_camera_controller, dlc_processor):
    w = window
    ctrl = multi_camera_controller
    dlc = dlc_processor

    # Start preview
    with qtbot.waitSignal(ctrl.all_started, timeout=3000):
        qtbot.mouseClick(w.preview_button, Qt.LeftButton)

    # Ensure preview is producing frames and GUI is rendering
    qtbot.waitUntil(
        lambda: w.video_label.pixmap() is not None and not w.video_label.pixmap().isNull(),
        timeout=5000,
    )

    # Must set model path so MainWindow._configure_dlc passes the "select a model" check
    w.model_path_edit.setText("dummy_model.pt")

    # Robust: count pose signals so we don't miss one emitted quickly after init
    pose_count = [0]

    def _on_pose(_result):
        pose_count[0] += 1

    dlc.pose_ready.connect(_on_pose)

    try:
        # Start inference
        # Use waitSignal around the click so we don't race the initialized emission
        with qtbot.waitSignal(dlc.initialized, timeout=5000) as init_blocker:
            qtbot.mouseClick(w.start_inference_button, Qt.LeftButton)

        assert init_blocker.args[0] is True

        # Wait until we have received at least one pose
        qtbot.waitUntil(lambda: pose_count[0] >= 1, timeout=5000)

    finally:
        # Avoid leaking connections across tests
        try:
            dlc.pose_ready.disconnect(_on_pose)
        except Exception:  # pragma: no cover
            pass

    # Stop preview
    with qtbot.waitSignal(ctrl.all_stopped, timeout=3000):
        qtbot.mouseClick(w.stop_preview_button, Qt.LeftButton)
