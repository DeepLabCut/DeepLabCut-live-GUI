import numpy as np
import pytest


class _StubRec:
    def stop(self):
        pass


@pytest.mark.gui
@pytest.mark.timeout(10)
def test_record_overlay_uses_identity_transform_for_per_camera_recording(window, draw_pose_stub):
    # Disable event timers to avoid GUI rendering pipelines interfering with test
    window._display_timer.stop()
    window._metrics_timer.stop()
    # Arrange: pretend we're recording overlay on inference camera
    cam_id = "fake:0"
    window._inference_camera_id = cam_id

    # Make tiled preview transform non-identity
    window._dlc_tile_offset = (100, 50)
    window._dlc_tile_scale = (0.5, 0.5)

    # Enable overlay recording
    window.record_with_overlays_checkbox.setChecked(True)

    # Provide a fake pose
    window._last_pose = type("Pose", (), {"pose": {"x": 10, "y": 20}})()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Act: call your helper directly
    out = window._render_overlays_for_recording(cam_id, frame)

    # Assert: the call happened
    assert "offset" in draw_pose_stub
    assert "scale" in draw_pose_stub

    # Expected behavior for per-camera recording:
    # offset=(0,0), scale=(1,1)
    assert draw_pose_stub["offset"] == (0, 0)
    assert draw_pose_stub["scale"] == (1.0, 1.0)

    # And green pixel should be at (x=10,y=20)
    assert (out[20, 10] == np.array([0, 255, 0])).all()


@pytest.mark.gui
@pytest.mark.timeout(10)
def test_record_overlay_toggle_affects_frames_sent_to_recorder(window, recording_frame_spy, draw_pose_stub):
    # Disable event timers to avoid GUI rendering pipelines interfering with test
    window._display_timer.stop()
    window._metrics_timer.stop()
    # Arrange: pretend we're recording overlay on inference camera
    cam_id = "fake:0"
    window._inference_camera_id = cam_id
    window._running_cams_ids = {cam_id}

    # Pretend recording is active
    window._rec_manager._recorders = {"dummy": _StubRec()}  # minimal: make is_active True via bool(dict)

    # Provide pose
    window._last_pose = type("Pose", (), {"pose": {"x": 10, "y": 20}})()

    # Provide a frame
    raw = np.zeros((100, 100, 3), dtype=np.uint8)

    # Build minimal frame_data to call _on_multi_frame_ready
    from dlclivegui.services.multi_camera_controller import MultiFrameData

    frame_data = MultiFrameData(
        frames={cam_id: raw},
        timestamps={cam_id: 1.0},
        source_camera_id=cam_id,
    )

    # 1) toggle OFF: should record raw
    window.record_with_overlays_checkbox.setChecked(False)
    window._on_multi_frame_ready(frame_data)

    assert cam_id in recording_frame_spy
    recorded_off = recording_frame_spy[cam_id]
    assert np.array_equal(recorded_off, raw)

    # 2) toggle ON: should record overlay frame (different)
    window.record_with_overlays_checkbox.setChecked(True)
    window._on_multi_frame_ready(frame_data)

    recorded_on = recording_frame_spy[cam_id]
    assert not np.array_equal(recorded_on, raw)
    # verify our stub drew the marker at expected pixel
    assert (recorded_on[20, 10] == np.array([0, 255, 0])).all()
