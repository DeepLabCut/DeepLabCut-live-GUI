import numpy as np
import pytest

from dlclivegui.services.multi_camera_controller import MultiFrameData, get_camera_id


@pytest.mark.gui
class TestPendingRecordingAfterPreview:
    def test_start_recording_when_preview_stopped_defers_until_preview_frames(
        self,
        window,
        monkeypatch,
    ):
        calls = {
            "start_preview": 0,
            "start_recording": 0,
        }

        monkeypatch.setattr(
            window.multi_camera_controller,
            "is_running",
            lambda: False,
        )

        def fake_start_preview():
            calls["start_preview"] += 1

        def fake_start_multi_camera_recording():
            calls["start_recording"] += 1

        monkeypatch.setattr(window, "_start_preview", fake_start_preview)
        monkeypatch.setattr(window, "_start_multi_camera_recording", fake_start_multi_camera_recording)

        window._pending_recording_after_preview = False

        window._start_recording()

        assert calls["start_preview"] == 1
        assert calls["start_recording"] == 0
        assert window._pending_recording_after_preview is True

    def test_pending_recording_waits_until_all_active_cameras_have_frames(
        self,
        window,
        monkeypatch,
    ):
        active_cams = window._config.multi_camera.get_active_cameras()
        assert len(active_cams) >= 2

        cam0_id = get_camera_id(active_cams[0])
        cam1_id = get_camera_id(active_cams[1])

        calls = {
            "start_recording": 0,
        }

        monkeypatch.setattr(
            window.multi_camera_controller,
            "is_running",
            lambda: True,
        )

        def fake_start_multi_camera_recording():
            calls["start_recording"] += 1

        monkeypatch.setattr(window, "_start_multi_camera_recording", fake_start_multi_camera_recording)

        window._pending_recording_after_preview = True
        window._multi_camera_frames = {
            cam0_id: np.zeros((10, 10, 3), dtype=np.uint8),
        }

        window._try_start_pending_recording()

        assert calls["start_recording"] == 0
        assert window._pending_recording_after_preview is True

        window._multi_camera_frames[cam1_id] = np.zeros((10, 10, 3), dtype=np.uint8)

        window._try_start_pending_recording()

        assert calls["start_recording"] == 1
        assert window._pending_recording_after_preview is False

    def test_pending_recording_is_triggered_from_multi_frame_processing_ready(
        self,
        window,
        monkeypatch,
    ):
        active_cams = window._config.multi_camera.get_active_cameras()
        assert len(active_cams) >= 2

        cam0_id = get_camera_id(active_cams[0])
        cam1_id = get_camera_id(active_cams[1])

        calls = {
            "start_recording": 0,
        }

        monkeypatch.setattr(
            window.multi_camera_controller,
            "is_running",
            lambda: True,
        )

        def fake_start_multi_camera_recording():
            calls["start_recording"] += 1

        monkeypatch.setattr(window, "_start_multi_camera_recording", fake_start_multi_camera_recording)

        window._pending_recording_after_preview = True

        frame0 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)

        frame_data = MultiFrameData(
            frames={
                cam0_id: frame0,
                cam1_id: frame1,
            },
            timestamps={
                cam0_id: 1.0,
                cam1_id: 1.0,
            },
            source_camera_id=cam0_id,
            display_ids={
                cam0_id: "Cam0",
                cam1_id: "Cam1",
            },
        )

        window._on_multi_frame_processing_ready(frame_data)

        assert calls["start_recording"] == 1
        assert window._pending_recording_after_preview is False

    def test_pending_recording_does_not_start_twice(
        self,
        window,
        monkeypatch,
    ):
        active_cams = window._config.multi_camera.get_active_cameras()
        assert len(active_cams) >= 2

        cam0_id = get_camera_id(active_cams[0])
        cam1_id = get_camera_id(active_cams[1])

        calls = {
            "start_recording": 0,
        }

        monkeypatch.setattr(
            window.multi_camera_controller,
            "is_running",
            lambda: True,
        )

        def fake_start_multi_camera_recording():
            calls["start_recording"] += 1

        monkeypatch.setattr(window, "_start_multi_camera_recording", fake_start_multi_camera_recording)

        window._pending_recording_after_preview = True
        window._multi_camera_frames = {
            cam0_id: np.zeros((10, 10, 3), dtype=np.uint8),
            cam1_id: np.zeros((10, 10, 3), dtype=np.uint8),
        }

        window._try_start_pending_recording()
        window._try_start_pending_recording()

        assert calls["start_recording"] == 1
        assert window._pending_recording_after_preview is False
