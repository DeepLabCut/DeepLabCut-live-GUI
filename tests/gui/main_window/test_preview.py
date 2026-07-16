# tests/gui/main_window/test_preview.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QPixmap

from dlclivegui.services.multi_camera_controller import get_camera_id


@pytest.mark.gui
class TestPreviewLifecycle:
    def test_start_preview_with_no_active_cameras_shows_error(self, monkeypatch, window):
        w = window
        for cam in w._config.multi_camera.cameras:
            cam.enabled = False

        messages: list[str] = []
        monkeypatch.setattr(w, "_show_error", messages.append)

        w._start_preview()

        assert messages == ["No cameras configured. Use 'Configure Cameras...' to add cameras."]
        assert not w.multi_camera_controller.is_running()
        assert w.preview_button.isEnabled()
        assert not w.stop_preview_button.isEnabled()

    def test_on_multi_camera_stopped_clears_runtime_state(self, monkeypatch, window):
        w = window
        monkeypatch.setattr(w, "_stop_multi_camera_recording", lambda: None)

        w.preview_button.setEnabled(False)
        w.stop_preview_button.setEnabled(True)
        w._current_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        w._multi_camera_frames = {"fake:index:0": np.zeros((4, 4, 3), dtype=np.uint8)}
        w._multi_camera_display_ids = {"fake:index:0": "Cam0"}
        w._running_cams_ids = {"fake:index:0"}
        w._display_dirty = True
        w.video_label.setPixmap(QPixmap(8, 8))

        w._on_multi_camera_stopped()

        assert w.preview_button.isEnabled()
        assert not w.stop_preview_button.isEnabled()
        assert w._current_frame is None
        assert w._multi_camera_frames == {}
        assert w._multi_camera_display_ids == {}
        assert w._running_cams_ids == set()
        assert w._display_dirty is False
        assert w.video_label.text() == "Camera preview not started"

    def test_stop_preview_requests_orderly_shutdown(self, monkeypatch, window):
        w = window
        calls: list[str] = []

        monkeypatch.setattr(w.multi_camera_controller, "is_running", lambda: True)
        monkeypatch.setattr(w, "_stop_multi_camera_recording", lambda: calls.append("recording"))
        monkeypatch.setattr(w, "_stop_inference", lambda show_message=False: calls.append("inference"))
        monkeypatch.setattr(
            w.multi_camera_controller,
            "stop",
            lambda *args, **kwargs: calls.append("controller"),
        )

        w.preview_button.setEnabled(True)
        w.stop_preview_button.setEnabled(True)
        w.start_inference_button.setEnabled(True)
        w.stop_inference_button.setEnabled(True)
        w._pending_recording_after_preview = True

        w._stop_preview()

        assert calls == ["recording", "inference", "controller"]
        assert w._pending_recording_after_preview is False
        assert not w.preview_button.isEnabled()
        assert not w.stop_preview_button.isEnabled()
        assert not w.start_inference_button.isEnabled()
        assert not w.stop_inference_button.isEnabled()
        assert w.camera_stats_label.text() == "Camera idle"

    def test_on_multi_camera_started_updates_primary_buttons(self, window):
        w = window

        w.preview_button.setEnabled(True)
        w.stop_preview_button.setEnabled(False)

        w._on_multi_camera_started()

        assert not w.preview_button.isEnabled()
        assert w.stop_preview_button.isEnabled()

    def test_processing_runtime_fallback_does_not_overwrite_preferred_inference_camera(self, window):
        w = window

        active_cams = w._config.multi_camera.get_active_cameras()
        if len(active_cams) < 2:
            pytest.skip("This regression test requires at least two active cameras.")

        fallback_cam = active_cams[0]
        preferred_cam = active_cams[1]

        fallback_id = get_camera_id(fallback_cam)
        preferred_id = get_camera_id(preferred_cam)

        w._inference_camera_id = preferred_id
        w._active_inference_camera_id = preferred_id
        w._running_cams_ids = set()
        w._dlc_active = False

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_data = SimpleNamespace(
            frames={fallback_id: frame},
            display_ids={fallback_id: "Fallback camera"},
            source_camera_id=fallback_id,
            timestamps={fallback_id: 123.0},
        )

        w._on_multi_frame_processing_ready(frame_data)

        assert w._inference_camera_id == preferred_id
        assert w._active_inference_camera_id == fallback_id
        assert w.dlc_camera_combo.currentData() == fallback_id
