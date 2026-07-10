# tests/gui/main_window/test_recording.py
from __future__ import annotations

import numpy as np
import pytest

from dlclivegui.services.multi_camera_controller import get_camera_id


@pytest.mark.gui
class TestRecordingLifecycle:
    def test_start_recording_auto_starts_preview_when_preview_is_not_running(self, monkeypatch, window):
        w = window
        calls: list[str] = []

        monkeypatch.setattr(w.multi_camera_controller, "is_running", lambda: False)
        monkeypatch.setattr(w, "_start_preview", lambda: calls.append("preview"))

        w._pending_recording_after_preview = False
        w._start_recording()

        assert calls == ["preview"]
        assert w._pending_recording_after_preview is True

    def test_start_recording_starts_immediately_when_preview_is_running(self, monkeypatch, window):
        w = window
        calls: list[str] = []

        monkeypatch.setattr(w.multi_camera_controller, "is_running", lambda: True)
        monkeypatch.setattr(w, "_start_multi_camera_recording", lambda: calls.append("recording"))

        w._pending_recording_after_preview = False
        w._start_recording()

        assert calls == ["recording"]
        assert w._pending_recording_after_preview is False

    def test_pending_recording_waits_until_all_expected_frames_are_available(self, monkeypatch, window):
        w = window
        calls: list[str] = []

        monkeypatch.setattr(w.multi_camera_controller, "is_running", lambda: True)
        monkeypatch.setattr(w, "_start_multi_camera_recording", lambda: calls.append("recording"))

        active = w._config.multi_camera.get_active_cameras()
        assert len(active) >= 2
        first_id = get_camera_id(active[0])

        w._pending_recording_after_preview = True
        w._multi_camera_frames = {first_id: np.zeros((4, 4, 3), dtype=np.uint8)}

        w._try_start_pending_recording()

        assert calls == []
        assert w._pending_recording_after_preview is True

    def test_pending_recording_starts_when_all_expected_frames_are_available(self, monkeypatch, window):
        w = window
        calls: list[str] = []

        monkeypatch.setattr(w.multi_camera_controller, "is_running", lambda: True)
        monkeypatch.setattr(w, "_start_multi_camera_recording", lambda: calls.append("recording"))

        w._pending_recording_after_preview = True
        w._multi_camera_frames = {
            get_camera_id(cam): np.zeros((4, 4, 3), dtype=np.uint8)
            for cam in w._config.multi_camera.get_active_cameras()
        }

        w._try_start_pending_recording()

        assert calls == ["recording"]
        assert w._pending_recording_after_preview is False

    def test_start_multi_camera_recording_success_sets_buttons_and_sink(self, window, start_all_spy):
        w = window
        active = w._config.multi_camera.get_active_cameras()
        w._multi_camera_frames = {get_camera_id(cam): np.zeros((4, 4, 3), dtype=np.uint8) for cam in active}

        w._start_multi_camera_recording()

        assert start_all_spy["active_cams"] == active
        assert not w.start_record_button.isEnabled()
        assert w.stop_record_button.isEnabled()

    def test_stop_multi_camera_recording_when_idle_is_noop(self, window):
        w = window
        w._recording_stopping = False

        w._stop_multi_camera_recording()

        assert w._recording_stopping is False
