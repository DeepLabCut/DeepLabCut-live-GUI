from __future__ import annotations

import numpy as np
import pytest

from dlclivegui.config import CameraSettings
from dlclivegui.gui.recording_manager import RecordingManager
from dlclivegui.services.multi_camera_controller import get_camera_id, get_display_id
from dlclivegui.utils.stats import RecorderStats


@pytest.fixture
def _active_cams_two(app_config_two_cams):
    """
    Active camera settings clone (two fake cams).
    """
    return [c.model_copy(deep=True) for c in app_config_two_cams.multi_camera.get_active_cameras()]


@pytest.fixture
def current_frames(_active_cams_two):
    """
    Provide deterministic current frames by camera id for frame_size inference.
    cam0: 480x640, cam1: 720x1280.
    """
    from dlclivegui.services.multi_camera_controller import get_camera_id

    frames = {}
    for cam in _active_cams_two:
        cam_id = get_camera_id(cam)
        if cam.index == 0:
            frames[cam_id] = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            frames[cam_id] = np.zeros((720, 1280, 3), dtype=np.uint8)
    return frames


@pytest.mark.unit
def test_start_all_creates_recorders_and_returns_run_dir(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    spy, expected_run_dir = patch_build_run_dir
    mgr = RecordingManager()

    run_dir = mgr.start_all(
        recording_settings,
        _active_cams_two,
        current_frames,
        session_name="Sess",
        use_timestamp=True,
        all_or_nothing=False,
    )

    assert run_dir == expected_run_dir
    assert mgr.is_active is True
    assert mgr.run_dir == expected_run_dir
    assert mgr.session_dir is not None
    assert len(mgr.recorders) == 2

    # build_run_dir called with correct use_timestamp
    assert spy["use_timestamp"] is True
    assert spy["session_dir"] is not None

    # Validate per-cam recorder construction
    for cam in _active_cams_two:
        cam_id = get_camera_id(cam)
        rec = mgr.recorders[cam_id]
        assert rec.codec == recording_settings.codec
        assert rec.crf == recording_settings.crf
        assert rec.frame_rate == float(cam.fps)
        assert rec.is_running is True
        # output file should be inside run dir
        assert rec.output.parent == expected_run_dir
        # filename should include backend + cam index
        assert f"_{cam.backend}_cam{cam.index}" in rec.output.name


@pytest.mark.unit
def test_start_all_passes_use_timestamp_flag(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    spy, _expected_run_dir = patch_build_run_dir
    mgr = RecordingManager()

    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess", use_timestamp=False)
    assert spy["use_timestamp"] is False


@pytest.mark.unit
def test_frame_size_is_inferred_from_current_frames(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    mgr = RecordingManager()
    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess")

    # cam0 -> 480x640, cam1 -> 720x1280
    for cam in _active_cams_two:
        cam_id = get_camera_id(cam)
        rec = mgr.recorders[cam_id]
        frame = current_frames[cam_id]
        assert rec.frame_size == (frame.shape[0], frame.shape[1])


@pytest.mark.unit
def test_missing_frame_results_in_none_frame_size(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    # Remove one frame
    cam1_id = get_camera_id(_active_cams_two[1])
    current_frames.pop(cam1_id)

    mgr = RecordingManager()
    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess")

    rec1 = mgr.recorders[cam1_id]
    assert rec1.frame_size is None


@pytest.mark.unit
def test_partial_failure_allowed_when_not_all_or_nothing(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    mgr = RecordingManager()

    original_start = patch_video_recorder.start

    def start_with_failure(self):
        if "_cam1" in self.output.name:
            raise RuntimeError("boom")
        return original_start(self)

    patch_video_recorder.start = start_with_failure
    try:
        run_dir = mgr.start_all(
            recording_settings,
            _active_cams_two,
            current_frames,
            session_name="Sess",
            all_or_nothing=False,
        )
        assert run_dir is not None
        assert len(mgr.recorders) == 1  # only cam0 should remain
    finally:
        patch_video_recorder.start = original_start
        mgr.stop_all()


@pytest.mark.unit
def test_all_or_nothing_stops_all_on_any_failure(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    mgr = RecordingManager()

    original_start = patch_video_recorder.start

    def start_with_failure(self):
        if "_cam1" in self.output.name:
            raise RuntimeError("boom")
        return original_start(self)

    patch_video_recorder.start = start_with_failure
    try:
        run_dir = mgr.start_all(
            recording_settings,
            _active_cams_two,
            current_frames,
            session_name="Sess",
            all_or_nothing=True,
        )
        assert run_dir is None
        assert mgr.is_active is False
        assert mgr.recorders == {}
        assert mgr.run_dir is None
        assert mgr.session_dir is None
    finally:
        patch_video_recorder.start = original_start


@pytest.mark.unit
def test_stop_all_clears_state(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    mgr = RecordingManager()
    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess")
    assert mgr.is_active is True

    mgr.stop_all()
    assert mgr.is_active is False
    assert mgr.recorders == {}
    assert mgr.run_dir is None
    assert mgr.session_dir is None


@pytest.mark.unit
def test_write_frame_uses_given_timestamp(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    mgr = RecordingManager()
    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess")

    cam0_id = get_camera_id(_active_cams_two[0])
    frame = current_frames[cam0_id]
    mgr.write_frame(cam0_id, frame, timestamp=123.0)

    rec = mgr.recorders[cam0_id]
    assert rec.write_calls[-1][1] == 123.0


@pytest.mark.unit
def test_write_frame_uses_time_when_timestamp_missing(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir, monkeypatch
):
    mgr = RecordingManager()
    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess")

    import dlclivegui.gui.recording_manager as rm_mod  # noqa: E402

    monkeypatch.setattr(rm_mod.time, "time", lambda: 999.0)

    cam0_id = get_camera_id(_active_cams_two[0])
    frame = current_frames[cam0_id]
    mgr.write_frame(cam0_id, frame, timestamp=None)

    rec = mgr.recorders[cam0_id]
    assert rec.write_calls[-1][1] == 999.0


@pytest.mark.unit
def test_write_frame_removes_recorder_on_exception(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    mgr = RecordingManager()
    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess")

    cam0_id = get_camera_id(_active_cams_two[0])
    rec = mgr.recorders[cam0_id]
    rec.raise_on_write = True

    mgr.write_frame(cam0_id, current_frames[cam0_id], timestamp=1.0)
    assert cam0_id not in mgr.recorders


@pytest.mark.unit
def test_get_stats_summary_single_recorder_uses_formatter(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir, monkeypatch
):
    mgr = RecordingManager()
    mgr.start_all(recording_settings, [_active_cams_two[0]], current_frames, session_name="Sess")

    cam0_id = get_camera_id(_active_cams_two[0])
    mgr.recorders[cam0_id]._stats = RecorderStats(frames_written=10, frames_enqueued=12)

    # Patch formatter to avoid depending on formatting implementation
    import dlclivegui.utils.stats as stats_mod

    monkeypatch.setattr(stats_mod, "format_recorder_stats", lambda s: "OK_SINGLE")

    assert mgr.get_stats_summary() == "OK_SINGLE"


@pytest.mark.unit
def test_get_stats_summary_multi_aggregates(
    recording_settings, _active_cams_two, current_frames, patch_video_recorder, patch_build_run_dir
):
    mgr = RecordingManager()
    mgr.start_all(recording_settings, _active_cams_two, current_frames, session_name="Sess")

    ids = [get_camera_id(c) for c in _active_cams_two]
    mgr.recorders[ids[0]]._stats = RecorderStats(
        frames_written=10, dropped_frames=1, queue_size=2, average_latency=0.01, last_latency=0.02
    )
    mgr.recorders[ids[1]]._stats = RecorderStats(
        frames_written=20, dropped_frames=3, queue_size=4, average_latency=0.03, last_latency=0.05
    )

    summary = mgr.get_stats_summary()
    assert "2 cams" in summary
    assert "30 frames" in summary  # 10 + 20
    assert "dropped 4" in summary  # 1 + 3
    assert "queue 6" in summary  # 2 + 4


@pytest.mark.unit
def test_recording_manager_uses_stable_camera_id_not_display_id(
    recording_settings,
    patch_video_recorder,
    patch_build_run_dir,
):
    mgr = RecordingManager()

    cam = CameraSettings(
        name="GenTL cam",
        backend="gentl",
        index=0,
        fps=30.0,
        enabled=True,
        properties={
            "gentl": {
                "device_id": "serial:SER0",
                "serial_number": "SER0",
            }
        },
    ).apply_defaults()

    stable_id = get_camera_id(cam)
    display_id = get_display_id(cam)

    assert stable_id == "gentl:serial:SER0"
    assert display_id == "GenTL cam"
    assert stable_id != display_id

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_frames = {stable_id: frame}

    run_dir = mgr.start_all(
        recording_settings,
        [cam],
        current_frames,
        session_name="Sess",
    )

    assert run_dir is not None
    assert stable_id in mgr.recorders
    assert display_id not in mgr.recorders

    rec = mgr.recorders[stable_id]
    assert rec.frame_size == (480, 640)

    mgr.write_frame(stable_id, frame, timestamp=123.0)
    assert len(rec.write_calls) == 1
    assert rec.write_calls[-1][1] == 123.0

    # Display ID is GUI-only and must not route frames internally.
    mgr.write_frame(display_id, frame, timestamp=456.0)
    assert len(rec.write_calls) == 1


@pytest.mark.unit
def test_start_all_does_not_infer_frame_size_from_display_id(
    recording_settings,
    patch_video_recorder,
    patch_build_run_dir,
):
    mgr = RecordingManager()

    cam = CameraSettings(
        name="GenTL cam",
        backend="gentl",
        index=0,
        fps=30.0,
        enabled=True,
        properties={
            "gentl": {
                "device_id": "serial:SER0",
                "serial_number": "SER0",
            }
        },
    ).apply_defaults()

    stable_id = get_camera_id(cam)
    display_id = get_display_id(cam)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulate the buggy situation: frames are keyed by display ID.
    current_frames = {display_id: frame}

    mgr.start_all(
        recording_settings,
        [cam],
        current_frames,
        session_name="Sess",
    )

    assert stable_id in mgr.recorders
    assert display_id not in mgr.recorders

    # Since RecordingManager uses stable IDs internally, it should not find this frame.
    rec = mgr.recorders[stable_id]
    assert rec.frame_size is None
