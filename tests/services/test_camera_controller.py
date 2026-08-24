from __future__ import annotations

import numpy as np

from dlclivegui.cameras.base import CapturedFrame
from dlclivegui.config import CameraSettings
from dlclivegui.services.camera_controller import SingleCameraWorker


def _capture_signals(worker: SingleCameraWorker) -> dict[str, list[tuple]]:
    """Collect worker Qt signal emissions synchronously."""
    seen: dict[str, list[tuple]] = {
        "runtime_info": [],
        "started": [],
        "frame_captured": [],
        "error_occurred": [],
        "stopped": [],
    }

    worker.runtime_info.connect(lambda *args: seen["runtime_info"].append(args))
    worker.started.connect(lambda *args: seen["started"].append(args))
    worker.frame_captured.connect(lambda *args: seen["frame_captured"].append(args))
    worker.error_occurred.connect(lambda *args: seen["error_occurred"].append(args))
    worker.stopped.connect(lambda *args: seen["stopped"].append(args))

    return seen


def test_worker_fake_backend(qtbot, patch_factory, camera_worker_settings: CameraSettings):
    worker = SingleCameraWorker("fake:index:0", camera_worker_settings)
    seen = _capture_signals(worker)

    # Stop after first frame so worker.run() returns synchronously.
    worker.frame_captured.connect(lambda *_args: worker.stop())

    worker.run()

    assert len(seen["error_occurred"]) == 0
    assert len(seen["runtime_info"]) == 1
    assert len(seen["started"]) == 1
    assert len(seen["frame_captured"]) == 1
    assert len(seen["stopped"]) == 1

    runtime_camera_id, runtime = seen["runtime_info"][0]
    assert runtime_camera_id == "fake:index:0"
    assert set(runtime) == {
        "actual_fps",
        "actual_resolution",
        "actual_pixel_format",
        "actual_output_format",
    }

    assert seen["started"][0] == ("fake:index:0",)

    frame_camera_id, frame, timestamp, timestamp_metadata = seen["frame_captured"][0]
    assert frame_camera_id == "fake:index:0"
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert isinstance(timestamp, float)
    assert timestamp_metadata is None

    assert seen["stopped"][0] == ("fake:index:0",)


def test_worker_recording_sink_receives_frame(qtbot, patch_factory, camera_worker_settings: CameraSettings):
    worker = SingleCameraWorker("fake:index:0", camera_worker_settings)
    seen = _capture_signals(worker)

    recorded: list[tuple] = []

    def recording_sink(camera_id, frame, timestamp, timestamp_metadata):
        recorded.append((camera_id, frame.copy(), timestamp, timestamp_metadata))

    worker.set_recording_sink(recording_sink)
    worker.set_recording_enabled(True)

    worker.frame_captured.connect(lambda *_args: worker.stop())

    worker.run()

    assert len(seen["error_occurred"]) == 0
    assert len(seen["frame_captured"]) == 1
    assert len(recorded) == 1

    rec_camera_id, rec_frame, rec_timestamp, rec_metadata = recorded[0]
    frame_camera_id, emitted_frame, emitted_timestamp, emitted_metadata = seen["frame_captured"][0]

    assert rec_camera_id == "fake:index:0"
    assert frame_camera_id == "fake:index:0"

    np.testing.assert_array_equal(rec_frame, emitted_frame)
    assert rec_timestamp == emitted_timestamp
    assert rec_metadata == emitted_metadata


def test_worker_recording_sink_disabled_does_not_receive_frame(qtbot, camera_worker_settings: CameraSettings):
    worker = SingleCameraWorker("fake:index:0", camera_worker_settings)

    recorded: list[tuple] = []

    def recording_sink(*args):
        recorded.append(args)

    worker.set_recording_sink(recording_sink)
    worker.set_recording_enabled(False)

    worker.frame_captured.connect(lambda *_args: worker.stop())

    worker.run()

    assert recorded == []


def test_worker_backend_creation_failure_emits_error(monkeypatch, qtbot, camera_worker_settings: CameraSettings):
    from dlclivegui.services import camera_controller as controller_mod

    def fail_create(_settings):
        raise RuntimeError("error")

    monkeypatch.setattr(controller_mod.CameraFactory, "create", staticmethod(fail_create))

    worker = SingleCameraWorker("fake:index:0", camera_worker_settings)
    seen = _capture_signals(worker)

    worker.run()

    assert len(seen["runtime_info"]) == 0
    assert len(seen["started"]) == 0
    assert len(seen["frame_captured"]) == 0

    assert len(seen["error_occurred"]) == 1
    camera_id, message = seen["error_occurred"][0]
    assert camera_id == "fake:index:0"
    assert "Failed to initialize camera" in message
    assert "error" in message

    assert seen["stopped"] == [("fake:index:0",)]


class _EmptyFrameBackend:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.open_called = False
        self.close_called = False

    def open(self):
        self.open_called = True

    def read(self):
        return CapturedFrame(frame=None, software_timestamp=123.0, timestamp_metadata=None)

    def close(self):
        self.close_called = True


def test_worker_too_many_empty_frames_emits_error(monkeypatch, qtbot, camera_worker_settings: CameraSettings):
    from dlclivegui.services import camera_controller as controller_mod

    backend = _EmptyFrameBackend(camera_worker_settings)

    monkeypatch.setattr(
        controller_mod.CameraFactory,
        "create",
        staticmethod(lambda _settings: backend),
    )

    worker = SingleCameraWorker("fake:index:0", camera_worker_settings)
    worker._max_consecutive_errors = 3
    worker._retry_delay = 0.0

    seen = _capture_signals(worker)

    worker.run()

    assert backend.open_called
    assert backend.close_called

    assert seen["started"] == [("fake:index:0",)]
    assert len(seen["frame_captured"]) == 0

    assert len(seen["error_occurred"]) == 1
    camera_id, message = seen["error_occurred"][0]
    assert camera_id == "fake:index:0"
    assert "Too many empty frames" in message

    assert seen["stopped"] == [("fake:index:0",)]


class _ReadExceptionBackend:
    waits_for_hardware_trigger = False

    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.open_called = False
        self.close_called = False
        self.read_count = 0

    def open(self):
        self.open_called = True

    def read(self):
        self.read_count += 1
        raise RuntimeError(f"read failed {self.read_count}")

    def close(self):
        self.close_called = True


def test_worker_read_exception_emits_error_after_retries(
    monkeypatch,
    qtbot,
    camera_worker_settings: CameraSettings,
):
    from dlclivegui.services import camera_controller as controller_mod

    backend = _ReadExceptionBackend(camera_worker_settings)

    monkeypatch.setattr(
        controller_mod.CameraFactory,
        "create",
        staticmethod(lambda _settings: backend),
    )

    worker = SingleCameraWorker("fake:index:0", camera_worker_settings)
    worker._max_consecutive_errors = 3
    worker._retry_delay = 0.0

    seen = _capture_signals(worker)

    worker.run()

    assert backend.open_called
    assert backend.close_called
    assert backend.read_count == 3

    assert seen["started"] == [("fake:index:0",)]
    assert len(seen["frame_captured"]) == 0

    assert len(seen["error_occurred"]) == 1
    camera_id, message = seen["error_occurred"][0]
    assert camera_id == "fake:index:0"
    assert "Camera read error" in message
    assert "read failed 3" in message

    assert seen["stopped"] == [("fake:index:0",)]


class _TimeoutTriggerBackend:
    waits_for_hardware_trigger = True

    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.open_called = False
        self.close_called = False
        self.read_count = 0

    def open(self):
        self.open_called = True

    def read(self):
        self.read_count += 1
        raise TimeoutError("waiting for trigger")

    def close(self):
        self.close_called = True


def test_worker_hardware_trigger_timeouts_do_not_emit_error(
    monkeypatch,
    qtbot,
    camera_worker_settings: CameraSettings,
):
    from dlclivegui.services import camera_controller as controller_mod

    backend = _TimeoutTriggerBackend(camera_worker_settings)

    monkeypatch.setattr(
        controller_mod.CameraFactory,
        "create",
        staticmethod(lambda _settings: backend),
    )

    worker = SingleCameraWorker("fake:index:0", camera_worker_settings)
    worker._trigger_timeout_delay = 0.0

    seen = _capture_signals(worker)

    original_read = backend.read

    def read_then_stop():
        if backend.read_count >= 3:
            worker.stop()
        return original_read()

    backend.read = read_then_stop

    worker.run()

    assert backend.open_called
    assert backend.close_called
    assert backend.read_count >= 3

    assert seen["started"] == [("fake:index:0",)]
    assert seen["error_occurred"] == []
    assert seen["frame_captured"] == []
    assert seen["stopped"] == [("fake:index:0",)]
