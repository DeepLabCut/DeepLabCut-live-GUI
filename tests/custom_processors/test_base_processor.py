# tests/processors/test_dlc_processor_socket.py
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _mock_dlclive(monkeypatch):
    """Provide a dummy dlclive.Processor so the module can import in tests."""
    fake = types.ModuleType("dlclive")

    class Processor:
        def __init__(self, *args, **kwargs):
            pass

    fake.Processor = Processor
    monkeypatch.setitem(sys.modules, "dlclive", fake)


@pytest.fixture
def socket_mod(monkeypatch):
    """
    Import the processor module with dlclive mocked.
    Adjust module name if your file lives elsewhere.
    """
    _mock_dlclive(monkeypatch)
    mod_name = "dlclivegui.processors.dlc_processor_socket"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _mk_pose(n_keypoints: int = 5) -> np.ndarray:
    """
    Create a small pose array (N, 3) that BaseProcessorSocket.process() accepts.
    Base class does not interpret pose content—only broadcasts/logs it.
    """
    pose = np.zeros((n_keypoints, 3), dtype=float)
    # Fill with simple coordinates & confidence
    for i in range(n_keypoints):
        pose[i, :] = [10.0 + i, 20.0 + i, 0.9]
    return pose


def test_base_init_and_stop(socket_mod):
    """
    Instantiate BaseProcessorSocket on an ephemeral port, verify core state,
    and ensure stop() is idempotent.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), use_perf_counter=True, save_original=False)
    try:
        # Core attributes exist
        assert hasattr(proc, "listener")
        assert callable(proc.timing_func)
        # perf_counter chosen
        import time as _t

        assert proc.timing_func is _t.perf_counter

        # Initial flags & counters
        assert proc.recording is False
        assert proc.video_recording is False
        assert proc.curr_step == 0
        assert isinstance(proc.conns, set)
    finally:
        # stop must be safe and idempotent
        proc.stop()
        proc.stop()  # second call should be a no-op


def test_base_recording_flags_and_session_name(socket_mod):
    """
    _handle_client_message should toggle recording/video flags and set session name.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0))
    try:
        # Start recording
        proc._handle_client_message({"cmd": "start_recording"})
        assert proc.recording is True
        assert proc.video_recording is True
        assert proc.curr_step == 0  # reset

        # Set a session name
        proc._handle_client_message({"cmd": "set_session_name", "session_name": "unit_test"})
        assert proc.session_name == "unit_test"
        assert proc.filename == "unit_test_dlc_processor_data.pkl"

        # Stop recording
        proc._handle_client_message({"cmd": "stop_recording"})
        assert proc.recording is False
        assert proc.video_recording is False

        # Unknown / invalid messages must not crash
        proc._handle_client_message(None)
        proc._handle_client_message({"cmd": "does_not_exist"})
    finally:
        proc.stop()


def test_base_process_without_and_with_recording(socket_mod):
    """
    BaseProcessorSocket.process() should:
      - increment curr_step always,
      - when recording, append time/step/frame_time/pose_time,
      - when save_original=True, store copies of pose arrays.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=True)
    try:
        pose = _mk_pose()

        # Not recording yet: curr_step increments, no logs appended
        before_step = proc.curr_step
        ret = proc.process(pose, frame_time=0.012, pose_time=0.013)
        assert ret is pose
        assert proc.curr_step == before_step + 1
        assert len(proc.time_stamp) == 0
        assert len(proc.step) == 0
        assert len(proc.frame_time) == 0
        assert len(proc.pose_time) == 0
        # When not recording, save_original is still respected
        assert proc.original_pose is not None
        assert len(proc.original_pose) == 1
        np.testing.assert_allclose(proc.original_pose[0], pose)

        # Start recording and push two frames
        proc._handle_client_message({"cmd": "start_recording"})
        for _ in range(2):
            proc.process(pose, frame_time=0.01, pose_time=0.011)

        assert len(proc.time_stamp) == 2
        assert len(proc.step) == 2
        assert len(proc.frame_time) == 2
        assert len(proc.pose_time) == 2

        # Data snapshot integrity
        data = proc.get_data()
        assert "start_time" in data
        assert isinstance(data["time_stamp"], np.ndarray)
        assert isinstance(data["step"], np.ndarray)
        assert isinstance(data["frame_time"], np.ndarray)
        # pose_time can be None if never provided; here it is provided.
        assert isinstance(data["pose_time"], np.ndarray)
        # original_pose is included when save_original=True
        assert isinstance(data["original_pose"], np.ndarray)

    finally:
        proc.stop()


def test_base_broadcast_handles_bad_connections(socket_mod):
    """
    broadcast() must handle failing connections gracefully and drop them.
    We simulate a conn that raises on send() and can't be closed cleanly.
    """

    class BadConn:
        def __init__(self):
            # Minimal attributes to satisfy _close_conn
            class Sock:
                def shutdown(self, *_args, **_kwargs):
                    raise RuntimeError("shutdown fail")

            self._socket = Sock()

        def send(self, _payload):
            raise RuntimeError("send fail")

        def close(self):
            raise RuntimeError("close fail")

        def __hash__(self):
            # allow put in a set
            return id(self)

        def __eq__(self, other):
            return self is other

    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0))
    try:
        bad = BadConn()
        proc.conns.add(bad)
        # Should not raise
        proc.broadcast(["ts", "payload"])
        # bad conn should be discarded
        assert bad not in proc.conns
    finally:
        proc.stop()
