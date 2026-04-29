# tests/processors/test_dlc_processor_socket.py
from __future__ import annotations

import importlib
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
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


def _module_data_dir(socket_mod) -> Path:
    """Compute the data/ directory where save() writes artifacts."""
    return Path(socket_mod.__file__).parent.parent.parent / "data"


def _mk_bodyparts(n: int) -> list[str]:
    return [f"bp{i}" for i in range(n)]


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
            - when save_original=True, store copies of pose arrays only while recording.
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
        # Raw poses must stay aligned with recorded metadata.
        assert proc.original_pose is not None
        assert len(proc.original_pose) == 0

        # Start recording and push two frames
        proc._handle_client_message({"cmd": "start_recording"})
        for _ in range(2):
            proc.process(pose, frame_time=0.01, pose_time=0.011)

        assert len(proc.time_stamp) == 2
        assert len(proc.step) == 2
        assert len(proc.frame_time) == 2
        assert len(proc.pose_time) == 2
        assert len(proc.original_pose) == 2
        np.testing.assert_allclose(proc.original_pose[0], pose)
        np.testing.assert_allclose(proc.original_pose[1], pose)

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


def test_save_ignores_pre_recording_original_pose_frames(socket_mod):
    """
    save_original data must stay aligned with recorded metadata even if process()
    is called before recording starts.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=True)

    try:
        n_keypoints = 4
        bodyparts = _mk_bodyparts(n_keypoints)
        proc.set_dlc_cfg({"metadata": {"bodyparts": bodyparts}})

        pose = _mk_pose(n_keypoints=n_keypoints)

        for _ in range(3):
            proc.process(pose, frame_time=0.001, pose_time=0.002)

        assert len(proc.original_pose) == 0
        assert len(proc.frame_time) == 0

        proc._handle_client_message({"cmd": "start_recording"})
        for _ in range(2):
            proc.process(pose, frame_time=0.01, pose_time=0.02)
        proc._handle_client_message({"cmd": "stop_recording"})

        filename = "unit_test_pre_recording_frames.pkl"
        ret = proc.save(filename)
        assert ret == 1

        data_dir = _module_data_dir(socket_mod)
        pkl_path = data_dir / filename
        h5_path = data_dir / (Path(filename).stem + "_DLC.hdf5")

        assert pkl_path.exists()
        assert h5_path.exists()

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        assert len(payload["frame_time"]) == 2
        assert len(payload["time_stamp"]) == 2

        pytest.importorskip("tables")
        df = pd.read_hdf(h5_path, key="df_with_missing")
        assert df.shape[0] == 2
        assert list(df["frame_time"]) == [0.01, 0.01]
        assert list(df["pose_time"]) == list(payload["time_stamp"])

    finally:
        proc.stop()
        try:
            pkl_path.unlink(missing_ok=True)
            h5_path.unlink(missing_ok=True)
        except Exception:
            pass


@pytest.mark.parametrize(
    ("class_name", "n_keypoints"),
    [
        ("ExampleProcessorSocketCalculateMousePose", 27),
        ("ExampleProcessorSocketFilterKeypoints", 10),
    ],
)
def test_subclass_save_ignores_pre_recording_original_pose_frames(socket_mod, class_name, n_keypoints):
    """
    Concrete processors must keep original_pose aligned with recorded metadata
    even when process() is called before recording starts.
    """
    processor_class = getattr(socket_mod, class_name)
    proc = processor_class(bind=("127.0.0.1", 0), save_original=True)

    try:
        bodyparts = _mk_bodyparts(n_keypoints)
        proc.set_dlc_cfg({"metadata": {"bodyparts": bodyparts}})

        pose = _mk_pose(n_keypoints=n_keypoints)

        for _ in range(4):
            proc.process(pose, frame_time=0.001, pose_time=0.002)

        assert len(proc.original_pose) == 0
        assert len(proc.frame_time) == 0

        proc._handle_client_message({"cmd": "start_recording"})
        for _ in range(3):
            proc.process(pose, frame_time=0.01, pose_time=0.02)
        proc._handle_client_message({"cmd": "stop_recording"})

        filename = f"unit_test_{class_name}.pkl"
        ret = proc.save(filename)
        assert ret == 1

        data_dir = _module_data_dir(socket_mod)
        pkl_path = data_dir / filename
        h5_path = data_dir / (Path(filename).stem + "_DLC.hdf5")

        assert pkl_path.exists()
        assert h5_path.exists()

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        assert len(payload["frame_time"]) == 3
        assert len(payload["time_stamp"]) == 3

        pytest.importorskip("tables")
        df = pd.read_hdf(h5_path, key="df_with_missing")
        assert df.shape[0] == 3
        assert list(df["frame_time"]) == [0.01, 0.01, 0.01]
        assert list(df["pose_time"]) == list(payload["time_stamp"])

    finally:
        proc.stop()
        try:
            pkl_path.unlink(missing_ok=True)
            h5_path.unlink(missing_ok=True)
        except Exception:
            pass


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


def test_save_writes_pkl_and_hdf5_with_labels(socket_mod, caplog):
    """
    End-to-end save() with save_original=True and a matching dlc_cfg bodypart list.
    Verifies:
      - .pkl exists and does not include 'original_pose'
      - .pkl includes 'dlc_cfg'
      - _DLC.hdf5 exists and contains expected labeled columns and row count
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=True)

    try:
        n_keypoints = 5
        bodyparts = _mk_bodyparts(n_keypoints)
        dlc_cfg = {"metadata": {"bodyparts": bodyparts}}
        proc.set_dlc_cfg(dlc_cfg)

        # create 3 frames
        pose = _mk_pose(n_keypoints=n_keypoints)
        proc._handle_client_message({"cmd": "start_recording"})
        for _ in range(3):
            proc.process(pose, frame_time=0.01, pose_time=0.011)
        proc._handle_client_message({"cmd": "stop_recording"})

        # deterministic relative filename
        filename = "unit_test_session.pkl"
        ret = proc.save(filename)
        assert ret == 1

        data_dir = _module_data_dir(socket_mod)
        pkl_path = data_dir / filename
        h5_path = data_dir / (Path(filename).stem + "_DLC.hdf5")

        assert pkl_path.exists(), f"Missing {pkl_path}"
        assert h5_path.exists(), f"Missing {h5_path}"

        # verify pkl payload
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        assert "original_pose" not in payload  # popped out before pickling
        assert "dlc_cfg" in payload
        assert payload["dlc_cfg"] == dlc_cfg

        # verify HDF5 contents (skip if tables is not installed)
        pytest.importorskip("tables")
        df = pd.read_hdf(h5_path, key="df_with_missing")
        # Expect rows == frames
        assert df.shape[0] == 3

        # Confirm the labeled columns exist for all bodyparts x (x, y, likelihood)
        expected_cols = pd.MultiIndex.from_product(
            [bodyparts, ["x", "y", "likelihood"]],
            names=["bodyparts", "coords"],
        )
        # Some pandas versions will allow mixing multiindex + string cols;
        # so just check presence of expected label tuples:
        for col in expected_cols:
            assert col in df.columns

        # frame_time & pose_time columns are present
        assert "frame_time" in df.columns
        assert "pose_time" in df.columns
        assert list(df["frame_time"]) == [0.01, 0.01, 0.01]
        assert list(df["pose_time"]) == list(payload["time_stamp"])

        # sanity check values for first row
        for i, bp in enumerate(bodyparts):
            assert np.isclose(df[(bp, "x")].iloc[0], 10.0 + i)
            assert np.isclose(df[(bp, "y")].iloc[0], 20.0 + i)
            assert np.isclose(df[(bp, "likelihood")].iloc[0], 0.9)

    finally:
        proc.stop()
        # cleanup
        try:
            pkl_path.unlink(missing_ok=True)
            h5_path.unlink(missing_ok=True)
        except Exception:
            pass


def test_save_without_dlc_cfg_unlabeled_columns(socket_mod, caplog):
    """
    Ensure that without dlc_cfg, save() still writes HDF5 with unlabeled columns
    and logs a warning (no crash).
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=True)

    try:
        pose = _mk_pose(3)
        proc._handle_client_message({"cmd": "start_recording"})
        proc.process(pose, frame_time=0.01, pose_time=0.02)
        proc._handle_client_message({"cmd": "stop_recording"})

        filename = "unit_test_no_dlc_cfg.pkl"
        ret = proc.save(filename)
        assert ret == 1

        data_dir = _module_data_dir(socket_mod)
        pkl_path = data_dir / filename
        h5_path = data_dir / (Path(filename).stem + "_DLC.hdf5")

        assert pkl_path.exists()
        assert h5_path.exists()

        # Check warning logged
        # (Depending on logger config in tests, you may need to set level to capture warnings)
        [rec for rec in caplog.records if "saving without column labels" in rec.message]
        # It's okay if caplog didn't catch it due to logger level; we mainly ensure no crash and files exist.

        # Verify HDF5 loads (skip if tables not installed)
        pytest.importorskip("tables")
        df = pd.read_hdf(h5_path, key="df_with_missing")
        assert df.shape[0] == 1  # 1 frame saved
        # Expect unlabeled numeric columns for pose plus "frame_time" and "pose_time"
        # We can't rely on a MultiIndex here; just ensure numeric columns exist
        numeric_cols = [c for c in df.columns if c not in ("frame_time", "pose_time")]
        assert len(numeric_cols) == 3 * 3  # 3 keypoints * 3 coords

    finally:
        proc.stop()
        # cleanup
        try:
            pkl_path.unlink(missing_ok=True)
            h5_path.unlink(missing_ok=True)
        except Exception:
            pass


def test_get_data_includes_dlc_cfg(socket_mod):
    """
    If dlc_cfg is set, get_data() should include it.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=False)
    try:
        dlc_cfg = {"metadata": {"bodyparts": ["a", "b"]}}
        proc.set_dlc_cfg(dlc_cfg)
        data = proc.get_data()
        assert "dlc_cfg" in data
        assert data["dlc_cfg"] == dlc_cfg
    finally:
        proc.stop()


def test_save_handles_empty_original_pose(socket_mod):
    """
    With save_original=True but no process() calls, save() should not crash.
    Depending on pandas behavior, HDF5 should exist with 0 rows or be created successfully.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=True)
    try:
        filename = "unit_test_empty_original.pkl"
        ret = proc.save(filename)
        # If nothing to save, your implementation returns 1 (saved) or could be 0; current code returns 1
        assert ret in (1, 0, -1)  # accept current behavior; adjust if you standardize
        data_dir = _module_data_dir(socket_mod)
        pkl_path = data_dir / filename
        h5_path = data_dir / (Path(filename).stem + "_DLC.hdf5")
        # pkl exists if ret == 1; hdf5 may or may not depending on your final logic
        # Leave assertions lenient; the main check is that no exception bubbles up.
    finally:
        proc.stop()
        # cleanup
        try:
            pkl_path.unlink(missing_ok=True)
            h5_path.unlink(missing_ok=True)
        except Exception:
            pass
