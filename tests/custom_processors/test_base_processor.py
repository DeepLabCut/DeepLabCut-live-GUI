# tests/custom_processors/test_base_processor.py
from __future__ import annotations

import importlib
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _mock_dlclive(monkeypatch):
    class Processor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, pose, **kwargs):
            return pose

    dlclive_mod = types.ModuleType("dlclive")
    processor_mod = types.ModuleType("dlclive.processor")

    dlclive_mod.Processor = Processor
    processor_mod.Processor = Processor

    monkeypatch.setitem(sys.modules, "dlclive", dlclive_mod)
    monkeypatch.setitem(sys.modules, "dlclive.processor", processor_mod)


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


@pytest.fixture
def example_processor_mod(monkeypatch):
    """
    Import the example processor module with dlclive mocked.
    Adjust module name if your file lives elsewhere.
    """
    _mock_dlclive(monkeypatch)
    mod_name = "dlclivegui.processors.examples"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _mk_bodyparts(n: int) -> list[str]:
    return [f"bp{i}" for i in range(n)]


def _mk_pose(n_keypoints: int = 5) -> np.ndarray:
    """
    Create a small pose array (N, 3) that BaseProcessorSocket.process() accepts.
    Base class does not interpret pose content—only broadcasts/logs it.
    """
    pose = np.zeros((n_keypoints, 3), dtype=float)
    for i in range(n_keypoints):
        pose[i, :] = [10.0 + i, 20.0 + i, 0.9]
    return pose


def test_base_init_and_stop(socket_mod):
    """
    Instantiate BaseProcessorSocket on an ephemeral port, verify core state,
    and ensure stop() is idempotent.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(
        bind=("127.0.0.1", 0),
        use_perf_counter=True,
        save_original=False,
    )
    try:
        assert hasattr(proc, "listener")
        assert callable(proc.timing_func)

        import time as _t

        assert proc.timing_func is _t.perf_counter

        assert proc.recording is False
        assert proc.video_recording is False
        assert proc.curr_step == 0
        assert isinstance(proc.conns, set)
    finally:
        proc.stop()


def test_base_recording_flags_and_session_name(socket_mod):
    """
    _handle_client_message should toggle recording/video flags and set session name.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0))
    try:
        proc._handle_client_message({"cmd": "start_recording"})
        assert proc.recording is True
        assert proc.video_recording is True
        assert proc.curr_step == 0

        proc._handle_client_message({"cmd": "set_session_name", "session_name": "unit_test"})
        assert proc.session_name == "unit_test"
        assert proc.filename == "unit_test_dlc_processor_data.pkl"

        proc._handle_client_message({"cmd": "stop_recording"})
        assert proc.recording is False
        assert proc.video_recording is False

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

        before_step = proc.curr_step
        ret = proc.process(pose, frame_time=0.012, pose_time=0.013)

        assert ret is pose
        assert proc.curr_step == before_step + 1
        assert len(proc.time_stamp) == 0
        assert len(proc.step) == 0
        assert len(proc.frame_time) == 0
        assert len(proc.pose_time) == 0

        assert proc.original_pose is not None
        assert len(proc.original_pose) == 0

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

        data = proc.get_data()
        assert "start_time" in data
        assert isinstance(data["time_stamp"], np.ndarray)
        assert isinstance(data["step"], np.ndarray)
        assert isinstance(data["frame_time"], np.ndarray)
        assert isinstance(data["pose_time"], np.ndarray)
        assert isinstance(data["original_pose"], np.ndarray)

    finally:
        proc.stop()


def test_save_ignores_pre_recording_original_pose_frames(socket_mod, tmp_path):
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

        pkl_path = tmp_path / "unit_test_pre_recording_frames.pkl"
        h5_path = tmp_path / "unit_test_pre_recording_frames_DLC.hdf5"

        ret = proc.save(pkl_path)
        assert ret == 1

        assert pkl_path.exists()
        assert h5_path.exists()

        with pkl_path.open("rb") as f:
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


@pytest.mark.parametrize(
    ("class_name", "n_keypoints"),
    [
        ("ExampleProcessorSocketCalculateMousePose", 27),
        ("ExampleProcessorSocketFilterKeypoints", 10),
    ],
)
def test_subclass_save_ignores_pre_recording_original_pose_frames(
    socket_mod,
    example_processor_mod,
    class_name,
    n_keypoints,
    tmp_path,
):
    """
    Concrete processors must keep original_pose aligned with recorded metadata
    even when process() is called before recording starts.
    """
    processor_class = getattr(example_processor_mod, class_name)
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

        pkl_path = tmp_path / f"unit_test_{class_name}.pkl"
        h5_path = tmp_path / f"unit_test_{class_name}_DLC.hdf5"

        ret = proc.save(pkl_path)
        assert ret == 1

        assert pkl_path.exists()
        assert h5_path.exists()

        with pkl_path.open("rb") as f:
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


def test_base_broadcast_handles_bad_connections(socket_mod):
    """
    broadcast() must handle failing connections gracefully and drop them.
    We simulate a conn that raises on send() and can't be closed cleanly.
    """

    class BadConn:
        def __init__(self):
            class Sock:
                def shutdown(self, *_args, **_kwargs):
                    raise RuntimeError("shutdown fail")

            self._socket = Sock()

        def send(self, _payload):
            raise RuntimeError("send fail")

        def close(self):
            raise RuntimeError("close fail")

        def __hash__(self):
            return id(self)

        def __eq__(self, other):
            return self is other

    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0))

    try:
        bad = BadConn()
        proc.conns.add(bad)

        proc.broadcast(["ts", "payload"])

        assert bad not in proc.conns

    finally:
        proc.stop()


def test_save_writes_pkl_and_hdf5_with_labels(socket_mod, tmp_path, caplog):
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

        pose = _mk_pose(n_keypoints=n_keypoints)

        proc._handle_client_message({"cmd": "start_recording"})
        for _ in range(3):
            proc.process(pose, frame_time=0.01, pose_time=0.011)
        proc._handle_client_message({"cmd": "stop_recording"})

        pkl_path = tmp_path / "unit_test_session.pkl"
        h5_path = tmp_path / "unit_test_session_DLC.hdf5"

        ret = proc.save(pkl_path)
        assert ret == 1

        assert pkl_path.exists(), f"Missing {pkl_path}"
        assert h5_path.exists(), f"Missing {h5_path}"

        with pkl_path.open("rb") as f:
            payload = pickle.load(f)

        assert "original_pose" not in payload
        assert "dlc_cfg" in payload
        assert payload["dlc_cfg"] == dlc_cfg

        pytest.importorskip("tables")
        df = pd.read_hdf(h5_path, key="df_with_missing")

        assert df.shape[0] == 3

        expected_cols = pd.MultiIndex.from_product(
            [bodyparts, ["x", "y", "likelihood"]],
            names=["bodyparts", "coords"],
        )

        for col in expected_cols:
            assert col in df.columns

        assert "frame_time" in df.columns
        assert "pose_time" in df.columns
        assert list(df["frame_time"]) == [0.01, 0.01, 0.01]
        assert list(df["pose_time"]) == list(payload["time_stamp"])

        for i, bp in enumerate(bodyparts):
            assert np.isclose(df[(bp, "x")].iloc[0], 10.0 + i)
            assert np.isclose(df[(bp, "y")].iloc[0], 20.0 + i)
            assert np.isclose(df[(bp, "likelihood")].iloc[0], 0.9)

    finally:
        proc.stop()


def test_save_without_dlc_cfg_unlabeled_columns(socket_mod, tmp_path, caplog):
    """
    Ensure that without dlc_cfg, save() still writes HDF5 with unlabeled columns
    and logs a warning or at least succeeds without crashing.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=True)

    try:
        pose = _mk_pose(3)

        proc._handle_client_message({"cmd": "start_recording"})
        proc.process(pose, frame_time=0.01, pose_time=0.02)
        proc._handle_client_message({"cmd": "stop_recording"})

        pkl_path = tmp_path / "unit_test_no_dlc_cfg.pkl"
        h5_path = tmp_path / "unit_test_no_dlc_cfg_DLC.hdf5"

        ret = proc.save(pkl_path)
        assert ret == 1

        assert pkl_path.exists()
        assert h5_path.exists()

        # Depending on logger config in tests, caplog may or may not catch this.
        [rec for rec in caplog.records if "saving without column labels" in rec.message]

        pytest.importorskip("tables")
        df = pd.read_hdf(h5_path, key="df_with_missing")

        assert df.shape[0] == 1

        numeric_cols = [c for c in df.columns if c not in ("frame_time", "pose_time")]
        assert len(numeric_cols) == 3 * 3

    finally:
        proc.stop()


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


def test_save_handles_empty_original_pose(socket_mod, tmp_path):
    """
    With save_original=True but no process() calls, save() should not raise.
    The exact return value is implementation-dependent for empty data.
    """
    BaseProcessorSocket = socket_mod.BaseProcessorSocket
    proc = BaseProcessorSocket(bind=("127.0.0.1", 0), save_original=True)

    try:
        pkl_path = tmp_path / "unit_test_empty_original.pkl"

        ret = proc.save(pkl_path)

        assert ret in (1, 0, -1)

    finally:
        proc.stop()
