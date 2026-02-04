# tests/conftest.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend
from dlclivegui.config import (
    DEFAULT_CONFIG,
    ApplicationSettings,
    CameraSettings,
    DLCProcessorSettings,
    MultiCameraSettings,
)
from dlclivegui.gui.main_window import DLCLiveMainWindow


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------
class FakeDLCLive:
    """A minimal fake DLCLive object for testing."""

    def __init__(self, **opts):
        self.opts = opts
        self.init_called = False
        self.pose_calls = 0

    def init_inference(self, frame):
        self.init_called = True

    def get_pose(self, frame, frame_time=None):
        self.pose_calls += 1
        # Deterministic small pose array
        return np.ones((2, 2), dtype=float)


@pytest.fixture
def fake_dlclive_factory():
    """A factory that creates FakeDLCLive instances."""

    def _factory(**opts):
        return FakeDLCLive(**opts)

    return _factory


class FakeBackend(CameraBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self._opened = False
        self._counter = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    def open(self) -> None:
        self._opened = True

    def read(self):
        # Produce a deterministic small frame
        if not self._opened:
            raise RuntimeError("not opened")
        self._counter += 1
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        ts = time.time()
        return frame, ts

    def close(self) -> None:
        self._opened = False

    def stop(self) -> None:
        pass


@pytest.fixture
def fake_backend_factory():
    """A factory that creates FakeBackend instances."""

    def _factory(settings):
        return FakeBackend(settings)

    return _factory


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def patch_factory(monkeypatch):
    def _create(settings):
        return FakeBackend(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create))
    return _create


@pytest.fixture
def monkeypatch_dlclive(monkeypatch):
    """
    Replace the dlclive.DLCLive import with FakeDLCLive *within* the dlc_processor module.

    Scope is function-level by default, which keeps tests isolated.
    """
    from dlclivegui.services import dlc_processor

    monkeypatch.setattr(dlc_processor, "DLCLive", FakeDLCLive)
    return FakeDLCLive


@pytest.fixture
def settings_model():
    """A standard Pydantic DLCProcessorSettingsModel for tests."""
    return DLCProcessorSettings(model_path="dummy.pt")


# ---------- Test helpers: application configuration with two fake cameras ----------
@pytest.fixture
def app_config_two_cams(tmp_path) -> ApplicationSettings:
    """An app config with two enabled cameras (fake backend) and writable recording dir."""
    cfg = ApplicationSettings.from_dict(DEFAULT_CONFIG.to_dict())

    cam_a = CameraSettings(name="CamA", backend="fake", index=0, enabled=True, fps=30.0)
    cam_b = CameraSettings(name="CamB", backend="fake", index=1, enabled=True, fps=30.0)

    cfg.multi_camera = MultiCameraSettings(cameras=[cam_a, cam_b], max_cameras=4, tile_layout="auto")
    cfg.camera = cam_a  # kept for backward-compat single-camera access in UI

    cfg.recording.directory = str(tmp_path / "videos")
    cfg.recording.enabled = True
    return cfg


# ---------- The main window fixture ----------
@pytest.fixture
def window(qtbot, app_config_two_cams) -> DLCLiveMainWindow:
    """
    Construct the real DLCLiveMainWindow with a valid two-camera config,
    make it headless, show it, and yield it. Threads and timers are managed by close().
    """
    w = DLCLiveMainWindow(config=app_config_two_cams)
    qtbot.addWidget(w)
    # Don't pop windows in CI:
    w.setAttribute(Qt.WA_DontShowOnScreen, True)
    w.show()

    try:
        yield w
    finally:
        # The window's closeEvent stops controllers, recorders, timers, etc.
        # Use .close() to trigger the standard shutdown path.
        try:
            w.close()
        except Exception:
            pass


@pytest.fixture
def draw_pose_stub(monkeypatch):
    """Fake pose drawing that records offset/scale and draws a bright pixel."""
    calls = {}

    def _stub_draw_pose(
        frame,
        pose,
        p_cutoff=None,
        colormap=None,
        offset=(0, 0),
        scale=(1.0, 1.0),
        **_ignored,
    ):
        # record args passed to draw_pose
        calls["offset"] = offset
        calls["scale"] = scale

        # pose format: {"x": int, "y": int}
        x = pose["x"]
        y = pose["y"]

        ox, oy = offset
        sx, sy = scale

        xx = int(x * sx + ox)
        yy = int(y * sy + oy)

        out = frame.copy()
        if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
            out[yy, xx] = (0, 255, 0)  # bright green pixel (BGR)
        return out

    # IMPORTANT: patch draw_pose where main_window imports it
    import dlclivegui.gui.main_window as mw_mod

    monkeypatch.setattr(mw_mod, "draw_pose", _stub_draw_pose)

    return calls


# ---------- Convenience fixtures that expose controller/processor from the window ----------
@pytest.fixture
def multi_camera_controller(window):
    """
    Return the *controller used by the window* so tests can wait on all_started/all_stopped.
    """
    return window.multi_camera_controller


@pytest.fixture
def dlc_processor(window):
    """
    Return the *processor used by the window* so tests can connect to pose/initialized.
    """
    return window._dlc


# ---------- Monkeypatch RecordingManager start_all to capture args and return fake path ----------
@pytest.fixture
def start_all_spy(monkeypatch, tmp_path):
    """
    Patch RecordingManager.start_all to capture args and return a fake run_dir.
    """
    calls = {}

    def _fake_start_all(self, recording, active_cams, current_frames, **kwargs):
        calls["recording"] = recording
        calls["active_cams"] = active_cams
        calls["current_frames"] = current_frames
        calls["kwargs"] = kwargs

        # deterministic fake path returned to GUI
        run_dir = tmp_path / "videos" / "Sess" / "run_TEST"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # IMPORTANT: patch the RecordingManager class that the GUI imports.
    from dlclivegui.gui import recording_manager as rm_mod

    monkeypatch.setattr(rm_mod.RecordingManager, "start_all", _fake_start_all)

    return calls


# ---------- Fake processor ----------
class _FakeProcessor:
    def __init__(self):
        self.conns = [object()]
        self._recording = True  # just needs to exist
        self._vid_recording = True  # attribute presence required by your code
        self.video_recording = True
        self.session_name = "auto_ABC"
        self.recording = True


@pytest.fixture
def fake_processor():
    """Return a simple fake processor for testing."""
    return _FakeProcessor()


# ---------- RecordingManager helpers/fixtures ----------
class FakeVideoRecorder:
    """Lightweight test double for VideoRecorder (no threads/ffmpeg)."""

    def __init__(self, output, frame_size=None, frame_rate=None, codec="libx264", crf=23, **kwargs):
        self.output = Path(output)
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.codec = codec
        self.crf = crf
        self.started = False
        self.stopped = False
        self.write_calls = []
        self.raise_on_start = False
        self.raise_on_write = False
        self._stats = None

    @property
    def is_running(self):
        return self.started and not self.stopped

    def start(self):
        if self.raise_on_start:
            raise RuntimeError("start failed")
        self.started = True

    def stop(self):
        self.stopped = True

    def write(self, frame, timestamp=None):
        if self.raise_on_write:
            raise RuntimeError("write failed")
        self.write_calls.append((frame, timestamp))
        return True

    def get_stats(self):
        return self._stats


@pytest.fixture
def recording_settings(app_config_two_cams):
    """
    RecordingSettingsModel clone derived from app_config_two_cams.
    Keeps tests isolated from mutation across runs.
    """
    return app_config_two_cams.recording.model_copy(deep=True)


@pytest.fixture
def patch_video_recorder(monkeypatch):
    """
    Patch the VideoRecorder symbol used inside dlclivegui.gui.recording_manager
    so RecordingManager tests don't invoke vidgear/ffmpeg.
    """
    import dlclivegui.gui.recording_manager as rm_mod

    monkeypatch.setattr(rm_mod, "VideoRecorder", FakeVideoRecorder)
    return FakeVideoRecorder


@pytest.fixture
def recording_frame_spy(monkeypatch, window):
    """Capture frames passed to RecordingManager.write_frame calls."""
    captured = {}

    def _fake_write_frame(cam_id, frame, timestamp=None):
        captured[cam_id] = frame.copy()

    monkeypatch.setattr(window._rec_manager, "write_frame", _fake_write_frame)
    return captured


@pytest.fixture
def patch_build_run_dir(monkeypatch, tmp_path):
    """
    Patch build_run_dir (resolved in dlclivegui.gui.recording_manager namespace)
    to return a deterministic run directory and capture the call args.
    """
    import dlclivegui.gui.recording_manager as rm_mod

    spy = {"session_dir": None, "use_timestamp": None}
    run_dir = tmp_path / "videos" / "Sess_SANITIZED" / "run_TEST"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fake_build_run_dir(session_dir: Path, *, use_timestamp: bool):
        spy["session_dir"] = Path(session_dir)
        spy["use_timestamp"] = use_timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(rm_mod, "build_run_dir", _fake_build_run_dir)
    return spy, run_dir
