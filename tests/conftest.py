# tests/conftest.py
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import (
    CameraBackend,
    SupportLevel,
    register_backend_direct,
    unregister_backend,
)
from dlclivegui.config import (
    DEFAULT_CONFIG,
    ApplicationSettings,
    CameraSettings,
    DLCProcessorSettings,
    MultiCameraSettings,
)
from dlclivegui.gui.main_window import DLCLiveMainWindow

# ---------------------------------------------------------------------
# Generic backend helpers (removes FakeBackend/temp_backend duplication)
# ---------------------------------------------------------------------

DEFAULT_TEST_CAPS: dict[str, SupportLevel] = {
    "set_resolution": SupportLevel.SUPPORTED,
    "set_fps": SupportLevel.SUPPORTED,
    "set_exposure": SupportLevel.SUPPORTED,
    "set_gain": SupportLevel.SUPPORTED,
    "device_discovery": SupportLevel.SUPPORTED,
    "stable_identity": SupportLevel.SUPPORTED,
}


def make_backend_class(
    name: str,
    *,
    caps: dict[str, SupportLevel] | None = None,
    frame_shape: tuple[int, int, int] = (48, 64, 3),
    timestamp_fn: Callable[[], float] = time.time,
) -> type[CameraBackend]:
    """
    Create a lightweight CameraBackend subclass for tests.

    - caps: static_capabilities returned to the GUI
    - frame_shape: deterministic black image returned on read()
    """
    caps = dict(caps) if caps is not None else dict(DEFAULT_TEST_CAPS)

    class _TestBackend(CameraBackend):
        OPTIONS_KEY = name

        def __init__(self, settings: CameraSettings):
            super().__init__(settings)
            self._opened = False
            self._counter = 0

        @classmethod
        def is_available(cls) -> bool:
            return True

        @classmethod
        def static_capabilities(cls) -> dict[str, SupportLevel]:
            return dict(caps)

        def open(self) -> None:
            self._opened = True

        def close(self) -> None:
            self._opened = False

        def stop(self) -> None:
            # Optional API; no-op for tests
            return

        def read(self):
            if not self._opened:
                raise RuntimeError("not opened")
            self._counter += 1
            frame = np.zeros(frame_shape, dtype=np.uint8)
            return frame, float(timestamp_fn())

    _TestBackend.__name__ = f"TestBackend_{name}"
    return _TestBackend


@contextmanager
def _temp_backend(name: str, *, caps: dict[str, SupportLevel], frame_shape=(10, 10, 3)):
    backend_cls = make_backend_class(name, caps=caps, frame_shape=frame_shape)
    register_backend_direct(name, backend_cls)
    try:
        yield backend_cls
    finally:
        unregister_backend(name)


@pytest.fixture
def temp_backend():
    return _temp_backend


@pytest.fixture(scope="session", autouse=True)
def register_fake_backend_session():
    """
    Register the "fake" backend once per test session.
    Your app config uses backend="fake", so this makes CameraFactory.create work naturally
    without monkeypatching CameraFactory.create everywhere.
    """
    fake_cls = make_backend_class("fake", caps=DEFAULT_TEST_CAPS, frame_shape=(48, 64, 3))
    register_backend_direct("fake", fake_cls)
    try:
        tuple(CameraFactory.backend_names())
    except Exception:
        pass
    try:
        yield fake_cls
    finally:
        unregister_backend("fake")


@pytest.fixture(scope="session")
def fake_backend_cls(register_fake_backend_session):
    """Return the registered fake backend class."""
    return register_fake_backend_session


@pytest.fixture
def fake_backend_factory(fake_backend_cls):
    """
    Return a factory(settings) -> backend instance.
    Always forces backend='fake' for deterministic identity/caps.
    """

    def _factory(settings: CameraSettings):
        try:
            s = settings.model_copy(deep=True)
        except Exception:
            s = settings
        s.backend = "fake"
        return fake_backend_cls(s)

    return _factory


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
        return np.ones((2, 2), dtype=float)


@pytest.fixture
def fake_dlclive_factory():
    """A factory that creates FakeDLCLive instances."""

    def _factory(**opts):
        return FakeDLCLive(**opts)

    return _factory


@pytest.fixture(scope="session")
def FakeDLCLiveClass():
    return FakeDLCLive


@pytest.fixture
def monkeypatch_dlclive(monkeypatch):
    """
    Replace dlclive.DLCLive import with FakeDLCLive within dlc_processor module.
    """
    from dlclivegui.services import dlc_processor

    monkeypatch.setattr(dlc_processor, "DLCLive", FakeDLCLive)
    return FakeDLCLive


@pytest.fixture
def settings_model():
    """A standard Pydantic DLCProcessorSettings for tests."""
    return DLCProcessorSettings(model_path="dummy.pt")


# ---------------------------------------------------------------------
# Reusable config builder (removes duplication in app_config_* fixtures)
# ---------------------------------------------------------------------


def make_app_config(
    *,
    tmp_path: Path,
    num_cams: int = 2,
    backend: str = "fake",
    enabled: bool = True,
    fps: float = 30.0,
    max_cameras: int = 4,
    tile_layout: str = "auto",
    recording_enabled: bool = True,
) -> ApplicationSettings:
    cfg = ApplicationSettings.from_dict(DEFAULT_CONFIG.to_dict())

    cams: list[CameraSettings] = []
    for i in range(num_cams):
        cams.append(CameraSettings(name=f"Cam{i}", backend=backend, index=i, enabled=enabled, fps=fps))

    cfg.multi_camera = MultiCameraSettings(cameras=cams, max_cameras=max_cameras, tile_layout=tile_layout)
    cfg.camera = cams[0] if cams else CameraSettings()  # backward compat

    cfg.recording.directory = str(tmp_path / "videos")
    cfg.recording.enabled = bool(recording_enabled)
    return cfg


@pytest.fixture
def app_config_two_cams(tmp_path) -> ApplicationSettings:
    """An app config with two enabled cameras and writable recording dir."""
    return make_app_config(tmp_path=tmp_path, num_cams=2, backend="fake", enabled=True, fps=30.0)


# ---------------------------------------------------------------------
# Main window fixture
# ---------------------------------------------------------------------


@pytest.fixture
def window(qtbot, app_config_two_cams):
    """
    Construct the real DLCLiveMainWindow with a valid config,
    make it headless, show it, and yield it.
    """
    w = DLCLiveMainWindow(config=app_config_two_cams)
    qtbot.addWidget(w)
    w.setAttribute(Qt.WA_DontShowOnScreen, True)
    w.show()

    try:
        yield w
    finally:
        try:
            w.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Drawing / recording helpers (unchanged, but still isolated)
# ---------------------------------------------------------------------


@pytest.fixture
def draw_pose_stub(monkeypatch):
    """Fake pose drawing that records offset/scale and draws a bright pixel."""
    calls = {}

    def _stub_draw_pose(frame, pose, p_cutoff=None, colormap=None, offset=(0, 0), scale=(1.0, 1.0), **_ignored):
        calls["offset"] = offset
        calls["scale"] = scale

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

    import dlclivegui.gui.main_window as mw_mod

    monkeypatch.setattr(mw_mod, "draw_pose", _stub_draw_pose)
    return calls


@pytest.fixture
def multi_camera_controller(window):
    return window.multi_camera_controller


@pytest.fixture
def dlc_processor(window):
    return window._dlc


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

        run_dir = tmp_path / "videos" / "Sess" / "run_TEST"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    from dlclivegui.gui import recording_manager as rm_mod

    monkeypatch.setattr(rm_mod.RecordingManager, "start_all", _fake_start_all)
    return calls


class _FakeProcessor:
    def __init__(self):
        self.conns = [object()]
        self._recording = True
        self._vid_recording = True
        self.video_recording = True
        self.session_name = "auto_ABC"
        self.recording = True


@pytest.fixture
def fake_processor():
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
    return app_config_two_cams.recording.model_copy(deep=True)


@pytest.fixture
def patch_video_recorder(monkeypatch):
    import dlclivegui.gui.recording_manager as rm_mod

    monkeypatch.setattr(rm_mod, "VideoRecorder", FakeVideoRecorder)
    return FakeVideoRecorder


@pytest.fixture
def recording_frame_spy(monkeypatch, window):
    captured = {}

    def _fake_write_frame(cam_id, frame, timestamp=None):
        captured[cam_id] = frame.copy()

    monkeypatch.setattr(window._rec_manager, "write_frame", _fake_write_frame)
    return captured


@pytest.fixture
def patch_build_run_dir(monkeypatch, tmp_path):
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


# ---------------------------------------------------------------------
# Optional legacy fixture: patch_factory (keep only if some tests still depend on it)
# ---------------------------------------------------------------------
@pytest.fixture
def patch_factory(monkeypatch, fake_backend_factory):
    """
    Patch CameraFactory.create to always return the fake backend, regardless of backend name.
    This supports tests that still create CameraSettings(backend="opencv", ...).
    """

    def _create(settings: CameraSettings):
        return fake_backend_factory(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create))
    return _create
