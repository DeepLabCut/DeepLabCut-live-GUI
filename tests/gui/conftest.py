# tests/services/gui/conftest.py
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from dlclivegui.cameras import CameraFactory
from dlclivegui.config import (
    DEFAULT_CONFIG,
    ApplicationSettings,
    CameraSettings,
    MultiCameraSettings,
)
from dlclivegui.gui.main_window import DLCLiveMainWindow
from tests.conftest import FakeBackend, FakeDLCLive  # noqa: F401


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


# ---------- Autouse patches to keep GUI tests fast and side-effect-free ----------
@pytest.fixture(autouse=True)
def _patch_camera_factory(monkeypatch):
    """
    Replace hardware backends with FakeBackend globally for GUI tests.
    We patch at the central creation point used by the controller.
    """

    def _create_stub(settings: CameraSettings):
        # FakeBackend ignores 'backend' and produces deterministic frames
        return FakeBackend(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create_stub))


@pytest.fixture(autouse=True)
def _patch_camera_validation(monkeypatch):
    """
    Accept all cameras regardless of backend and silence warning/error dialogs in the window.
    """
    # 1) Pretend all cameras are available
    monkeypatch.setattr(
        CameraFactory,
        "check_camera_available",
        staticmethod(lambda cam: (True, "")),
    )

    # 2) Silence GUI dialogs during tests
    monkeypatch.setattr(DLCLiveMainWindow, "_show_warning", lambda self, msg: None)
    monkeypatch.setattr(DLCLiveMainWindow, "_show_error", lambda self, msg: None)


@pytest.fixture(autouse=True)
def _patch_dlclive_to_fake(monkeypatch):
    """
    Ensure dlclive is replaced by the test double in the DLCLiveProcessor module.
    (The window will instantiate DLCLiveProcessor internally, which imports DLCLive.)
    """
    from dlclivegui.services import dlc_processor as dlcp_mod

    monkeypatch.setattr(dlcp_mod, "DLCLive", FakeDLCLive)


@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path):
    """
    Redirect QSettings to a temp directory so persistence tests are deterministic
    and do not touch real user settings.
    """
    from PySide6.QtCore import QSettings

    # Use INI backend for easy temp path redirection
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))

    # Clear keys for this app/org to avoid leakage between tests
    s = QSettings("DeepLabCut", "DLCLiveGUI")
    s.clear()
    s.sync()

    yield


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
