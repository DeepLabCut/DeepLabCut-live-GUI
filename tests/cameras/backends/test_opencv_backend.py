from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
import dlclivegui.cameras.backends.opencv_backend as ob  # noqa: E402


class FakeCapture:
    """A controllable fake cv2.VideoCapture."""

    def __init__(self, opened=True, backend_name="FAKE"):
        self._opened = opened
        self._released = False
        self._backend_name = backend_name

        # Emulate common capture properties
        self.props = {
            ob.cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            ob.cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            ob.cv2.CAP_PROP_FPS: 30.0,
            ob.cv2.CAP_PROP_FOURCC: 0.0,
        }

        # Behavior toggles for read path
        self.grab_ok = True
        self.retrieve_ok = True
        self.retrieve_frame = None  # if None, create a dummy frame on retrieve()

        # Introspection
        self.set_calls = []
        self.get_calls = []
        self.grab_calls = 0
        self.retrieve_calls = 0

    def isOpened(self):
        return self._opened and not self._released

    def release(self):
        self._released = True

    def getBackendName(self):
        return self._backend_name

    def get(self, prop_id):
        self.get_calls.append(prop_id)
        return self.props.get(prop_id, 0.0)

    def set(self, prop_id, value):
        self.set_calls.append((prop_id, value))
        self.props[prop_id] = float(value)
        return True

    def grab(self):
        self.grab_calls += 1
        return self.grab_ok

    def retrieve(self):
        self.retrieve_calls += 1
        if not self.retrieve_ok:
            return False, None
        if self.retrieve_frame is None:
            import numpy as np

            self.retrieve_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        return True, self.retrieve_frame


def make_settings(index=0, fps=30.0, properties=None):
    """Minimal settings object compatible with CameraBackend usage."""
    if properties is None:
        properties = {}
    return SimpleNamespace(index=index, fps=fps, properties=properties)


def test_parse_resolution_defaults_and_invalid_values():
    backend = ob.OpenCVCameraBackend(make_settings(properties={}))
    assert backend._parse_resolution(None) == (720, 540)
    assert backend._parse_resolution([1280, 720]) == (1280, 720)
    assert backend._parse_resolution(("1920", "1080")) == (1920, 1080)
    assert backend._parse_resolution(("bad", 123)) == (720, 540)
    assert backend._parse_resolution("nope") == (720, 540)


def test_normalize_resolution_windows(monkeypatch):
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")
    backend = ob.OpenCVCameraBackend(make_settings(properties={"resolution": (800, 600)}))
    assert backend._normalize_resolution(800, 600) == (1280, 720)
    assert backend._normalize_resolution(1920, 1080) == (1920, 1080)


def test_try_open_windows_fallback_to_msmf(monkeypatch):
    """If preferred backend fails on Windows, try MSMF then ANY."""
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    calls = []

    def fake_videocapture(index, flag):
        calls.append((index, flag))
        if flag == getattr(ob.cv2, "CAP_DSHOW", ob.cv2.CAP_ANY):
            return FakeCapture(opened=False)
        if flag == getattr(ob.cv2, "CAP_MSMF", ob.cv2.CAP_ANY):
            return FakeCapture(opened=True, backend_name="MSMF")
        return FakeCapture(opened=False)

    monkeypatch.setattr(ob.cv2, "VideoCapture", fake_videocapture)

    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    preferred = getattr(ob.cv2, "CAP_DSHOW", ob.cv2.CAP_ANY)
    cap = backend._try_open(0, preferred)
    assert cap is not None and cap.isOpened()
    assert cap.getBackendName() == "MSMF"
    assert calls[0][1] == preferred
    assert calls[1][1] == getattr(ob.cv2, "CAP_MSMF", ob.cv2.CAP_ANY)


def test_open_uses_alt_index_probe_on_windows(monkeypatch):
    """If initial open fails and alt_index_probe is enabled, try index+1."""
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    calls = []

    def fake_videocapture(index, flag):
        calls.append((index, flag))
        if index == 1:
            return FakeCapture(opened=True, backend_name="DSHOW")
        return FakeCapture(opened=False)

    monkeypatch.setattr(ob.cv2, "VideoCapture", fake_videocapture)

    settings = make_settings(index=0, fps=30.0, properties={"alt_index_probe": True})
    backend = ob.OpenCVCameraBackend(settings)
    backend.open()

    assert any(idx == 0 for idx, _ in calls)
    assert any(idx == 1 for idx, _ in calls)
    assert "camera" in backend.device_name().lower()


def test_open_raises_when_unable_to_open(monkeypatch):
    monkeypatch.setattr(ob.platform, "system", lambda: "Linux")

    def fake_videocapture(index, flag):
        return FakeCapture(opened=False)

    monkeypatch.setattr(ob.cv2, "VideoCapture", fake_videocapture)

    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    with pytest.raises(RuntimeError, match="Unable to open camera index"):
        backend.open()


def test_read_returns_none_on_grab_failure():
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = FakeCapture(opened=True)
    cap.grab_ok = False
    backend._capture = cap

    frame, ts = backend.read()
    assert frame is None
    assert isinstance(ts, float)


def test_read_returns_none_on_retrieve_failure():
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = FakeCapture(opened=True)
    cap.retrieve_ok = False
    backend._capture = cap

    frame, ts = backend.read()
    assert frame is None
    assert isinstance(ts, float)


def test_read_never_raises_on_exception():
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = FakeCapture(opened=True)

    def boom():
        raise RuntimeError("transient")

    cap.grab = boom
    backend._capture = cap

    frame, ts = backend.read()
    assert frame is None
    assert isinstance(ts, float)


def test_configure_capture_sets_resolution_and_fps_non_faststart_windows(monkeypatch):
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    cap = FakeCapture(opened=True, backend_name="DSHOW")
    cap.props[ob.cv2.CAP_PROP_FRAME_WIDTH] = 640.0
    cap.props[ob.cv2.CAP_PROP_FRAME_HEIGHT] = 480.0
    cap.props[ob.cv2.CAP_PROP_FPS] = 30.0

    settings = make_settings(index=0, fps=60.0, properties={"resolution": (800, 600)})
    backend = ob.OpenCVCameraBackend(settings)
    backend._capture = cap

    backend._configure_capture()

    assert backend.actual_resolution == (1280, 720)
    assert settings.properties["resolution"] == (1280, 720)
    assert backend.actual_fps is not None
    assert isinstance(backend.actual_fps, float)


def test_configure_capture_fast_start_does_not_force_resolution(monkeypatch):
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    cap = FakeCapture(opened=True, backend_name="DSHOW")
    cap.props[ob.cv2.CAP_PROP_FRAME_WIDTH] = 1920.0
    cap.props[ob.cv2.CAP_PROP_FRAME_HEIGHT] = 1080.0

    settings = make_settings(index=0, fps=30.0, properties={"resolution": (1280, 720), "fast_start": True})
    backend = ob.OpenCVCameraBackend(settings)
    backend._capture = cap

    backend._configure_capture()

    assert backend.actual_resolution == (1920, 1080)
    assert settings.properties["resolution"] == (1920, 1080)


def test_configure_capture_applies_only_safe_numeric_properties(monkeypatch):
    monkeypatch.setattr(ob.platform, "system", lambda: "Linux")

    cap = FakeCapture(opened=True)
    settings = make_settings(
        index=0,
        fps=30.0,
        properties={
            "resolution": (640, 480),
            "api": "ANY",
            "fast_start": False,
            "alt_index_probe": False,
            str(int(getattr(ob.cv2, "CAP_PROP_GAIN", 14))): 7,
            "999": 123,
            "not-a-number": 1,
        },
    )

    backend = ob.OpenCVCameraBackend(settings)
    backend._capture = cap
    backend._configure_capture()

    gain_id = int(getattr(ob.cv2, "CAP_PROP_GAIN", 14))
    assert any(pid == gain_id for pid, _ in cap.set_calls)
    assert not any(pid == 999 for pid, _ in cap.set_calls)


def test_close_and_stop_release_capture():
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = FakeCapture(opened=True)
    backend._capture = cap

    backend.close()
    assert backend._capture is None
    assert cap._released is True

    cap2 = FakeCapture(opened=True)
    backend._capture = cap2
    backend.stop()
    assert backend._capture is None
    assert cap2._released is True
