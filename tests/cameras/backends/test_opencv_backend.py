# tests/cameras/backends/test_opencv_backend.py
from types import SimpleNamespace

import pytest

import dlclivegui.cameras.backends.opencv_backend as ob

pytestmark = pytest.mark.unit


def make_settings(index=0, fps=30.0, properties=None, *, width=0, height=0, backend="opencv", name="Test"):
    """Minimal settings object compatible with OpenCVCameraBackend usage."""
    if properties is None:
        properties = {}
    return SimpleNamespace(
        index=index,
        fps=fps,
        properties=properties,
        width=width,
        height=height,
        backend=backend,
        name=name,
    )


def test_requested_resolution_precedence_property_over_width_height():
    settings = make_settings(
        properties={"resolution": (800, 600)},
        width=1280,
        height=720,
    )
    backend = ob.OpenCVCameraBackend(settings)
    assert backend._get_requested_resolution() == (800, 600)


def test_requested_resolution_uses_width_height_when_no_property_resolution():
    settings = make_settings(
        properties={},
        width=1280,
        height=720,
    )
    backend = ob.OpenCVCameraBackend(settings)
    assert backend._get_requested_resolution() == (1280, 720)


def test_requested_resolution_defaults_when_no_request():
    settings = make_settings(properties={}, width=0, height=0)
    backend = ob.OpenCVCameraBackend(settings)
    assert backend._get_requested_resolution() == (0, 0)


def test_try_open_windows_fallback_to_msmf(monkeypatch, fake_capture_factory):
    """If preferred backend fails on Windows, try MSMF then ANY."""
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    calls = []

    def fake_videocapture(index, flag):
        calls.append((index, flag))
        if flag == getattr(ob.cv2, "CAP_DSHOW", ob.cv2.CAP_ANY):
            return fake_capture_factory(opened=False)
        if flag == getattr(ob.cv2, "CAP_MSMF", ob.cv2.CAP_ANY):
            return fake_capture_factory(opened=True, backend_name="MSMF")
        return fake_capture_factory(opened=False)

    monkeypatch.setattr(ob.cv2, "VideoCapture", fake_videocapture)

    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    preferred = getattr(ob.cv2, "CAP_DSHOW", ob.cv2.CAP_ANY)
    cap = backend._try_open(0, preferred)
    assert cap is not None and cap.isOpened()
    assert cap.getBackendName() == "MSMF"
    assert calls[0][1] == preferred
    assert calls[1][1] == getattr(ob.cv2, "CAP_MSMF", ob.cv2.CAP_ANY)


def test_open_does_not_use_alt_index_probe_when_disabled_in_code(monkeypatch, fake_capture_factory):
    """alt_index_probe is currently commented out in backend.open(); ensure no index+1 attempt."""
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    # Prevent discovery from changing index/backend
    monkeypatch.setattr(ob, "list_cameras", lambda *_a, **_k: [])
    monkeypatch.setattr(ob, "select_camera", lambda *_a, **_k: None)

    calls = []

    def fake_videocapture(index, flag):
        calls.append((index, flag))
        # Only index 0 succeeds; if code tries index 1, we'd see it in calls and fail.
        if index == 0:
            return fake_capture_factory(opened=True, backend_name="DSHOW")
        return fake_capture_factory(opened=False)

    monkeypatch.setattr(ob.cv2, "VideoCapture", fake_videocapture)

    settings = make_settings(index=0, fps=30.0, properties={"opencv": {"alt_index_probe": True}})
    backend = ob.OpenCVCameraBackend(settings)
    backend.open()

    assert any(idx == 0 for idx, _ in calls)
    assert not any(idx == 1 for idx, _ in calls)  # since alt index probe is commented out
    assert "camera" in backend.device_name().lower()


def test_open_raises_when_unable_to_open(monkeypatch, fake_capture_factory):
    monkeypatch.setattr(ob.platform, "system", lambda: "Linux")

    monkeypatch.setattr(ob, "list_cameras", lambda *_a, **_k: [])
    monkeypatch.setattr(ob, "select_camera", lambda *_a, **_k: None)

    def fake_videocapture(index, flag):
        return fake_capture_factory(opened=False)

    monkeypatch.setattr(ob.cv2, "VideoCapture", fake_videocapture)

    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    with pytest.raises(RuntimeError, match="Unable to open camera index"):
        backend.open()


def test_read_returns_none_on_grab_failure(fake_capture_factory):
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = fake_capture_factory(opened=True)
    cap.grab_ok = False
    backend._capture = cap

    frame, ts = backend.read()
    assert frame is None
    assert isinstance(ts, float)


def test_read_returns_none_on_retrieve_failure(fake_capture_factory):
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = fake_capture_factory(opened=True)
    cap.retrieve_ok = False
    backend._capture = cap

    frame, ts = backend.read()
    assert frame is None
    assert isinstance(ts, float)


def test_read_never_raises_on_exception(fake_capture_factory):
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = fake_capture_factory(opened=True)

    def boom():
        raise RuntimeError("transient")

    cap.grab = boom
    backend._capture = cap

    frame, ts = backend.read()
    assert frame is None
    assert isinstance(ts, float)


def test_configure_capture_sets_resolution_and_fps_non_faststart_windows(monkeypatch, fake_capture_factory):
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    cap = fake_capture_factory(opened=True, backend_name="DSHOW")
    cap.props[ob.cv2.CAP_PROP_FRAME_WIDTH] = 640.0
    cap.props[ob.cv2.CAP_PROP_FRAME_HEIGHT] = 480.0
    cap.props[ob.cv2.CAP_PROP_FPS] = 30.0

    settings = make_settings(index=0, fps=60.0, properties={"resolution": (800, 600)})
    backend = ob.OpenCVCameraBackend(settings)
    backend._capture = cap

    backend._configure_capture()

    assert backend.actual_resolution == (800, 600)
    assert backend.actual_fps is not None
    assert isinstance(backend.actual_fps, float)


def test_configure_capture_fast_start_does_not_force_resolution(monkeypatch, fake_capture_factory):
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    cap = fake_capture_factory(opened=True, backend_name="DSHOW")
    cap.props[ob.cv2.CAP_PROP_FRAME_WIDTH] = 1920.0
    cap.props[ob.cv2.CAP_PROP_FRAME_HEIGHT] = 1080.0

    settings = make_settings(
        index=0,
        fps=30.0,
        properties={"resolution": (1280, 720), "opencv": {"fast_start": True}},
    )
    backend = ob.OpenCVCameraBackend(settings)
    backend._capture = cap

    backend._configure_capture()

    # Fast-start still applies requested resolution, just without verification
    assert backend.actual_resolution == (1280, 720)

    # Requested intent must remain unchanged
    assert settings.properties["resolution"] == (1280, 720)


def test_configure_capture_applies_only_safe_numeric_properties(monkeypatch, fake_capture_factory):
    monkeypatch.setattr(ob.platform, "system", lambda: "Linux")

    cap = fake_capture_factory(opened=True)
    settings = make_settings(
        index=0,
        fps=30.0,
        properties={
            "resolution": (640, 480),
            "opencv": {"api": "ANY", "fast_start": False, "alt_index_probe": False},
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


def test_close_and_stop_release_capture(fake_capture_factory):
    backend = ob.OpenCVCameraBackend(make_settings(index=0, properties={}))
    cap = fake_capture_factory(opened=True)
    backend._capture = cap

    backend.close()
    assert backend._capture is None
    assert cap.released is True

    cap2 = fake_capture_factory(opened=True)
    backend._capture = cap2
    backend.stop()
    assert backend._capture is None
    assert cap2.released is True


def test_configure_capture_uses_width_height_when_no_legacy_resolution(monkeypatch, fake_capture_factory):
    monkeypatch.setattr(ob.platform, "system", lambda: "Windows")

    cap = fake_capture_factory(opened=True, backend_name="DSHOW")
    cap.props[ob.cv2.CAP_PROP_FRAME_WIDTH] = 640.0
    cap.props[ob.cv2.CAP_PROP_FRAME_HEIGHT] = 480.0
    cap.props[ob.cv2.CAP_PROP_FPS] = 30.0

    settings = make_settings(index=0, fps=30.0, properties={}, width=1024, height=768)
    backend = ob.OpenCVCameraBackend(settings)
    backend._capture = cap

    backend._configure_capture()

    assert backend.actual_resolution == (1024, 768)
