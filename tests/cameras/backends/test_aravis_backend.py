# tests/cameras/backends/test_aravis_backend.py
from unittest.mock import MagicMock

import numpy as np
import pytest

from dlclivegui.cameras.backends.aravis_backend import AravisCameraBackend


@pytest.fixture(autouse=True)
def _patch_aravis_module(monkeypatch):
    """
    Ensure the backend sees a working Aravis module during unit tests.
    This is required because tests bypass open() and still call read(),
    which references Aravis.BufferStatus and pixel format constants.
    """
    import dlclivegui.cameras.backends.aravis_backend as ar

    # Replace optional dependency with our in-file fake
    monkeypatch.setattr(ar, "Aravis", FakeAravis, raising=False)
    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", True, raising=False)
    yield


# -----------------------------------------------------------------------------
# Fake Aravis backend (module-level)
# -----------------------------------------------------------------------------
class FakeAravis:
    """Minimal fake Aravis module."""

    class BufferStatus:
        SUCCESS = "SUCCESS"
        ERROR = "ERROR"

    PIXEL_FORMAT_MONO_8 = "MONO8"
    PIXEL_FORMAT_MONO_12 = "MONO12"
    PIXEL_FORMAT_MONO_16 = "MONO16"
    PIXEL_FORMAT_RGB_8_PACKED = "RGB8"
    PIXEL_FORMAT_BGR_8_PACKED = "BGR8"

    class Auto:
        OFF = "OFF"

    devices = ["dev0"]

    @classmethod
    def update_device_list(cls):
        pass

    @classmethod
    def get_n_devices(cls) -> int:
        return len(cls.devices)

    @classmethod
    def get_device_id(cls, index: int) -> str:
        return cls.devices[index]

    class Camera:
        def __init__(self, device_id="dev0"):
            self.device_id = device_id
            self.pixel_format = None
            self._exposure = 0.0
            self._gain = 0.0
            self._fps = 0.0
            self.payload = 100
            self.stream = None  # should be a FakeStream

            # Device default "features"
            self._features_int = {
                "Width": 1920,
                "Height": 1080,
            }
            self._features_float = {
                "AcquisitionFrameRate": 30.0,
            }

        @classmethod
        def new(cls, device_id):
            return cls(device_id)

        # GenICam feature-style access used by backend
        def set_integer(self, name: str, value: int):
            self._features_int[name] = int(value)

        def get_integer(self, name: str) -> int:
            return int(self._features_int[name])

        def set_float(self, name: str, value: float):
            self._features_float[name] = float(value)

        def get_float(self, name: str) -> float:
            return float(self._features_float[name])

        # Pixel format
        def set_pixel_format(self, fmt):
            self.pixel_format = fmt

        def set_pixel_format_from_string(self, s):
            self.pixel_format = s

        # Exposure
        def set_exposure_time_auto(self, mode):
            pass

        def set_exposure_time(self, v):
            self._exposure = v

        def get_exposure_time(self):
            return self._exposure

        # Gain
        def set_gain_auto(self, mode):
            pass

        def set_gain(self, v):
            self._gain = v

        def get_gain(self):
            return self._gain

        # FPS (legacy methods still used by your backend)
        def set_frame_rate(self, v):
            self._fps = v
            self._features_float["AcquisitionFrameRate"] = float(v)

        def get_frame_rate(self):
            return self._fps

        # Metadata
        def get_model_name(self):
            return "FakeModel"

        def get_vendor_name(self):
            return "FakeVendor"

        def get_device_serial_number(self):
            return "12345"

        # Streaming
        def get_payload(self):
            return self.payload

        def create_stream(self, *_):
            return self.stream

        def start_acquisition(self):
            pass

        def stop_acquisition(self):
            pass

    class Buffer:
        def __init__(self, data, w, h, fmt, status="SUCCESS"):
            self._data = data
            self._w = w
            self._h = h
            self._fmt = fmt
            self._status = status

        @classmethod
        def new_allocate(cls, size):
            # Just provide a placeholder object for open() buffer queue
            return MagicMock()

        def get_status(self):
            return self._status

        def get_data(self):
            return self._data

        def get_image_width(self):
            return self._w

        def get_image_height(self):
            return self._h

        def get_image_pixel_format(self):
            return self._fmt


class FakeStream:
    def __init__(self, buffers):
        self._buffers = list(buffers)
        self.pushed = 0

    def timeout_pop_buffer(self, timeout):
        return self._buffers.pop(0) if self._buffers else None

    def try_pop_buffer(self):
        return self._buffers.pop(0) if self._buffers else None

    def push_buffer(self, buf):
        self.pushed += 1


# -----------------------------------------------------------------------------
# Helper to instantiate a backend with fake Aravis (bypasses open())
# -----------------------------------------------------------------------------


class Settings:
    """Mimic the settings object used by CameraBackend."""

    def __init__(
        self,
        properties=None,
        index=0,
        exposure=0,
        gain=0.0,
        fps=None,
        width=0,
        height=0,
        name="Test",
    ):
        self.properties = properties or {}
        self.index = index
        self.exposure = exposure
        self.gain = gain
        self.fps = fps
        self.width = width
        self.height = height
        self.name = name
        self.backend = "aravis"


def make_backend(settings, buffers):
    fake_camera = FakeAravis.Camera()
    stream = FakeStream(buffers)
    fake_camera.stream = stream

    backend = AravisCameraBackend(settings)
    # Shortcut the heavy open(): directly set fake camera/stream/label
    backend._camera = fake_camera
    backend._stream = stream
    backend._device_label = "FakeVendor FakeModel (12345)"
    return backend, fake_camera, stream


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.integration
def test_device_name():
    be, cam, s = make_backend(Settings(), [])
    assert be.device_name() == "FakeVendor FakeModel (12345)"


@pytest.mark.unit
@pytest.mark.integration
def test_read_mono8():
    w, h = 4, 3
    data = (np.arange(w * h) % 256).astype(np.uint8).tobytes()

    buf = FakeAravis.Buffer(data, w, h, FakeAravis.PIXEL_FORMAT_MONO_8)
    be, cam, s = make_backend(Settings(), [buf])

    frame, ts = be.read()
    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8
    # Ensure grayscale expanded to 3 channels
    assert np.all(frame[..., 0] == frame[..., 1])
    assert np.all(frame[..., 1] == frame[..., 2])
    # Buffer should be pushed back in finally
    assert s.pushed >= 1


@pytest.mark.unit
@pytest.mark.integration
def test_read_rgb8_converts_to_bgr():
    w, h = 2, 1
    # RGB: red=[255,0,0], green=[0,255,0]
    data = np.array([255, 0, 0, 0, 255, 0], dtype=np.uint8).tobytes()

    buf = FakeAravis.Buffer(data, w, h, FakeAravis.PIXEL_FORMAT_RGB_8_PACKED)
    be, cam, s = make_backend(Settings(), [buf])

    frame, _ = be.read()
    assert frame.shape == (1, 2, 3)
    # BGR conversion: red → [0,0,255], green → [0,255,0]
    assert (frame[0, 0] == np.array([0, 0, 255])).all()
    assert (frame[0, 1] == np.array([0, 255, 0])).all()
    assert s.pushed >= 1


@pytest.mark.unit
@pytest.mark.integration
def test_read_bgr8_passthrough():
    w, h = 2, 1
    data = np.array([10, 20, 30, 40, 50, 60], dtype=np.uint8).tobytes()

    buf = FakeAravis.Buffer(data, w, h, FakeAravis.PIXEL_FORMAT_BGR_8_PACKED)
    be, cam, s = make_backend(Settings(), [buf])

    frame, _ = be.read()
    assert frame.shape == (1, 2, 3)
    assert (frame.flatten() == np.array([10, 20, 30, 40, 50, 60])).all()
    assert s.pushed >= 1


@pytest.mark.unit
@pytest.mark.integration
def test_read_mono16_scaling():
    w, h = 3, 1
    raw = np.array([0, 32768, 65535], dtype=np.uint16)

    buf = FakeAravis.Buffer(raw.tobytes(), w, h, FakeAravis.PIXEL_FORMAT_MONO_16)
    be, cam, s = make_backend(Settings(), [buf])

    frame, _ = be.read()
    assert frame.shape == (1, 3, 3)

    # scaling: 0 → 0, max → 255, mid → ~128
    assert frame[0, 0, 0] == 0
    assert 120 <= frame[0, 1, 0] <= 135
    assert frame[0, 2, 0] == 255
    assert s.pushed >= 1


@pytest.mark.unit
@pytest.mark.integration
def test_read_unknown_format_fallback_to_mono8():
    w, h = 2, 2
    data = (np.arange(w * h) % 256).astype(np.uint8).tobytes()
    # Unknown token
    buf = FakeAravis.Buffer(data, w, h, "SOME_UNKNOWN_FMT")
    be, cam, s = make_backend(Settings(), [buf])

    frame, _ = be.read()
    assert frame.shape == (h, w, 3)
    assert np.all(frame[..., 0] == frame[..., 1])
    assert np.all(frame[..., 1] == frame[..., 2])
    assert s.pushed >= 1


@pytest.mark.unit
@pytest.mark.integration
def test_read_timeout_raises():
    be, cam, s = make_backend(Settings(), [])
    with pytest.raises(TimeoutError):
        be.read()


@pytest.mark.unit
@pytest.mark.integration
def test_read_status_error_raises_and_pushes_back():
    w, h = 1, 1
    data = b"\x00"
    buf = FakeAravis.Buffer(data, w, h, FakeAravis.PIXEL_FORMAT_MONO_8, status="ERROR")
    be, cam, s = make_backend(Settings(), [buf])

    with pytest.raises(TimeoutError):
        be.read()
    assert s.pushed >= 1


@pytest.mark.unit
@pytest.mark.integration
def test_close_is_idempotent():
    be, cam, s = make_backend(Settings(), [])
    be.close()
    be.close()  # should not raise


# -----------------------------------------------------------------------------
# Availability & device count surface tests (no SDK required)
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.integration
def test_is_available_false_when_aravis_missing(monkeypatch):
    import dlclivegui.cameras.backends.aravis_backend as ar

    # Simulate missing optional dependency
    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", False, raising=False)
    assert not ar.AravisCameraBackend.is_available()


@pytest.mark.unit
@pytest.mark.integration
def test_get_device_count_when_unavailable(monkeypatch):
    import dlclivegui.cameras.backends.aravis_backend as ar

    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", False, raising=False)
    assert ar.AravisCameraBackend.get_device_count() == -1


@pytest.mark.unit
@pytest.mark.integration
def test_get_device_count_when_available(monkeypatch):
    import dlclivegui.cameras.backends.aravis_backend as ar

    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ar, "Aravis", FakeAravis, raising=False)
    FakeAravis.devices = ["a", "b", "c"]
    assert ar.AravisCameraBackend.get_device_count() == 3


# -----------------------------------------------------------------------------
# Open path tests (with FakeAravis injected)
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.integration
def test_open_index_out_of_range(monkeypatch):
    # Patch Aravis module inside backend
    fake = FakeAravis
    monkeypatch.setattr("dlclivegui.cameras.backends.aravis_backend.Aravis", fake, raising=False)
    monkeypatch.setattr("dlclivegui.cameras.backends.aravis_backend.ARAVIS_AVAILABLE", True, raising=False)

    fake.devices = ["only_one"]
    with pytest.raises(RuntimeError):
        AravisCameraBackend(Settings(index=5)).open()


@pytest.mark.unit
@pytest.mark.integration
def test_open_success_pushes_initial_buffers_and_configures(monkeypatch):
    import dlclivegui.cameras.backends.aravis_backend as ar

    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ar, "Aravis", FakeAravis, raising=False)

    # Prepare a camera instance we can inspect and a stream to receive initial buffers
    cam = FakeAravis.Camera("dev0")
    stream = FakeStream([])
    cam.stream = stream

    # Ensure the `new` factory returns our prepared camera with stream
    def new_camera(device_id):
        return cam

    monkeypatch.setattr(FakeAravis.Camera, "new", staticmethod(new_camera))

    # Use a pixel_format and runtime settings to test configuration calls
    settings = Settings(
        properties={"aravis": {"pixel_format": "Mono8", "n_buffers": 4}},  # speed up test
        index=0,
        fps=15.0,
        exposure=1200.0,
        gain=5.5,
    )

    be = AravisCameraBackend(settings)
    be.open()

    # Stream should have received initial buffers
    assert stream.pushed == 4

    # Configurations should have been applied
    assert cam.pixel_format in (
        FakeAravis.PIXEL_FORMAT_MONO_8,  # via map
        "Mono8",  # or via from_string fallback
    )
    assert cam.get_frame_rate() == pytest.approx(15.0)
    assert cam.get_exposure_time() == pytest.approx(1200.0)
    assert cam.get_gain() == pytest.approx(5.5)

    # Device label should be resolved
    assert be.device_name().startswith("FakeVendor FakeModel")
    be.close()


@pytest.mark.unit
@pytest.mark.integration
def test_open_device_default_resolution_sets_actual_resolution(monkeypatch):
    import dlclivegui.cameras.backends.aravis_backend as ar

    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ar, "Aravis", FakeAravis, raising=False)

    cam = FakeAravis.Camera("dev0")
    cam.set_integer("Width", 1600)
    cam.set_integer("Height", 900)
    stream = FakeStream([])
    cam.stream = stream

    monkeypatch.setattr(FakeAravis.Camera, "new", staticmethod(lambda device_id: cam))

    settings = Settings(
        properties={"aravis": {"pixel_format": "Mono8", "n_buffers": 1}},
        index=0,
        width=0,
        height=0,
    )
    be = AravisCameraBackend(settings)
    be.open()

    assert be.actual_resolution == (1600, 900)
    be.close()


@pytest.mark.unit
@pytest.mark.integration
def test_open_requested_resolution_applies_and_reports_actual(monkeypatch):
    import dlclivegui.cameras.backends.aravis_backend as ar

    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ar, "Aravis", FakeAravis, raising=False)

    cam = FakeAravis.Camera("dev0")
    stream = FakeStream([])
    cam.stream = stream

    monkeypatch.setattr(FakeAravis.Camera, "new", staticmethod(lambda device_id: cam))

    settings = Settings(
        properties={"aravis": {"pixel_format": "Mono8", "n_buffers": 1}},
        index=0,
        width=640,
        height=480,
    )
    be = AravisCameraBackend(settings)
    be.open()

    assert cam.get_integer("Width") == 640
    assert cam.get_integer("Height") == 480
    assert be.actual_resolution == (640, 480)
    be.close()


@pytest.mark.unit
@pytest.mark.integration
def test_close_flushes_stream_and_clears_state(monkeypatch):
    import dlclivegui.cameras.backends.aravis_backend as ar

    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ar, "Aravis", FakeAravis, raising=False)

    cam = FakeAravis.Camera("dev0")
    # Preload some buffers that the close() loop should flush via try_pop_buffer
    stream = FakeStream([FakeAravis.Buffer(b"", 1, 1, FakeAravis.PIXEL_FORMAT_MONO_8) for _ in range(3)])
    cam.stream = stream

    def new_camera(device_id):
        return cam

    monkeypatch.setattr(FakeAravis.Camera, "new", staticmethod(new_camera))

    be = AravisCameraBackend(Settings(properties={"n_buffers": 1}))
    be.open()
    # Pretend runtime has placed some extra buffers in the stream
    stream._buffers.extend([FakeAravis.Buffer(b"", 1, 1, FakeAravis.PIXEL_FORMAT_MONO_8) for _ in range(2)])
    be.close()

    # State cleared
    assert be._camera is None
    assert be._stream is None
    # Stream emptied (no more buffers to pop)
    assert stream.try_pop_buffer() is None
