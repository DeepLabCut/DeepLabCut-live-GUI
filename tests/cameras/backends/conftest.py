# tests/cameras/backends/conftest.py
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


# -----------------------------
# Dependency detection helpers
# -----------------------------
def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _has_aravis_gi() -> bool:
    """
    GI can exist without the Aravis typelib. Be representative:
    check that gi.repository.Aravis is importable and versionable.
    """
    try:
        import gi  # type: ignore

        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis  # noqa: F401

        return True
    except Exception:
        return False


ARAVIS_AVAILABLE = _has_aravis_gi()
PYPYLON_AVAILABLE = _has_module("pypylon")
HARVESTERS_AVAILABLE = _has_module("harvesters")


# -----------------------------
# Pytest configuration
# -----------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require hardware/SDKs (aravis/pypylon/gentl). "
        "By default these are skipped. You can also set BACKENDS_RUN_HARDWARE=1.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Document custom markers
    config.addinivalue_line("markers", "hardware: tests that touch real devices or SDKs")
    config.addinivalue_line("markers", "aravis: tests for Aravis backend")
    config.addinivalue_line("markers", "pypylon: tests for Basler/pypylon backend")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Auto-skip tests if the corresponding dependency is not present,
    and only run hardware-marked tests when explicitly requested.
    """
    run_hardware_flag = bool(config.getoption("--run-hardware"))
    run_hardware_env = os.getenv("BACKENDS_RUN_HARDWARE", "").strip() in {"1", "true", "yes"}
    run_hardware = run_hardware_flag or run_hardware_env

    skip_no_aravis = pytest.mark.skip(reason="Aravis/gi is not available")
    skip_no_pypylon = pytest.mark.skip(reason="Basler pypylon is not available")
    skip_hardware = pytest.mark.skip(
        reason="Hardware/SDK tests disabled. Use --run-hardware or set BACKENDS_RUN_HARDWARE=1"
    )

    for item in items:
        # Per-backend availability skips
        if "aravis" in item.keywords and not ARAVIS_AVAILABLE:
            item.add_marker(skip_no_aravis)
        if "pypylon" in item.keywords and not PYPYLON_AVAILABLE:
            item.add_marker(skip_no_pypylon)

        # Global hardware gate (only applies to tests marked 'hardware')
        if "hardware" in item.keywords and not run_hardware:
            item.add_marker(skip_hardware)


# -----------------------------
# Useful fixtures for backends
# -----------------------------
@pytest.fixture
def reset_backend_registry():
    """
    Ensure backend registry is clean for tests that rely on registration behavior.
    Automatically imports the package module that registers backends.
    """
    from dlclivegui.cameras.base import reset_backends

    reset_backends()
    try:
        # Import once so decorators run and register built-ins where possible.
        import dlclivegui.cameras.backends  # noqa: F401
    except Exception:
        # If import fails (optional deps), tests can still register backends directly.
        pass
    yield
    reset_backends()  # cleanup


@pytest.fixture
def force_aravis_unavailable(monkeypatch):
    """
    Force the Aravis backend to behave as if Aravis is not installed.
    Useful for testing error paths without modifying the environment.
    """
    import dlclivegui.cameras.backends.aravis_backend as ar

    # Simulate missing optional dependency
    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", False, raising=False)
    # Make sure the module symbol itself is treated as absent
    monkeypatch.setattr(ar, "Aravis", None, raising=False)
    yield


@pytest.fixture
def force_pypylon_unavailable(monkeypatch):
    """
    Force Basler/pypylon to be unavailable for error-path testing.
    Basler backend availability is based on 'pylon is not None'.
    """
    try:
        import dlclivegui.cameras.backends.basler_backend as bas
    except Exception:
        yield
        return

    monkeypatch.setattr(bas, "pylon", None, raising=False)
    yield


# -----------------------------------------------------------------------------
# Fake Aravis SDK (module-like) + fixtures
# -----------------------------------------------------------------------------
class FakeAravis:
    """Minimal fake Aravis module used for SDK-less unit/contract tests."""

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

    # Mutable "device list" that tests can override
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

    # Optional metadata used by snapshot/rebind logic (safe defaults)
    @classmethod
    def get_device_physical_id(cls, index: int) -> str:
        return f"PHYS-{cls.devices[index]}"

    @classmethod
    def get_device_vendor(cls, index: int) -> str:
        return "FakeVendor"

    @classmethod
    def get_device_model(cls, index: int) -> str:
        return "FakeModel"

    @classmethod
    def get_device_serial_nbr(cls, index: int) -> str:
        return "12345"

    @classmethod
    def get_device_protocol(cls, index: int) -> str:
        return "FakeProtocol"

    @classmethod
    def get_device_address(cls, index: int) -> str:
        return f"ADDR-{index}"

    class Camera:
        def __init__(self, device_id="dev0"):
            self.device_id = device_id
            self.pixel_format = None
            self._exposure = 0.0
            self._gain = 0.0
            self._fps = 0.0
            self.payload = 100
            self.stream = None  # should be a FakeAravisStream

            self._features_int = {"Width": 1920, "Height": 1080}
            self._features_float = {"AcquisitionFrameRate": 30.0}

        @classmethod
        def new(cls, device_id):
            return cls(device_id)

        # GenICam-like int/float access
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
            self._exposure = float(v)

        def get_exposure_time(self):
            return float(self._exposure)

        # Gain
        def set_gain_auto(self, mode):
            pass

        def set_gain(self, v):
            self._gain = float(v)

        def get_gain(self):
            return float(self._gain)

        # FPS
        def set_frame_rate(self, v):
            self._fps = float(v)
            self._features_float["AcquisitionFrameRate"] = float(v)

        def get_frame_rate(self):
            return float(self._fps)

        # Metadata
        def get_model_name(self):
            return "FakeModel"

        def get_vendor_name(self):
            return "FakeVendor"

        def get_device_serial_number(self):
            return "12345"

        # Streaming
        def get_payload(self):
            return int(self.payload)

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
            # Placeholder buffer object for open() buffer queue
            return object()

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


class FakeAravisStream:
    def __init__(self, buffers):
        self._buffers = list(buffers)
        self.pushed = 0

    def timeout_pop_buffer(self, timeout):
        return self._buffers.pop(0) if self._buffers else None

    def try_pop_buffer(self):
        return self._buffers.pop(0) if self._buffers else None

    def push_buffer(self, buf):
        self.pushed += 1


@pytest.fixture()
def fake_aravis_module():
    """
    Returns the FakeAravis 'module' and resets its mutable state for isolation.
    Tests may mutate FakeAravis.devices safely.
    """
    FakeAravis.devices = ["dev0"]
    return FakeAravis


@pytest.fixture()
def patch_aravis_sdk(monkeypatch, fake_aravis_module):
    """
    Patch the Aravis backend module so it behaves as if the SDK is installed,
    but uses our FakeAravis implementation.

    Usage:
        def test_something(patch_aravis_sdk):
            ...  # aravis backend sees ARAVIS_AVAILABLE=True and Aravis=FakeAravis
    """
    import dlclivegui.cameras.backends.aravis_backend as ar

    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ar, "Aravis", fake_aravis_module, raising=False)
    return fake_aravis_module


@pytest.fixture()
def fake_aravis_stream():
    """
    Small helper fixture to create a FakeAravisStream with a list of buffers.
    """

    def _make(buffers):
        return FakeAravisStream(buffers)

    return _make


# -----------------------------------------------------------------------------
# Fake Basler / pypylon SDK (module-like) + fixtures
# -----------------------------------------------------------------------------


class FakePylon:
    """Minimal fake for 'from pypylon import pylon' usage in basler_backend."""

    # Constants used by Basler backend
    GrabStrategy_LatestImageOnly = 1
    TimeoutHandling_ThrowException = 1
    PixelType_BGR8packed = 0x02180014  # arbitrary token
    OutputBitAlignment_MsbAligned = 1

    class _Feature:
        def __init__(self, value=0):
            self._value = value

        def SetValue(self, v):
            self._value = v

        def GetValue(self):
            return self._value

    class _DeviceInfo:
        def __init__(self, serial: str):
            self._serial = serial

        def GetSerialNumber(self):
            return self._serial

    class _Device:
        def __init__(self, info):
            self.info = info

    class TlFactory:
        _instance = None

        def __init__(self):
            self._devices = [FakePylon._DeviceInfo("FAKE-BASLER-0")]

        @classmethod
        def GetInstance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        def EnumerateDevices(self):
            return list(self._devices)

        def CreateDevice(self, device_info):
            return FakePylon._Device(device_info)

    class _GrabResult:
        def __init__(self, ok=True, array=None):
            self._ok = ok
            self._array = array

        def GrabSucceeded(self):
            return bool(self._ok)

        def Release(self):
            return None

    class InstantCamera:
        def __init__(self, device):
            self._device = device
            self._open = False
            self._grabbing = False

            # Feature nodes the backend uses
            self.ExposureTime = FakePylon._Feature(1000.0)
            self.Gain = FakePylon._Feature(0.0)
            self.Width = FakePylon._Feature(1920)
            self.Height = FakePylon._Feature(1080)

            self.AcquisitionFrameRateEnable = FakePylon._Feature(False)
            self.AcquisitionFrameRate = FakePylon._Feature(30.0)

        def Open(self):
            self._open = True

        def Close(self):
            self._open = False

        def IsOpen(self):
            return bool(self._open)

        def StartGrabbing(self, *_args, **_kwargs):
            self._grabbing = True

        def StopGrabbing(self):
            self._grabbing = False

        def IsGrabbing(self):
            return bool(self._grabbing)

        def RetrieveResult(self, *_args, **_kwargs):
            # Always succeed with a small dummy image (BGR)
            import numpy as np

            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            return FakePylon._GrabResult(ok=True, array=frame)

    class _ConvertedImage:
        def __init__(self, array):
            self._array = array

        def GetArray(self):
            return self._array

    class ImageFormatConverter:
        def __init__(self):
            self.OutputPixelFormat = None
            self.OutputBitAlignment = None

        def Convert(self, grab_result):
            return FakePylon._ConvertedImage(grab_result._array)


@pytest.fixture()
def fake_pylon_module():
    """
    Returns the FakePylon 'module' and resets singleton devices for isolation.
    """
    # reset singleton factory so devices list resets per test
    FakePylon.TlFactory._instance = None
    return FakePylon


@pytest.fixture()
def patch_basler_sdk(monkeypatch, fake_pylon_module):
    """
    Patch Basler backend to behave as if pypylon is installed, using FakePylon.
    """
    import dlclivegui.cameras.backends.basler_backend as bb

    monkeypatch.setattr(bb, "pylon", fake_pylon_module, raising=False)
    return fake_pylon_module


# -----------------------------------------------------------------------------
# Fake GenTL / harvesters SDK (SDK-free) + fixtures for strict lifecycle tests
# -----------------------------------------------------------------------------


class FakeGenTLTimeoutException(TimeoutError):
    """Fake timeout/error type used as HarvesterTimeoutError in backend tests."""

    pass


def _info_get(info: Any, key: str, default=None):
    """Read a device-info field from dict-like or attribute-like entries."""
    try:
        if hasattr(info, "get"):
            v = info.get(key)
            if v is not None:
                return v
    except Exception:
        pass
    try:
        v = getattr(info, key, None)
        if v is not None:
            return v
    except Exception:
        pass
    return default


class _FakeNode:
    """Minimal GenICam node: .value plus optional constraints and symbolics."""

    def __init__(self, value=None, *, min=None, max=None, inc=1, symbolics=None):
        self.value = value
        self.min = min
        self.max = max
        self.inc = inc
        self.symbolics = symbolics or []


class _FakeNodeMap:
    """Node map with the attributes your GenTLCameraBackend touches."""

    def __init__(
        self,
        *,
        width=1920,
        height=1080,
        fps=30.0,
        exposure=10000.0,
        gain=0.0,
        pixel_format="Mono8",
        model="FakeGenTLModel",
        serial="FAKE-GENTL-0",
        display="FakeGenTLDisplay",
    ):
        # Label fields used by _resolve_device_label()
        self.DeviceModelName = _FakeNode(model)
        self.DeviceSerialNumber = _FakeNode(serial)
        self.DeviceDisplayName = _FakeNode(display)

        # Pixel format node
        self.PixelFormat = _FakeNode(
            pixel_format,
            symbolics=["Mono8", "Mono16", "RGB8", "BGR8"],
        )

        # Width/Height constraints for increment alignment logic
        self.Width = _FakeNode(int(width), min=64, max=4096, inc=2)
        self.Height = _FakeNode(int(height), min=64, max=4096, inc=2)

        # FPS / actual fps
        self.AcquisitionFrameRateEnable = _FakeNode(True)
        self.AcquisitionFrameRate = _FakeNode(float(fps))
        self.ResultingFrameRate = _FakeNode(float(fps))

        # Exposure / gain
        self.ExposureAuto = _FakeNode("Off")
        self.ExposureTime = _FakeNode(float(exposure))
        self.GainAuto = _FakeNode("Off")
        self.Gain = _FakeNode(float(gain))


class _FakeRemoteDevice:
    def __init__(self, node_map: _FakeNodeMap):
        self.node_map = node_map


class _FakeComponent:
    """
    Component with .data, .width, .height like Harvesters component2D image.
    Your backend does np.asarray(component.data) and reshape using height/width.
    """

    def __init__(self, width: int, height: int, channels: int, dtype=np.uint8):
        self.width = int(width)
        self.height = int(height)
        self._channels = int(channels)

        n = self.width * self.height * self._channels
        if dtype == np.uint8:
            arr = (np.arange(n) % 255).astype(np.uint8)
        else:
            arr = (np.arange(n) % 65535).astype(np.uint16)
        self.data = arr


class _FakePayload:
    def __init__(self, component: _FakeComponent):
        self.components = [component]


class _FakeFetchedBufferCtx:
    """Context manager returned by fetch(). Must have .payload."""

    def __init__(self, payload: _FakePayload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class FakeImageAcquirer:
    """
    Minimal ImageAcquirer:
      - remote_device.node_map
      - node_map shortcut (backend uses self._acquirer.node_map in read())
      - start/stop/destroy
      - fetch(timeout=...) -> ctx manager yielding buffer-like object
    Strict rule: fetch fails unless started=True.
    """

    serial: str = "FAKE-GENTL-0"
    width: int = 1920
    height: int = 1080
    pixel_format: str = "Mono8"

    def __post_init__(self):
        self.remote_device = _FakeRemoteDevice(
            _FakeNodeMap(width=self.width, height=self.height, pixel_format=self.pixel_format, serial=self.serial)
        )
        self.node_map = self.remote_device.node_map

        self._started = False
        self._destroyed = False
        self._queue: list[_FakePayload] = []

        # Call tracing
        self.start_calls = 0
        self.stop_calls = 0
        self.destroy_calls = 0
        self.fetch_calls: list[float] = []

        # Prepare one default frame
        self._enqueue_default_frame()

    def _enqueue_default_frame(self):
        pf = str(self.node_map.PixelFormat.value or "Mono8")
        if pf in ("RGB8", "BGR8"):
            channels, dtype = 3, np.uint8
        elif pf == "Mono16":
            channels, dtype = 1, np.uint16
        else:
            channels, dtype = 1, np.uint8

        comp = _FakeComponent(self.node_map.Width.value, self.node_map.Height.value, channels, dtype=dtype)
        self._queue.append(_FakePayload(comp))

    def start(self):
        self.start_calls += 1
        self._started = True

    def stop(self):
        self.stop_calls += 1
        self._started = False

    def destroy(self):
        self.destroy_calls += 1
        self._destroyed = True

    def fetch(self, timeout: float = 2.0):
        self.fetch_calls.append(float(timeout))

        # Strict rule: cannot fetch unless started
        if not self._started:
            raise FakeGenTLTimeoutException("fetch called while not started")

        if not self._queue:
            raise FakeGenTLTimeoutException(f"timeout after {timeout}s")

        payload = self._queue.pop(0)
        return _FakeFetchedBufferCtx(payload)


class FakeHarvester:
    """
    Minimal Harvester:
      - add_file/update/reset
      - device_info_list
      - create(index) or create({"serial_number": ...})
    Inventory-driven so tests can control enumeration.
    """

    def __init__(self, inventory: list[dict[str, Any]] | None = None):
        self._files: list[str] = []
        self._inventory: list[dict[str, Any]] = list(inventory or [])
        self.device_info_list: list[Any] = []

        # Call tracing
        self.add_file_calls: list[str] = []
        self.update_calls = 0
        self.reset_calls = 0
        self.create_calls: list[Any] = []

    def add_file(self, file_path: str):
        self._files.append(str(file_path))
        self.add_file_calls.append(str(file_path))

    def update(self):
        self.update_calls += 1
        # If not provided, default to a single fake device
        if not self._inventory:
            self._inventory = [
                {
                    "display_name": "TLSimuMono (FAKE-GENTL-0)",
                    "model": "FakeGenTLModel",
                    "vendor": "FakeVendor",
                    "serial_number": "FAKE-GENTL-0",
                    "id_": "FakeDeviceId",
                    "tl_type": "Custom",
                    "user_defined_name": "Center",
                    "version": "1.0.0",
                    "access_status": 1000,
                }
            ]
        self.device_info_list = list(self._inventory)

    def reset(self):
        self.reset_calls += 1
        self.device_info_list = []
        self._files = []

    def create(self, selector=None, index: int | None = None, *args, **kwargs):
        # Record call for verification
        self.create_calls.append({"selector": selector, "index": index, "args": args, "kwargs": kwargs})

        if not self.device_info_list:
            self.update()

        serial = None
        if isinstance(selector, dict):
            serial = selector.get("serial_number")

        if serial is None:
            if index is None:
                # allow create(0) style
                if isinstance(selector, int):
                    index = selector
                else:
                    index = 0
            if index < 0 or index >= len(self.device_info_list):
                raise RuntimeError("Index out of range")
            info = self.device_info_list[index]
            serial = str(_info_get(info, "serial_number", "FAKE-GENTL-0"))

        return FakeImageAcquirer(serial=str(serial))

    # Keep compatibility if anything uses the older name
    def create_image_acquirer(self, *args, **kwargs):
        return self.create(*args, **kwargs)


# -----------------------------------------------------------------------------
# GentL fixtures: inventory, patching, settings factory
# -----------------------------------------------------------------------------


@pytest.fixture()
def gentl_inventory():
    """
    Mutable inventory list used by FakeHarvester.update().
    Tests can replace contents to simulate multiple devices, ambiguity, missing fields, etc.
    """
    inv: list[dict[str, Any]] = [
        {
            "display_name": "TLSimuMono (FAKE-GENTL-0)",
            "model": "FakeGenTLModel",
            "vendor": "FakeVendor",
            "serial_number": "FAKE-GENTL-0",
            "id_": "FakeDeviceId",
            "tl_type": "Custom",
            "user_defined_name": "Center",
            "version": "1.0.0",
            "access_status": 1000,
        }
    ]
    return inv


@pytest.fixture()
def fake_harvester_factory(gentl_inventory):
    """
    Factory that returns a FakeHarvester bound to the current gentl_inventory.
    Allows tests to mutate gentl_inventory before calling backend.open().
    """

    def _make():
        return FakeHarvester(inventory=gentl_inventory)

    return _make


@pytest.fixture()
def patch_gentl_sdk(monkeypatch, fake_harvester_factory, tmp_path):
    """
    Patch dlclivegui.cameras.backends.gentl_backend to use FakeHarvester + Fake timeout.
    Also ensure CTI discovery succeeds for classmethods (discover_devices/quick_ping)
    by creating a real dummy .cti and exposing it via GENICAM_GENTL64_PATH.
    """
    import dlclivegui.cameras.backends.gentl_backend as gb

    # Patch Harvester symbol (the backend calls Harvester() directly)
    monkeypatch.setattr(gb, "Harvester", lambda: fake_harvester_factory(), raising=False)

    # Keep your backend timeout contract as-is: it catches HarvesterTimeoutError
    monkeypatch.setattr(gb, "HarvesterTimeoutError", FakeGenTLTimeoutException, raising=False)

    # Create a real CTI file and advertise it via env var (cross-platform via os.pathsep)
    cti_file = tmp_path / "dummy.cti"
    if not cti_file.exists():
        cti_file.write_text("fake", encoding="utf-8")

    monkeypatch.setenv("GENICAM_GENTL64_PATH", str(tmp_path))
    monkeypatch.delenv("GENICAM_GENTL32_PATH", raising=False)

    return gb


@pytest.fixture()
def gentl_settings_factory(tmp_path):
    """
    Convenience factory for CameraSettings for gentl backend tests.
    """
    from dlclivegui.config import CameraSettings

    def _make(
        *,
        index=0,
        name="TestCam",
        width=0,
        height=0,
        fps=0.0,
        exposure=0,
        gain=0.0,
        enabled=True,
        properties=None,
    ):
        cti = tmp_path / "dummy.cti"
        if not cti.exists():
            cti.write_text("fake", encoding="utf-8")
        props = properties if isinstance(properties, dict) else {}
        props.setdefault("gentl", {})
        props["gentl"] = dict(props["gentl"])
        props["gentl"].setdefault("cti_file", str(cti))
        return CameraSettings(
            name=name,
            index=index,
            backend="gentl",
            width=width,
            height=height,
            fps=fps,
            exposure=exposure,
            gain=gain,
            enabled=enabled,
            properties=props,
        )

    return _make


# -----------------------------------------------------------------------------
# Generic patcher mapping fixture for test_generic_contracts.py
# -----------------------------------------------------------------------------
@pytest.fixture()
def backend_sdk_patchers(patch_aravis_sdk, patch_basler_sdk, patch_gentl_sdk):
    """
    Mapping from backend name -> patcher callable (best-effort SDK stubs).

    This fixture intentionally reuses existing per-backend patch fixtures
    to avoid duplication. Patch side effects occur when this fixture is
    requested (because patch_aravis_sdk is injected).
    """
    return {
        # Calling it is harmless; patching already applied by fixture injection.
        "aravis": (lambda: patch_aravis_sdk),
        "basler": (lambda: patch_basler_sdk),
        "gentl": (lambda: patch_gentl_sdk),
        # No patch needed: OpenCV is assumed present
        # "opencv": None,
    }
