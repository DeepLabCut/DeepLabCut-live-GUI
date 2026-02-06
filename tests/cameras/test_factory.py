# tests/cameras/test_factory.py

import pytest

from dlclivegui.cameras import CameraFactory, DetectedCamera, base
from dlclivegui.config import CameraSettings


@pytest.fixture
def register_backend_clean():
    """
    Register a backend name for the duration of a test and clean it up afterwards.

    This prevents global registry leakage across tests.
    """
    created = []

    def _register(name: str, cls):
        base.register_backend_direct(name, cls)
        created.append(name)
        return cls

    yield _register

    for name in created:
        base._BACKEND_REGISTRY.pop(name, None)


# -----------------------------------------------------------------------------
# Rich discovery path
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_cameras_prefers_rich_discovery(register_backend_clean):
    class RichBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def discover_devices(cls, *, max_devices=10, should_cancel=None, progress_cb=None):
            # Note: factory should pass max_devices through.
            if progress_cb:
                progress_cb("rich discovery called")
            return [
                DetectedCamera(
                    index=1,
                    label="Cam A",
                    device_id="usb:1234:5678:CamA",
                    vid=0x1234,
                    pid=0x5678,
                    path="fake/path",
                    backend_hint=42,
                ),
                DetectedCamera(
                    index=0,
                    label="Cam B",
                    device_id="usb:aaaa:bbbb:CamB",
                ),
            ]

        # These should never be called if rich discovery returns a list.
        @staticmethod
        def quick_ping(i):
            raise AssertionError("Probing path should not run when rich discovery returns a list")

        def open(self):
            raise AssertionError("Probing path should not open when rich discovery returns a list")

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("rich", RichBackend)

    detected = CameraFactory.detect_cameras("rich", max_devices=99)
    assert [c.index for c in detected] == [0, 1]  # factory sorts by index
    assert detected[0].device_id == "usb:aaaa:bbbb:CamB"
    assert detected[1].vid == 0x1234
    assert detected[1].backend_hint == 42


@pytest.mark.unit
def test_detect_cameras_rich_discovery_receives_cancel_and_progress(register_backend_clean):
    calls = {"progress": [], "cancel_checked": 0}

    def progress_cb(msg: str):
        calls["progress"].append(msg)

    def should_cancel():
        calls["cancel_checked"] += 1
        return False

    class RichBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def discover_devices(cls, *, max_devices=10, should_cancel=None, progress_cb=None):
            assert max_devices == 5
            assert should_cancel is not None
            assert progress_cb is not None
            progress_cb("hello")
            _ = should_cancel()
            return [DetectedCamera(index=0, label="ok", device_id="id")]

        def open(self):
            pass

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("rich2", RichBackend)

    detected = CameraFactory.detect_cameras(
        "rich2",
        max_devices=5,
        should_cancel=should_cancel,
        progress_cb=progress_cb,
    )
    assert detected[0].device_id == "id"
    assert calls["progress"] == ["hello"]
    assert calls["cancel_checked"] == 1


@pytest.mark.unit
def test_detect_cameras_rich_discovery_none_falls_back_to_probing(register_backend_clean):
    class ProbeBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def discover_devices(cls, **kwargs):
            return None  # triggers fallback to probing

        @staticmethod
        def quick_ping(i):
            return i in (0, 2)

        def open(self):
            if self.settings.index not in (0, 2):
                raise RuntimeError("no device")

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("probe", ProbeBackend)

    detected = CameraFactory.detect_cameras("probe", max_devices=4)
    assert [c.index for c in detected] == [0, 2]
    assert all(isinstance(c, DetectedCamera) for c in detected)


@pytest.mark.unit
def test_detect_cameras_rich_discovery_error_falls_back_to_probing(register_backend_clean):
    class FlakyBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def discover_devices(cls, **kwargs):
            raise RuntimeError("boom")

        @staticmethod
        def quick_ping(i):
            return i == 1

        def open(self):
            if self.settings.index != 1:
                raise RuntimeError("no device")

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("flaky", FlakyBackend)

    detected = CameraFactory.detect_cameras("flaky", max_devices=3)
    assert [c.index for c in detected] == [1]


# -----------------------------------------------------------------------------
# Rebinding behavior (stable identity)
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_check_camera_available_applies_rebind_settings_before_quick_ping(register_backend_clean):
    class RebindBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def rebind_settings(cls, settings):
            # simulate stable-id rebind: index 9 should map to 0
            if settings.index == 9:
                settings.index = 0
            return settings

        @staticmethod
        def quick_ping(i):
            return i == 0

        def open(self):
            pass

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("rebind", RebindBackend)

    ok, msg = CameraFactory.check_camera_available(CameraSettings(backend="rebind", index=9))
    assert ok is True
    assert msg == ""


@pytest.mark.unit
def test_create_applies_rebind_settings(register_backend_clean):
    class RebindCreateBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def rebind_settings(cls, settings):
            settings.index = 7
            # optionally: store stable id in backend namespace
            if isinstance(settings.properties, dict):
                ns = settings.properties.setdefault("rebindcreate", {})
                ns["device_id"] = "usb:dead:why"
            return settings

        def open(self):
            pass

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("rebindcreate", RebindCreateBackend)

    cam = CameraSettings(backend="rebindcreate", index=0, properties={})
    backend = CameraFactory.create(cam)
    assert backend.settings.index == 7
    assert backend.settings.properties["rebindcreate"]["device_id"] == "usb:dead:why"


@pytest.mark.unit
def test_create_rebind_failure_is_non_fatal(register_backend_clean):
    class BadRebindBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def rebind_settings(cls, settings):
            raise RuntimeError("rebind broke")

        def open(self):
            pass

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("badrebind", BadRebindBackend)

    backend = CameraFactory.create(CameraSettings(backend="badrebind", index=0, properties={}))
    assert backend.settings.backend == "badrebind"


# -----------------------------------------------------------------------------
# Legacy / baseline behavior: quick_ping and probe path
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_check_camera_available_quick_ping(register_backend_clean):
    class MockBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @staticmethod
        def quick_ping(i):
            return i == 0

        def open(self):
            pass

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("mock", MockBackend)

    ok, _ = CameraFactory.check_camera_available(CameraSettings(backend="mock", index=0))
    assert ok is True

    ok, _ = CameraFactory.check_camera_available(CameraSettings(backend="mock", index=3))
    assert ok is False


@pytest.mark.unit
def test_detect_cameras_probe_path(register_backend_clean):
    class DetectBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        @staticmethod
        def quick_ping(i):
            return i in (0, 2)  # pretend devices 0 and 2 exist

        def open(self):
            if self.settings.index not in (0, 2):
                raise RuntimeError("no device")

        def read(self):
            return None, 0.0

        def close(self):
            pass

    register_backend_clean("detect", DetectBackend)

    detected = CameraFactory.detect_cameras("detect", max_devices=4)
    assert isinstance(detected, list)
    assert [c.index for c in detected] == [0, 2]
    assert all(isinstance(c, DetectedCamera) for c in detected)
