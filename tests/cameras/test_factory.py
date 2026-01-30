# tests/cameras/test_factory_basic.py
import sys
import types

import pytest

from dlclivegui.cameras import CameraFactory, DetectedCamera, base

# from dlclivegui.config import CameraSettings
from dlclivegui.utils.config_models import CameraSettingsModel


@pytest.mark.unit
def test_check_camera_available_quick_ping():
    mod = types.ModuleType("mock_mod")

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

    mod.MockBackend = MockBackend
    sys.modules["mock_mod"] = mod
    base.register_backend_direct("mock", MockBackend)

    ok, msg = CameraFactory.check_camera_available(CameraSettingsModel(backend="mock", index=0))
    assert ok is True

    ok, msg = CameraFactory.check_camera_available(CameraSettingsModel(backend="mock", index=3))
    assert ok is False


@pytest.mark.unit
def test_detect_cameras():
    mod = types.ModuleType("detect_mod")

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
            return None, 0

        def close(self):
            pass

        def stop(self):
            pass

    mod.DetectBackend = DetectBackend
    sys.modules["detect_mod"] = mod
    base.register_backend_direct("detect", DetectBackend)

    detected = CameraFactory.detect_cameras("detect", max_devices=4)
    assert isinstance(detected, list)
    assert [c.index for c in detected] == [0, 2]
    assert all(isinstance(c, DetectedCamera) for c in detected)
