# tests/cameras/test_fake_backend.py
import sys
import types

import numpy as np
import pytest

from dlclivegui.cameras import CameraFactory, base
from dlclivegui.config import CameraSettings


@pytest.mark.functional
def test_fake_backend_e2e():
    mod = types.ModuleType("fake_mod")

    class FakeBackend(base.CameraBackend):
        @classmethod
        def is_available(cls):
            return True

        def open(self):
            self._opened = True

        def read(self):
            assert self._opened
            img = np.zeros((10, 20, 3), dtype=np.uint8)
            return img, 123.456

        def close(self):
            self._opened = False

        def stop(self):
            pass

    mod.FakeBackend = FakeBackend
    sys.modules["fake_mod"] = mod
    base.register_backend_direct("fake2", FakeBackend)

    s = CameraSettings(backend="fake2", name="X")
    cam = CameraFactory.create(s)
    cam.open()
    frame, ts = cam.read()

    assert frame.shape == (10, 20, 3)
    assert ts == 123.456

    cam.close()
