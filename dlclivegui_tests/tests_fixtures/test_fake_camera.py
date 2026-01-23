import numpy as np
import pytest

from dlclivegui_tests.fixtures.fake_cameras_backend import StubCameraBackend, StubCameraBehavior


@pytest.mark.integration
def test_stub_camera_backend_basic():
    cam = StubCameraBackend(camera_id="fake:0", shape=(120, 160, 3))
    cam.open()
    frame, ts = cam.read()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
    assert isinstance(ts, float)
    cam.close()


@pytest.mark.integration
def test_stub_camera_backend_empty_frames():
    cam = StubCameraBackend(camera_id="fake:0", behavior=StubCameraBehavior(empty_first_n=2))
    cam.open()
    f1, _ = cam.read()
    f2, _ = cam.read()
    f3, _ = cam.read()
    assert f1.size == 0
    assert f2.size == 0
    assert f3.size > 0
    cam.close()


@pytest.mark.integration
def test_stub_camera_backend_raises():
    cam = StubCameraBackend(camera_id="fake:0", behavior=StubCameraBehavior(raise_first_n=1))
    cam.open()
    try:
        cam.read()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass
    cam.close()
