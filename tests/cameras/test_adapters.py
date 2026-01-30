# tests/cameras/test_adapters.py
import pytest

from dlclivegui.cameras.config_adapters import ensure_dc_camera
from dlclivegui.config import CameraSettings

# If available:
try:
    from dlclivegui.utils.config_models import CameraSettingsModel

    HAS_PYD = True
except Exception:
    HAS_PYD = False


@pytest.mark.unit
def test_ensure_dc_from_dataclass():
    dc = CameraSettings(name="TestCam", index=2, fps=0)
    out = ensure_dc_camera(dc)
    assert isinstance(out, CameraSettings)
    assert out is not dc  # must be deep-copied
    assert out.fps > 0  # apply_defaults triggers replacement of 0fps


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYD, reason="Pydantic models not installed yet")
def test_ensure_dc_from_pydantic():
    pm = CameraSettingsModel(name="PM", index=1, fps=15)
    out = ensure_dc_camera(pm)
    assert isinstance(out, CameraSettings)
    assert out.index == 1
    assert out.fps == 15.0


@pytest.mark.unit
def test_ensure_dc_from_dict():
    d = {"name": "DictCam", "index": 5, "fps": 60, "backend": "opencv"}
    out = ensure_dc_camera(d)
    assert isinstance(out, CameraSettings)
    assert out.index == 5
    assert out.backend == "opencv"
