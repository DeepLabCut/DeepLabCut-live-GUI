# tests/services/test_multicam_controller.py
import pytest

from dlclivegui.cameras.factory import CameraFactory
from dlclivegui.services.multi_camera_controller import MultiCameraController, get_camera_id

# from dlclivegui.config import CameraSettings
from dlclivegui.utils.config_models import CameraSettingsModel


@pytest.mark.unit
def test_start_and_frames(qtbot, patch_factory):
    mc = MultiCameraController()

    # One dataclass + one dict (simulate mixed inputs)
    cam1 = CameraSettingsModel(name="C1", backend="opencv", index=0, fps=25.0).apply_defaults()
    cam2 = {"name": "C2", "backend": "opencv", "index": 1, "fps": 30.0, "enabled": True}
    cam2 = CameraSettingsModel.from_dict(cam2).apply_defaults()

    frames_seen = []

    def on_ready(mfd):
        frames_seen.append((mfd.source_camera_id, {k: v.shape for k, v in mfd.frames.items()}))

    mc.frame_ready.connect(on_ready)

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([cam1, cam2])

        # Wait for at least one composite emission
        qtbot.waitUntil(lambda: len(frames_seen) >= 1, timeout=2000)

        assert mc.is_running()
        # We should have at least one entry with 1 or 2 frames (depending on timing)
        assert any(len(shape_map) >= 1 for _, shape_map in frames_seen)

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)


@pytest.mark.unit
def test_rotation_and_crop(qtbot, patch_factory):
    mc = MultiCameraController()

    # 64x48 frame; rotate 90 => 48x64 then crop to 32x32 box
    cam = CameraSettingsModel(
        name="C",
        backend="opencv",
        index=0,
        enabled=True,
        rotation=90,
        crop_x0=0,
        crop_y0=0,
        crop_x1=32,
        crop_y1=32,
    ).apply_defaults()

    last_shape = {"shape": None}

    def on_ready(mfd):
        f = mfd.frames.get(get_camera_id(cam))
        if f is not None:
            last_shape["shape"] = f.shape

    mc.frame_ready.connect(on_ready)

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([cam])

        # Wait until a rotated+cropped frame arrives
        qtbot.waitUntil(lambda: last_shape["shape"] is not None, timeout=2000)

        # Expect height=32, width=32, 3 channels
        assert last_shape["shape"] == (32, 32, 3)

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)


@pytest.mark.unit
def test_initialization_failure(qtbot, monkeypatch):
    # Make factory.create raise
    def _create(_settings):
        raise RuntimeError("no device")

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create))

    mc = MultiCameraController()
    cam = CameraSettingsModel(name="C", backend="opencv", index=0, enabled=True).apply_defaults()

    # Expect initialization_failed with the camera id
    with qtbot.waitSignals([mc.initialization_failed, mc.all_stopped], timeout=2000) as _:
        mc.start([cam])
