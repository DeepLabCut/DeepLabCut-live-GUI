# tests/services/test_multicam_controller.py
import pytest

from dlclivegui.cameras.factory import CameraFactory

# from dlclivegui.config import CameraSettings
from dlclivegui.config import CameraSettings
from dlclivegui.services.multi_camera_controller import MultiCameraController, get_camera_id, get_display_id


@pytest.mark.unit
def test_start_and_frames(qtbot, patch_factory):
    mc = MultiCameraController()

    cam1 = CameraSettings(name="C1", backend="opencv", index=0, fps=25.0).apply_defaults()
    cam2 = {"name": "C2", "backend": "opencv", "index": 1, "fps": 30.0, "enabled": True}
    cam2 = CameraSettings.from_dict(cam2).apply_defaults()

    cam1_id = get_camera_id(cam1)
    cam2_id = get_camera_id(cam2)

    cam1_display = get_display_id(cam1)
    cam2_display = get_display_id(cam2)

    frames_seen = []

    def on_ready(mfd):
        frames_seen.append((mfd.source_camera_id, {k: v.shape for k, v in mfd.frames.items()}))

    mc.frame_ready.connect(on_ready)

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([cam1, cam2])

        qtbot.waitUntil(lambda: len(frames_seen) >= 1, timeout=2000)

        assert mc.is_running()

        # Internal IDs should be used as frame keys.
        seen_keys = set()
        seen_sources = set()
        for source_id, shape_map in frames_seen:
            seen_sources.add(source_id)
            seen_keys.update(shape_map.keys())

        assert seen_keys <= {cam1_id, cam2_id}
        assert seen_sources <= {cam1_id, cam2_id}

        # Display IDs should not be used as internal frame keys.
        assert cam1_display not in seen_keys
        assert cam2_display not in seen_keys

        assert any(len(shape_map) >= 1 for _, shape_map in frames_seen)

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)


@pytest.mark.unit
def test_rotation_and_crop(qtbot, patch_factory):
    mc = MultiCameraController()

    # 64x48 frame; rotate 90 => 48x64 then crop to 32x32 box
    cam = CameraSettings(
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
    cam = CameraSettings(name="C", backend="opencv", index=0, enabled=True).apply_defaults()

    # Expect initialization_failed with the camera id
    with qtbot.waitSignals([mc.initialization_failed, mc.all_stopped], timeout=2000) as _:
        mc.start([cam])


@pytest.mark.unit
def test_controller_uses_stable_camera_id_not_display_id(qtbot, patch_factory):
    mc = MultiCameraController()

    cam = CameraSettings(
        name="C1",
        backend="gentl",
        index=0,
        fps=30.0,
        enabled=True,
        properties={
            "gentl": {
                "device_id": "serial:SER0",
                "serial_number": "SER0",
            }
        },
    ).apply_defaults()

    stable_id = get_camera_id(cam)
    display_id = get_display_id(cam)

    assert stable_id == "gentl:serial:SER0"
    assert display_id == "gentl:0"
    assert stable_id != display_id

    seen = []

    def on_ready(mfd):
        seen.append(mfd)

    mc.frame_ready.connect(on_ready)

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([cam])

        qtbot.waitUntil(lambda: bool(seen), timeout=2000)

        mfd = seen[-1]

        assert mfd.source_camera_id == stable_id
        assert stable_id in mfd.frames
        assert stable_id in mfd.timestamps

        assert display_id not in mfd.frames
        assert display_id not in mfd.timestamps

        assert mfd.display_ids is not None
        assert mfd.display_ids[stable_id] == display_id

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)
