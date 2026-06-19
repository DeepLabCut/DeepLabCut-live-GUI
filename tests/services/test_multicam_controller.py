import pytest

from dlclivegui.cameras.factory import CameraFactory
from dlclivegui.config import CameraSettings
from dlclivegui.services.multi_camera_controller import (
    MultiCameraController,
    _camera_start_priority,
    _trigger_role_from_settings,
    get_camera_id,
    get_display_id,
)


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

        # Internal stable IDs should be used as frame keys and source IDs.
        seen_keys = set()
        seen_sources = set()
        for source_id, shape_map in frames_seen:
            seen_sources.add(source_id)
            seen_keys.update(shape_map.keys())

        assert seen_keys <= {cam1_id, cam2_id}
        assert seen_sources <= {cam1_id, cam2_id}

        # Human/display IDs should not be used as internal frame keys.
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

        qtbot.waitUntil(lambda: last_shape["shape"] is not None, timeout=2000)

        assert last_shape["shape"] == (32, 32, 3)

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)


@pytest.mark.unit
def test_initialization_failure(qtbot, monkeypatch):
    def _create(_settings):
        raise RuntimeError("no device")

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create))

    mc = MultiCameraController()
    cam = CameraSettings(name="C", backend="opencv", index=0, enabled=True).apply_defaults()

    with qtbot.waitSignals([mc.initialization_failed, mc.all_stopped], timeout=2000):
        mc.start([cam])


@pytest.mark.unit
def test_get_camera_id_prefers_stable_device_id():
    cam = CameraSettings(
        name="GenTL Cam",
        backend="gentl",
        index=0,
        properties={
            "gentl": {
                "device_id": "serial:30220469",
                "serial_number": "30220469",
            }
        },
    ).apply_defaults()

    assert get_camera_id(cam) == "gentl:serial:30220469"


@pytest.mark.unit
def test_get_camera_id_falls_back_to_index_without_stable_identity():
    cam = CameraSettings(
        name="Cam",
        backend="opencv",
        index=2,
    ).apply_defaults()

    assert get_camera_id(cam) == "opencv:index:2"


@pytest.mark.unit
def test_get_display_id_is_human_index_label():
    cam = CameraSettings(
        name="GenTL Cam",
        backend="gentl",
        index=3,
        properties={
            "gentl": {
                "device_id": "serial:30220469",
                "serial_number": "30220469",
            }
        },
    ).apply_defaults()

    assert get_camera_id(cam) == "gentl:serial:30220469"
    assert get_display_id(cam) == "gentl:3"
    assert get_camera_id(cam) != get_display_id(cam)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("off", "off"),
        ("disabled", "off"),
        ("on", "external"),
        ("triggered", "external"),
        ("external", "external"),
        ("follower", "follower"),
        ("slave", "follower"),
        ("master", "master"),
        ("main", "master"),
    ],
)
def test_trigger_role_from_settings_aliases(role, expected):
    cam = CameraSettings(
        name="C",
        backend="gentl",
        index=0,
        properties={
            "gentl": {
                "trigger": {
                    "role": role,
                }
            }
        },
    ).apply_defaults()

    assert _trigger_role_from_settings(cam) == expected


@pytest.mark.unit
def test_camera_start_priority_orders_trigger_roles():
    external = CameraSettings(
        name="External",
        backend="gentl",
        index=0,
        properties={"gentl": {"trigger": {"role": "external"}}},
    ).apply_defaults()

    normal = CameraSettings(
        name="Normal",
        backend="gentl",
        index=1,
        properties={"gentl": {"trigger": {"role": "off"}}},
    ).apply_defaults()

    master = CameraSettings(
        name="Master",
        backend="gentl",
        index=2,
        properties={"gentl": {"trigger": {"role": "master"}}},
    ).apply_defaults()

    assert _camera_start_priority(external) == 0
    assert _camera_start_priority(normal) == 1
    assert _camera_start_priority(master) == 2


@pytest.mark.unit
def test_start_preserves_user_display_order_even_when_trigger_start_order_differs(qtbot, patch_factory):
    mc = MultiCameraController()

    # User wants master first in display/tile order, follower second.
    # Startup order should still be follower first internally.
    master = CameraSettings(
        name="Master",
        backend="opencv",
        index=0,
        enabled=True,
        properties={
            "opencv": {
                "device_id": "master-cam",
                "trigger": {"role": "master"},
            }
        },
    ).apply_defaults()

    follower = CameraSettings(
        name="Follower",
        backend="opencv",
        index=1,
        enabled=True,
        properties={
            "opencv": {
                "device_id": "follower-cam",
                "trigger": {"role": "follower"},
            }
        },
    ).apply_defaults()

    expected_display_order = [get_camera_id(master), get_camera_id(follower)]

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([master, follower])

        # Display order follows user order, but stores stable IDs.
        assert mc._camera_display_order == expected_display_order

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)


@pytest.mark.unit
def test_frame_ready_emits_frames_in_user_configured_order(qtbot, patch_factory):
    mc = MultiCameraController()

    cam_a = CameraSettings(
        name="A",
        backend="opencv",
        index=0,
        enabled=True,
        properties={"opencv": {"device_id": "cam-a"}},
    ).apply_defaults()

    cam_b = CameraSettings(
        name="B",
        backend="opencv",
        index=1,
        enabled=True,
        properties={"opencv": {"device_id": "cam-b"}},
    ).apply_defaults()

    expected_order = [get_camera_id(cam_a), get_camera_id(cam_b)]
    seen_orders: list[list[str]] = []

    def on_ready(mfd):
        if len(mfd.frames) >= 2:
            seen_orders.append(list(mfd.frames.keys()))

    mc.frame_ready.connect(on_ready)

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([cam_a, cam_b])

        qtbot.waitUntil(lambda: bool(seen_orders), timeout=2500)

        assert seen_orders[-1] == expected_order

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)


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


@pytest.mark.unit
def test_display_order_is_cleared_on_stop(qtbot, patch_factory):
    mc = MultiCameraController()

    cam = CameraSettings(
        name="C",
        backend="opencv",
        index=0,
        enabled=True,
        properties={"opencv": {"device_id": "cam-0"}},
    ).apply_defaults()

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([cam])

        assert mc._camera_display_order == [get_camera_id(cam)]

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)

    assert mc._camera_display_order == []


@pytest.mark.unit
def test_hardware_trigger_timeouts_are_not_fatal(qtbot, monkeypatch):
    class WaitingTriggerBackend:
        waits_for_hardware_trigger = True

        def __init__(self, settings):
            self.settings = settings
            self.opened = False
            self.closed = False

        def open(self):
            self.opened = True

        def read(self):
            raise TimeoutError("waiting for hardware trigger")

        def close(self):
            self.closed = True

    def _create(settings):
        return WaitingTriggerBackend(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create))

    mc = MultiCameraController()
    cam = CameraSettings(
        name="Triggered",
        backend="gentl",
        index=0,
        enabled=True,
        properties={
            "gentl": {
                "device_id": "serial:30220469",
                "trigger": {"role": "external", "timeout": 0.1},
            }
        },
    ).apply_defaults()

    errors: list[tuple[str, str]] = []
    mc.camera_error.connect(lambda cam_id, msg: errors.append((cam_id, msg)))

    try:
        with qtbot.waitSignal(mc.all_started, timeout=1500):
            mc.start([cam])

        # Let several timeout cycles happen.
        qtbot.wait(500)

        assert mc.is_running()
        assert errors == []

    finally:
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)


@pytest.mark.unit
def test_non_trigger_timeouts_are_fatal_after_retries(qtbot, monkeypatch):
    class TimeoutBackend:
        waits_for_hardware_trigger = False

        def __init__(self, settings):
            self.settings = settings

        def open(self):
            pass

        def read(self):
            raise TimeoutError("camera timeout")

        def close(self):
            pass

    def _create(settings):
        return TimeoutBackend(settings)

    monkeypatch.setattr(CameraFactory, "create", staticmethod(_create))

    mc = MultiCameraController()
    cam = CameraSettings(name="TimeoutCam", backend="opencv", index=0, enabled=True).apply_defaults()

    expected_id = get_camera_id(cam)

    with qtbot.waitSignal(mc.camera_error, timeout=3000) as blocker:
        mc.start([cam])

    cam_id, msg = blocker.args
    assert cam_id == expected_id
    assert "Camera read timeout" in msg

    # Cleanup if still running.
    if mc.is_running():
        with qtbot.waitSignal(mc.all_stopped, timeout=2000):
            mc.stop(wait=True)