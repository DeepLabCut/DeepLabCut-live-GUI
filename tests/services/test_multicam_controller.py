# tests/services/test_multicam_controller.py
import pytest

from dlclivegui.config import CameraSettings
from dlclivegui.services.multi_camera_controller import MultiCameraController


@pytest.mark.unit
def test_start_and_frames(qtbot, patch_factory):
    mc = MultiCameraController()

    # One dataclass + one dict (simulate mixed inputs)
    cam1 = CameraSettings(name="C1", backend="opencv", index=0, fps=25.0).apply_defaults()
    cam2 = {"name": "C2", "backend": "opencv", "index": 1, "fps": 30.0, "enabled": True}

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
