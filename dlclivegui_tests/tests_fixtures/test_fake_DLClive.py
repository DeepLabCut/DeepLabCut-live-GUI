# tests/test_fixtures/test_real_dlc_processor_smoke.py
import numpy as np
import pytest

from dlclivegui.config import DLCProcessorSettings


@pytest.mark.integration
def test_real_dlc_processor_smoke(qtbot, dlc_processor):
    proc = dlc_processor
    proc.configure(DLCProcessorSettings(model_path="dummy_model.pt"), processor=None)
    proc.reset()

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    # First enqueue triggers init + init-frame pose
    # Don't wait sequentially; signals can arrive close together.
    seen = {"init": [], "pose": []}
    proc.initialized.connect(lambda ok: seen["init"].append(ok))
    proc.pose_ready.connect(lambda res: seen["pose"].append(res))

    proc.enqueue_frame(frame, 1.0)

    qtbot.waitUntil(lambda: len(seen["init"]) >= 1 and len(seen["pose"]) >= 1, timeout=3000)
    assert seen["init"][-1] is True
    assert seen["pose"][-1].pose is not None
