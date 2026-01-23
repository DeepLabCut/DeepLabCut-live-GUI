# dlclivegui_tests/fixtures/fake_DLClive.py
import numpy as np


class StubDLCLive:
    """Minimal DLCLive replacement for tests."""

    def __init__(self, **options):
        self.options = options
        self.inited = False

    def init_inference(self, init_frame):
        self.inited = True

    def get_pose(self, frame, frame_time=None):
        # Return a deterministic (K,3) pose: x,y,p
        h, w = frame.shape[:2]
        k = 4
        xs = np.linspace(0.25 * w, 0.75 * w, k)
        ys = np.linspace(0.25 * h, 0.75 * h, k)
        ps = np.ones(k) * 0.99
        return np.stack([xs, ys, ps], axis=1).astype(np.float32)
