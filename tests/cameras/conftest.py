# tests/cameras/conftest.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Mock classes and fixtures for testing camera backends
# -----------------------------------------------------------------------------


@dataclass
class FakeVideoCapture:
    """
    Merged fake for cv2.VideoCapture used across:
      - backend tests (isOpened/release/getBackendName/grab/retrieve)
      - mode-probe tests (grant_map influences get() after set()).

    Key idea:
      - `props` is the source of truth for cap.get(...)
      - `grant_map` optionally overrides width/height readback based on what was last set
    """

    opened: bool = True
    backend_name: str = "FAKE"
    fps: float = 30.0

    # Optional: map requested (w,h) -> granted (w,h)
    grant_map: dict[tuple[int, int], tuple[int, int]] | None = None

    # Behavior toggles
    grab_ok: bool = True
    retrieve_ok: bool = True
    retrieve_frame: np.ndarray | None = None

    # If you want to emulate "device lost"
    released: bool = False

    def __post_init__(self):
        # Introspection
        self.set_calls: list[tuple[int, float]] = []
        self.get_calls: list[int] = []
        self.grab_calls: int = 0
        self.retrieve_calls: int = 0

        # Track last requested size (used for grant_map)
        self._set_w: int = 0
        self._set_h: int = 0

        # Default props store (works with cv2 constants or raw ids)
        # We'll fill with common OpenCV prop ids used in tests:
        self.props: dict[int, float] = {}

        # Keep defaults consistent with your existing tests:
        # Use numeric "canonical" ids 3/4/5 for W/H/FPS in case tests use those.
        self.props[3] = 640.0  # CAP_PROP_FRAME_WIDTH
        self.props[4] = 480.0  # CAP_PROP_FRAME_HEIGHT
        self.props[5] = float(self.fps)  # CAP_PROP_FPS
        self.props[6] = 0.0  # CAP_PROP_FOURCC (common id), may differ; we also handle via passed constant.

    # --- OpenCV-like lifecycle ---
    def isOpened(self) -> bool:
        return bool(self.opened) and not self.released

    def release(self) -> None:
        self.released = True

    def getBackendName(self) -> str:
        return self.backend_name

    # --- Core set/get ---
    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((int(prop_id), float(value)))

        pid = int(prop_id)

        # Treat both canonical ids and any cv2 constants the same by writing into props[pid].
        # Also mirror canonical ids for width/height/fps if a cv2 build uses different constants.
        if pid in (3, 4, 5):
            # Remember last requested W/H for grant_map logic
            if pid == 3:
                self._set_w = int(value)
            elif pid == 4:
                self._set_h = int(value)
            elif pid == 5:
                self.fps = float(value)

        # FOURCC: store as int (some code reads it back and bit-shifts)
        # OpenCV constant value varies; treat any integer-ish value as int if it looks like FourCC.
        if pid == 6 or (isinstance(value, (int, float)) and float(value).is_integer() and pid not in (3, 4, 5)):
            # We only coerce to int for common FOURCC prop or when caller sets an integer-like.
            # (keeps behavior stable for normal float props like exposure.)
            self.props[pid] = float(int(value))
        else:
            self.props[pid] = float(value)

        # Mirror to canonical ids when tests pass cv2 constants that differ from 3/4/5/6
        # (Many OpenCV builds use the same canonical numbers, but this makes it resilient.)
        if pid != 3 and pid != 4 and pid != 5:
            # If caller used a non-canonical id for width/height/fps, try to keep both in sync
            # by detecting if it matches the current stored canonical prop (best-effort).
            pass

        return True

    def get(self, prop_id: int) -> float:
        pid = int(prop_id)
        self.get_calls.append(pid)

        # If we have a grant_map and the property is width/height,
        # return the granted mode for the last requested (w,h).
        if self.grant_map and pid in (3, 4):
            req = (int(self._set_w), int(self._set_h))
            granted = self.grant_map.get(req, req)
            if pid == 3:
                return float(granted[0])
            return float(granted[1])

        # FPS canonical
        if pid == 5:
            return float(self.fps)

        return float(self.props.get(pid, 0.0))

    # --- Read path ---
    def grab(self) -> bool:
        self.grab_calls += 1
        return bool(self.grab_ok)

    def retrieve(self):
        self.retrieve_calls += 1
        if not self.retrieve_ok:
            return False, None
        if self.retrieve_frame is None:
            self.retrieve_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        return True, self.retrieve_frame


@pytest.fixture()
def fake_capture_factory():
    """
    Factory fixture to create configured FakeVideoCapture instances.
    """

    def _factory(
        *,
        opened=True,
        backend_name="FAKE",
        fps=30.0,
        grab_ok=True,
        retrieve_ok=True,
        retrieve_frame=None,
        grant_map=None,
    ):
        cap = FakeVideoCapture(
            opened=opened,
            backend_name=backend_name,
            fps=fps,
            grab_ok=grab_ok,
            retrieve_ok=retrieve_ok,
            retrieve_frame=retrieve_frame,
            grant_map=grant_map,
        )
        return cap

    return _factory
