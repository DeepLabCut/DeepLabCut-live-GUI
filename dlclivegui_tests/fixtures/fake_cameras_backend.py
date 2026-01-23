# dlclivegui_tests/fixtures/fake_cameras_backend.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class StubCameraBehavior:
    """
    Controls fault injection.
    - empty_every_n: every Nth frame is empty (size==0)
    - raise_every_n: every Nth frame raises RuntimeError
    - empty_first_n: first N frames are empty
    - raise_first_n: first N frames raise
    """

    empty_every_n: int = 0
    raise_every_n: int = 0
    empty_first_n: int = 0
    raise_first_n: int = 0


class StubCameraBackend:
    """
    Minimal camera backend stub compatible with SingleCameraWorker.

    Methods:
      - open()
      - read() -> (frame, timestamp)
      - close()

    Frames:
      - default BGR uint8 (H,W,3)
      - can optionally emit grayscale or BGRA for conversion-path tests
    """

    def __init__(
        self,
        *,
        camera_id: str,
        shape: Tuple[int, int, int] = (240, 320, 3),
        base_color_bgr: Tuple[int, int, int] = (30, 30, 200),
        behavior: Optional[StubCameraBehavior] = None,
        emit_mode: str = "bgr",  # "bgr" | "gray" | "bgra"
        properties: Optional[Dict[str, Any]] = None,
        fake_fps: float = 0.0,  # if >0, timestamps advance at 1/fps; no sleeping
    ):
        self.camera_id = camera_id
        self.shape = shape
        self.base_color_bgr = base_color_bgr
        self.behavior = behavior or StubCameraBehavior()
        self.emit_mode = emit_mode
        self.properties = properties or {}
        self.fake_fps = float(fake_fps)

        self._opened = False
        self._frame_idx = 0
        self._t0 = time.time()

    def open(self) -> None:
        self._opened = True
        self._frame_idx = 0
        self._t0 = time.time()

    def close(self) -> None:
        self._opened = False

    def read(self):
        if not self._opened:
            raise RuntimeError(f"StubCameraBackend({self.camera_id}) read() called before open()")

        self._frame_idx += 1

        # Fault injection: raise
        if self.behavior.raise_first_n and self._frame_idx <= self.behavior.raise_first_n:
            raise RuntimeError(f"Simulated camera read error (first_n) for {self.camera_id}")
        if self.behavior.raise_every_n and (self._frame_idx % self.behavior.raise_every_n) == 0:
            raise RuntimeError(f"Simulated camera read error (every_n) for {self.camera_id}")

        # Fault injection: empty frame
        if self.behavior.empty_first_n and self._frame_idx <= self.behavior.empty_first_n:
            return np.array([], dtype=np.uint8), self._timestamp()
        if self.behavior.empty_every_n and (self._frame_idx % self.behavior.empty_every_n) == 0:
            return np.array([], dtype=np.uint8), self._timestamp()

        # Normal frame
        frame = self._make_frame()
        return frame, self._timestamp()

    def _timestamp(self) -> float:
        if self.fake_fps > 0:
            # deterministic timestamps without sleeping
            return self._t0 + (self._frame_idx / self.fake_fps)
        return time.time()

    def _make_frame(self) -> np.ndarray:
        h, w, c = self.shape
        if c not in (1, 3, 4):
            raise ValueError(f"Unsupported channel count in shape: {self.shape}")

        if self.emit_mode == "gray":
            frame = np.zeros((h, w), dtype=np.uint8)
            frame[:] = int(np.mean(self.base_color_bgr))
            self._stamp_text_gray(frame)
            return frame

        # BGR or BGRA
        if self.emit_mode == "bgra":
            frame = np.zeros((h, w, 4), dtype=np.uint8)
            frame[..., :3] = self.base_color_bgr
            frame[..., 3] = 255
            self._stamp_text_bgr(frame[..., :3])
            return frame

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = self.base_color_bgr
        self._stamp_text_bgr(frame)
        return frame

    def _stamp_text_bgr(self, bgr: np.ndarray) -> None:
        # If cv2 isn't available in environment, just draw simple pixels deterministically
        if cv2 is None:
            y = min(10, bgr.shape[0] - 1)
            bgr[y : y + 3, 0:50] = (255, 255, 255)
            return

        cv2.putText(
            bgr,
            self.camera_id,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            bgr,
            f"f={self._frame_idx}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _stamp_text_gray(self, gray: np.ndarray) -> None:
        # Simple deterministic marker when cv2 not available
        if cv2 is None:
            y = min(10, gray.shape[0] - 1)
            gray[y : y + 3, 0:50] = 255
            return

        cv2.putText(
            gray,
            self.camera_id,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            255,
            2,
        )
        cv2.putText(
            gray,
            f"f={self._frame_idx}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            255,
            2,
        )
