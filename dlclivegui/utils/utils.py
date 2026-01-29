from __future__ import annotations

import time
from collections import deque
from pathlib import Path

SUPPORTED_MODELS = [".pt", ".pth", ".pb"]


def is_model_file(file_path: Path | str) -> bool:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not file_path.is_file():
        return False
    return file_path.suffix.lower() in SUPPORTED_MODELS


class FPSTracker:
    """Track per-camera FPS within a sliding time window."""

    def __init__(self, window_seconds: float = 5.0, maxlen: int = 240):
        self.window_seconds = window_seconds
        self._times: dict[str, deque[float]] = {}
        self._maxlen = maxlen

    def clear(self) -> None:
        self._times.clear()

    def note_frame(self, camera_id: str) -> None:
        now = time.perf_counter()
        dq = self._times.get(camera_id)
        if dq is None:
            dq = deque(maxlen=self._maxlen)
            self._times[camera_id] = dq
        dq.append(now)
        while dq and (now - dq[0]) > self.window_seconds:
            dq.popleft()

    def fps(self, camera_id: str) -> float:
        dq = self._times.get(camera_id)
        if not dq or len(dq) < 2:
            return 0.0
        duration = dq[-1] - dq[0]
        if duration <= 0:
            return 0.0
        return (len(dq) - 1) / duration
