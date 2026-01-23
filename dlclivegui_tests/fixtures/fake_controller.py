# dlclivegui_tests/fixtures/fake_controller.py
from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from dlclivegui.multi_camera_controller import MultiFrameData, get_camera_id


class FakeMultiCameraController(QObject):
    """
    Thread-free, deterministic MultiCameraController for GUI tests.

    - start/stop lifecycle
    - emits all_started/all_stopped
    - emits frame_ready(MultiFrameData)
    - timer-based streaming of synthetic frames
    """

    frame_ready = Signal(object)  # MultiFrameData
    camera_started = Signal(str, object)  # camera_id, settings
    camera_stopped = Signal(str)  # camera_id
    camera_error = Signal(str, str)  # camera_id, error_message
    all_started = Signal()
    all_stopped = Signal()
    initialization_failed = Signal(list)  # List[(camera_id, error_message)]

    MAX_CAMERAS = 4

    def __init__(self):
        super().__init__()
        self._running: bool = False
        self._settings: Dict[str, object] = {}
        self._started: Set[str] = set()
        self._failed: Dict[str, str] = {}
        self._frames: Dict[str, np.ndarray] = {}
        self._timestamps: Dict[str, float] = {}
        self._timer: Optional[QTimer] = None
        self._frame_counter: int = 0

        # Allow tests to force certain cameras to fail on start
        self.fail_on_start: Set[str] = set()

    def is_running(self) -> bool:
        return self._running and len(self._started) > 0

    def get_active_count(self) -> int:
        return len(self._started)

    def start(self, camera_settings: List[object]) -> None:
        if self._running:
            return

        active = [s for s in camera_settings if getattr(s, "enabled", True)][: self.MAX_CAMERAS]
        if not active:
            return

        self._running = True
        self._settings.clear()
        self._started.clear()
        self._failed.clear()
        self._frames.clear()
        self._timestamps.clear()
        self._frame_counter = 0

        failures: List[Tuple[str, str]] = []
        for s in active:
            cam_id = get_camera_id(s)
            self._settings[cam_id] = s

            if cam_id in self.fail_on_start:
                msg = "Simulated initialization failure"
                self._failed[cam_id] = msg
                failures.append((cam_id, msg))
                self.camera_error.emit(cam_id, msg)
                continue

            self._started.add(cam_id)
            self.camera_started.emit(cam_id, s)

        # Match real controller behavior
        if self._started:
            self.all_started.emit()
        else:
            self._running = False
            if failures:
                self.initialization_failed.emit(failures)
            self.all_stopped.emit()

    def stop(self, wait: bool = True) -> None:
        # wait ignored (no threads), kept for API compatibility
        if not self._running and not self._started:
            return

        self._running = False

        for cam_id in list(self._started):
            self.camera_stopped.emit(cam_id)

        self._started.clear()
        self._settings.clear()
        self._frames.clear()
        self._timestamps.clear()
        self._failed.clear()

        self.stop_streaming()
        self.all_stopped.emit()

    def emit_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        if not self._running or camera_id not in self._started:
            return

        ts = time.time() if timestamp is None else float(timestamp)
        self._frames[camera_id] = frame
        self._timestamps[camera_id] = ts

        self.frame_ready.emit(
            MultiFrameData(
                frames=dict(self._frames),
                timestamps=dict(self._timestamps),
                source_camera_id=camera_id,
                tiled_frame=None,
            )
        )

    def start_streaming(self, fps: float = 30.0, frame_shape=(240, 320, 3)) -> None:
        self.stop_streaming()
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, int(1000 / fps)))
        self._timer.timeout.connect(lambda: self._tick(frame_shape))
        self._timer.start()

    def stop_streaming(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None

    def _tick(self, frame_shape) -> None:
        if not self.is_running():
            return

        self._frame_counter += 1
        for idx, cam_id in enumerate(sorted(self._started)):
            frame = self._make_frame(cam_id, frame_shape, idx, self._frame_counter)
            self.emit_frame(cam_id, frame)

    def _make_frame(self, cam_id: str, shape, cam_index: int, counter: int) -> np.ndarray:
        h, w, c = shape
        frame = np.zeros((h, w, c), dtype=np.uint8)

        # Different base color per camera for easy debugging
        base_colors = [
            (30, 30, 200),  # reddish
            (30, 200, 30),  # greenish
            (200, 30, 30),  # bluish
            (160, 160, 30),
        ]
        frame[:] = base_colors[cam_index % len(base_colors)]

        # Stamp text so frames differ over time
        cv2.putText(frame, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            frame, f"t={counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        return frame
