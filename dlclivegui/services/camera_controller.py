from __future__ import annotations

import copy
import logging
import time
from threading import Event, Lock

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend

# from dlclivegui.config import CameraSettings
from dlclivegui.config import (
    SINGLE_CAMERA_WORKER_DO_LOG_TIMING,
    CameraSettings,
)
from dlclivegui.utils.stats import WorkerTimingStats

logger = logging.getLogger(__name__)


class SingleCameraWorker(QObject):
    """Worker for a single camera in multi-camera mode."""

    frame_captured = Signal(str, object, float)  # camera_id, frame, timestamp
    error_occurred = Signal(str, str)  # camera_id, error_message
    runtime_info = Signal(str, object)  # camera_id, dict of runtime info
    started = Signal(str)  # camera_id
    stopped = Signal(str)  # camera_id

    def __init__(self, camera_id: str, settings: CameraSettings):
        super().__init__()
        self._camera_id = camera_id
        self._settings = copy.deepcopy(settings)
        self._stop_event = Event()
        self._backend: CameraBackend | None = None
        self._max_consecutive_errors = 5
        self._retry_delay = 0.1
        self._trigger_timeout_delay = 0.05
        self._trigger_wait_log_interval = 2.0
        self._last_trigger_wait_log = 0.0
        self._trigger_wait_suppressed_count = 0

        self._recording_sink = None
        self._recording_enabled = False
        self._recording_sink_lock = Lock()

        # Performance logs
        self._timing = WorkerTimingStats(
            camera_id, logger=logger, log_interval=1.0, enabled=SINGLE_CAMERA_WORKER_DO_LOG_TIMING
        )

    def set_recording_sink(self, sink) -> None:
        with self._recording_sink_lock:
            self._recording_sink = sink

    def set_recording_enabled(self, enabled: bool) -> None:
        with self._recording_sink_lock:
            self._recording_enabled = bool(enabled)

    @Slot()
    def run(self) -> None:
        self._stop_event.clear()

        try:
            logger.debug(
                "[Worker %s] before create: backend=%s index=%s properties=%s",
                self._camera_id,
                self._settings.backend,
                self._settings.index,
                self._settings.properties,
            )

            self._backend = CameraFactory.create(self._settings)

            logger.debug(
                "[Worker %s] after create: backend=%s index=%s properties=%s",
                self._camera_id,
                self._backend.settings.backend,
                self._backend.settings.index,
                self._backend.settings.properties,
            )

            self._backend.open()
            self.runtime_info.emit(
                self._camera_id,
                {
                    "actual_fps": getattr(self._backend, "actual_fps", None),
                    "actual_resolution": getattr(self._backend, "actual_resolution", None),
                    "actual_pixel_format": getattr(self._backend, "actual_pixel_format", None),
                    "actual_output_format": getattr(self._backend, "actual_output_format", None),
                },
            )
        except Exception as exc:
            logger.exception(f"Failed to initialize camera {self._camera_id}", exc_info=exc)
            self.error_occurred.emit(self._camera_id, f"Failed to initialize camera: {exc}")
            self.stopped.emit(self._camera_id)
            return

        self.started.emit(self._camera_id)
        consecutive_errors = 0

        while not self._stop_event.is_set():
            try:
                with self._timing.measure("Single.read"):
                    captured = self._backend.read()
                    frame = captured.frame
                    timestamp = captured.software_timestamp
                    timestamp_metadata = captured.timestamp_metadata
                if frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors >= self._max_consecutive_errors:
                        self.error_occurred.emit(
                            self._camera_id, "Too many empty frames.\nWas the device disconnected ?"
                        )
                        break
                    if self._stop_event.wait(self._retry_delay):
                        break
                    continue

                consecutive_errors = 0
                with self._timing.measure("Single.transforms"):
                    frame = self._apply_worker_transforms(frame)

                with self._recording_sink_lock:
                    recording_enabled = self._recording_enabled
                    recording_sink = self._recording_sink

                if recording_enabled and recording_sink is not None:
                    try:
                        with self._timing.measure("Single.recording_sink"):
                            recording_sink(self._camera_id, frame, timestamp, timestamp_metadata)
                    except Exception as exc:
                        logger.exception(f"Failed to write frame for camera {self._camera_id}: {exc}")

                with self._timing.measure("Single.emit"):
                    self.frame_captured.emit(self._camera_id, frame, timestamp, timestamp_metadata)

                self._timing.note_frame()
                self._timing.maybe_log()

            except TimeoutError as exc:
                self._timing.note_timeout()
                self._timing.maybe_log()
                if self._stop_event.is_set():
                    break

                # In hardware-trigger mode, a timeout usually means:
                # "no trigger pulse arrived during this poll interval".
                # This is expected and should not count as a camera failure.
                if bool(getattr(self._backend, "waits_for_hardware_trigger", False)):
                    self._log_trigger_wait_throttled(exc)
                    consecutive_errors = 0

                    if self._stop_event.wait(self._trigger_timeout_delay):
                        break  # Stop event set during wait
                    continue

                consecutive_errors += 1
                if consecutive_errors >= self._max_consecutive_errors:
                    self.error_occurred.emit(self._camera_id, f"Camera read timeout: {exc}")
                    break
                if self._stop_event.wait(self._retry_delay):
                    break
                continue

            except Exception as exc:
                self._timing.note_error()
                self._timing.maybe_log()
                consecutive_errors += 1
                if self._stop_event.is_set():
                    break
                if consecutive_errors >= self._max_consecutive_errors:
                    self.error_occurred.emit(self._camera_id, f"Camera read error: {exc}")
                    break
                if self._stop_event.wait(self._retry_delay):
                    break
                continue

        # Cleanup
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
        self.stopped.emit(self._camera_id)

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def apply_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
        """Apply rotation to frame."""
        if degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def apply_crop(frame: np.ndarray, crop_region: tuple[int, int, int, int]) -> np.ndarray:
        """Apply crop to frame."""
        x0, y0, x1, y1 = crop_region
        height, width = frame.shape[:2]

        x0 = max(0, min(x0, width))
        y0 = max(0, min(y0, height))
        x1 = max(x0, min(x1, width)) if x1 > 0 else width
        y1 = max(y0, min(y1, height)) if y1 > 0 else height

        if x0 < x1 and y0 < y1:
            return frame[y0:y1, x0:x1]
        return frame

    def _apply_worker_transforms(self, frame: np.ndarray) -> np.ndarray:
        if self._settings.rotation:
            frame = self.apply_rotation(frame, self._settings.rotation)

        crop_region = self._settings.get_crop_region()
        if crop_region:
            frame = self.apply_crop(frame, crop_region)

        return frame

    def _log_trigger_wait_throttled(self, exc: BaseException) -> None:
        """Log hardware-trigger wait timeouts at a controlled rate.

        In trigger-waiting modes, read timeouts are expected polling misses.
        Without throttling, the log can be flooded at ~10-20 messages/sec/camera.
        """
        now = time.monotonic()

        if now - self._last_trigger_wait_log < self._trigger_wait_log_interval:
            self._trigger_wait_suppressed_count += 1
            return

        suppressed = self._trigger_wait_suppressed_count
        self._trigger_wait_suppressed_count = 0
        self._last_trigger_wait_log = now

        if suppressed:
            logger.debug(
                "[Worker %s] waiting for hardware trigger: %s (suppressed %d repeated timeout logs)",
                self._camera_id,
                exc,
                suppressed,
            )
        else:
            logger.debug(
                "[Worker %s] waiting for hardware trigger: %s",
                self._camera_id,
                exc,
            )
