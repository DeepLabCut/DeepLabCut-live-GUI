"""Camera management for the DLC Live GUI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Event
from typing import Optional

import numpy as np
from PyQt6.QtCore import QMetaObject, QObject, Qt, QThread, pyqtSignal, pyqtSlot

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend
from dlclivegui.config import CameraSettings

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for a captured frame."""

    image: np.ndarray
    timestamp: float


class CameraWorker(QObject):
    """Worker object running inside a :class:`QThread`."""

    frame_captured = pyqtSignal(object)
    started = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    warning_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, settings: CameraSettings):
        super().__init__()
        self._settings = settings
        self._stop_event = Event()
        self._backend: Optional[CameraBackend] = None

        # Error recovery settings
        self._max_consecutive_errors = 5
        self._max_reconnect_attempts = 3
        self._retry_delay = 0.1  # seconds
        self._reconnect_delay = 1.0  # seconds

        # Frame validation
        self._expected_frame_size: Optional[tuple[int, int]] = None  # (height, width)

    @pyqtSlot()
    def run(self) -> None:
        self._stop_event.clear()

        # Initialize camera
        if not self._initialize_camera():
            self.finished.emit()
            return

        self.started.emit(self._settings)

        consecutive_errors = 0
        reconnect_attempts = 0

        while not self._stop_event.is_set():
            try:
                frame, timestamp = self._backend.read()

                # Validate frame size
                if not self._validate_frame_size(frame):
                    consecutive_errors += 1
                    LOGGER.warning(
                        f"Frame size validation failed ({consecutive_errors}/{self._max_consecutive_errors})"
                    )
                    if consecutive_errors >= self._max_consecutive_errors:
                        self.error_occurred.emit("Too many frames with incorrect size")
                        break
                    time.sleep(self._retry_delay)
                    continue

                consecutive_errors = 0  # Reset error count on success
                reconnect_attempts = 0  # Reset reconnect attempts on success

            except TimeoutError as exc:
                consecutive_errors += 1
                LOGGER.warning(
                    f"Camera frame timeout ({consecutive_errors}/{self._max_consecutive_errors}): {exc}"
                )

                if self._stop_event.is_set():
                    break

                # Handle timeout with retry logic
                if consecutive_errors < self._max_consecutive_errors:
                    self.warning_occurred.emit(
                        f"Frame timeout (retry {consecutive_errors}/{self._max_consecutive_errors})"
                    )
                    time.sleep(self._retry_delay)
                    continue
                else:
                    # Too many consecutive errors, try to reconnect
                    LOGGER.error(f"Too many consecutive timeouts, attempting reconnection...")
                    if self._attempt_reconnection():
                        consecutive_errors = 0
                        reconnect_attempts += 1
                        self.warning_occurred.emit(
                            f"Camera reconnected (attempt {reconnect_attempts})"
                        )
                        continue
                    else:
                        reconnect_attempts += 1
                        if reconnect_attempts >= self._max_reconnect_attempts:
                            self.error_occurred.emit(
                                f"Camera reconnection failed after {reconnect_attempts} attempts"
                            )
                            break
                        else:
                            consecutive_errors = 0  # Reset to try again
                            self.warning_occurred.emit(
                                f"Reconnection attempt {reconnect_attempts} failed, retrying..."
                            )
                            time.sleep(self._reconnect_delay)
                            continue

            except Exception as exc:
                consecutive_errors += 1
                LOGGER.warning(
                    f"Camera read error ({consecutive_errors}/{self._max_consecutive_errors}): {exc}"
                )

                if self._stop_event.is_set():
                    break

                # Handle general errors with retry logic
                if consecutive_errors < self._max_consecutive_errors:
                    self.warning_occurred.emit(
                        f"Frame read error (retry {consecutive_errors}/{self._max_consecutive_errors})"
                    )
                    time.sleep(self._retry_delay)
                    continue
                else:
                    # Too many consecutive errors, try to reconnect
                    LOGGER.error(f"Too many consecutive errors, attempting reconnection...")
                    if self._attempt_reconnection():
                        consecutive_errors = 0
                        reconnect_attempts += 1
                        self.warning_occurred.emit(
                            f"Camera reconnected (attempt {reconnect_attempts})"
                        )
                        continue
                    else:
                        reconnect_attempts += 1
                        if reconnect_attempts >= self._max_reconnect_attempts:
                            self.error_occurred.emit(
                                f"Camera failed after {reconnect_attempts} reconnection attempts: {exc}"
                            )
                            break
                        else:
                            consecutive_errors = 0  # Reset to try again
                            self.warning_occurred.emit(
                                f"Reconnection attempt {reconnect_attempts} failed, retrying..."
                            )
                            time.sleep(self._reconnect_delay)
                            continue

            if self._stop_event.is_set():
                break

            self.frame_captured.emit(FrameData(frame, timestamp))

        # Cleanup
        self._cleanup_camera()
        self.finished.emit()

    def _initialize_camera(self) -> bool:
        """Initialize the camera backend. Returns True on success, False on failure."""
        try:
            self._backend = CameraFactory.create(self._settings)
            self._backend.open()
            # Don't set expected frame size - will be established from first frame
            self._expected_frame_size = None
            LOGGER.info(
                "Camera initialized successfully, frame size will be determined from camera"
            )
            return True
        except Exception as exc:
            LOGGER.exception("Failed to initialize camera", exc_info=exc)
            self.error_occurred.emit(f"Failed to initialize camera: {exc}")
            return False

    def _validate_frame_size(self, frame: np.ndarray) -> bool:
        """Validate that the frame has the expected size. Returns True if valid."""
        if frame is None or frame.size == 0:
            LOGGER.warning("Received empty frame")
            return False

        actual_size = (frame.shape[0], frame.shape[1])  # (height, width)

        if self._expected_frame_size is None:
            # First frame - establish expected size
            self._expected_frame_size = actual_size
            LOGGER.info(
                f"Established expected frame size: (h={actual_size[0]}, w={actual_size[1]})"
            )
            return True

        if actual_size != self._expected_frame_size:
            LOGGER.warning(
                f"Frame size mismatch: expected (h={self._expected_frame_size[0]}, w={self._expected_frame_size[1]}), "
                f"got (h={actual_size[0]}, w={actual_size[1]}). Camera may have reconnected with different resolution."
            )
            # Update expected size for future frames after reconnection
            self._expected_frame_size = actual_size
            LOGGER.info(f"Updated expected frame size to: (h={actual_size[0]}, w={actual_size[1]})")
            # Emit warning so GUI can restart recording if needed
            self.warning_occurred.emit(
                f"Camera resolution changed to {actual_size[1]}x{actual_size[0]}"
            )
            return True  # Accept the new size

        return True

    def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to the camera. Returns True on success, False on failure."""
        if self._stop_event.is_set():
            return False

        LOGGER.info("Attempting camera reconnection...")

        # Close existing connection
        self._cleanup_camera()

        # Wait longer before reconnecting to let the device fully release
        LOGGER.info(f"Waiting {self._reconnect_delay}s before reconnecting...")
        time.sleep(self._reconnect_delay)

        if self._stop_event.is_set():
            return False

        # Try to reinitialize (this will also reset expected frame size)
        try:
            self._backend = CameraFactory.create(self._settings)
            self._backend.open()
            # Reset expected frame size - will be re-established on first frame
            self._expected_frame_size = None
            LOGGER.info("Camera reconnection successful, frame size will be determined from camera")
            return True
        except Exception as exc:
            LOGGER.warning(f"Camera reconnection failed: {exc}")
            return False

    def _cleanup_camera(self) -> None:
        """Clean up camera backend resources."""
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception as exc:
                LOGGER.warning(f"Error closing camera: {exc}")
            self._backend = None

    @pyqtSlot()
    def stop(self) -> None:
        self._stop_event.set()
        if self._backend is not None:
            try:
                self._backend.stop()
            except Exception:
                pass


class CameraController(QObject):
    """High level controller that manages a camera worker thread."""

    frame_ready = pyqtSignal(object)
    started = pyqtSignal(object)
    stopped = pyqtSignal()
    error = pyqtSignal(str)
    warning = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._thread: Optional[QThread] = None
        self._worker: Optional[CameraWorker] = None
        self._pending_settings: Optional[CameraSettings] = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, settings: CameraSettings) -> None:
        if self.is_running():
            self._pending_settings = settings
            self.stop(preserve_pending=True)
            return
        self._pending_settings = None
        self._start_worker(settings)

    def stop(self, wait: bool = False, *, preserve_pending: bool = False) -> None:
        if not self.is_running():
            if not preserve_pending:
                self._pending_settings = None
            return
        assert self._worker is not None
        assert self._thread is not None
        if not preserve_pending:
            self._pending_settings = None
        QMetaObject.invokeMethod(
            self._worker,
            "stop",
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker.stop()
        self._thread.quit()
        if wait:
            self._thread.wait()

    def _start_worker(self, settings: CameraSettings) -> None:
        self._thread = QThread()
        self._worker = CameraWorker(settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_captured.connect(self.frame_ready)
        self._worker.started.connect(self.started)
        self._worker.error_occurred.connect(self.error)
        self._worker.warning_occurred.connect(self.warning)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    @pyqtSlot()
    def _cleanup(self) -> None:
        self._thread = None
        self._worker = None
        self.stopped.emit()
        if self._pending_settings is not None:
            pending = self._pending_settings
            self._pending_settings = None
            self.start(pending)
