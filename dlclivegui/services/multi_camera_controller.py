"""Multi-camera management for the DLC Live GUI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Event, Lock

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend

# from dlclivegui.config import CameraSettings
from dlclivegui.config import CameraSettings

LOGGER = logging.getLogger(__name__)


@dataclass
class MultiFrameData:
    """Container for frames from multiple cameras."""

    frames: dict[str, np.ndarray]  # camera_id -> frame
    timestamps: dict[str, float]  # camera_id -> timestamp
    source_camera_id: str = ""  # ID of camera that triggered this emission
    tiled_frame: np.ndarray | None = None  # Combined tiled frame (deprecated, done in GUI)


class SingleCameraWorker(QObject):
    """Worker for a single camera in multi-camera mode."""

    frame_captured = Signal(str, object, float)  # camera_id, frame, timestamp
    error_occurred = Signal(str, str)  # camera_id, error_message
    started = Signal(str)  # camera_id
    stopped = Signal(str)  # camera_id

    def __init__(self, camera_id: str, settings: CameraSettings):
        super().__init__()
        self._camera_id = camera_id
        self._settings = settings
        self._stop_event = Event()
        self._backend: CameraBackend | None = None
        self._max_consecutive_errors = 5
        self._retry_delay = 0.1

    @Slot()
    def run(self) -> None:
        self._stop_event.clear()

        try:
            self._backend = CameraFactory.create(self._settings)
            self._backend.open()
        except Exception as exc:
            LOGGER.exception(f"Failed to initialize camera {self._camera_id}", exc_info=exc)
            self.error_occurred.emit(self._camera_id, f"Failed to initialize camera: {exc}")
            self.stopped.emit(self._camera_id)
            return

        self.started.emit(self._camera_id)
        consecutive_errors = 0

        while not self._stop_event.is_set():
            try:
                frame, timestamp = self._backend.read()
                if frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors >= self._max_consecutive_errors:
                        self.error_occurred.emit(
                            self._camera_id, "Too many empty frames.\nWas the device disconnected ?"
                        )
                        break
                    time.sleep(self._retry_delay)
                    continue

                consecutive_errors = 0
                self.frame_captured.emit(self._camera_id, frame, timestamp)

            except Exception as exc:
                consecutive_errors += 1
                if self._stop_event.is_set():
                    break
                if consecutive_errors >= self._max_consecutive_errors:
                    self.error_occurred.emit(self._camera_id, f"Camera read error: {exc}")
                    break
                time.sleep(self._retry_delay)
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


def get_camera_id(settings: CameraSettings) -> str:
    """Generate a unique camera ID from settings."""
    return f"{settings.backend}:{settings.index}"


class MultiCameraController(QObject):
    """Controller for managing multiple cameras simultaneously."""

    # Signals
    frame_ready = Signal(object)  # MultiFrameData
    camera_started = Signal(str, object)  # camera_id, settings
    camera_stopped = Signal(str)  # camera_id
    camera_error = Signal(str, str)  # camera_id, error_message
    all_started = Signal()
    all_stopped = Signal()
    initialization_failed = Signal(list)  # List of (camera_id, error_message) tuples

    MAX_CAMERAS = 4

    def __init__(self):
        super().__init__()
        self._workers: dict[str, SingleCameraWorker] = {}
        self._threads: dict[str, QThread] = {}
        self._settings: dict[str, CameraSettings] = {}
        self._frames: dict[str, np.ndarray] = {}
        self._timestamps: dict[str, float] = {}
        self._frame_lock = Lock()
        self._running = False
        self._started_cameras: set = set()
        self._failed_cameras: dict[str, str] = {}  # camera_id -> error message
        self._expected_cameras: int = 0  # Number of cameras we're trying to start

    def is_running(self) -> bool:
        """Check if any camera is currently running."""
        return self._running and len(self._started_cameras) > 0

    def get_active_count(self) -> int:
        """Get the number of active cameras."""
        return len(self._started_cameras)

    def start(self, camera_settings: list[CameraSettings]) -> None:
        """Start multiple cameras."""
        if self._running:
            LOGGER.warning("Multi-camera controller already running")
            return

        active_settings = [s for s in camera_settings if s.enabled][: self.MAX_CAMERAS]
        if not active_settings:
            LOGGER.warning("No active cameras to start")
            return

        self._running = True
        self._frames.clear()
        self._timestamps.clear()
        self._started_cameras.clear()
        self._failed_cameras.clear()
        self._expected_cameras = len(active_settings)

        for settings in active_settings:
            self._start_camera(settings)

    def _start_camera(self, settings: CameraSettings) -> None:
        """Start a single camera."""
        cam_id = get_camera_id(settings)
        if cam_id in self._workers:
            LOGGER.warning(f"Camera {cam_id} already has a worker")
            return

        # Normalize and store the dataclass once
        self._settings[cam_id] = settings
        dc = self._settings[cam_id]
        worker = SingleCameraWorker(cam_id, dc)
        thread = QThread()
        worker.moveToThread(thread)

        # Connections unchanged
        thread.started.connect(worker.run)
        worker.frame_captured.connect(self._on_frame_captured)
        worker.started.connect(self._on_camera_started)
        worker.stopped.connect(self._on_camera_stopped)
        worker.error_occurred.connect(self._on_camera_error)

        self._workers[cam_id] = worker
        self._threads[cam_id] = thread
        thread.start()

    def stop(self, wait: bool = True) -> None:
        """Stop all cameras."""
        if not self._running:
            return

        self._running = False

        # Signal all workers to stop
        for worker in self._workers.values():
            worker.stop()

        # Wait for threads to finish
        if wait:
            for cam_id, thread in list(self._threads.items()):
                if not thread.isRunning():
                    continue

                thread.quit()
                if not thread.wait(5000):
                    LOGGER.error("Frozen camera thread %s; Forcing terminate()", cam_id)
                    thread.terminate()
                    thread.wait(1000)

        self._workers.clear()
        self._threads.clear()
        self._settings.clear()
        self._started_cameras.clear()
        self._failed_cameras.clear()
        self._expected_cameras = 0
        self.all_stopped.emit()

    def _on_frame_captured(self, camera_id: str, frame: np.ndarray, timestamp: float) -> None:
        """Handle a frame from one camera."""
        # Apply rotation if configured
        settings = self._settings.get(camera_id)
        if settings and settings.rotation:
            frame = MultiCameraController.apply_rotation(frame, settings.rotation)

        # Apply cropping if configured
        if settings:
            crop_region = settings.get_crop_region()
            if crop_region:
                frame = MultiCameraController.apply_crop(frame, crop_region)

        with self._frame_lock:
            self._frames[camera_id] = frame
            self._timestamps[camera_id] = timestamp

            # Emit frame data without tiling (tiling done in GUI for performance)
            if self._frames:
                frame_data = MultiFrameData(
                    frames=dict(self._frames),
                    timestamps=dict(self._timestamps),
                    source_camera_id=camera_id,  # Track which camera triggered this
                    tiled_frame=None,
                )
                self.frame_ready.emit(frame_data)

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

    @staticmethod
    def apply_resize(frame: np.ndarray, max_w: int, max_h: int, allow_upscale: bool = False) -> np.ndarray:
        """Resize frame to fit within max dimensions while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            LOGGER.warning("Cannot resize frame with zero width or height")
            return frame

        scale = min(max_w / w, max_h / h)
        if not allow_upscale:
            scale = min(scale, 1.0)

        if scale == 1.0:
            return frame

        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def ensure_color_bgr(frame: np.ndarray) -> np.ndarray:
        """Ensure frame is 3-channel color."""
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    @staticmethod
    def ensure_color_rgb(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    @staticmethod
    def to_display_pixmap(frame: np.ndarray) -> QPixmap:
        """Convert a frame to QPixmap for display."""
        frame = MultiCameraController.ensure_color_rgb(frame)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(q_img)

    def _create_tiled_frame(self) -> np.ndarray:
        """Create a tiled frame from all camera frames.

        The tiled frame is scaled to fit within a maximum canvas size
        while maintaining aspect ratio of individual camera frames.
        """
        if not self._frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frames_list = [self._frames[idx] for idx in sorted(self._frames.keys())]
        num_frames = len(frames_list)

        if num_frames == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Determine grid layout
        if num_frames == 1:
            rows, cols = 1, 1
        elif num_frames == 2:
            rows, cols = 1, 2
        elif num_frames <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 2  # Limit to 4

        # Maximum canvas size to fit on screen (leaving room for UI elements)
        max_canvas_width = 1200
        max_canvas_height = 800

        # Calculate tile size based on frame aspect ratio and available space
        first_frame = frames_list[0]
        frame_h, frame_w = first_frame.shape[:2]
        frame_aspect = frame_w / frame_h if frame_h > 0 else 1.0

        # Calculate tile dimensions that fit within the canvas
        tile_w = max_canvas_width // cols
        tile_h = max_canvas_height // rows

        # Maintain aspect ratio of original frames
        tile_aspect = tile_w / tile_h if tile_h > 0 else 1.0

        if frame_aspect > tile_aspect:
            # Frame is wider than tile slot - constrain by width
            tile_h = int(tile_w / frame_aspect)
        else:
            # Frame is taller than tile slot - constrain by height
            tile_w = int(tile_h * frame_aspect)

        # Ensure minimum size
        tile_w = max(160, tile_w)
        tile_h = max(120, tile_h)

        # Create canvas
        canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        # Get sorted camera IDs for consistent ordering
        cam_ids = sorted(self._frames.keys())
        frames_list = [self._frames[cam_id] for cam_id in cam_ids]

        # Place each frame in the grid
        for idx, frame in enumerate(frames_list[: rows * cols]):
            row = idx // cols
            col = idx % cols

            # Ensure frame is 3-channel
            frame = MultiCameraController.ensure_color_bgr(frame)

            # Resize to tile size
            resized = MultiCameraController.apply_resize(frame, tile_w, tile_h, allow_upscale=True)

            # Add camera ID label
            if idx < len(cam_ids):
                label = cam_ids[idx]
                cv2.putText(
                    resized,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Place in canvas
            y_start = row * tile_h
            y_end = y_start + tile_h
            x_start = col * tile_w
            x_end = x_start + tile_w
            canvas[y_start:y_end, x_start:x_end] = resized

        return canvas

    def _on_camera_started(self, camera_id: str) -> None:
        """Handle camera start event."""
        self._started_cameras.add(camera_id)
        settings = self._settings.get(camera_id)
        self.camera_started.emit(camera_id, settings)
        LOGGER.info(f"Camera {camera_id} started")

        # Check if all cameras have reported (started or failed)
        total_reported = len(self._started_cameras) + len(self._failed_cameras)
        if total_reported == self._expected_cameras:
            if self._started_cameras:
                # At least some cameras started successfully
                self.all_started.emit()
            # If no cameras started but all failed, that's handled in _on_camera_stopped

    def _on_camera_stopped(self, camera_id: str) -> None:
        """Handle camera stop event."""
        # Check if this camera never started (initialization failure)
        was_started = camera_id in self._started_cameras
        self._started_cameras.discard(camera_id)
        self.camera_stopped.emit(camera_id)
        LOGGER.info(f"Camera {camera_id} stopped (was_started={was_started})")

        # Cleanup thread
        if camera_id in self._threads:
            thread = self._threads[camera_id]
            if thread.isRunning():
                thread.quit()
                thread.wait(1000)
            del self._threads[camera_id]

        if camera_id in self._workers:
            del self._workers[camera_id]

        # Remove frame data
        with self._frame_lock:
            self._frames.pop(camera_id, None)
            self._timestamps.pop(camera_id, None)

        # Check if all cameras have reported and none started
        total_reported = len(self._started_cameras) + len(self._failed_cameras)
        if total_reported == self._expected_cameras and not self._started_cameras:
            # All cameras failed to start
            if self._running and self._failed_cameras:
                self._running = False
                failure_list = list(self._failed_cameras.items())
                self.initialization_failed.emit(failure_list)
                self.all_stopped.emit()
                return

        # Check if all running cameras have stopped (normal shutdown)
        if not self._started_cameras and self._running and not self._workers:
            self._running = False
            self.all_stopped.emit()

    def _on_camera_error(self, camera_id: str, message: str) -> None:
        """Handle camera error event."""
        LOGGER.error(f"Camera {camera_id} error: {message}")
        # Track failed cameras (only if not already started - i.e., initialization failure)
        if camera_id not in self._started_cameras:
            self._failed_cameras[camera_id] = message
        self.camera_error.emit(camera_id, message)

    def get_frame(self, camera_id: str) -> np.ndarray | None:
        """Get the latest frame from a specific camera."""
        with self._frame_lock:
            return self._frames.get(camera_id)

    def get_all_frames(self) -> dict[str, np.ndarray]:
        """Get the latest frames from all cameras."""
        with self._frame_lock:
            return dict(self._frames)

    def get_tiled_frame(self) -> np.ndarray | None:
        """Get a tiled view of all camera frames."""
        with self._frame_lock:
            if self._frames:
                return self._create_tiled_frame()
        return None
