"""Multi-camera management for the DLC Live GUI."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from functools import partial
from threading import Lock

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from dlclivegui.cameras.factory import camera_identity_key

# from dlclivegui.config import CameraSettings
from dlclivegui.config import (
    GUI_MAX_DISPLAY_FPS,
    MULTI_CAMERA_WORKER_DO_LOG_TIMING,
    CameraSettings,
)
from dlclivegui.utils.stats import WorkerTimingStats

from .camera_controller import SingleCameraWorker

LOGGER = logging.getLogger(__name__)

QUIT_WAIT_MS = 5000  # wait for cooperative quit (5s)
TERMINATE_WAIT_MS = 1000  # wait after terminate (1s)


@dataclass
class MultiFrameData:
    """Container for frames from multiple cameras."""

    frames: dict[str, np.ndarray]  # camera_id -> frame
    timestamps: dict[str, float]  # camera_id -> timestamp
    source_camera_id: str = ""  # ID of camera that triggered this emission
    tiled_frame: np.ndarray | None = None  # Combined tiled frame (deprecated, done in GUI)
    display_ids: dict[str, str] = None  # camera_id -> display_id (for labeling)


def get_display_id(settings: CameraSettings) -> str:
    """Return the human-friendly camera label used for GUI display.
    Intentionally different from get_camera_id(), which should return a stable
    internal, reliable and unambiguous identity and may contain serials or machine paths.
    """
    name = str(getattr(settings, "name", "") or "").strip()
    if name:
        return name

    backend = (settings.backend or "").lower()
    props = settings.properties if isinstance(settings.properties, dict) else {}
    ns = props.get(backend, {}) if isinstance(props.get(backend), dict) else {}

    device_name = str(ns.get("device_name", "") or "").strip()
    if device_name:
        return device_name

    return f"{settings.backend}:{settings.index}"


def get_camera_id(settings: CameraSettings) -> str:
    """Generate a unique camera ID from stable backend identity."""
    backend = (settings.backend or "").lower()
    props = settings.properties if isinstance(settings.properties, dict) else {}
    ns = props.get(backend, {}) if isinstance(props.get(backend), dict) else {}

    device_id = ns.get("device_id")
    if device_id:
        return f"{backend}:{device_id}"

    serial = ns.get("serial_number") or ns.get("device_serial_number") or ns.get("serial")
    if serial:
        return f"{backend}:serial:{serial}"

    return f"{backend}:index:{int(settings.index)}"


def _trigger_role_from_settings(settings: CameraSettings) -> str:
    backend = (settings.backend or "").lower()
    props = settings.properties if isinstance(settings.properties, dict) else {}
    ns = props.get(backend, {}) if isinstance(props.get(backend), dict) else {}

    trigger = ns.get("trigger", {})
    if not isinstance(trigger, dict):
        return "off"

    role = str(trigger.get("role", "off") or "off").strip().lower()

    # Match CameraTriggerSettings aliases enough for controller ordering.
    if role in {"true", "on", "trigger", "triggered"}:
        return "external"
    if role in {"slave"}:
        return "follower"
    if role in {"main"}:
        return "master"
    if role in {"false", "none", "disabled", "disable", ""}:
        return "off"

    return role


def _camera_start_priority(settings: CameraSettings) -> int:
    """Start trigger-waiting cameras before trigger-generating cameras.

    Priority:
      0: external/follower cameras, which should be armed first
      1: normal/free-run cameras
      2: master cameras, which may generate trigger pulses
    """
    role = _trigger_role_from_settings(settings)
    if role in {"external", "follower"}:
        return 0
    if role == "master":
        return 2
    return 1


class MultiCameraController(QObject):
    """Controller for managing multiple cameras simultaneously."""

    # Signals
    frame_ready = Signal(object)  # MultiFrameData (full cam FPS; inference only)
    # recording_frame_ready = Signal(
    #     str, object, float, object
    # )  # camera_id, frame, timestamp, timestamp_metadata (full cam FPS; for recording)
    display_ready = Signal(object)  # MultiFrameData for GUI display (throttled to GUI_MAX_DISPLAY_FPS)
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
        self._runtime_info: dict[str, dict] = {}
        self._frames: dict[str, np.ndarray] = {}
        self._timestamps: dict[str, float] = {}
        self._frame_lock = Lock()
        self._running = False
        self._recording_frame_emission_enabled: bool = False
        self._recording_sink = None
        self._started_cameras: set = set()
        self._camera_display_order: list[str] = []
        self._display_ids: dict[str, str] = {}  # camera_id -> display_id (for labeling)
        self._failed_cameras: dict[str, str] = {}  # camera_id -> error message
        self._expected_cameras: int = 0  # Number of cameras we're trying to start

        # GUI display max FPS (for throttling display updates when many cameras are active)
        self._gui_display_max_fps: float = GUI_MAX_DISPLAY_FPS
        self._gui_display_last_emit: float = 0.0
        # Performance logs
        self._timing_per_cam: dict[str, WorkerTimingStats] = {}

    def is_running(self) -> bool:
        """Check if any camera is currently running."""
        return self._running and len(self._started_cameras) > 0

    def get_active_count(self) -> int:
        """Get the number of active cameras."""
        return len(self._started_cameras)

    def _timing_for_camera(self, camera_id: str) -> WorkerTimingStats:
        timing = self._timing_per_cam.get(camera_id)
        if timing is None:
            timing = WorkerTimingStats(
                f"Controller {camera_id}",
                logger=LOGGER,
                log_interval=1.0,
                enabled=MULTI_CAMERA_WORKER_DO_LOG_TIMING,
            )
            self._timing_per_cam[camera_id] = timing
        return timing

    def set_recording_frame_do_emit(self, enabled: bool) -> None:
        self._recording_frame_emission_enabled = bool(enabled)
        for worker in list(self._workers.values()):
            worker.set_recording_enabled(enabled)

    def _should_emit_display_ready(self) -> bool:
        """Return True when the UI/display path should be updated.

        This only throttles display_ready. It must not throttle frame_ready,
        because frame_ready is used for full-rate consumers such as recording.
        """
        if self._gui_display_max_fps <= 0:
            return True

        now = time.perf_counter()
        min_interval = 1.0 / max(self._gui_display_max_fps, 1e-9)

        if now - self._gui_display_last_emit < min_interval:
            return False

        self._gui_display_last_emit = now
        return True

    def start(self, camera_settings: list[CameraSettings]) -> None:
        """Start multiple cameras."""
        if self._running:
            LOGGER.warning("Multi-camera controller already running")
            return

        active_settings_user_order = [s for s in camera_settings if s.enabled][: self.MAX_CAMERAS]
        if not active_settings_user_order:
            LOGGER.warning("No active cameras to start")
            return

        # Display/tile order follows the user-configured camera order.
        self._camera_display_order = [get_camera_id(s) for s in active_settings_user_order]

        # Startup order may differ for trigger safety:
        # followers/external first, master last.
        active_settings = sorted(active_settings_user_order, key=_camera_start_priority)
        if not active_settings:
            LOGGER.warning("No active cameras to start")
            return

        # Check for dupes
        seen = {}
        for s in active_settings:
            camera_id = get_camera_id(s)
            try:
                key = camera_identity_key(s)
            except Exception:
                LOGGER.exception(
                    "Failed to compute camera identity key for %s; falling back to camera_id",
                    camera_id,
                )
                key = camera_id

            if key in seen:
                self.initialization_failed.emit(
                    [
                        (
                            camera_id,
                            f"Duplicate camera configuration. Conflicts with {seen[key]}",
                        )
                    ]
                )
                return

            seen[key] = camera_id

        self._running = True
        self._recording_frame_emission_enabled = False
        self._frames.clear()
        self._timestamps.clear()
        self._started_cameras.clear()
        self._failed_cameras.clear()
        self._display_ids.clear()
        self._expected_cameras = len(active_settings)

        for settings in active_settings:
            self._start_camera(settings)

    def _start_camera(self, settings: CameraSettings) -> None:
        """Start a single camera."""
        settings_copy = copy.deepcopy(settings)
        cam_id = get_camera_id(settings_copy)
        display_id = get_display_id(settings_copy)

        if cam_id in self._workers:
            LOGGER.warning(f"Camera {cam_id} is already running, skipping start")
            return

        LOGGER.info(f"[MultiCameraController] Starting {cam_id} with settings: {settings_copy}")

        # Normalize and store the dataclass once
        self._settings[cam_id] = settings_copy
        self._display_ids[cam_id] = display_id
        dc = self._settings[cam_id]
        worker = SingleCameraWorker(cam_id, dc)
        worker.set_recording_sink(self._recording_sink)
        worker.set_recording_enabled(self._recording_frame_emission_enabled)
        thread = QThread()
        worker.moveToThread(thread)

        # Connections unchanged
        thread.started.connect(worker.run)
        worker.runtime_info.connect(self._on_camera_runtime_info)
        worker.frame_captured.connect(self._on_frame_captured)
        worker.started.connect(self._on_camera_started)
        worker.stopped.connect(self._on_camera_stopped)
        worker.error_occurred.connect(self._on_camera_error)

        self._workers[cam_id] = worker
        self._threads[cam_id] = thread
        thread.finished.connect(partial(self._cleanup_camera, cam_id))
        worker.stopped.connect(thread.quit)
        thread.start()

    def set_recording_sink(self, sink) -> None:
        self._recording_sink = sink
        for worker in list(self._workers.values()):
            worker.set_recording_sink(sink)

    def _cleanup_camera(self, camera_id: str) -> None:
        # remove stored frame data
        with self._frame_lock:
            self._frames.pop(camera_id, None)
            self._timestamps.pop(camera_id, None)

        worker = self._workers.pop(camera_id, None)
        thread = self._threads.pop(camera_id, None)
        self._settings.pop(camera_id, None)

        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    def stop(self, wait: bool = True) -> None:
        """Stop all cameras."""
        if not self._running:
            return

        self._running = False
        self._recording_frame_emission_enabled = False

        # Signal all workers to stop
        for worker in self._workers.values():
            worker.stop()

        # Wait for threads to finish
        if wait:
            still_running: list[str] = []
            for cam_id, thread in list(self._threads.items()):
                if thread is None:
                    self._cleanup_camera(cam_id)
                    continue
                if not thread.isRunning():
                    self._cleanup_camera(cam_id)
                    continue

                thread.quit()
                if thread.wait(QUIT_WAIT_MS):
                    self._cleanup_camera(cam_id)
                    continue  # Clean exit

                LOGGER.error(
                    "Camera thread %s did not quit within %dms; forcing terminate()",
                    cam_id,
                    QUIT_WAIT_MS,
                )

                thread.terminate()
                if thread.wait(TERMINATE_WAIT_MS):
                    self._cleanup_camera(cam_id)
                    continue  # Terminated successfully

                LOGGER.critical(
                    "Camera thread %s refused to terminate after terminate()+wait(%dms). "
                    "Keeping references to avoid use-after-free/segfaults. "
                    "Application restart may be required.",
                    cam_id,
                    TERMINATE_WAIT_MS,
                )
                still_running.append(cam_id)

            if still_running:
                self._started_cameras.clear()
                self._failed_cameras.clear()
                with self._frame_lock:
                    self._frames.clear()
                    self._timestamps.clear()
                self._expected_cameras = 0

                LOGGER.critical(
                    "Force stop failed for camera threads: %s. "
                    "Sending all_stopped to leave GUI in running state, "
                    "but background camera threads may still be active and interfere with future camera starts. "
                    "Application restart may be required.",
                    still_running,
                )
                self.all_stopped.emit()
                return

        self._timing_per_cam.clear()
        self._gui_display_last_emit = 0.0
        self._workers.clear()
        self._threads.clear()
        self._settings.clear()
        self._started_cameras.clear()
        self._failed_cameras.clear()
        self._camera_display_order.clear()
        self._display_ids.clear()
        with self._frame_lock:
            self._frames.clear()
            self._timestamps.clear()
        self._expected_cameras = 0

        self.all_stopped.emit()

    def _on_frame_captured(
        self, camera_id: str, frame: np.ndarray, timestamp: float, timestamp_metadata: object | None = None
    ) -> None:
        """Handle a frame from one camera."""
        timing = self._timing_for_camera(camera_id)
        frame_data: MultiFrameData | None = None

        with timing.measure("Multi.slot.total"):
            # self._settings.get(camera_id)

            # with timing.measure("Multi.apply_transforms"):
            #     if settings and settings.rotation:
            #         frame = MultiCameraController.apply_rotation(frame, settings.rotation)

            #     if settings:
            #         crop_region = settings.get_crop_region()
            #         if crop_region:
            #             frame = MultiCameraController.apply_crop(frame, crop_region)

            # if self._recording_frame_emission_enabled:
            #     with timing.measure("Multi.emit.recording_frame_ready"):
            #         self.recording_frame_ready.emit(camera_id, frame, timestamp)

            with self._frame_lock:
                with timing.measure("Multi.store_latest"):
                    self._frames[camera_id] = frame
                    self._timestamps[camera_id] = timestamp

                with timing.measure("Multi.build_ordered"):
                    ordered_frames: dict[str, np.ndarray] = {}
                    ordered_timestamps: dict[str, float] = {}

                    for cam_id in self._camera_display_order:
                        if cam_id in self._frames:
                            ordered_frames[cam_id] = self._frames[cam_id]
                        if cam_id in self._timestamps:
                            ordered_timestamps[cam_id] = self._timestamps[cam_id]

                    # Any unexpected/legacy IDs, appended deterministically.
                    for cam_id in self._frames:
                        if cam_id not in ordered_frames:
                            ordered_frames[cam_id] = self._frames[cam_id]
                    for cam_id in self._timestamps:
                        if cam_id not in ordered_timestamps:
                            ordered_timestamps[cam_id] = self._timestamps[cam_id]

                with timing.measure("Multi.construct_frame_data"):
                    frame_data = MultiFrameData(
                        frames=ordered_frames,
                        timestamps=ordered_timestamps,
                        source_camera_id=camera_id,
                        tiled_frame=None,
                        display_ids=dict(self._display_ids),
                    )

            if frame_data is not None:
                with timing.measure("Multi.emit.frame_ready"):
                    self.frame_ready.emit(frame_data)

                # GUI-only path: throttled display updates
                if self._should_emit_display_ready():
                    with timing.measure("Multi.emit.display_ready"):
                        self.display_ready.emit(frame_data)

        timing.note_frame()
        timing.maybe_log()

    def _on_camera_runtime_info(self, camera_id: str, info: object) -> None:
        if not isinstance(info, dict):
            return

        self._runtime_info[camera_id] = dict(info)

        actual_fps = info.get("actual_fps")
        LOGGER.info(
            "Camera %s runtime info: actual_fps=%s actual_resolution=%s pixel_format=%s output_format=%s",
            camera_id,
            actual_fps,
            info.get("actual_resolution"),
            info.get("actual_pixel_format"),
            info.get("actual_output_format"),
        )

    def actual_fps_by_camera_id(self) -> dict[str, float]:
        out: dict[str, float] = {}

        for camera_id, info in self._runtime_info.items():
            try:
                fps = float(info.get("actual_fps") or 0.0)
            except Exception:
                fps = 0.0

            if fps > 0.0:
                out[camera_id] = fps

        return out

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
        self._display_ids.pop(camera_id, None)
        self.camera_stopped.emit(camera_id)
        LOGGER.info(f"Camera {camera_id} stopped (was_started={was_started})")

        # Cleanup thread
        if camera_id in self._threads:
            thread = self._threads.get(camera_id)
            if thread is not None and thread.isRunning():
                thread.quit()

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
        if (
            not self._started_cameras
            and self._running
            and all(not t.isRunning() for t in self._threads.values() if t is not None)
        ):
            self._running = False
            self.all_stopped.emit()

    def _on_camera_error(self, camera_id: str, message: str) -> None:
        """Handle camera error event."""
        LOGGER.error(f"Camera {camera_id} error: {message}")
        # Track failed cameras (only if not already started - i.e., initialization failure)
        if camera_id not in self._started_cameras:
            self._failed_cameras[camera_id] = message
        self.camera_error.emit(camera_id, message)
