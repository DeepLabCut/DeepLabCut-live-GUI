"""Workers and state logic for loading cameras in the GUI."""
# dlclivegui/gui/camera_loaders.py

import copy
import logging
from dataclasses import dataclass
from enum import Enum, auto

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QWidget

from ...cameras.base import CameraSettings
from ...cameras.factory import CameraBackend, CameraFactory

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class PreviewState(Enum):
    """Preview lifecycle state.."""

    IDLE = auto()  # No loader, no backend, no timer.
    LOADING = auto()  # Loader started; waiting for success/error/canceled.
    ACTIVE = auto()  # Backend open + preview timer running.
    STOPPING = auto()  # Tearing down loader/backend/timer.
    ERROR = auto()  # Terminal error state (optional; can just go back to IDLE)


# -------------------------------
# Background worker to detect cameras
# -------------------------------
class DetectCamerasWorker(QThread):
    """Background worker to detect cameras for the selected backend."""

    progress = Signal(str)  # human-readable text
    result = Signal(list)  # list[DetectedCamera]
    error = Signal(str)
    finished = Signal()

    def __init__(self, backend: str, max_devices: int = 10, parent: QWidget | None = None):
        super().__init__(parent)
        self.backend = backend
        self.max_devices = max_devices

    def run(self):
        try:
            # Initial message
            self.progress.emit(f"Scanning {self.backend} cameras…")

            cams = CameraFactory.detect_cameras(
                self.backend,
                max_devices=self.max_devices,
                should_cancel=self.isInterruptionRequested,
                progress_cb=self.progress.emit,
            )
            self.result.emit(cams)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class CameraProbeWorker(QThread):
    """Request a quick device probe (open/close) without starting preview."""

    progress = Signal(str)
    success = Signal(object)  # emits CameraSettings
    error = Signal(str)
    finished = Signal()

    def __init__(self, cam: CameraSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self._cam = copy.deepcopy(cam)
        self._cancel = False

        # Enable fast_start when supported (backend reads namespace options)
        if isinstance(self._cam.properties, dict):
            ns = self._cam.properties.setdefault(self._cam.backend.lower(), {})
            if isinstance(ns, dict):
                ns.setdefault("fast_start", True)

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.progress.emit("Probing device defaults…")
            if self._cancel:
                return
            self.success.emit(self._cam)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


# -------------------------------
# Singleton camera preview loader worker
# -------------------------------
class CameraLoadWorker(QThread):
    """Open/configure a camera backend off the UI thread with progress and cancel support."""

    progress = Signal(str)  # Human-readable status updates
    success = Signal(object)  # Emits the ready backend (CameraBackend)
    error = Signal(str)  # Emits error message
    canceled = Signal()  # Emits when canceled before success

    def __init__(self, cam: CameraSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self._cam = copy.deepcopy(cam)

        self._cancel = False
        self._backend: CameraBackend | None = None

        # Do not use fast_start here as we want to actually open the camera to probe capabilities
        # If you want a quick probe without full open, use CameraProbeWorker instead which sets fast_start=True
        # Ensure preview open never uses fast_start probe mode
        if isinstance(self._cam.properties, dict):
            ns = self._cam.properties.setdefault(self._cam.backend.lower(), {})
            if isinstance(ns, dict):
                ns["fast_start"] = False

    def request_cancel(self):
        self._cancel = True

    def _check_cancel(self) -> bool:
        if self._cancel:
            self.progress.emit("Canceled by user.")
            return True
        return False

    def run(self):
        try:
            self.progress.emit("Creating backend…")
            if self._check_cancel():
                self.canceled.emit()
                return

            LOGGER.debug("Creating camera backend for %s:%d", self._cam.backend, self._cam.index)
            self.progress.emit("Opening device…")
            # Open only in GUI thread to avoid simultaneous opens
            self.success.emit(self._cam)

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            try:
                if self._backend:
                    self._backend.close()
            except Exception:
                pass
            self.error.emit(msg)


@dataclass
class PreviewSession:
    """
    Owns all runtime objects for preview and defines intent.

    epoch:
      Monotonically increasing integer used to invalidate stale signals from previous loaders.
      Any signal handler must check that the epoch matches the current session epoch.

    state:
      PreviewState that replaces multiple booleans.

    requested_cam:
      The CameraSettings snapshot used to start the current LOADING request.

    backend / timer / loader:
      Runtime handles. Only valid in states where they should exist.
    """

    epoch: int = 0
    state: PreviewState = PreviewState.IDLE
    requested_cam: CameraSettings | None = None
    loader: CameraLoadWorker | None = None
    backend: CameraBackend | None = None
    timer: QTimer | None = None

    pending_restart: CameraSettings | None = None
    restart_scheduled: bool = False  # Coalesces restarts to “at most once in the queue”.
