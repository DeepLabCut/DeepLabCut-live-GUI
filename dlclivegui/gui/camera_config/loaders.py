"""Workers and state logic for loading cameras in the GUI."""

# dlclivegui/gui/loaders.py
from __future__ import annotations

import copy
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from ...cameras.factory import CameraBackend, CameraFactory
from ...config import CameraSettings

if TYPE_CHECKING:
    pass  # only for typing

LOGGER = logging.getLogger(__name__)


class CameraScanState(Enum):
    IDLE = auto()
    RUNNING = auto()
    CANCELING = auto()
    DONE = auto()


# -------------------------------
# Background worker to detect cameras
# -------------------------------
class DetectCamerasWorker(QThread):
    """Background worker to detect cameras for the selected backend.

    Signals:
      - progress(str): human-readable status
      - result(list): list of DetectedCamera (may be empty)
      - error(str): error message (on exception)
      - canceled(): emitted if interruption was requested during/after discovery
    """

    progress = Signal(str)
    result = Signal(list)  # list[DetectedCamera] at runtime
    error = Signal(str)
    canceled = Signal()

    def __init__(self, backend: str, max_devices: int = 10, parent: QWidget | None = None):
        super().__init__(parent)
        self.backend = backend
        self.max_devices = max_devices

    def run(self) -> None:
        try:
            self.progress.emit(f"Scanning {self.backend} cameras…")

            cams = CameraFactory.detect_cameras(
                self.backend,
                max_devices=self.max_devices,
                should_cancel=self.isInterruptionRequested,
                progress_cb=self.progress.emit,
            )

            # Always emit result (even if empty) so UI can stabilize deterministically.
            self.result.emit(cams or [])

            # If canceled, emit canceled so UI can set ScanState.CANCELING/DONE if desired.
            if self.isInterruptionRequested():
                self.canceled.emit()

        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        # No custom finished signal: QThread.finished is emitted automatically when run() returns.


class CameraProbeWorker(QThread):
    """Request a quick device probe (open/close) without starting preview."""

    progress = Signal(str)
    success = Signal(object)  # emits CameraSettings
    error = Signal(str)

    def __init__(self, cam: CameraSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self._cam = copy.deepcopy(cam)
        self._cancel = False

        # Enable fast_start when supported (backend reads namespace options)
        if isinstance(self._cam.properties, dict):
            ns = self._cam.properties.setdefault(self._cam.backend.lower(), {})
            if isinstance(ns, dict):
                ns.setdefault("fast_start", True)

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.progress.emit("Probing device defaults…")
            if self._cancel:
                return
            self.success.emit(self._cam)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        # QThread.finished will fire automatically.


# -------------------------------
# Singleton camera preview loader worker
# -------------------------------
class CameraLoadWorker(QThread):
    """Open/configure a camera backend off the UI thread with progress and cancel support."""

    progress = Signal(str)
    success = Signal(object)  # emits CameraSettings for GUI-thread open
    error = Signal(str)
    canceled = Signal()

    def __init__(self, cam: CameraSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self._cam = copy.deepcopy(cam)
        self._cancel = False
        self._backend: CameraBackend | None = None

        # Ensure preview open never uses fast_start probe mode
        if isinstance(self._cam.properties, dict):
            ns = self._cam.properties.setdefault(self._cam.backend.lower(), {})
            if isinstance(ns, dict):
                ns["fast_start"] = False

    def request_cancel(self) -> None:
        self._cancel = True

    def _check_cancel(self) -> bool:
        if self._cancel:
            self.progress.emit("Canceled by user.")
            return True
        return False

    def run(self) -> None:
        try:
            self.progress.emit("Creating backend…")
            if self._check_cancel():
                self.canceled.emit()
                return

            LOGGER.debug("Preparing camera open for %s:%d", self._cam.backend, self._cam.index)
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
