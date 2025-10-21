"""Camera management for the DLC Live GUI."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, QMetaObject, Qt, pyqtSignal, pyqtSlot

from .cameras import CameraFactory
from .cameras.base import CameraBackend
from .config import CameraSettings


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
    finished = pyqtSignal()

    def __init__(self, settings: CameraSettings):
        super().__init__()
        self._settings = settings
        self._stop_event = Event()
        self._backend: Optional[CameraBackend] = None

    @pyqtSlot()
    def run(self) -> None:
        self._stop_event.clear()
        try:
            self._backend = CameraFactory.create(self._settings)
            self._backend.open()
        except Exception as exc:  # pragma: no cover - device specific
            self.error_occurred.emit(str(exc))
            self.finished.emit()
            return

        self.started.emit(self._settings)

        while not self._stop_event.is_set():
            try:
                frame, timestamp = self._backend.read()
            except Exception as exc:  # pragma: no cover - device specific
                self.error_occurred.emit(str(exc))
                break
            self.frame_captured.emit(FrameData(frame, timestamp))

        if self._backend is not None:
            try:
                self._backend.close()
            except Exception as exc:  # pragma: no cover - device specific
                self.error_occurred.emit(str(exc))
            self._backend = None
        self.finished.emit()

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
