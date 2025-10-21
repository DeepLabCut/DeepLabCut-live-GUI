"""DLCLive integration helpers."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .config import DLCProcessorSettings

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from dlclive import DLCLive  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    DLCLive = None  # type: ignore[assignment]


@dataclass
class PoseResult:
    pose: Optional[np.ndarray]
    timestamp: float


class DLCLiveProcessor(QObject):
    """Background pose estimation using DLCLive."""

    pose_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    initialized = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._settings = DLCProcessorSettings()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._dlc: Optional[DLCLive] = None
        self._init_future: Optional[Future[Any]] = None
        self._pending: Optional[Future[Any]] = None
        self._lock = threading.Lock()

    def configure(self, settings: DLCProcessorSettings) -> None:
        self._settings = settings

    def shutdown(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
            if self._init_future is not None:
                self._init_future.cancel()
                self._init_future = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._dlc = None

    def enqueue_frame(self, frame: np.ndarray, timestamp: float) -> None:
        with self._lock:
            if self._dlc is None and self._init_future is None:
                self._init_future = self._executor.submit(
                    self._initialise_model, frame.copy(), timestamp
                )
                self._init_future.add_done_callback(self._on_initialised)
                return
            if self._dlc is None:
                return
            if self._pending is not None and not self._pending.done():
                return
            self._pending = self._executor.submit(
                self._run_inference, frame.copy(), timestamp
            )
            self._pending.add_done_callback(self._on_pose_ready)

    def _initialise_model(self, frame: np.ndarray, timestamp: float) -> bool:
        if DLCLive is None:
            raise RuntimeError(
                "The 'dlclive' package is required for pose estimation. Install it to enable DLCLive support."
            )
        if not self._settings.model_path:
            raise RuntimeError("No DLCLive model path configured.")
        options = {
            "model_path": self._settings.model_path,
            "processor": self._settings.processor,
        }
        options.update(self._settings.additional_options)
        if self._settings.shuffle is not None:
            options["shuffle"] = self._settings.shuffle
        if self._settings.trainingsetindex is not None:
            options["trainingsetindex"] = self._settings.trainingsetindex
        if self._settings.processor_args:
            options["processor_config"] = {
                "object": self._settings.processor,
                **self._settings.processor_args,
            }
        model = DLCLive(**options)
        model.init_inference(frame, frame_time=timestamp, record=False)
        self._dlc = model
        return True

    def _on_initialised(self, future: Future[Any]) -> None:
        try:
            result = future.result()
            self.initialized.emit(bool(result))
        except Exception as exc:  # pragma: no cover - runtime behaviour
            LOGGER.exception("Failed to initialise DLCLive", exc_info=exc)
            self.error.emit(str(exc))

    def _run_inference(self, frame: np.ndarray, timestamp: float) -> PoseResult:
        if self._dlc is None:
            raise RuntimeError("DLCLive model not initialised")
        pose = self._dlc.get_pose(frame, frame_time=timestamp, record=False)
        return PoseResult(pose=pose, timestamp=timestamp)

    def _on_pose_ready(self, future: Future[Any]) -> None:
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - runtime behaviour
            LOGGER.exception("Pose inference failed", exc_info=exc)
            self.error.emit(str(exc))
            return
        self.pose_ready.emit(result)
