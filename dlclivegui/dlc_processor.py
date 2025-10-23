"""DLCLive integration helpers."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from dlclivegui.config import DLCProcessorSettings

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from dlclive import DLCLive  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    DLCLive = None  # type: ignore[assignment]


@dataclass
class PoseResult:
    pose: Optional[np.ndarray]
    timestamp: float


@dataclass
class ProcessorStats:
    """Statistics for DLC processor performance."""
    frames_enqueued: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    queue_size: int = 0
    processing_fps: float = 0.0
    average_latency: float = 0.0
    last_latency: float = 0.0


_SENTINEL = object()


class DLCLiveProcessor(QObject):
    """Background pose estimation using DLCLive with queue-based threading."""

    pose_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    initialized = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._settings = DLCProcessorSettings()
        self._dlc: Optional[Any] = None
        self._queue: Optional[queue.Queue[Any]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._initialized = False
        
        # Statistics tracking
        self._frames_enqueued = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._latencies: deque[float] = deque(maxlen=60)
        self._processing_times: deque[float] = deque(maxlen=60)
        self._stats_lock = threading.Lock()

    def configure(self, settings: DLCProcessorSettings) -> None:
        self._settings = settings

    def reset(self) -> None:
        """Stop the worker thread and drop the current DLCLive instance."""
        self._stop_worker()
        self._dlc = None
        self._initialized = False
        with self._stats_lock:
            self._frames_enqueued = 0
            self._frames_processed = 0
            self._frames_dropped = 0
            self._latencies.clear()
            self._processing_times.clear()

    def shutdown(self) -> None:
        self._stop_worker()
        self._dlc = None
        self._initialized = False

    def enqueue_frame(self, frame: np.ndarray, timestamp: float) -> None:
        if not self._initialized and self._worker_thread is None:
            # Start worker thread with initialization
            self._start_worker(frame.copy(), timestamp)
            return
        
        if self._queue is not None:
            try:
                # Non-blocking put - drop frame if queue is full
                self._queue.put_nowait((frame.copy(), timestamp, time.perf_counter()))
                with self._stats_lock:
                    self._frames_enqueued += 1
            except queue.Full:
                LOGGER.debug("DLC queue full, dropping frame")
                with self._stats_lock:
                    self._frames_dropped += 1

    def get_stats(self) -> ProcessorStats:
        """Get current processing statistics."""
        queue_size = self._queue.qsize() if self._queue is not None else 0
        
        with self._stats_lock:
            avg_latency = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies
                else 0.0
            )
            last_latency = self._latencies[-1] if self._latencies else 0.0
            
            # Compute processing FPS from processing times
            if len(self._processing_times) >= 2:
                duration = self._processing_times[-1] - self._processing_times[0]
                processing_fps = (len(self._processing_times) - 1) / duration if duration > 0 else 0.0
            else:
                processing_fps = 0.0
            
            return ProcessorStats(
                frames_enqueued=self._frames_enqueued,
                frames_processed=self._frames_processed,
                frames_dropped=self._frames_dropped,
                queue_size=queue_size,
                processing_fps=processing_fps,
                average_latency=avg_latency,
                last_latency=last_latency,
            )

    def _start_worker(self, init_frame: np.ndarray, init_timestamp: float) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        
        self._queue = queue.Queue(maxsize=5)
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(init_frame, init_timestamp),
            name="DLCLiveWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_worker(self) -> None:
        if self._worker_thread is None:
            return
        
        self._stop_event.set()
        if self._queue is not None:
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass
        
        self._worker_thread.join(timeout=2.0)
        if self._worker_thread.is_alive():
            LOGGER.warning("DLC worker thread did not terminate cleanly")
        
        self._worker_thread = None
        self._queue = None

    def _worker_loop(self, init_frame: np.ndarray, init_timestamp: float) -> None:
        try:
            # Initialize model
            if DLCLive is None:
                raise RuntimeError(
                    "The 'dlclive' package is required for pose estimation."
                )
            if not self._settings.model_path:
                raise RuntimeError("No DLCLive model path configured.")
            
            options = {
                "model_path": self._settings.model_path,
                "model_type": self._settings.model_type,
                "processor": None,
                "dynamic": [False,0.5,10],
                "resize": 1.0,
            }
            self._dlc = DLCLive(**options)
            self._dlc.init_inference(init_frame)
            self._initialized = True
            self.initialized.emit(True)
            LOGGER.info("DLCLive model initialized successfully")
            
            # Process the initialization frame
            enqueue_time = time.perf_counter()
            pose = self._dlc.get_pose(init_frame, frame_time=init_timestamp)
            process_time = time.perf_counter()
            
            with self._stats_lock:
                self._frames_enqueued += 1
                self._frames_processed += 1
                self._processing_times.append(process_time)
            
            self.pose_ready.emit(PoseResult(pose=pose, timestamp=init_timestamp))
            
        except Exception as exc:
            LOGGER.exception("Failed to initialize DLCLive", exc_info=exc)
            self.error.emit(str(exc))
            self.initialized.emit(False)
            return
        
        # Main processing loop
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if item is _SENTINEL:
                break
            
            frame, timestamp, enqueue_time = item
            try:
                start_process = time.perf_counter()
                pose = self._dlc.get_pose(frame, frame_time=timestamp)
                end_process = time.perf_counter()
                
                latency = end_process - enqueue_time
                
                with self._stats_lock:
                    self._frames_processed += 1
                    self._latencies.append(latency)
                    self._processing_times.append(end_process)
                
                self.pose_ready.emit(PoseResult(pose=pose, timestamp=timestamp))
            except Exception as exc:
                LOGGER.exception("Pose inference failed", exc_info=exc)
                self.error.emit(str(exc))
            finally:
                self._queue.task_done()
        
        LOGGER.info("DLC worker thread exiting")
