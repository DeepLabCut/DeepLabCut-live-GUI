from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)
ENABLE_PROFILING = True


@dataclass
class PoseResult:
    pose: np.ndarray | None
    timestamp: float


@dataclass
class ProcessorStats:
    frames_enqueued: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    queue_size: int = 0
    processing_fps: float = 0.0
    average_latency: float = 0.0
    last_latency: float = 0.0
    avg_queue_wait: float = 0.0
    avg_inference_time: float = 0.0
    avg_signal_emit_time: float = 0.0
    avg_total_process_time: float = 0.0


class PoseProcessor(QObject):
    """
    Background pose estimation using a pluggable backend.

    backend_factory: () -> backend implementing init_inference() and get_pose()
    """

    pose_ready = Signal(object)
    error = Signal(str)
    initialized = Signal(bool)
    frame_processed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._backend_factory: Callable[[], Any] | None = None
        self._backend: Any | None = None

        self._queue: queue.Queue[Any] | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._frames_enqueued = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._latencies: deque[float] = deque(maxlen=60)
        self._processing_times: deque[float] = deque(maxlen=60)
        self._queue_wait_times: deque[float] = deque(maxlen=60)
        self._inference_times: deque[float] = deque(maxlen=60)
        self._signal_emit_times: deque[float] = deque(maxlen=60)
        self._total_process_times: deque[float] = deque(maxlen=60)
        self._stats_lock = threading.Lock()

    def configure(self, backend_factory: Callable[[], Any]) -> None:
        self._backend_factory = backend_factory

    def reset(self) -> None:
        self._stop_worker()
        self._backend = None
        with self._stats_lock:
            self._frames_enqueued = 0
            self._frames_processed = 0
            self._frames_dropped = 0
            self._latencies.clear()
            self._processing_times.clear()
            self._queue_wait_times.clear()
            self._inference_times.clear()
            self._signal_emit_times.clear()
            self._total_process_times.clear()

    def enqueue_frame(self, frame: np.ndarray, timestamp: float) -> None:
        if self._worker_thread is None:
            self._start_worker(frame.copy(), timestamp)
            return

        if self._queue is None:
            return

        try:
            self._queue.put_nowait((frame.copy(), timestamp, time.perf_counter()))
            with self._stats_lock:
                self._frames_enqueued += 1
        except queue.Full:
            with self._stats_lock:
                self._frames_dropped += 1

    def get_stats(self) -> ProcessorStats:
        queue_size = self._queue.qsize() if self._queue is not None else 0
        with self._stats_lock:
            avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
            last_latency = self._latencies[-1] if self._latencies else 0.0

            if len(self._processing_times) >= 2:
                duration = self._processing_times[-1] - self._processing_times[0]
                processing_fps = (len(self._processing_times) - 1) / duration if duration > 0 else 0.0
            else:
                processing_fps = 0.0

            avg_queue_wait = (
                sum(self._queue_wait_times) / len(self._queue_wait_times) if self._queue_wait_times else 0.0
            )
            avg_inference = sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0.0
            avg_signal_emit = (
                sum(self._signal_emit_times) / len(self._signal_emit_times) if self._signal_emit_times else 0.0
            )
            avg_total = (
                sum(self._total_process_times) / len(self._total_process_times) if self._total_process_times else 0.0
            )

            return ProcessorStats(
                frames_enqueued=self._frames_enqueued,
                frames_processed=self._frames_processed,
                frames_dropped=self._frames_dropped,
                queue_size=queue_size,
                processing_fps=processing_fps,
                average_latency=avg_latency,
                last_latency=last_latency,
                avg_queue_wait=avg_queue_wait,
                avg_inference_time=avg_inference,
                avg_signal_emit_time=avg_signal_emit,
                avg_total_process_time=avg_total,
            )

    def _start_worker(self, init_frame: np.ndarray, init_timestamp: float) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._queue = queue.Queue(maxsize=1)
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(init_frame, init_timestamp),
            name="PoseWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_worker(self) -> None:
        if self._worker_thread is None:
            return
        self._stop_event.set()
        self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self._queue = None
        if self._backend is not None and hasattr(self._backend, "close"):
            try:
                self._backend.close()
            except Exception:
                pass

    def _process_frame(
        self, frame: np.ndarray, timestamp: float, enqueue_time: float, *, queue_wait_time: float
    ) -> None:
        inf_start = time.perf_counter()
        pose = self._backend.get_pose(frame, frame_time=timestamp)
        if self._processor is not None:
            pose = self._processor.process(pose)
        inf_time = time.perf_counter() - inf_start

        sig_start = time.perf_counter()
        self.pose_ready.emit(PoseResult(pose=pose, timestamp=timestamp))
        sig_time = time.perf_counter() - sig_start

        end = time.perf_counter()
        latency = end - enqueue_time
        total = end - enqueue_time

        with self._stats_lock:
            self._frames_processed += 1
            self._latencies.append(latency)
            self._processing_times.append(end)
            if ENABLE_PROFILING:
                self._queue_wait_times.append(queue_wait_time)
                self._inference_times.append(inf_time)
                self._signal_emit_times.append(sig_time)
                self._total_process_times.append(total)

        self.frame_processed.emit()

    def _worker_loop(self, init_frame: np.ndarray, init_timestamp: float) -> None:
        try:
            if self._backend_factory is None:
                raise RuntimeError("No backend configured.")
            self._backend = self._backend_factory()

            self._backend.init_inference(init_frame)
            self.initialized.emit(True)

            self._process_frame(init_frame, init_timestamp, time.perf_counter(), queue_wait_time=0.0)
            with self._stats_lock:
                self._frames_enqueued += 1

        except Exception as exc:
            logger.exception("Failed to initialize pose backend", exc_info=exc)
            self.error.emit(str(exc))
            self.initialized.emit(False)
            return

        while True:
            if self._stop_event.is_set():
                if self._queue is not None:
                    try:
                        frame, ts, enq = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        try:
                            self._process_frame(frame, ts, enq, queue_wait_time=0.0)
                        except Exception as exc:
                            logger.exception("Pose inference failed", exc_info=exc)
                            self.error.emit(str(exc))
                        finally:
                            try:
                                self._queue.task_done()
                            except ValueError:
                                pass
                        continue

            try:
                wait_start = time.perf_counter()
                frame, ts, enq = self._queue.get(timeout=0.05)
                qwait = time.perf_counter() - wait_start
            except queue.Empty:
                continue

            try:
                self._process_frame(frame, ts, enq, queue_wait_time=qwait)
            except Exception as exc:
                logger.exception("Pose inference failed", exc_info=exc)
                self.error.emit(str(exc))
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

        logger.info("Pose worker thread exiting")
