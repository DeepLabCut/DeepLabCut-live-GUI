"""DLCLive integration helpers."""

# dlclivegui/services/dlc_processor.py
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from dlclive import DLCLive
from PySide6.QtCore import QObject, Signal

from dlclivegui.config import DLCProcessorSettings, ModelType
from dlclivegui.processors.processor_utils import instantiate_from_scan
from dlclivegui.temp import Engine  # type: ignore # TODO use main package enum when released

logger = logging.getLogger(__name__)

# Enable profiling to get more detailed timing metrics for debugging and optimization.
ENABLE_PROFILING = True


class PoseBackends(Enum):
    DLC_LIVE = auto()


@dataclass
class PoseResult:
    pose: np.ndarray | None
    timestamp: float
    packet: PosePacket | None = None


@dataclass(slots=True, frozen=True)
class PoseSource:
    backend: PoseBackends  # e.g. "DLCLive"
    model_type: ModelType | None = None


@dataclass(slots=True, frozen=True)
class PosePacket:
    schema_version: int = 0
    keypoints: np.ndarray | None = None
    keypoint_names: list[str] | None = None
    individual_ids: list[str] | None = None
    source: PoseSource = PoseSource(backend=PoseBackends.DLC_LIVE)
    raw: Any | None = None


def validate_pose_array(pose: Any, *, source_backend: PoseBackends = PoseBackends.DLC_LIVE) -> np.ndarray:
    """
    Validate pose output shape and dtype.

    Accepted runner output shapes:
    - (K, 3): single-animal
    - (N, K, 3): multi-animal
    """
    try:
        arr = np.asarray(pose)
    except Exception as exc:
        raise ValueError(
            f"{source_backend} returned an invalid pose output format: could not convert to array ({exc})"
        ) from exc

    if arr.ndim not in (2, 3):
        raise ValueError(
            f"{source_backend} returned an invalid pose output format:"
            f" expected a 2D or 3D array, got ndim={arr.ndim}, shape={arr.shape!r}"
        )

    if arr.shape[-1] != 3:
        raise ValueError(
            f"{source_backend} returned an invalid pose output format:"
            f" expected last dimension size 3 (x, y, likelihood), got shape={arr.shape!r}"
        )

    if arr.ndim == 2 and arr.shape[0] <= 0:
        raise ValueError(f"{source_backend} returned an invalid pose output format: expected at least one keypoint")
    if arr.ndim == 3 and (arr.shape[0] <= 0 or arr.shape[1] <= 0):
        raise ValueError(
            f"{source_backend} returned an invalid pose output format:"
            f" expected at least one individual and one keypoint, got shape={arr.shape!r}"
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(
            f"{source_backend} returned an invalid pose output format: expected numeric values, got dtype={arr.dtype}"
        )

    return arr


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
    # Profiling metrics
    avg_queue_wait: float = 0.0
    avg_inference_time: float = 0.0
    avg_signal_emit_time: float = 0.0
    avg_total_process_time: float = 0.0
    # Separated timing for GPU vs socket processor
    avg_gpu_inference_time: float = 0.0  # Pure model inference
    avg_processor_overhead: float = 0.0  # Socket processor overhead


class DLCLiveProcessor(QObject):
    """Background pose estimation using DLCLive with queue-based threading."""

    pose_ready = Signal(object)
    error = Signal(str)
    initialized = Signal(bool)
    frame_processed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._settings = DLCProcessorSettings()
        self._dlc: Any | None = None
        self._processor: Any | None = None
        self._queue: queue.Queue[Any] | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._initialized = False

        # Statistics tracking
        self._frames_enqueued = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._latencies: deque[float] = deque(maxlen=60)
        self._processing_times: deque[float] = deque(maxlen=60)
        self._stats_lock = threading.Lock()

        # Profiling metrics
        self._queue_wait_times: deque[float] = deque(maxlen=60)
        self._inference_times: deque[float] = deque(maxlen=60)
        self._signal_emit_times: deque[float] = deque(maxlen=60)
        self._total_process_times: deque[float] = deque(maxlen=60)
        self._gpu_inference_times: deque[float] = deque(maxlen=60)
        self._processor_overhead_times: deque[float] = deque(maxlen=60)

    @staticmethod
    def get_model_backend(model_path: str) -> Engine:
        return Engine.from_model_path(model_path)

    def configure(self, settings: DLCProcessorSettings, processor: Any | None = None) -> None:
        self._settings = settings
        self._processor = processor

    def reset(self) -> None:
        """Stop the worker thread and drop the current DLCLive instance."""
        stopped = self._stop_worker()
        if not stopped:
            logger.warning(
                "Reset requested but worker thread is still alive; skipping DLCLive reset to avoid potential issues."
            )
            return
        self._dlc = None
        self._initialized = False
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
            self._gpu_inference_times.clear()
            self._processor_overhead_times.clear()

    def shutdown(self) -> None:
        self._stop_worker()
        self._dlc = None
        self._initialized = False

    def enqueue_frame(self, frame: np.ndarray, timestamp: float) -> None:
        # Start worker on first frame
        if self._worker_thread is None:
            self._start_worker(frame.copy(), timestamp)
            return

        # As long as worker and queue are ready, ALWAYS enqueue
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
        """Get current processing statistics."""
        queue_size = self._queue.qsize() if self._queue is not None else 0

        with self._stats_lock:
            avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
            last_latency = self._latencies[-1] if self._latencies else 0.0

            # Compute processing FPS from processing times
            if len(self._processing_times) >= 2:
                duration = self._processing_times[-1] - self._processing_times[0]
                processing_fps = (len(self._processing_times) - 1) / duration if duration > 0 else 0.0
            else:
                processing_fps = 0.0

            # Profiling metrics
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
            avg_gpu = (
                sum(self._gpu_inference_times) / len(self._gpu_inference_times) if self._gpu_inference_times else 0.0
            )
            avg_proc_overhead = (
                sum(self._processor_overhead_times) / len(self._processor_overhead_times)
                if self._processor_overhead_times
                else 0.0
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
                avg_gpu_inference_time=avg_gpu,
                avg_processor_overhead=avg_proc_overhead,
            )

    def _start_worker(self, init_frame: np.ndarray, init_timestamp: float) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._queue = queue.Queue(maxsize=1)
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
            return True

        self._stop_event.set()

        # Wait for timed get() loop to observe the flag and drain
        self._worker_thread.join(timeout=2.0)
        if self._worker_thread.is_alive():
            logger.warning("DLC worker thread did not terminate cleanly")
            # IMPORTANT: do not clear references; thread may still be using them
            return False

        self._worker_thread = None
        self._queue = None
        return True

    @contextmanager
    def _timed_processor(self):
        """
        If a socket processor is attached, temporarily wrap its .process()
        to measure processor overhead time independently of GPU inference.
        Yields a one-element list [processor_overhead_seconds] or None when no processor.
        Always restores the original .process reference.
        """
        if self._processor is None:
            yield None
            return

        original = self._processor.process
        holder = [0.0]

        def timed_process(pose, _op=original, _holder=holder, **kwargs):
            start = time.perf_counter()
            try:
                return _op(pose, **kwargs)
            finally:
                _holder[0] = time.perf_counter() - start

        self._processor.process = timed_process
        try:
            yield holder
        finally:
            # Restore even if inference/errors occur
            self._processor.process = original

    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        enqueue_time: float,
        *,
        queue_wait_time: float = 0.0,
    ) -> None:
        """
        Single source of truth for: inference -> (optional) processor timing -> signal emit -> stats.
        Updates: frames_processed, latency, processing timeline, profiling metrics.
        """
        # Time GPU inference (and processor overhead when present)
        with self._timed_processor() as proc_holder:
            inference_start = time.perf_counter()
            raw_pose: Any = self._dlc.get_pose(frame, frame_time=timestamp)
            inference_time = time.perf_counter() - inference_start
        pose_arr: np.ndarray = validate_pose_array(raw_pose, source_backend=PoseBackends.DLC_LIVE)
        pose_packet = PosePacket(
            schema_version=0,
            keypoints=pose_arr,
            keypoint_names=None,
            individual_ids=None,
            source=PoseSource(backend=PoseBackends.DLC_LIVE, model_type=self._settings.model_type),
            raw=raw_pose,
        )

        processor_overhead = 0.0
        gpu_inference_time = inference_time
        if proc_holder is not None:
            processor_overhead = proc_holder[0]
            gpu_inference_time = max(0.0, inference_time - processor_overhead)

        # Emit pose (measure signal overhead)
        signal_start = time.perf_counter()
        self.pose_ready.emit(PoseResult(pose=pose_packet.keypoints, timestamp=timestamp, packet=pose_packet))
        signal_time = time.perf_counter() - signal_start

        end_ts = time.perf_counter()
        latency = end_ts - enqueue_time
        # service_time_no_queue = signal_time + inference_time (includes processor overhead when present)
        # Actual end-to-end time from enqueue to signal emit
        total_process_time = end_ts - enqueue_time

        with self._stats_lock:
            self._frames_processed += 1
            self._latencies.append(latency)
            self._processing_times.append(end_ts)
            if ENABLE_PROFILING:
                self._queue_wait_times.append(queue_wait_time)
                self._inference_times.append(inference_time)
                self._signal_emit_times.append(signal_time)
                self._total_process_times.append(total_process_time)
                self._gpu_inference_times.append(gpu_inference_time)
                self._processor_overhead_times.append(processor_overhead)

        self.frame_processed.emit()

    def _worker_loop(self, init_frame: np.ndarray, init_timestamp: float) -> None:
        try:
            # -------- Initialization (unchanged) --------
            if DLCLive is None:
                raise RuntimeError("The 'dlclive' package is required for pose estimation.")
            if not self._settings.model_path:
                raise RuntimeError("No DLCLive model path configured.")

            init_start = time.perf_counter()
            dyn = self._settings.dynamic
            if not isinstance(dyn, (list, tuple)) or len(dyn) != 3:
                try:
                    dyn = dyn.to_tuple()
                except Exception as e:
                    raise RuntimeError("Invalid dynamic crop settings format.") from e
            enabled, margin, max_missing = dyn

            options = {
                "model_path": self._settings.model_path,
                "model_type": self._settings.model_type,
                "processor": self._processor,
                "dynamic": [enabled, margin, max_missing],
                "resize": self._settings.resize,
                "precision": self._settings.precision,
                "single_animal": self._settings.single_animal,
            }
            if self._settings.device is not None:
                options["device"] = self._settings.device

            self._dlc = DLCLive(**options)

            # First inference to initialize
            init_inference_start = time.perf_counter()
            self._dlc.init_inference(init_frame)
            init_inference_time = time.perf_counter() - init_inference_start

            # Pass DLCLive cfg to processor if available
            if hasattr(self._dlc, "processor") and hasattr(self._dlc.processor, "set_dlc_cfg"):
                self._dlc.processor.set_dlc_cfg(getattr(self._dlc, "cfg", None))

            self._initialized = True
            self.initialized.emit(True)

            total_init_time = time.perf_counter() - init_start
            logger.info(
                "DLCLive model initialized successfully (total: %.3fs, init_inference: %.3fs)",
                total_init_time,
                init_inference_time,
            )

            # Emit pose for init frame & update stats (not dequeued)
            self._process_frame(init_frame, init_timestamp, time.perf_counter(), queue_wait_time=0.0)
            with self._stats_lock:
                self._frames_enqueued += 1

        except Exception as exc:
            logger.exception("Failed to initialize DLCLive", exc_info=exc)
            self.error.emit(str(exc))
            self.initialized.emit(False)
            return

        # -------- Main processing loop: stop-flag + timed get + drain --------
        # NOTE: We never exit early unless _stop_event is set.
        while True:
            # If stop requested, only exit when queue is empty
            if self._stop_event.is_set():
                if self._queue is not None:
                    try:
                        frame, ts, enq = self._queue.get_nowait()
                    except queue.Empty:
                        # NOW it is safe to exit
                        break
                    else:
                        # Still work to do, process one
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
                        continue  # check stop_event again WITHOUT breaking

            # Normal operation: timed get
            try:
                wait_start = time.perf_counter()
                item = self._queue.get(timeout=0.05)
                queue_wait_time = time.perf_counter() - wait_start
            except queue.Empty:
                continue

            try:
                frame, ts, enq = item
                self._process_frame(frame, ts, enq, queue_wait_time=queue_wait_time)
            except Exception as exc:
                logger.exception("Pose inference failed", exc_info=exc)
                self.error.emit(str(exc))
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

        logger.info("DLC worker thread exiting")


class DLCService:
    """Wrap DLCLiveProcessor lifecycle & configuration."""

    def __init__(self):
        self._proc = DLCLiveProcessor()
        self.active = False
        self._last_pose: PoseResult | None = None
        self._processor_info = None

    @property
    def processor(self):
        return self._proc._processor

    # Expose key signals (to let MainWindow connect easily)
    @property
    def pose_ready(self):
        return self._proc.pose_ready

    @property
    def error(self):
        return self._proc.error

    @property
    def initialized(self):
        return self._proc.initialized

    def enqueue(self, frame, ts):
        self._proc.enqueue_frame(frame, ts)

    def configure(self, settings: DLCProcessorSettings, scanned_processors: dict, selected_key) -> bool:
        processor = None
        if selected_key is not None and scanned_processors:
            try:
                processor = instantiate_from_scan(scanned_processors, selected_key)
            except Exception as exc:
                logger.error("Failed to instantiate processor: %s", exc)
                return False
        self._proc.configure(settings, processor=processor)
        return True

    def start(self):
        self._proc.reset()
        self.active = True
        self.initialized = False

    def stop(self):
        self.active = False
        self.initialized = False
        self._proc.reset()
        self._last_pose = None

    def stats(self) -> ProcessorStats:
        return self._proc.get_stats()

    def last_pose(self) -> PoseResult | None:
        return self._last_pose
