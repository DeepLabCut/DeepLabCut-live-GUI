"""DLCLive integration helpers."""

# dlclivegui/services/dlc_processor.py
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal

# from dlclivegui.config import DLCProcessorSettings
from dlclivegui.processors.processor_utils import instantiate_from_scan
from dlclivegui.utils.config_models import DLCProcessorSettingsModel

logger = logging.getLogger(__name__)

# Enable profiling
ENABLE_PROFILING = True

try:  # pragma: no cover - optional dependency
    from dlclive import DLCLive  # type: ignore
except Exception as e:  # pragma: no cover - handled gracefully
    logger.error(f"dlclive package could not be imported: {e}")
    DLCLive = None  # type: ignore[assignment]


def ensure_dc_dlc(settings: DLCProcessorSettingsModel) -> DLCProcessorSettingsModel:
    if isinstance(settings, DLCProcessorSettingsModel):
        settings = DLCProcessorSettingsModel.model_validate(settings)
        data = settings.model_dump()
        dyn = data.get("dynamic")
        # Convert DynamicCropModel -> tuple expected by dataclass
        if hasattr(dyn, "enabled"):
            data["dynamic"] = (dyn.enabled, dyn.margin, dyn.max_missing_frames)
        elif isinstance(dyn, dict) and {"enabled", "margin", "max_missing_frames"} <= set(dyn):
            data["dynamic"] = (dyn["enabled"], dyn["margin"], dyn["max_missing_frames"])
        return DLCProcessorSettingsModel(**data)
    raise TypeError("Unsupported DLC settings type")


@dataclass
class PoseResult:
    pose: np.ndarray | None
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
    # Profiling metrics
    avg_queue_wait: float = 0.0
    avg_inference_time: float = 0.0
    avg_signal_emit_time: float = 0.0
    avg_total_process_time: float = 0.0
    # Separated timing for GPU vs socket processor
    avg_gpu_inference_time: float = 0.0  # Pure model inference
    avg_processor_overhead: float = 0.0  # Socket processor overhead


_SENTINEL = object()


class DLCLiveProcessor(QObject):
    """Background pose estimation using DLCLive with queue-based threading."""

    pose_ready = Signal(object)
    error = Signal(str)
    initialized = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._settings = DLCProcessorSettingsModel()
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

    def configure(self, settings: DLCProcessorSettingsModel, processor: Any | None = None) -> None:
        self._settings = ensure_dc_dlc(settings)
        self._processor = processor

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
        if not self._initialized and self._worker_thread is None:
            # Start worker thread with initialization
            self._start_worker(frame.copy(), timestamp)
            return

        # Don't count dropped frames until processor is initialized
        if not self._initialized:
            return

        if self._queue is not None:
            try:
                # Non-blocking put - drop frame if queue is full
                self._queue.put_nowait((frame.copy(), timestamp, time.perf_counter()))
                with self._stats_lock:
                    self._frames_enqueued += 1
            except queue.Full:
                logger.debug("DLC queue full, dropping frame")
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
            return

        self._stop_event.set()
        if self._queue is not None:
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass

        self._worker_thread.join(timeout=2.0)
        if self._worker_thread.is_alive():
            logger.warning("DLC worker thread did not terminate cleanly")

        self._worker_thread = None
        self._queue = None

    def _worker_loop(self, init_frame: np.ndarray, init_timestamp: float) -> None:
        try:
            # Initialize model
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
            # Add device if specified in settings
            if self._settings.device is not None:
                # FIXME @C-Achard make sure this is ok for tf
                # maybe add smth in utils or config to validate device strings
                options["device"] = self._settings.device
            self._dlc = DLCLive(**options)

            init_inference_start = time.perf_counter()
            self._dlc.init_inference(init_frame)
            init_inference_time = time.perf_counter() - init_inference_start

            self._initialized = True
            self.initialized.emit(True)

            total_init_time = time.perf_counter() - init_start
            logger.info(
                f"DLCLive model initialized successfully "
                f"(total: {total_init_time:.3f}s, init_inference: {init_inference_time:.3f}s)"
            )

            # Process the initialization frame
            enqueue_time = time.perf_counter()

            inference_start = time.perf_counter()
            pose = self._dlc.get_pose(init_frame, frame_time=init_timestamp)
            inference_time = time.perf_counter() - inference_start

            signal_start = time.perf_counter()
            self.pose_ready.emit(PoseResult(pose=pose, timestamp=init_timestamp))
            signal_time = time.perf_counter() - signal_start

            process_time = time.perf_counter()

            with self._stats_lock:
                self._frames_enqueued += 1
                self._frames_processed += 1
                self._processing_times.append(process_time)
                if ENABLE_PROFILING:
                    self._inference_times.append(inference_time)
                    self._signal_emit_times.append(signal_time)

        except Exception as exc:
            logger.exception("Failed to initialize DLCLive", exc_info=exc)
            self.error.emit(str(exc))
            self.initialized.emit(False)
            return

        # Main processing loop
        frame_count = 0
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            # Time spent waiting for queue
            queue_wait_start = time.perf_counter()
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            queue_wait_time = time.perf_counter() - queue_wait_start

            if item is _SENTINEL:
                break

            frame, timestamp, enqueue_time = item

            try:
                # Time the inference - we need to separate GPU from processor overhead
                # If processor exists, wrap its process method to time it separately
                processor_overhead_time = 0.0
                gpu_inference_time = 0.0

                original_process = None  # bind for finally safety

                if self._processor is not None:
                    # Wrap processor.process() to time it
                    original_process = self._processor.process
                    processor_time_holder = [0.0]  # Use list to allow modification in nested scope

                    # Bind original_process and holder into defaults to satisfy flake8-bugbear B023
                    def timed_process(pose, _op=original_process, _holder=processor_time_holder, **kwargs):
                        proc_start = time.perf_counter()
                        try:
                            return _op(pose, **kwargs)
                        finally:
                            _holder[0] = time.perf_counter() - proc_start

                    self._processor.process = timed_process

                try:
                    inference_start = time.perf_counter()
                    pose = self._dlc.get_pose(frame, frame_time=timestamp)
                    inference_time = time.perf_counter() - inference_start
                finally:
                    # Always restore the original process method if we wrapped it
                    if original_process is not None and self._processor is not None:
                        self._processor.process = original_process

                if original_process is not None:
                    processor_overhead_time = processor_time_holder[0]
                    gpu_inference_time = inference_time - processor_overhead_time
                else:
                    # No processor, all time is GPU inference
                    gpu_inference_time = inference_time

                # Time the signal emission
                signal_start = time.perf_counter()
                self.pose_ready.emit(PoseResult(pose=pose, timestamp=timestamp))
                signal_time = time.perf_counter() - signal_start

                end_process = time.perf_counter()
                total_process_time = end_process - loop_start
                latency = end_process - enqueue_time

                with self._stats_lock:
                    self._frames_processed += 1
                    self._latencies.append(latency)
                    self._processing_times.append(end_process)

                    if ENABLE_PROFILING:
                        self._queue_wait_times.append(queue_wait_time)
                        self._inference_times.append(inference_time)
                        self._signal_emit_times.append(signal_time)
                        self._total_process_times.append(total_process_time)
                        self._gpu_inference_times.append(gpu_inference_time)
                        self._processor_overhead_times.append(processor_overhead_time)

                # Log profiling every 100 frames
                frame_count += 1
                if ENABLE_PROFILING and frame_count % 100 == 0:
                    logger.info(
                        f"[Profile] Frame {frame_count}: "
                        f"queue_wait={queue_wait_time * 1000:.2f}ms, "
                        f"inference={inference_time * 1000:.2f}ms "
                        f"(GPU={gpu_inference_time * 1000:.2f}ms, processor={processor_overhead_time * 1000:.2f}ms), "
                        f"signal_emit={signal_time * 1000:.2f}ms, "
                        f"total={total_process_time * 1000:.2f}ms, "
                        f"latency={latency * 1000:.2f}ms"
                    )

            except Exception as exc:
                logger.exception("Pose inference failed", exc_info=exc)
                self.error.emit(str(exc))
            finally:
                if item is not _SENTINEL:
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

    def configure(self, settings: DLCProcessorSettingsModel, scanned_processors: dict, selected_key) -> bool:
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
