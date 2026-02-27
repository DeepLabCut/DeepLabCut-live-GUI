# dlclivegui/services/poet_processor.py
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal

from .dlc_processor import PoseResult

try:
    from poet_live import POET, PostProcess
    from poet_live.models.backbone import Backbone, Joiner
    from poet_live.models.position_encoding import PositionEmbeddingSine
    from poet_live.models.transformer import Transformer
except ImportError as e:
    raise ImportError(
        "POET imports failed. Ensure POET code is in yourPYTHONPATH and dependencies are installed."
    ) from e

logger = logging.getLogger(__name__)
ENABLE_PROFILING = True


# FIMXE @C-Achard Duplicated code - refactor to a shared module
@dataclass
class ProcessorStats:
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


class POETProcessor(QObject):
    """Background pose estimation using POET with queue-based threading."""

    pose_ready = Signal(object)
    error = Signal(str)
    initialized = Signal(bool)
    frame_processed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._checkpoint_path: str | None = None
        self._device: str = "cuda"
        self._threshold: float = 0.7
        self._use_amp: bool = True

        self._model: Any | None = None
        self._post: Any | None = None

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

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @staticmethod
    def _resolve_torch_device(device_str: str | None) -> torch.device:
        """
        Accepts 'auto', 'cuda', 'cuda:0', 'cpu', etc.
        Falls back safely if requested device is unavailable.
        """
        if not device_str:
            device_str = "auto"

        d = device_str.strip().lower()

        if d in {"auto", "best"}:
            if torch.cuda.is_available():
                return torch.device("cuda")
            # optional: support Apple MPS if you ever run macOS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        # If user explicitly requests cuda but it's not available, fall back
        if d.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("Requested device '%s' but CUDA is not available; falling back to CPU.", device_str)
            return torch.device("cpu")

        # Otherwise let torch parse it (cpu, cuda:0, mps, etc.)
        return torch.device(device_str)

    def configure(self, checkpoint_path: str, *, device: str = "cuda", threshold: float = 0.7, use_amp: bool = True):
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._threshold = threshold
        self._use_amp = use_amp

    def reset(self) -> None:
        self._stop_worker()
        self._model = None
        self._post = None
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

    def shutdown(self) -> None:
        self.reset()

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
            avg_inference_time = (
                sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0.0
            )
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
                avg_inference_time=avg_inference_time,
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
            name="POETWorker",
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

    def _build_model(self):
        if not self._checkpoint_path:
            raise RuntimeError("No POET checkpoint configured.")
        device = self._resolve_torch_device(self._device)

        hidden_dim = 256
        backbone = Backbone(
            "resnet50", train_backbone=False, return_interm_layers=False, dilation5=False, dilation4=False
        )
        pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = Joiner(backbone, pos_enc)
        backbone_with_pos_enc.num_channels = backbone.num_channels
        transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
        model = POET(backbone_with_pos_enc, transformer, num_classes=2, num_queries=25, aux_loss=False)

        ckpt = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)

        model.to(device).eval()
        post = PostProcess().to(device)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        return model, post, device

    @torch.no_grad()
    def _infer(self, frame: np.ndarray, timestamp: float) -> np.ndarray | None:
        # OpenCV frames are BGR -> convert to RGB
        rgb = frame[..., ::-1].copy()
        h, w = rgb.shape[:2]

        device = next(self._model.parameters()).device  # type: ignore[union-attr]
        x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = (x - self._mean) / self._std

        use_amp = self._use_amp and device.type == "cuda"
        autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

        with autocast(enabled=use_amp):
            outputs = self._model(x)  # type: ignore[misc]

        target_sizes = torch.tensor([[h, w]], device=device)
        res = self._post(outputs, target_sizes=target_sizes)[0]  # type: ignore[misc]

        scores = res["scores"]  # (num_queries,)
        kpts = res["keypoints"]  # (num_queries, 51)

        keep = scores >= float(self._threshold)
        if keep.sum().item() == 0:
            return None

        scores = scores[keep]
        kpts = kpts[keep].reshape(-1, 17, 3)  # (N,17,3)

        # Replace visibility with detection confidence (same for all keypoints)
        kpts = kpts.clone()
        kpts[:, :, 2] = scores[:, None].clamp(0, 1)

        # Sort by score (stable ordering)
        order = torch.argsort(scores, descending=True)
        kpts = kpts[order]

        return kpts.detach().cpu().numpy().astype(np.float32)  # (N,17,3)

    def _process_frame(self, frame: np.ndarray, timestamp: float, enqueue_time: float, *, queue_wait_time: float):
        inf_start = time.perf_counter()
        pose = self._infer(frame, timestamp)
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
            self._model, self._post, _device = self._build_model()

            # Warmup
            _ = self._infer(init_frame, init_timestamp)

            self.initialized.emit(True)
            self._process_frame(init_frame, init_timestamp, time.perf_counter(), queue_wait_time=0.0)
            with self._stats_lock:
                self._frames_enqueued += 1
        except Exception as exc:
            logger.exception("Failed to initialize POET", exc_info=exc)
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
                            logger.exception("POET inference failed", exc_info=exc)
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
                logger.exception("POET inference failed", exc_info=exc)
                self.error.emit(str(exc))
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass
