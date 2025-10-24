"""Video recording support using the vidgear library."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from vidgear.gears import WriteGear
except ImportError:  # pragma: no cover - handled at runtime
    WriteGear = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class RecorderStats:
    """Snapshot of recorder throughput metrics."""

    frames_enqueued: int
    frames_written: int
    dropped_frames: int
    queue_size: int
    average_latency: float
    last_latency: float
    write_fps: float
    buffer_seconds: float


_SENTINEL = object()


class VideoRecorder:
    """Thin wrapper around :class:`vidgear.gears.WriteGear`."""

    def __init__(
        self,
        output: Path | str,
        frame_size: Optional[Tuple[int, int]] = None,
        frame_rate: Optional[float] = None,
        codec: str = "libx264",
        crf: int = 23,
        buffer_size: int = 240,
    ):
        self._output = Path(output)
        self._writer: Optional[Any] = None
        self._frame_size = frame_size
        self._frame_rate = frame_rate
        self._codec = codec
        self._crf = int(crf)
        self._buffer_size = max(1, int(buffer_size))
        self._queue: Optional[queue.Queue[Any]] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._frames_enqueued = 0
        self._frames_written = 0
        self._dropped_frames = 0
        self._total_latency = 0.0
        self._last_latency = 0.0
        self._written_times: deque[float] = deque(maxlen=600)
        self._encode_error: Optional[Exception] = None
        self._last_log_time = 0.0
        self._frame_timestamps: List[float] = []

    @property
    def is_running(self) -> bool:
        return self._writer_thread is not None and self._writer_thread.is_alive()

    def start(self) -> None:
        if WriteGear is None:
            raise RuntimeError(
                "vidgear is required for video recording. Install it with 'pip install vidgear'."
            )
        if self._writer is not None:
            return
        fps_value = float(self._frame_rate) if self._frame_rate else 30.0

        writer_kwargs: Dict[str, Any] = {
            "compression_mode": True,
            "logging": False,
            "-input_framerate": fps_value,
            "-vcodec": (self._codec or "libx264").strip() or "libx264",
            "-crf": int(self._crf),
        }
        # TODO deal with pixel format

        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._writer = WriteGear(output=str(self._output), **writer_kwargs)
        self._queue = queue.Queue(maxsize=self._buffer_size)
        self._frames_enqueued = 0
        self._frames_written = 0
        self._dropped_frames = 0
        self._total_latency = 0.0
        self._last_latency = 0.0
        self._written_times.clear()
        self._frame_timestamps.clear()
        self._encode_error = None
        self._stop_event.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="VideoRecorderWriter",
            daemon=True,
        )
        self._writer_thread.start()

    def configure_stream(self, frame_size: Tuple[int, int], frame_rate: Optional[float]) -> None:
        self._frame_size = frame_size
        self._frame_rate = frame_rate

    def write(self, frame: np.ndarray, timestamp: Optional[float] = None) -> bool:
        if not self.is_running or self._queue is None:
            return False
        error = self._current_error()
        if error is not None:
            raise RuntimeError(f"Video encoding failed: {error}") from error

        # Record timestamp for this frame
        if timestamp is None:
            timestamp = time.time()
        with self._stats_lock:
            self._frame_timestamps.append(timestamp)

        # Convert frame to uint8 if needed
        if frame.dtype != np.uint8:
            frame_float = frame.astype(np.float32, copy=False)
            max_val = float(frame_float.max()) if frame_float.size else 0.0
            scale = 1.0
            if max_val > 0:
                scale = 255.0 / max_val if max_val > 255.0 else (255.0 if max_val <= 1.0 else 1.0)
            frame = np.clip(frame_float * scale, 0.0, 255.0).astype(np.uint8)

        # Convert grayscale to RGB if needed
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)

        # Ensure contiguous array
        frame = np.ascontiguousarray(frame)

        # Check if frame size matches expected size
        if self._frame_size is not None:
            expected_h, expected_w = self._frame_size
            actual_h, actual_w = frame.shape[:2]
            if (actual_h, actual_w) != (expected_h, expected_w):
                logger.warning(
                    f"Frame size mismatch: expected (h={expected_h}, w={expected_w}), "
                    f"got (h={actual_h}, w={actual_w}). "
                    "Stopping recorder to prevent encoding errors."
                )
                # Set error to stop recording gracefully
                with self._stats_lock:
                    self._encode_error = ValueError(
                        f"Frame size changed from (h={expected_h}, w={expected_w}) to (h={actual_h}, w={actual_w})"
                    )
                return False

        try:
            assert self._queue is not None
            self._queue.put(frame, block=False)
        except queue.Full:
            with self._stats_lock:
                self._dropped_frames += 1
            queue_size = self._queue.qsize() if self._queue is not None else -1
            logger.warning(
                "Video recorder queue full; dropping frame. queue=%d buffer=%d",
                queue_size,
                self._buffer_size,
            )
            return False
        with self._stats_lock:
            self._frames_enqueued += 1
        return True

    def stop(self) -> None:
        if self._writer is None and not self.is_running:
            return
        self._stop_event.set()
        if self._queue is not None:
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                self._queue.put(_SENTINEL)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
            if self._writer_thread.is_alive():
                logger.warning("Video recorder thread did not terminate cleanly")
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.exception("Failed to close WriteGear cleanly")

        # Save timestamps to JSON file
        self._save_timestamps()

        self._writer = None
        self._writer_thread = None
        self._queue = None

    def get_stats(self) -> Optional[RecorderStats]:
        if (
            self._writer is None
            and not self.is_running
            and self._queue is None
            and self._frames_enqueued == 0
            and self._frames_written == 0
            and self._dropped_frames == 0
        ):
            return None
        queue_size = self._queue.qsize() if self._queue is not None else 0
        with self._stats_lock:
            frames_enqueued = self._frames_enqueued
            frames_written = self._frames_written
            dropped = self._dropped_frames
            avg_latency = (
                self._total_latency / self._frames_written if self._frames_written else 0.0
            )
            last_latency = self._last_latency
            write_fps = self._compute_write_fps_locked()
        buffer_seconds = queue_size * avg_latency if avg_latency > 0 else 0.0
        return RecorderStats(
            frames_enqueued=frames_enqueued,
            frames_written=frames_written,
            dropped_frames=dropped,
            queue_size=queue_size,
            average_latency=avg_latency,
            last_latency=last_latency,
            write_fps=write_fps,
            buffer_seconds=buffer_seconds,
        )

    def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if item is _SENTINEL:
                self._queue.task_done()
                break
            frame = item
            start = time.perf_counter()
            try:
                assert self._writer is not None
                self._writer.write(frame)
            except OSError as exc:
                with self._stats_lock:
                    self._encode_error = exc
                logger.exception("Video encoding failed while writing frame")
                self._queue.task_done()
                self._stop_event.set()
                break
            elapsed = time.perf_counter() - start
            now = time.perf_counter()
            with self._stats_lock:
                self._frames_written += 1
                self._total_latency += elapsed
                self._last_latency = elapsed
                self._written_times.append(now)
                if now - self._last_log_time >= 1.0:
                    fps = self._compute_write_fps_locked()
                    queue_size = self._queue.qsize()
                    # logger.info(
                    #     "Recorder throughput: %.2f fps, latency %.2f ms, queue=%d",
                    #     fps,
                    #     elapsed * 1000.0,
                    #     queue_size,
                    # )
                    self._last_log_time = now
            self._queue.task_done()
        self._finalize_writer()

    def _finalize_writer(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                logger.exception("Failed to close WriteGear during finalisation")

    def _compute_write_fps_locked(self) -> float:
        if len(self._written_times) < 2:
            return 0.0
        duration = self._written_times[-1] - self._written_times[0]
        if duration <= 0:
            return 0.0
        return (len(self._written_times) - 1) / duration

    def _current_error(self) -> Optional[Exception]:
        with self._stats_lock:
            return self._encode_error

    def _save_timestamps(self) -> None:
        """Save frame timestamps to a JSON file alongside the video."""
        if not self._frame_timestamps:
            logger.info("No timestamps to save")
            return

        # Create timestamps file path
        timestamp_file = self._output.with_suffix("").with_suffix(
            self._output.suffix + "_timestamps.json"
        )

        try:
            with self._stats_lock:
                timestamps = self._frame_timestamps.copy()

            # Prepare metadata
            data = {
                "video_file": str(self._output.name),
                "num_frames": len(timestamps),
                "timestamps": timestamps,
                "start_time": timestamps[0] if timestamps else None,
                "end_time": timestamps[-1] if timestamps else None,
                "duration_seconds": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
            }

            # Write to JSON
            with open(timestamp_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(timestamps)} frame timestamps to {timestamp_file}")
        except Exception as exc:
            logger.exception(f"Failed to save timestamps to {timestamp_file}: {exc}")
