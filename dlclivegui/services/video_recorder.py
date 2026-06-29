"""Video recording support using the vidgear library."""

# dlclivegui/services/video_recorder.py
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from dlclivegui.config import REC_DO_LOG_TIMING
from dlclivegui.utils.stats import RecorderStats, WorkerTimingStats

try:
    from vidgear.gears import WriteGear
except ImportError:  # pragma: no cover - handled at runtime
    WriteGear = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

STOP_JOIN_TIMEOUT = 5.0  # seconds


_SENTINEL = object()


class VideoRecorder:
    """Asynchronous video recorder backed by VidGear/FFmpeg.

    `VideoRecorder` wraps VidGear's `WriteGear` writer with a bounded in-memory
    queue and a dedicated writer thread. Calls to `write()` perform minimal frame
    validation/preprocessing, enqueue accepted frames without blocking, and return
    immediately. The writer thread consumes queued frames and writes them to disk,
    while also recording timestamps for successfully written frames.

    The recorder is intended for high-throughput camera pipelines where frame
    acquisition should not block on video encoding. If the internal queue fills,
    incoming frames are dropped and counted in recorder statistics. Timestamp
    sidecar files are written on `stop()` for frames that were actually written.

    Args:
        output: Output video path.
        frame_size: Expected frame size as `(height, width)`. If provided,
            incoming frames with different dimensions are rejected and the
            recorder enters an error state.
        frame_rate: Output video frame rate. If missing or non-positive, the
            recorder falls back to 30 FPS and logs a warning.
        codec: FFmpeg video codec name passed to WriteGear, for example
            `"libx264"`.
        crf: Constant Rate Factor passed to compatible FFmpeg encoders. Lower
            values generally increase quality and file size.
        buffer_size: Maximum number of frames that may wait in the recorder
            queue before new frames are dropped.
        convert_grayscale_to_rgb: Whether 2D grayscale frames should be expanded
            to 3-channel RGB before writing. Set to `False` to preserve mono
            frames when supported by the chosen writer/codec path.
        writer_options: Optional dictionary of additional keyword arguments passed
            to `WriteGear`. If provided, this overrides the default options.

    Attributes:
        is_running: Whether the writer thread is currently alive.

    Raises:
        RuntimeError: If VidGear is unavailable, if the recorder is abandoned
            after a failed stop, or if a previous encoding error is detected
            during `write()`.

    Notes:
        This class does not guarantee that every submitted frame is written.
        Frames may be dropped when the queue is full, and timestamps are only
        saved for frames successfully consumed by the writer thread.
    """

    def __init__(
        self,
        output: Path | str,
        frame_size: tuple[int, int] | None = None,
        frame_rate: float | None = None,
        codec: str = "libx264",
        crf: int = 23,
        buffer_size: int = 240,
        convert_grayscale_to_rgb: bool = True,
        writer_options: dict[str, Any] | None = None,
    ):
        # Config
        self._output = Path(output)
        self._writer: Any | None = None
        self._frame_size = frame_size
        self._frame_rate = frame_rate
        self._hardware_timestamp_source: dict[str, Any] | None = None
        self._codec = codec
        self._crf = int(crf)
        self._buffer_size = max(1, int(buffer_size))
        self._convert_grayscale_to_rgb = bool(convert_grayscale_to_rgb)
        self._writer_options = dict(writer_options) if writer_options is not None else None
        # Worker state
        self._queue: queue.Queue[Any] | None = None
        self._writer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._abandoned = False
        # Stats
        self._frames_enqueued = 0
        self._frames_written = 0
        self._dropped_frames = 0
        self._total_latency = 0.0
        self._last_latency = 0.0
        self._written_times: deque[float] = deque(maxlen=600)
        self._encode_error: Exception | None = None
        self._last_log_time = 0.0
        self._frame_timestamps: list[dict[str, Any]] = []
        # Timing
        self._process_timing = WorkerTimingStats(
            f"RecorderProcess[{self._output.name}]", logger=logger, log_interval=1.0, enabled=REC_DO_LOG_TIMING
        )
        self._writer_timing = WorkerTimingStats(
            f"RecorderWriter[{self._output.name}]", logger=logger, log_interval=1.0, enabled=REC_DO_LOG_TIMING
        )
        self._logged_first_frame = False

    @property
    def is_running(self) -> bool:
        return self._writer_thread is not None and self._writer_thread.is_alive()

    def start(self) -> None:
        if WriteGear is None:
            raise RuntimeError("vidgear is required for video recording. Install it with 'pip install vidgear'.")

        with self._lifecycle_lock:
            if self._abandoned:
                raise RuntimeError("Cannot restart VideoRecorder, as a leftover thread is still running.")
            if self.is_running:
                return
            if self._writer is not None:
                # Best-effort cleanup of a stale writer to avoid leaking resources.
                logger.warning(
                    "VideoRecorder.start() found an existing writer while not running; "
                    "attempting to close the stale writer before restarting."
                )
                try:
                    close_method = getattr(self._writer, "close", None)
                    if callable(close_method):
                        close_method()
                except Exception:
                    logger.exception("Error while closing stale video writer in start().")
                finally:
                    self._writer = None
                    self._queue = None
                    self._writer_thread = None

            if self._frame_rate and float(self._frame_rate) > 0.0:
                fps_value = float(self._frame_rate)
            else:
                fps_value = 30.0
                logger.warning(
                    "VideoRecorder frame_rate missing/zero for %s; falling back to %.3f FPS. "
                    "Video playback duration may not match capture timestamps.",
                    self._output.name,
                    fps_value,
                )

            logger.info(
                "Starting VideoRecorder output=%s frame_size=%s frame_rate=%.3f "
                "codec=%s crf=%s buffer_size=%s convert_grayscale_to_rgb=%s writer_options=%s",
                self._output,
                self._frame_size,
                fps_value,
                self._codec,
                self._crf,
                self._buffer_size,
                self._convert_grayscale_to_rgb,
                self._writer_options,
            )

            codec_value = (self._codec or "libx264").strip() or "libx264"
            writer_kwargs: dict[str, Any] = {
                "compression_mode": True,
                "logging": False,
                "-input_framerate": fps_value,
                "-vcodec": codec_value,
                "-crf": int(self._crf),
            }

            if self._writer_options is not None:
                writer_kwargs.update(self._writer_options)

            # if not self._convert_grayscale_to_rgb:
            #     writer_kwargs.update(
            #         {
            #             "-pix_fmt": "yuv420p",
            #         }
            #     )
            #     if self._frame_size is not None:
            #         h, w = self._frame_size
            #         writer_kwargs["-output_dimensions"] = (int(w), int(h))

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
            self._hardware_timestamp_source = None
            self._encode_error = None
            self._stop_event.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="VideoRecorderWriter",
                daemon=True,
            )
            self._writer_thread.start()

    def configure_stream(self, frame_size: tuple[int, int], frame_rate: float | None) -> None:
        self._frame_size = frame_size
        self._frame_rate = frame_rate

    def write(
        self, frame: np.ndarray, timestamp: float | None = None, timestamp_metadata: object | None = None
    ) -> bool:
        error = self._current_error()
        if error is not None:
            raise RuntimeError(f"Video encoding failed: {error}") from error

        q = self._queue
        if not self.is_running or q is None:
            return False
        if self._stop_event.is_set():
            return False

        # Capture timestamp now, but only record it if frame is successfully enqueued
        if timestamp is None:
            timestamp = time.time()

        with self._process_timing.measure("Recorder.preprocess"):
            # Convert frame to uint8 if needed
            if frame.dtype != np.uint8:
                frame_float = frame.astype(np.float32, copy=False)
                max_val = float(frame_float.max()) if frame_float.size else 0.0
                scale = 1.0
                if max_val > 0:
                    scale = 255.0 / max_val if max_val > 255.0 else (255.0 if max_val <= 1.0 else 1.0)
                frame = np.clip(frame_float * scale, 0.0, 255.0).astype(np.uint8)

            # Convert grayscale to RGB if needed
            if self._convert_grayscale_to_rgb and frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)

            # Ensure contiguous array
            frame = np.ascontiguousarray(frame)

            if not self._logged_first_frame:
                self._logged_first_frame = True
                logger.info(
                    "Recorder %s first frame: shape=%s dtype=%s "
                    "contiguous=%s nbytes=%.2f MB convert_grayscale_to_rgb=%s",
                    self._output.name,
                    frame.shape,
                    frame.dtype,
                    frame.flags.c_contiguous,
                    frame.nbytes / (1024 * 1024),
                    self._convert_grayscale_to_rgb,
                )

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
                    with self._stats_lock:
                        self._encode_error = ValueError(
                            f"Frame size changed from (h={expected_h}, w={expected_w}) to (h={actual_h}, w={actual_w})"
                        )
                    self._process_timing.note_error()
                    self._process_timing.maybe_log()
                    return False

        try:
            with self._process_timing.measure("Recorder.queue_put"):
                q.put((frame, timestamp, timestamp_metadata), block=False)
        except queue.Full:
            with self._stats_lock:
                self._dropped_frames += 1
            queue_size = q.qsize()
            logger.warning(
                "Video recorder queue full; dropping frame. queue=%d buffer=%d",
                queue_size,
                self._buffer_size,
            )
            self._process_timing.note_error()
            self._process_timing.maybe_log()
            return False

        with self._stats_lock:
            self._frames_enqueued += 1

        self._process_timing.note_frame()
        self._process_timing.maybe_log()

        return True

    def stop(self) -> None:
        with self._lifecycle_lock:
            already_stopped = (self._writer is None) and (not self.is_running)
            if already_stopped:
                # If the recorder was previously marked as abandoned because the
                # writer thread did not stop in time, but the thread has since
                # exited, perform cleanup so the recorder can become fully stopped
                # and restartable.
                t = self._writer_thread
                if self._abandoned and (t is None or not t.is_alive()):
                    self._writer_thread = None
                    self._queue = None
                    self._stop_event.clear()
                    self._abandoned = False
                return

            self._stop_event.set()
            q = self._queue
            t = self._writer_thread

        if q is not None:
            try:
                q.put_nowait(_SENTINEL)
            except queue.Full:
                pass

        if t is not None:
            t.join(timeout=STOP_JOIN_TIMEOUT)
            if t.is_alive():
                with self._stats_lock:
                    self._encode_error = RuntimeError(
                        "Failed to stop VideoRecorder within timeout; thread is still alive."
                    )

                with self._lifecycle_lock:
                    self._abandoned = True

                self._save_timestamps()

                logger.critical(
                    "Failed to stop VideoRecorder within timeout; thread is still alive. "
                    "Marking recorder as abandoned to prevent restart. "
                    "Timestamps were saved, but may be incomplete."
                )
                return

        self._save_timestamps()

        with self._lifecycle_lock:
            self._writer = None
            self._writer_thread = None
            self._queue = None
            self._stop_event.clear()
            self._abandoned = False

    def get_stats(self) -> RecorderStats | None:
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
            avg_latency = self._total_latency / self._frames_written if self._frames_written else 0.0
            last_latency = self._last_latency
            write_fps = self._compute_write_fps_locked()

        if write_fps > 0:
            buffer_seconds = queue_size / write_fps
        elif avg_latency > 0:
            buffer_seconds = queue_size * avg_latency
        elif last_latency > 0:
            buffer_seconds = queue_size * last_latency
        else:
            buffer_seconds = 0.0
        return RecorderStats(
            frames_enqueued=frames_enqueued,
            frames_written=frames_written,
            dropped_frames=dropped,
            queue_size=queue_size,
            buffer_size=self._buffer_size,
            average_latency=avg_latency,
            last_latency=last_latency,
            write_fps=write_fps,
            buffer_seconds=buffer_seconds,
        )

    def _writer_loop(self) -> None:
        q = self._queue
        if q is None:
            with self._stats_lock:
                self._encode_error = RuntimeError("Writer loop started without a queue")
            logger.error("Writer loop started without a queue; exiting")
            return

        try:
            while True:
                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                except Exception as exc:
                    with self._stats_lock:
                        self._encode_error = exc
                    logger.exception("Could not retrieve item from queue", exc_info=exc)
                    self._stop_event.set()
                    break

                try:
                    if item is _SENTINEL:
                        break
                    else:
                        frame, timestamp, timestamp_metadata = item
                        start = time.perf_counter()

                        try:
                            writer = self._writer
                            if writer is None:
                                raise RuntimeError("WriteGear writer is not initialized")

                            with self._writer_timing.measure("Recorder.writer_write"):
                                writer.write(frame)

                            record: dict[str, Any] = {
                                "frame_index": self._frames_written,
                                "software_timestamp": float(timestamp),
                            }

                            if timestamp_metadata is not None:
                                if (
                                    hasattr(timestamp_metadata, "to_source_dict")
                                    and self._hardware_timestamp_source is None
                                ):
                                    self._hardware_timestamp_source = timestamp_metadata.to_source_dict()

                                if hasattr(timestamp_metadata, "to_frame_dict"):
                                    record["hardware_timestamp"] = timestamp_metadata.to_frame_dict()
                                    default_value = timestamp_metadata.get_default_reported()
                                    if default_value is not None:
                                        record["hardware_timestamp_default"] = default_value
                                elif isinstance(timestamp_metadata, dict):
                                    record["hardware_timestamp"] = dict(timestamp_metadata)
                                else:
                                    record["hardware_timestamp"] = repr(timestamp_metadata)

                            self._frame_timestamps.append(record)

                        except Exception as exc:
                            with self._stats_lock:
                                self._encode_error = exc
                            logger.exception("Video encoding failed while writing frame", exc_info=exc)
                            self._stop_event.set()
                            self._writer_timing.note_error()
                            self._writer_timing.maybe_log()
                            break
                        else:
                            elapsed = time.perf_counter() - start
                            now = time.perf_counter()
                            with self._stats_lock:
                                self._frames_written += 1
                                self._total_latency += elapsed
                                self._last_latency = elapsed
                                self._written_times.append(now)
                                if now - self._last_log_time >= 1.0:
                                    self._compute_write_fps_locked()
                                    self._last_log_time = now

                            self._writer_timing.note_frame()
                            self._writer_timing.maybe_log()

                finally:
                    # Ensure queue accounting is correct for every item pulled from q
                    try:
                        q.task_done()
                    except ValueError:
                        logger.warning("Queue task_done() called too many times in writer loop")
                        pass

        finally:
            self._finalize_writer()

    def _finalize_writer(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.close()
                time.sleep(0.2)  # give some time to finalize
            except Exception:
                logger.exception("Failed to close WriteGear during finalisation")

    def _compute_write_fps_locked(self) -> float:
        if len(self._written_times) < 2:
            return 0.0
        duration = self._written_times[-1] - self._written_times[0]
        if duration <= 0:
            return 0.0
        return (len(self._written_times) - 1) / duration

    def _current_error(self) -> Exception | None:
        with self._stats_lock:
            return self._encode_error

    def _save_timestamps(self) -> None:
        """Save frame timestamps to a JSON file alongside the video."""
        if not self._frame_timestamps:
            logger.info("No timestamps to save")
            return

        timestamp_file = self._output.with_suffix("").with_suffix(self._output.suffix + "_timestamps.json")

        try:
            with self._stats_lock:
                frame_timestamps = self._frame_timestamps.copy()
                hardware_timestamp_source = (
                    dict(self._hardware_timestamp_source) if self._hardware_timestamp_source is not None else None
                )

            software_timestamps = [
                float(rec["software_timestamp"]) for rec in frame_timestamps if "software_timestamp" in rec
            ]

            data = {
                "schema_version": 2,
                "video_file": str(self._output.name),
                "num_frames": len(frame_timestamps),
                # Backward-compatible host/software timestamp list.
                "timestamps": software_timestamps,
                # New descriptive schema.
                "timestamp_sources": {
                    "software_timestamp": {
                        "source": "host_time.time",
                        "backend": "host",
                        "kind": "software_wall_clock",
                        "timebase": "Unix epoch",
                        "unit": "seconds",
                        "description": "Host-side software timestamp captured during acquisition.",
                    },
                    "hardware_timestamp": hardware_timestamp_source,
                },
                "frame_timestamps": frame_timestamps,
                "start_time": software_timestamps[0] if software_timestamps else None,
                "end_time": software_timestamps[-1] if software_timestamps else None,
                "duration_seconds": (
                    software_timestamps[-1] - software_timestamps[0] if len(software_timestamps) > 1 else 0.0
                ),
            }

            with open(timestamp_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("Saved %d frame timestamps to %s", len(frame_timestamps), timestamp_file)
        except Exception as exc:
            logger.exception("Failed to save timestamps to %s: %s", timestamp_file, exc)
