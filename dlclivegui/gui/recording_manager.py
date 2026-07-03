from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from dlclivegui.config import CameraSettings, RecordingSettings
from dlclivegui.services.multi_camera_controller import get_camera_id
from dlclivegui.services.video_recorder import VideoRecorder
from dlclivegui.utils.stats import RecorderStats
from dlclivegui.utils.utils import build_run_dir, sanitize_name

log = logging.getLogger(__name__)

_FRAME_SENTINEL = object()


class RecordingManager:
    """Handle multi-camera recording lifecycle and filenames."""

    def __init__(self):
        self._recorders: dict[str, VideoRecorder] = {}
        self._session_dir: Path | None = None
        self._run_dir: Path | None = None

        self._lock = threading.RLock()
        self._frame_queue: queue.Queue | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._dispatch_stop = threading.Event()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return bool(self._recorders)

    @property
    def recorders(self) -> dict[str, VideoRecorder]:
        with self._lock:
            return dict(self._recorders)

    @property
    def session_dir(self) -> Path | None:
        with self._lock:
            return self._session_dir

    @property
    def run_dir(self) -> Path | None:
        with self._lock:
            return self._run_dir

    @staticmethod
    def _backend_ns(cam: CameraSettings) -> dict:
        backend = (cam.backend or "").lower()
        props = cam.properties if isinstance(cam.properties, dict) else {}
        ns = props.get(backend, {})
        return ns if isinstance(ns, dict) else {}

    @classmethod
    def _resolve_recording_fps(
        cls,
        cam: CameraSettings,
        cam_id: str,
        frame_rates: dict[str, float] | None,
    ) -> float | None:
        """Resolve writer FPS.

        Prefer runtime measured FPS, then backend-probed detected_fps,
        then explicit requested cam.fps. Auto/unknown returns None.
        """
        measured_fps = 0.0
        if frame_rates:
            try:
                measured_fps = float(frame_rates.get(cam_id, 0.0) or 0.0)
            except Exception:
                measured_fps = 0.0

        if measured_fps > 0.0:
            return measured_fps

        ns = cls._backend_ns(cam)

        try:
            detected_fps = float(ns.get("detected_fps", 0.0) or 0.0)
        except Exception:
            detected_fps = 0.0

        if detected_fps > 0.0:
            return detected_fps

        try:
            requested_fps = float(getattr(cam, "fps", 0.0) or 0.0)
        except Exception:
            requested_fps = 0.0

        if requested_fps > 0.0:
            return requested_fps

        return None

    def pop(self, cam_id: str, default=None) -> VideoRecorder | None:
        with self._lock:
            return self._recorders.pop(cam_id, default)

    def _start_dispatcher(self) -> None:
        with self._lock:
            if self._dispatch_thread is not None and self._dispatch_thread.is_alive():
                return

            self._dispatch_stop.clear()
            self._frame_queue = queue.Queue(maxsize=4096)
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_loop,
                name="RecordingManagerDispatcher",
                daemon=True,
            )
            self._dispatch_thread.start()

    def _stop_dispatcher(self, timeout: float = 2.0) -> None:
        self._dispatch_stop.set()

        with self._lock:
            q = self._frame_queue
            t = self._dispatch_thread

        if q is not None:
            try:
                q.put_nowait(_FRAME_SENTINEL)
            except queue.Full:
                pass

        if t is not None:
            t.join(timeout=timeout)
            if t.is_alive():
                log.warning("Recording frame dispatcher did not stop within %.1fs", timeout)

        with self._lock:
            self._dispatch_thread = None
            self._frame_queue = None
            self._dispatch_stop.clear()

    def _dispatch_loop(self) -> None:
        with self._lock:
            q = self._frame_queue

        if q is None:
            return

        while not self._dispatch_stop.is_set():
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if item is _FRAME_SENTINEL:
                    break

                cam_id, frame, timestamp = item
                self._write_frame_now(cam_id, frame, timestamp)

            finally:
                try:
                    q.task_done()
                except ValueError:
                    pass

    def start_all(
        self,
        recording: RecordingSettings,
        active_cams: list[CameraSettings],
        current_frames: dict[str, np.ndarray],
        *,
        frame_rates: dict[str, float] | None = None,
        session_name: str = "session",
        use_timestamp: bool = True,
        all_or_nothing: bool = False,
    ) -> Path | None:
        """Start recording for all active cameras.

        Record into <directory>/<session_name>/<unique_run_dir>/

        Args:
            recording: Recording settings including output directory and codec.
            active_cams: List of active camera settings to record.
            current_frames: Dict of current frames by camera ID for size reference.
            session_name: Name of the recording session (used in directory name).
            use_timestamp: Whether to use timestamp-based run directories instead of indexed.
            all_or_nothing: If True, stop all and return None if any recorder fails to start.

        Returns:
            run_dir if at least one recorder started, else None.
        """
        with self._lock:
            if self._recorders:
                return self._run_dir

        if not active_cams:
            return None

        base_path = recording.output_path()
        base_stem = base_path.stem

        # create session/run directories
        session_safe = sanitize_name(session_name, fallback="session")
        session_dir = base_path.parent / session_safe
        try:
            run_dir = build_run_dir(session_dir, use_timestamp=use_timestamp)
        except Exception as exc:
            log.error("Failed to create run dir: %s", exc)
            return None

        with self._lock:
            self._session_dir = session_dir
            self._run_dir = run_dir

        started_any = False

        for cam in active_cams:
            cam_id = get_camera_id(cam)
            cam_filename = f"{base_stem}_{cam.backend}_cam{cam.index}{base_path.suffix}"
            cam_path = run_dir / cam_filename

            frame = current_frames.get(cam_id)
            frame_size = (frame.shape[0], frame.shape[1]) if frame is not None else None
            recorder_fps = self._resolve_recording_fps(cam, cam_id, frame_rates)
            writer_options = recording.writegear_options(recorder_fps)

            log.debug(
                "Starting recorder %s -> %s frame_size=%s requested_fps=%s detected_fps=%s "
                "recorder_fps=%s fast_encoding=%s writer_options=%s",
                cam_id,
                cam_path,
                frame_size,
                getattr(cam, "fps", None),
                self._backend_ns(cam).get("detected_fps"),
                f"{recorder_fps:.3f}" if recorder_fps else "auto/fallback",
                bool(getattr(recording, "fast_encoding", False)),
                writer_options,
            )

            recorder = VideoRecorder(
                cam_path,
                frame_size=frame_size,
                frame_rate=recorder_fps,
                codec=recording.codec,
                crf=recording.crf,
                convert_grayscale_to_rgb=not bool(getattr(cam, "preserve_mono", False)),
                writer_options=writer_options,
            )
            try:
                recorder.start()
                with self._lock:
                    self._recorders[cam_id] = recorder
                started_any = True
                log.info("Started recording %s -> %s", cam_id, cam_path)
            except Exception as exc:
                log.error("Failed to start recording for %s: %s", cam_id, exc)
                if all_or_nothing:
                    self.stop_all()
                    return None

        if not started_any:
            with self._lock:
                self._recorders.clear()
                self._session_dir = None
                self._run_dir = None
            return None

        self._start_dispatcher()
        return run_dir

    def stop_all(self) -> None:
        self._stop_dispatcher()

        with self._lock:
            recorders = list(self._recorders.items())
            self._recorders.clear()

        for cam_id, rec in recorders:
            try:
                rec.stop()
                log.info("Stopped recording %s", cam_id)
            except Exception as exc:
                log.warning("Error stopping recorder for %s: %s", cam_id, exc)

        with self._lock:
            self._session_dir = None
            self._run_dir = None

    def _write_frame_now(
        self, cam_id: str, frame: np.ndarray, timestamp: float | None = None, timestamp_metadata: object | None = None
    ) -> None:
        with self._lock:
            rec = self._recorders.get(cam_id)

        if not rec or not rec.is_running:
            return

        try:
            rec.write(
                frame,
                timestamp=timestamp if timestamp is not None else time.time(),
                timestamp_metadata=timestamp_metadata,
            )
        except Exception as exc:
            log.warning(
                "Failed to write frame for %s: %s: %s frame_shape=%s dtype=%s. Removing recorder.",
                cam_id,
                type(exc).__name__,
                str(exc) or repr(exc),
                getattr(frame, "shape", None),
                getattr(frame, "dtype", None),
            )

            with self._lock:
                rec = self._recorders.pop(cam_id, None)

            if rec is not None:
                try:
                    rec.stop()
                except Exception:
                    log.exception("Failed to stop recorder for %s after write error.", cam_id)

    def write_frame(
        self, cam_id: str, frame: np.ndarray, timestamp: float | None = None, timestamp_metadata: object | None = None
    ) -> None:
        with self._lock:
            q = self._frame_queue
            active = cam_id in self._recorders

        if not active or q is None:
            return

        try:
            q.put_nowait((cam_id, frame, timestamp if timestamp is not None else time.time(), timestamp_metadata))
        except queue.Full:
            log.warning(
                "Recording manager frame queue full; dropping frame for %s. frame_shape=%s dtype=%s",
                cam_id,
                getattr(frame, "shape", None),
                getattr(frame, "dtype", None),
            )

    def get_stats_summary(self) -> str:
        totals = {
            "enqueued": 0,
            "written": 0,
            "dropped": 0,
            "queue": 0,
            "buffer": 0,
            "backlog": 0,
            "write_fps": 0.0,
            "max_latency": 0.0,
            "avg_latencies": [],
        }

        with self._lock:
            recorders = list(self._recorders.values())

        for rec in recorders:
            stats: RecorderStats | None = rec.get_stats()
            if not stats:
                continue
            totals["enqueued"] += stats.frames_enqueued
            totals["written"] += stats.frames_written
            totals["dropped"] += stats.dropped_frames
            totals["queue"] += stats.queue_size
            totals["buffer"] += stats.buffer_size
            totals["backlog"] += stats.backlog_frames
            totals["write_fps"] += stats.write_fps
            totals["max_latency"] = max(totals["max_latency"], stats.last_latency)
            totals["avg_latencies"].append(stats.average_latency)

        if len(recorders) == 1:
            rec = recorders[0]
            stats = rec.get_stats()
            if stats:
                from dlclivegui.utils.stats import format_recorder_stats

                return format_recorder_stats(stats)
            return "Recording..."
        else:
            avg = sum(totals["avg_latencies"]) / len(totals["avg_latencies"]) if totals["avg_latencies"] else 0.0

            buffer = totals["buffer"]
            queue_text = f"{totals['queue']}/{buffer}" if buffer > 0 else str(totals["queue"])
            fill_pct = (100.0 * totals["queue"] / buffer) if buffer > 0 else 0.0

            return (
                f"{len(recorders)} cams | {totals['written']}/{totals['enqueued']} frames | "
                f"writer {totals['write_fps']:.1f} fps | "
                f"latency {totals['max_latency'] * 1000:.1f}ms (avg {avg * 1000:.1f}ms) | "
                f"queue {queue_text} ({fill_pct:.0f}%) | "
                f"backlog {totals['backlog']} | "
                f"dropped {totals['dropped']}"
            )
