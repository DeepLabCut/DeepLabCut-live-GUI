from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from dlclivegui.services.multi_camera_controller import get_camera_id
from dlclivegui.services.video_recorder import RecorderStats, VideoRecorder
from dlclivegui.utils.config_models import CameraSettingsModel, RecordingSettingsModel
from dlclivegui.utils.utils import build_run_dir, sanitize_name

log = logging.getLogger(__name__)


class RecordingManager:
    """Handle multi-camera recording lifecycle and filenames.

    Directory structure:
      output_dir / <session_name> / <unique_run_dir> / <files...>
    """

    def __init__(self):
        self._recorders: dict[str, VideoRecorder] = {}
        self._session_dir: Path | None = None
        self._run_dir: Path | None = None

    @property
    def is_active(self) -> bool:
        return bool(self._recorders)

    @property
    def recorders(self) -> dict[str, VideoRecorder]:
        return self._recorders

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    def pop(self, cam_id: str, default=None) -> VideoRecorder | None:
        return self._recorders.pop(cam_id, default)

    def start_all(
        self,
        recording: RecordingSettingsModel,
        active_cams: list[CameraSettingsModel],
        current_frames: dict[str, np.ndarray],
        *,
        session_name: str | None = None,
        use_timestamp: bool = True,
    ) -> Path | None:
        """Start recording for all cameras.

        Returns:
            Path to the created run directory, or None if start failed.
        """
        if self._recorders:
            log.debug("Recording already active; start_all ignored.")
            return self._run_dir

        if not active_cams:
            log.warning("No active cameras provided; nothing to record.")
            return None

        base_path = recording.output_path()  # expected to include directory + filename + suffix
        base_stem = base_path.stem
        output_dir = base_path.parent

        sess = sanitize_name(session_name or "session", fallback="session")
        session_dir = output_dir / sess

        try:
            run_dir = build_run_dir(session_dir, use_timestamp=use_timestamp)
        except Exception as exc:
            log.error("Failed to create run directory in %s: %s", session_dir, exc)
            return None

        self._session_dir = session_dir
        self._run_dir = run_dir

        started_any = False
        errors: list[str] = []

        for cam in active_cams:
            cam_id = get_camera_id(cam)
            # Stable per-camera filename. No timestamp needed because run_dir is unique.
            cam_filename = f"{base_stem}_{cam.backend}_cam{cam.index}{base_path.suffix}"
            cam_path = run_dir / cam_filename

            frame = current_frames.get(cam_id)
            frame_size = (frame.shape[0], frame.shape[1]) if frame is not None else None

            recorder = VideoRecorder(
                cam_path,
                frame_size=frame_size,
                frame_rate=float(cam.fps),
                codec=recording.codec,
                crf=recording.crf,
            )

            try:
                recorder.start()
                self._recorders[cam_id] = recorder
                started_any = True
                log.info("Started recording %s -> %s", cam_id, cam_path)
            except Exception as exc:
                err = f"{cam_id}: {exc}"
                errors.append(err)
                log.error("Failed to start recording for %s: %s", cam_id, exc)

        # If nothing started, clean up and return None
        if not started_any:
            self._recorders.clear()
            self._session_dir = None
            self._run_dir = None
            log.error("No recorders started. Errors: %s", "; ".join(errors) if errors else "unknown")
            return None

        # If partial failures occurred, we keep successful recorders running,
        # but log clearly. You can choose to stop_all() here if you prefer "all-or-nothing".
        if errors:
            log.warning("Some cameras failed to start recording: %s", "; ".join(errors))

        return run_dir

    def stop_all(self) -> None:
        for cam_id, rec in list(self._recorders.items()):
            try:
                rec.stop()
                log.info("Stopped recording %s", cam_id)
            except Exception as exc:
                log.warning("Error stopping recorder for %s: %s", cam_id, exc)
        self._recorders.clear()
        self._session_dir = None
        self._run_dir = None

    def write_frame(self, cam_id: str, frame: np.ndarray, timestamp: float | None = None) -> None:
        rec = self._recorders.get(cam_id)
        if not rec or not rec.is_running:
            return
        try:
            rec.write(frame, timestamp=timestamp or time.time())
        except Exception as exc:
            log.warning("Failed to write frame for %s: %s", cam_id, exc)
            try:
                rec.stop()
            except Exception:
                log.exception("Failed to stop recorder for %s after write error.")
            self._recorders.pop(cam_id, None)

    def get_stats_summary(self) -> str:
        # Aggregate stats across recorders
        totals = {
            "written": 0,
            "dropped": 0,
            "queue": 0,
            "max_latency": 0.0,
            "avg_latencies": [],
        }
        for rec in self._recorders.values():
            stats: RecorderStats | None = rec.get_stats()
            if not stats:
                continue
            totals["written"] += stats.frames_written
            totals["dropped"] += stats.dropped_frames
            totals["queue"] += stats.queue_size
            totals["max_latency"] = max(totals["max_latency"], stats.last_latency)
            totals["avg_latencies"].append(stats.average_latency)

        if len(self._recorders) == 1:
            rec = next(iter(self._recorders.values()))
            stats = rec.get_stats()
            if stats:
                from dlclivegui.utils.stats import format_recorder_stats

                return format_recorder_stats(stats)
            return "Recording..."
        else:
            avg = sum(totals["avg_latencies"]) / len(totals["avg_latencies"]) if totals["avg_latencies"] else 0.0
            return (
                f"{len(self._recorders)} cams | {totals['written']} frames | "
                f"latency {totals['max_latency'] * 1000:.1f}ms (avg {avg * 1000:.1f}ms) | "
                f"queue {totals['queue']} | dropped {totals['dropped']}"
            )
