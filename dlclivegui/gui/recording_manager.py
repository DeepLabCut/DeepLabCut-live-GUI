# dlclivegui/services/recording_manager.py
from __future__ import annotations

import logging
import time

import numpy as np

from dlclivegui.services.multi_camera_controller import get_camera_id
from dlclivegui.services.video_recorder import RecorderStats, VideoRecorder

# from dlclivegui.config import CameraSettings, RecordingSettings
from dlclivegui.utils.config_models import CameraSettingsModel, RecordingSettingsModel

log = logging.getLogger(__name__)


class RecordingManager:
    """Handle multi-camera recording lifecycle and filenames."""

    def __init__(self):
        self._recorders: dict[str, VideoRecorder] = {}

    @property
    def is_active(self) -> bool:
        return bool(self._recorders)

    @property
    def recorders(self) -> dict[str, VideoRecorder]:
        return self._recorders

    def pop(self, cam_id: str, default=None) -> VideoRecorder | None:
        return self._recorders.pop(cam_id, default)

    def start_all(
        self,
        recording: RecordingSettingsModel,
        active_cams: list[CameraSettingsModel],
        current_frames: dict[str, np.ndarray],
    ) -> None:
        if self._recorders:
            return
        base_path = recording.output_path()
        base_stem = base_path.stem

        for cam in active_cams:
            cam_id = get_camera_id(cam)
            cam_filename = f"{base_stem}_{cam.backend}_cam{cam.index}{base_path.suffix}"
            cam_path = base_path.parent / cam_filename
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
                log.info("Started recording %s -> %s", cam_id, cam_path)
            except Exception as exc:
                log.error("Failed to start recording for %s: %s", cam_id, exc)

    def stop_all(self) -> None:
        for cam_id, rec in self._recorders.items():
            try:
                rec.stop()
                log.info("Stopped recording %s", cam_id)
            except Exception as exc:
                log.warning("Error stopping recorder for %s: %s", cam_id, exc)
        self._recorders.clear()

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
