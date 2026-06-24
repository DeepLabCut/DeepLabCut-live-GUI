# dlclivegui/utils/stats.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from dlclivegui.services.dlc_processor import ProcessorStats


@dataclass
class RecorderStats:
    """Snapshot of recorder throughput metrics."""

    frames_enqueued: int = 0
    frames_written: int = 0
    dropped_frames: int = 0
    queue_size: int = 0
    average_latency: float = 0.0
    last_latency: float = 0.0
    write_fps: float = 0.0
    buffer_seconds: float = 0.0


class WorkerTimingStats:
    """Tiny timing accumulator for camera worker performance diagnostics.

    Usage:
        with stats.measure("read"):
            frame, ts = backend.read()

    Logs aggregate timings once per log_interval seconds.
    """

    def __init__(
        self, camera_id: str, *, logger: logging.Logger | None = None, log_interval: float = 1.0, enabled: bool = True
    ):
        self.camera_id = camera_id
        self.log_interval = float(log_interval)
        self.enabled = bool(enabled)
        self.logger = logger or logging.getLogger(__name__)
        if self.enabled:  # force logger to proper level
            if not self.logger.isEnabledFor(logging.DEBUG):
                self.logger.setLevel(logging.DEBUG)

        self._last_log = time.perf_counter()
        self._frames = 0
        self._timeouts = 0
        self._errors = 0
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    class _Measure:
        def __init__(self, parent: WorkerTimingStats, name: str):
            self.parent = parent
            self.name = name
            self.t0 = 0.0

        def __enter__(self):
            if self.parent.enabled:
                self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            if not self.parent.enabled:
                return False

            dt = time.perf_counter() - self.t0
            self.parent._totals[self.name] = self.parent._totals.get(self.name, 0.0) + dt
            self.parent._counts[self.name] = self.parent._counts.get(self.name, 0) + 1
            return False

    def measure(self, name: str):
        return self._Measure(self, name)

    def note_frame(self) -> None:
        if self.enabled:
            self._frames += 1

    def note_timeout(self) -> None:
        if self.enabled:
            self._timeouts += 1

    def note_error(self) -> None:
        if self.enabled:
            self._errors += 1

    def maybe_log(self) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        elapsed = now - self._last_log
        if elapsed < self.log_interval:
            return

        fps = self._frames / max(elapsed, 1e-9)

        parts = [
            f"[Worker {self.camera_id}]",
            f"fps={fps:.1f}",
            f"frames={self._frames}",
        ]

        if self._timeouts:
            parts.append(f"timeouts={self._timeouts}")
        if self._errors:
            parts.append(f"errors={self._errors}")

        for name in sorted(self._totals):
            count = max(self._counts.get(name, 0), 1)
            avg_ms = 1000.0 * self._totals[name] / count
            parts.append(f"avg_{name}_ms={avg_ms:.3f}")

        self.logger.debug(" ".join(parts))

        self._last_log = now
        self._frames = 0
        self._timeouts = 0
        self._errors = 0
        self._totals.clear()
        self._counts.clear()


def format_recorder_stats(stats: RecorderStats) -> str:
    latency_ms = stats.last_latency * 1000.0
    avg_ms = stats.average_latency * 1000.0
    buffer_ms = stats.buffer_seconds * 1000.0
    return (
        f"{stats.frames_written}/{stats.frames_enqueued} frames | "
        f"write {stats.write_fps:.1f} fps | "
        f"latency {latency_ms:.1f} ms (avg {avg_ms:.1f} ms) | "
        f"queue {stats.queue_size} (~{buffer_ms:.0f} ms) | "
        f"dropped {stats.dropped_frames}"
    )


def format_dlc_stats(stats: ProcessorStats) -> str:
    latency_ms = stats.last_latency * 1000.0
    avg_ms = stats.average_latency * 1000.0
    profile = ""
    if stats.avg_inference_time > 0:
        inf_ms = stats.avg_inference_time * 1000.0
        queue_ms = stats.avg_queue_wait * 1000.0
        signal_ms = stats.avg_signal_emit_time * 1000.0
        total_ms = stats.avg_total_process_time * 1000.0
        gpu_breakdown = ""
        if stats.avg_gpu_inference_time > 0 or stats.avg_processor_overhead > 0:
            gpu_ms = stats.avg_gpu_inference_time * 1000.0
            proc_ms = stats.avg_processor_overhead * 1000.0
            gpu_breakdown = f" (GPU:{gpu_ms:.1f}ms+proc:{proc_ms:.1f}ms)"
        profile = (
            f"\n[Profile] inf:{inf_ms:.1f}ms{gpu_breakdown} "
            f"queue:{queue_ms:.1f}ms signal:{signal_ms:.1f}ms total:{total_ms:.1f}ms"
        )

    return (
        f"{stats.frames_processed}/{stats.frames_enqueued} frames | "
        f"inference {stats.processing_fps:.1f} fps | "
        f"latency {latency_ms:.1f} ms (avg {avg_ms:.1f} ms) | "
        f"queue {stats.queue_size} | dropped {stats.frames_dropped}{profile}"
    )
