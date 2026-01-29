# dlclivegui/utils/stats.py
from __future__ import annotations

from dlclivegui.services.dlc_processor import ProcessorStats
from dlclivegui.services.video_recorder import RecorderStats


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
