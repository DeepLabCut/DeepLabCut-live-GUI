from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from dlclivegui.config import ModelType


class PoseBackends(Enum):
    DLC_LIVE = auto()


class WorkerState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    FAULTED = auto()


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
    skeleton_id: str | None = None
    skeleton_edges: tuple[tuple[str, str], ...] | None = None
    source: PoseSource = PoseSource(backend=PoseBackends.DLC_LIVE)
    raw: Any | None = None


@dataclass
class PoseResult:
    pose: np.ndarray | None
    timestamp: float
    packet: PosePacket | None = None


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
