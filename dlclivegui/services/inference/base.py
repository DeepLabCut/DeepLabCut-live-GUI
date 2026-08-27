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
