# dlclivegui/gui/camera_config/preview.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer

from ...services.multi_camera_controller import MultiCameraController

if TYPE_CHECKING:
    from ...cameras.base import CameraBackend
    from ...config import CameraSettings
    from .loaders import CameraLoadWorker


class PreviewState(Enum):
    """Preview lifecycle state.."""

    IDLE = auto()  # No loader, no backend, no timer.
    LOADING = auto()  # Loader started; waiting for success/error/canceled.
    ACTIVE = auto()  # Backend open + preview timer running.
    STOPPING = auto()  # Tearing down loader/backend/timer.
    ERROR = auto()  # Terminal error state (optional; can just go back to IDLE)


@dataclass
class PreviewSession:
    """
    Owns all runtime objects for preview and defines intent.

    epoch:
      Monotonically increasing integer used to invalidate stale signals from previous loaders.
      Any signal handler must check that the epoch matches the current session epoch.

    state:
      PreviewState that replaces multiple booleans.

    requested_cam:
      The CameraSettings snapshot used to start the current LOADING request.

    backend / timer / loader:
      Runtime handles. Only valid in states where they should exist.
    """

    epoch: int = 0
    state: PreviewState = PreviewState.IDLE
    requested_cam: CameraSettings | None = None
    loader: CameraLoadWorker | None = None
    backend: CameraBackend | None = None
    timer: QTimer | None = None

    pending_restart: CameraSettings | None = None
    restart_scheduled: bool = False  # Coalesces restarts to “at most once in the queue”.


def apply_rotation(frame, rotation):
    return MultiCameraController.apply_rotation(frame, rotation)


def apply_crop(frame, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(x0, min(x1, w))
    y1 = max(y0, min(y1, h))

    return MultiCameraController.apply_crop(frame, (x0, y0, x1, y1))


def resize_to_fit(frame, max_w=400, max_h=300):
    return MultiCameraController.apply_resize(frame, max_w, max_h)


def to_display_pixmap(frame):
    return MultiCameraController.to_display_pixmap(frame)
