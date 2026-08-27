# dlclivegui/display/overlays.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlclivegui.config import BGR

from .display import draw_bbox, draw_pose
from .skeleton import ResolvedSkeleton, SkeletonStyle


@dataclass(frozen=True, slots=True)
class PoseOverlaySettings:
    visible: bool
    p_cutoff: float
    colormap: str


@dataclass(frozen=True, slots=True)
class BoundingBoxOverlaySettings:
    visible: bool
    coordinates: tuple[int, int, int, int]
    color_bgr: BGR


@dataclass(frozen=True, slots=True)
class SkeletonOverlaySettings:
    visible: bool
    resolved: ResolvedSkeleton | None
    style: SkeletonStyle


@dataclass(frozen=True, slots=True)
class OverlaySettings:
    pose: PoseOverlaySettings
    bounding_box: BoundingBoxOverlaySettings


def render_overlays(
    frame: np.ndarray,
    *,
    pose: np.ndarray | None,
    settings: OverlaySettings,
    offset: tuple[int, int] = (0, 0),
    scale: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Return a frame containing the requested overlays."""
    output = frame.copy()

    if settings.pose.visible and pose is not None:
        output = draw_pose(
            output,
            pose,
            p_cutoff=settings.pose.p_cutoff,
            colormap=settings.pose.colormap,
            offset=offset,
            scale=scale,
        )

    if settings.bounding_box.visible:
        output = draw_bbox(
            output,
            settings.bounding_box.coordinates,
            color_bgr=settings.bounding_box.color_bgr,
            offset=offset,
            scale=scale,
        )

    return output
