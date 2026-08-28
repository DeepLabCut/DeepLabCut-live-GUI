"""Composition of pose, skeleton, and bounding-box overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dlclivegui.config import (
    BGR,
    SkeletonColorMode,
    SkeletonStyle,
)

from .display import (
    draw_bbox,
    draw_pose,
    keypoint_colors_bgr,
)
from .skeleton import (
    ResolvedSkeleton,
    SkeletonRenderCode,
    SkeletonResolutionError,
    draw_skeleton,
    resolve_packet_skeleton,
)


class PosePacketLike(Protocol):
    keypoint_names: list[str] | None
    skeleton_id: str | None
    skeleton_edges: tuple[tuple[str, str], ...] | None


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
    style: SkeletonStyle


@dataclass(frozen=True, slots=True)
class OverlaySettings:
    pose: PoseOverlaySettings
    bounding_box: BoundingBoxOverlaySettings
    skeleton: SkeletonOverlaySettings


@dataclass(frozen=True, slots=True)
class OverlayRenderResult:
    frame: np.ndarray
    warning: str | None = None


class OverlayRenderer:
    """Render overlays and cache packet-derived skeleton resolution."""

    def __init__(self) -> None:
        self._skeleton_signature: object | None = None
        self._resolved_skeleton: ResolvedSkeleton | None = None
        self._last_warning: str | None = None

    def clear_runtime_state(self) -> None:
        self._skeleton_signature = None
        self._resolved_skeleton = None
        self._last_warning = None

    def render(
        self,
        frame: np.ndarray,
        *,
        pose: np.ndarray | None,
        packet: PosePacketLike | None,
        overlay_settings: OverlaySettings,
        offset: tuple[int, int] = (0, 0),
        scale: tuple[float, float] = (1.0, 1.0),
    ) -> OverlayRenderResult:
        output = frame.copy()
        warning: str | None = None

        if overlay_settings.pose.visible and pose is not None:
            output = draw_pose(
                output,
                pose,
                p_cutoff=overlay_settings.pose.p_cutoff,
                colormap=overlay_settings.pose.colormap,
                offset=offset,
                scale=scale,
            )

        if overlay_settings.skeleton.visible and pose is not None:
            try:
                resolved = self._resolve_packet_skeleton(packet)
            except SkeletonResolutionError as exc:
                warning = self._new_warning(str(exc))
            else:
                if resolved is None:
                    warning = self._new_warning("Skeleton metadata is unavailable for this pose output.")
                else:
                    style = overlay_settings.skeleton.style
                    colors = None

                    if style.color_mode == SkeletonColorMode.GRADIENT_KEYPOINTS:
                        colors = keypoint_colors_bgr(
                            overlay_settings.pose.colormap,
                            len(resolved.keypoint_names),
                        )

                    result = draw_skeleton(
                        output,
                        pose,
                        resolved,
                        style,
                        p_cutoff=overlay_settings.pose.p_cutoff,
                        offset=offset,
                        scale=scale,
                        keypoint_colors=colors,
                    )

                    if result.code in {
                        SkeletonRenderCode.RENDERED,
                        SkeletonRenderCode.NO_POSE,
                    }:
                        self._last_warning = None
                    else:
                        warning = self._new_warning(result.message)

        if overlay_settings.bounding_box.visible:
            output = draw_bbox(
                output,
                overlay_settings.bounding_box.coordinates,
                color_bgr=overlay_settings.bounding_box.color_bgr,
                offset=offset,
                scale=scale,
            )

        return OverlayRenderResult(
            frame=output,
            warning=warning,
        )

    def _resolve_packet_skeleton(
        self,
        packet: PosePacketLike | None,
    ) -> ResolvedSkeleton | None:
        if packet is None:
            self._skeleton_signature = None
            self._resolved_skeleton = None
            return None

        signature = (
            packet.skeleton_id,
            tuple(packet.keypoint_names or ()),
            tuple(packet.skeleton_edges or ()),
        )

        if signature == self._skeleton_signature:
            return self._resolved_skeleton

        resolved = resolve_packet_skeleton(packet)

        self._skeleton_signature = signature
        self._resolved_skeleton = resolved
        return resolved

    def _new_warning(
        self,
        message: str,
    ) -> str | None:
        if not message or message == self._last_warning:
            return None

        self._last_warning = message
        return message
