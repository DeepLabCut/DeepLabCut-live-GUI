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
    skeleton: SkeletonOverlaySettings
    bounding_box: BoundingBoxOverlaySettings


@dataclass(frozen=True, slots=True)
class OverlayRenderResult:
    frame: np.ndarray
    warning: str | None = None


class OverlayRenderer:
    """Compose overlays and cache packet-derived skeleton resolution."""

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
        settings: OverlaySettings,
        offset: tuple[int, int] = (0, 0),
        scale: tuple[float, float] = (1.0, 1.0),
    ) -> OverlayRenderResult:
        output = frame.copy()
        warning: str | None = None

        if settings.pose.visible and pose is not None:
            output = draw_pose(
                output,
                pose,
                p_cutoff=settings.pose.p_cutoff,
                colormap=settings.pose.colormap,
                offset=offset,
                scale=scale,
            )

        if settings.skeleton.visible and pose is not None:
            warning = self._draw_packet_skeleton(
                output,
                pose=pose,
                packet=packet,
                settings=settings,
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

        return OverlayRenderResult(
            frame=output,
            warning=warning,
        )

    def _draw_packet_skeleton(
        self,
        frame: np.ndarray,
        *,
        pose: np.ndarray,
        packet: PosePacketLike | None,
        settings: OverlaySettings,
        offset: tuple[int, int],
        scale: tuple[float, float],
    ) -> str | None:
        try:
            resolved = self._resolve_packet(packet)
        except SkeletonResolutionError as exc:
            return self._deduplicate_warning(str(exc))

        if resolved is None:
            return self._deduplicate_warning("Skeleton metadata is unavailable for this pose output.")

        style = settings.skeleton.style
        keypoint_colors = None

        if style.color_mode == SkeletonColorMode.GRADIENT_KEYPOINTS:
            keypoint_colors = tuple(
                keypoint_colors_bgr(
                    settings.pose.colormap,
                    len(resolved.keypoint_names),
                )
            )

        result = draw_skeleton(
            frame,
            pose,
            resolved,
            style,
            p_cutoff=settings.pose.p_cutoff,
            offset=offset,
            scale=scale,
            keypoint_colors=keypoint_colors,
        )

        if result.code in {
            SkeletonRenderCode.RENDERED,
            SkeletonRenderCode.NO_POSE,
        }:
            self._last_warning = None
            return None

        return self._deduplicate_warning(result.message)

    def _resolve_packet(
        self,
        packet: PosePacketLike | None,
    ) -> ResolvedSkeleton | None:
        if packet is None:
            self._clear_skeleton_cache()
            return None

        signature = (
            packet.skeleton_id,
            tuple(packet.keypoint_names or ()),
            tuple(packet.skeleton_edges or ()),
        )

        if signature == self._skeleton_signature:
            return self._resolved_skeleton

        try:
            resolved = resolve_packet_skeleton(packet)
        except SkeletonResolutionError:
            self._clear_skeleton_cache()
            raise

        self._skeleton_signature = signature
        self._resolved_skeleton = resolved
        return resolved

    def _clear_skeleton_cache(self) -> None:
        self._skeleton_signature = None
        self._resolved_skeleton = None

    def _deduplicate_warning(
        self,
        message: str,
    ) -> str | None:
        if not message or message == self._last_warning:
            return None

        self._last_warning = message
        return message
