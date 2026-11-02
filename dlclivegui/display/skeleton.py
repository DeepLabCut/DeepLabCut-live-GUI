"""Skeleton topology resolution and rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import cv2
import numpy as np

from dlclivegui.config import BGR, SkeletonColorMode, SkeletonStyle

# ####################### #
#  Skeleton definitions  #
# ####################### #


class SkeletonResolutionError(ValueError):
    """Raised when a skeleton cannot be aligned with pose keypoints."""


@dataclass(frozen=True, slots=True)
class SkeletonEdge:
    """An edge expressed using semantic keypoint names."""

    start: str
    end: str


@dataclass(frozen=True, slots=True)
class SkeletonDefinition:
    """Immutable, backend-independent skeleton topology."""

    identifier: str
    display_name: str
    edges: tuple[SkeletonEdge, ...]


@dataclass(frozen=True, slots=True)
class ResolvedSkeleton:
    """Skeleton topology resolved against a specific keypoint order."""

    definition: SkeletonDefinition
    keypoint_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]


def resolve_skeleton(
    definition: SkeletonDefinition,
    keypoint_names: list[str] | tuple[str, ...],
) -> ResolvedSkeleton:
    """Resolve named skeleton edges against an ordered keypoint list."""
    names = tuple(keypoint_names)

    if not names:
        raise SkeletonResolutionError("Cannot resolve a skeleton without keypoint names.")

    if len(set(names)) != len(names):
        raise SkeletonResolutionError("Cannot resolve a skeleton against duplicate keypoint names.")

    name_to_index = {name: index for index, name in enumerate(names)}
    resolved_edges: list[tuple[int, int]] = []
    missing_names: set[str] = set()

    for edge in definition.edges:
        start_index = name_to_index.get(edge.start)
        end_index = name_to_index.get(edge.end)

        if start_index is None:
            missing_names.add(edge.start)
        if end_index is None:
            missing_names.add(edge.end)

        if start_index is not None and end_index is not None:
            resolved_edges.append((start_index, end_index))

    if missing_names:
        missing = ", ".join(sorted(missing_names))
        raise SkeletonResolutionError(f"Skeleton keypoints are absent from the pose output: {missing}.")

    return ResolvedSkeleton(
        definition=definition,
        keypoint_names=names,
        edges=tuple(resolved_edges),
    )


def skeleton_definition_from_metadata(
    *,
    identifier: str,
    display_name: str,
    edges: tuple[tuple[str, str], ...],
) -> SkeletonDefinition:
    """Create a validated display topology from pose metadata."""
    if not identifier.strip():
        raise SkeletonResolutionError("Skeleton identifier cannot be empty.")

    if not edges:
        raise SkeletonResolutionError("Skeleton definition does not contain any edges.")

    skeleton_edges: list[SkeletonEdge] = []

    for start, end in edges:
        if not start or not end:
            raise SkeletonResolutionError("Skeleton edge names cannot be empty.")

        if start == end:
            raise SkeletonResolutionError(f"Skeleton contains a self-loop at {start!r}.")

        skeleton_edges.append(
            SkeletonEdge(
                start=start,
                end=end,
            )
        )

    return SkeletonDefinition(
        identifier=identifier,
        display_name=display_name,
        edges=tuple(skeleton_edges),
    )


# ###################### #
#  Skeleton I/O          #
# ###################### #


class SkeletonPacket(Protocol):
    keypoint_names: list[str] | None
    skeleton_id: str | None
    skeleton_edges: tuple[tuple[str, str], ...] | None


def resolve_packet_skeleton(
    packet: SkeletonPacket,
) -> ResolvedSkeleton | None:
    """Resolve skeleton metadata supplied by a pose packet."""
    if not packet.keypoint_names:
        return None

    if not packet.skeleton_id or not packet.skeleton_edges:
        return None

    definition = skeleton_definition_from_metadata(
        identifier=packet.skeleton_id,
        display_name=packet.skeleton_id,
        edges=packet.skeleton_edges,
    )

    return resolve_skeleton(
        definition,
        packet.keypoint_names,
    )


# ###################### #
#  Rendering outcomes   #
# ###################### #


class SkeletonRenderCode(Enum):
    RENDERED = auto()
    NO_POSE = auto()
    INVALID_POSE = auto()
    KEYPOINT_COUNT_MISMATCH = auto()
    COLOR_COUNT_MISMATCH = auto()


@dataclass(frozen=True, slots=True)
class SkeletonRenderResult:
    code: SkeletonRenderCode
    edges_drawn: int = 0
    message: str = ""

    @property
    def rendered(self) -> bool:
        return self.code == SkeletonRenderCode.RENDERED


# ###################### #
#  Rendering utilities  #
# ###################### #


def _effective_thickness(
    style: SkeletonStyle,
    scale: tuple[float, float],
) -> int:
    scale_x, scale_y = scale
    return style.effective_thickness(scale_x, scale_y)


def _draw_gradient_edge(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    start_color: BGR,
    end_color: BGR,
    *,
    thickness: int,
    steps: int,
) -> None:
    start_x, start_y = start
    end_x, end_y = end

    for step in range(steps):
        alpha_start = step / steps
        alpha_end = (step + 1) / steps
        color_alpha = (step + 0.5) / steps

        segment_start = (
            round(start_x + (end_x - start_x) * alpha_start),
            round(start_y + (end_y - start_y) * alpha_start),
        )
        segment_end = (
            round(start_x + (end_x - start_x) * alpha_end),
            round(start_y + (end_y - start_y) * alpha_end),
        )

        color: BGR = tuple(
            round(component_start + (component_end - component_start) * color_alpha)
            for component_start, component_end in zip(
                start_color,
                end_color,
                strict=True,
            )
        )

        cv2.line(
            frame,
            segment_start,
            segment_end,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def draw_skeleton(
    frame: np.ndarray,
    poses: np.ndarray | None,
    skeleton: ResolvedSkeleton,
    style: SkeletonStyle,
    *,
    p_cutoff: float,
    offset: tuple[int, int] = (0, 0),
    scale: tuple[float, float] = (1.0, 1.0),
    keypoint_colors: tuple[BGR, ...] | None = None,
) -> SkeletonRenderResult:
    """Draw a resolved skeleton over one or more poses.

    Accepted pose shapes are:

    - ``(K, 3)`` for one individual
    - ``(N, K, 3)`` for multiple individuals

    The function modifies ``frame`` in place and returns a structured result.
    """
    if poses is None:
        return SkeletonRenderResult(
            code=SkeletonRenderCode.NO_POSE,
        )

    pose_array = np.asarray(poses)

    if pose_array.ndim == 2:
        individuals = pose_array[np.newaxis, ...]
    elif pose_array.ndim == 3:
        individuals = pose_array
    else:
        return SkeletonRenderResult(
            code=SkeletonRenderCode.INVALID_POSE,
            message=(f"Skeleton poses must have shape (K, 3) or (N, K, 3); received {pose_array.shape!r}."),
        )

    if individuals.shape[-1] != 3:
        return SkeletonRenderResult(
            code=SkeletonRenderCode.INVALID_POSE,
            message=(f"Skeleton poses must contain x, y, and likelihood; received {pose_array.shape!r}."),
        )

    expected_keypoints = len(skeleton.keypoint_names)
    actual_keypoints = individuals.shape[1]

    if actual_keypoints != expected_keypoints:
        return SkeletonRenderResult(
            code=SkeletonRenderCode.KEYPOINT_COUNT_MISMATCH,
            message=(f"Skeleton expects {expected_keypoints} keypoints, but the pose contains {actual_keypoints}."),
        )

    uses_gradient = style.color_mode == SkeletonColorMode.GRADIENT_KEYPOINTS

    if uses_gradient and (keypoint_colors is None or len(keypoint_colors) != expected_keypoints):
        return SkeletonRenderResult(
            code=SkeletonRenderCode.COLOR_COUNT_MISMATCH,
            message=(f"Keypoint-gradient mode requires exactly {expected_keypoints} keypoint colors."),
        )

    offset_x, offset_y = offset
    scale_x, scale_y = scale
    thickness = _effective_thickness(style, scale)
    edges_drawn = 0

    for pose in individuals:
        for start_index, end_index in skeleton.edges:
            start_x, start_y, start_likelihood = pose[start_index]
            end_x, end_y, end_likelihood = pose[end_index]

            values = (
                start_x,
                start_y,
                start_likelihood,
                end_x,
                end_y,
                end_likelihood,
            )

            if not np.isfinite(values).all() or start_likelihood < p_cutoff or end_likelihood < p_cutoff:
                continue

            start_point = (
                round(start_x * scale_x + offset_x),
                round(start_y * scale_y + offset_y),
            )
            end_point = (
                round(end_x * scale_x + offset_x),
                round(end_y * scale_y + offset_y),
            )

            if uses_gradient:
                assert keypoint_colors is not None

                _draw_gradient_edge(
                    frame,
                    start_point,
                    end_point,
                    keypoint_colors[start_index],
                    keypoint_colors[end_index],
                    thickness=thickness,
                    steps=style.gradient_steps,
                )
            else:
                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    style.color_bgr,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            edges_drawn += 1

    return SkeletonRenderResult(
        code=SkeletonRenderCode.RENDERED,
        edges_drawn=edges_drawn,
    )
