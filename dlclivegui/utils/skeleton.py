"""Skeleton definition, validation, and drawing utilities."""

# dlclivegui/utils/skeleton.py
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from dlclivegui.config import BGR, SkeletonColorMode, SkeletonStyle


# ############### #
#  Status & code  #
# ############### #
class SkeletonRenderCode(Enum):
    OK = auto()
    POSE_SHAPE_INVALID = auto()
    KEYPOINT_COUNT_MISMATCH = auto()


@dataclass(frozen=True)
class SkeletonRenderStatus:
    code: SkeletonRenderCode
    message: str = ""

    @property
    def rendered(self) -> bool:
        return self.code == SkeletonRenderCode.OK

    @property
    def should_disable(self) -> bool:
        # GUI can switch off skeleton drawing if True
        return self.code in {
            SkeletonRenderCode.POSE_SHAPE_INVALID,
            SkeletonRenderCode.KEYPOINT_COUNT_MISMATCH,
        }


# ############ #
#  Exceptions  #
# ############ #


class SkeletonError(ValueError):
    """Raised when a skeleton definition is invalid."""


class SkeletonLoadError(Exception):
    """High-level skeleton loading error (safe for GUI display)."""


class SkeletonValidationError(SkeletonLoadError):
    """Schema or semantic validation error."""


# ################## #
#  Skeleton display  #
# ################## #


class SkeletonStyleModel(BaseModel):
    mode: SkeletonColorMode = SkeletonColorMode.SOLID
    color: BGR = (0, 255, 255)  # default if SOLID
    thickness: int = Field(2, ge=1, description="Base thickness in pixels")
    gradient_steps: int = Field(16, ge=2, description="Segments per edge when gradient")
    scale_with_zoom: bool = True

    @field_validator("thickness")
    @classmethod
    def _thickness_positive(cls, v):
        if v < 1:
            raise ValueError("Thickness must be at least 1 pixel")
        return v

    @field_validator("gradient_steps")
    @classmethod
    def _steps_positive(cls, v):
        if v < 2:
            raise ValueError("gradient_steps must be >= 2")
        return v


# ############# #
#  Skeleton IO  #
# ############# #
class SkeletonModel(BaseModel):
    """Validated skeleton definition (IO + schema)."""

    name: str | None = None

    keypoints: list[str] = Field(..., min_length=1, description="Ordered list of keypoint names")

    edges: list[tuple[int, int]] = Field(
        default_factory=list,
        description="List of (i, j) keypoint index pairs",
    )

    style: SkeletonStyleModel = Field(default_factory=SkeletonStyleModel)
    default_color: BGR = (0, 255, 255)  # used if style.color is None or in SOLID mode
    edge_colors: dict[tuple[int, int], BGR] = Field(default_factory=dict)

    schema_version: int = 1

    @field_validator("keypoints")
    @classmethod
    def validate_unique_keypoints(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("Duplicate keypoint names detected")
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, edges, info):
        keypoints = info.data.get("keypoints", [])
        n = len(keypoints)

        for i, j in edges:
            if i == j:
                raise ValueError(f"Self-loop detected in edge ({i}, {j})")
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"Edge ({i}, {j}) out of range for {n} keypoints")
        return edges


def _load_raw_skeleton_data(path: Path) -> dict:
    if not path.exists():
        raise SkeletonLoadError(f"Skeleton file not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())

    if path.suffix == ".json":
        return json.loads(path.read_text())

    raise SkeletonLoadError(f"Unsupported file type: {path.suffix}")


def _format_pydantic_error(err: ValidationError) -> str:
    lines = ["Invalid skeleton definition:"]
    for e in err.errors():
        loc = " → ".join(map(str, e["loc"]))
        msg = e["msg"]
        lines.append(f"• {loc}: {msg}")
    return "\n".join(lines)


def load_skeleton(path: Path) -> Skeleton:
    try:
        data = _load_raw_skeleton_data(path)
        model = SkeletonModel.model_validate(data)
        return Skeleton(model)

    except ValidationError as e:
        raise SkeletonValidationError(_format_pydantic_error(e)) from None

    except Exception as e:
        raise SkeletonLoadError(str(e)) from None


def save_skeleton(path: Path, model: SkeletonModel) -> None:
    if path.suffix in {".yaml", ".yml"}:
        data = model.model_dump()
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    elif path.suffix == ".json":
        # Use Pydantic's JSON serialization to ensure Enums and other types
        # are converted to JSON-friendly values.
        path.write_text(model.model_dump_json(indent=2))
    else:
        raise SkeletonLoadError(f"Unsupported skeleton file type: {path.suffix}")


def load_dlc_skeleton(config_path: Path) -> Skeleton | None:
    if not config_path.exists():
        raise SkeletonLoadError(f"DLC config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())

    bodyparts = cfg.get("bodyparts")
    if not bodyparts:
        return None  # No pose info

    edges = []

    # Newer DLC format
    if "skeleton" in cfg:
        for a, b in cfg["skeleton"]:
            edges.append((bodyparts.index(a), bodyparts.index(b)))

    # Older / alternative formats
    elif "skeleton_edges" in cfg:
        edges = [tuple(e) for e in cfg["skeleton_edges"]]

    if not edges:
        return None

    model = SkeletonModel(
        name=cfg.get("Task", "DeepLabCut"),
        keypoints=bodyparts,
        edges=edges,
    )

    return Skeleton(model)


class Skeleton:
    """Runtime skeleton optimized for drawing."""

    def __init__(self, model: SkeletonModel):
        self.name = model.name
        self.keypoints = model.keypoints
        self.edges = model.edges

        self.style = SkeletonStyle(
            mode=model.style.mode,
            color=model.style.color,
            thickness=model.style.thickness,
            gradient_steps=model.style.gradient_steps,
            scale_with_zoom=model.style.scale_with_zoom,
        )
        self.default_color = model.default_color
        self.edge_colors = model.edge_colors

    def check_pose_compat(self, pose: np.ndarray) -> SkeletonRenderStatus:
        pose = np.asarray(pose)

        if pose.ndim != 2 or pose.shape[1] not in (2, 3):
            return SkeletonRenderStatus(
                SkeletonRenderCode.POSE_SHAPE_INVALID,
                f"Pose must be (N,2) or (N,3); got shape={pose.shape}",
            )

        expected = len(self.keypoints)
        got = pose.shape[0]
        if got != expected:
            return SkeletonRenderStatus(
                SkeletonRenderCode.KEYPOINT_COUNT_MISMATCH,
                f"Skeleton expects {expected} keypoints, but pose has {got}.",
            )

        return SkeletonRenderStatus(SkeletonRenderCode.OK, "")

    def _draw_gradient_edge(
        self,
        img: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        c1: BGR,
        c2: BGR,
        thickness: int,
        steps: int,
    ):
        x1, y1 = p1
        x2, y2 = p2

        for s in range(steps):
            a0 = s / steps
            a1 = (s + 1) / steps
            xs0 = int(x1 + (x2 - x1) * a0)
            ys0 = int(y1 + (y2 - y1) * a0)
            xs1 = int(x1 + (x2 - x1) * a1)
            ys1 = int(y1 + (y2 - y1) * a1)

            t = (s + 0.5) / steps
            b = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            r = int(c1[2] + (c2[2] - c1[2]) * t)

            cv2.line(img, (xs0, ys0), (xs1, ys1), (b, g, r), thickness, lineType=cv2.LINE_AA)

    def draw(
        self,
        overlay: np.ndarray,
        pose: np.ndarray,
        p_cutoff: float,
        offset: tuple[int, int],
        scale: tuple[float, float],
        *,
        style: SkeletonStyle | None = None,
        color_override: BGR | None = None,
        keypoint_colors: list[BGR] | None = None,
    ) -> SkeletonRenderStatus:
        status = self.check_pose_compat(pose)
        if not status.rendered:
            return status

        st = style or self.style
        ox, oy = offset
        sx, sy = scale
        th = st.effective_thickness(sx, sy)

        # if gradient mode, require keypoint_colors aligned with keypoint order
        if st.mode == SkeletonColorMode.GRADIENT_KEYPOINTS:
            if keypoint_colors is None or len(keypoint_colors) != len(self.keypoints):
                return SkeletonRenderStatus(
                    SkeletonRenderCode.KEYPOINT_COUNT_MISMATCH,
                    f"Gradient mode requires keypoint_colors of length {len(self.keypoints)}.",
                )

        for i, j in self.edges:
            xi, yi = pose[i][:2]
            xj, yj = pose[j][:2]
            ci = pose[i][2] if pose.shape[1] > 2 else 1.0
            cj = pose[j][2] if pose.shape[1] > 2 else 1.0
            if np.isnan(xi) or np.isnan(yi) or ci < p_cutoff or np.isnan(xj) or np.isnan(yj) or cj < p_cutoff:
                continue

            p1 = (int(xi * sx + ox), int(yi * sy + oy))
            p2 = (int(xj * sx + ox), int(yj * sy + oy))

            if st.mode == SkeletonColorMode.GRADIENT_KEYPOINTS:
                c1 = keypoint_colors[i]
                c2 = keypoint_colors[j]
                self._draw_gradient_edge(overlay, p1, p2, c1, c2, th, st.gradient_steps)
            else:
                # SOLID: priority edge_colors > override > style.color > default_color
                color = self.edge_colors.get((i, j), color_override or st.color or self.default_color)
                cv2.line(overlay, p1, p2, color, th, lineType=cv2.LINE_AA)

        return SkeletonRenderStatus(SkeletonRenderCode.OK, "")

    def draw_many(
        self,
        overlay: np.ndarray,
        poses: np.ndarray,
        p_cutoff: float,
        offset: tuple[int, int],
        scale: tuple[float, float],
        *,
        style: SkeletonStyle | None = None,
        color_override: BGR | None = None,
        keypoint_colors: list[BGR] | None = None,
    ) -> SkeletonRenderStatus:
        poses = np.asarray(poses)
        if poses.ndim != 3:
            return SkeletonRenderStatus(
                SkeletonRenderCode.POSE_SHAPE_INVALID,
                f"Multi-pose must be (A,N,2/3); got shape={poses.shape}",
            )

        expected = len(self.keypoints)
        if poses.shape[1] != expected:
            return SkeletonRenderStatus(
                SkeletonRenderCode.KEYPOINT_COUNT_MISMATCH,
                f"Skeleton expects {expected} keypoints, but poses have N={poses.shape[1]}.",
            )

        for pose in poses:
            st = self.draw(
                overlay,
                pose,
                p_cutoff,
                offset,
                scale,
                style=style,
                color_override=color_override,
                keypoint_colors=keypoint_colors,
            )
            if not st.rendered:
                return st

        return SkeletonRenderStatus(SkeletonRenderCode.OK, "")
