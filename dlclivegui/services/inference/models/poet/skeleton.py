# dlclivegui/services/inference/models/poet/skeleton.py
from __future__ import annotations

from dlclivegui.display.skeleton import (
    SkeletonDefinition,
    SkeletonEdge,
)

POET_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


POET_SKELETON = SkeletonDefinition(
    identifier="poet.coco17",
    display_name="POET COCO-17",
    edges=(
        SkeletonEdge("left_ear", "left_eye"),
        SkeletonEdge("right_ear", "right_eye"),
        SkeletonEdge("left_eye", "nose"),
        SkeletonEdge("nose", "right_eye"),
        SkeletonEdge("left_shoulder", "right_shoulder"),
        SkeletonEdge("left_shoulder", "left_elbow"),
        SkeletonEdge("right_shoulder", "right_elbow"),
        SkeletonEdge("nose", "left_shoulder"),
        SkeletonEdge("nose", "right_shoulder"),
        SkeletonEdge("left_shoulder", "left_hip"),
        SkeletonEdge("right_shoulder", "right_hip"),
        SkeletonEdge("left_elbow", "left_wrist"),
        SkeletonEdge("right_elbow", "right_wrist"),
        SkeletonEdge("left_hip", "right_hip"),
        SkeletonEdge("left_hip", "left_knee"),
        SkeletonEdge("right_hip", "right_knee"),
        SkeletonEdge("left_knee", "left_ankle"),
        SkeletonEdge("right_knee", "right_ankle"),
    ),
)
