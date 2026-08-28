import numpy as np
import pytest

from dlclivegui.display import (
    draw_bbox,
    draw_keypoints,
    draw_pose,
)
from dlclivegui.display.skeleton import (
    ResolvedSkeleton,
    SkeletonColorMode,
    SkeletonDefinition,
    SkeletonEdge,
    SkeletonRenderCode,
    SkeletonStyle,
    draw_skeleton,
)


def _frame(h, w, c=3, value=0, dtype=np.uint8):
    """Helper to create test frames with predictable content."""
    if c == 1:
        return (np.ones((h, w), dtype=dtype) * value).astype(dtype)
    return (np.ones((h, w, c), dtype=dtype) * value).astype(dtype)


def test_draw_bbox_invalid_bbox_returns_same_object():
    frame = _frame(100, 100, 3)
    out = draw_bbox(frame, (10, 10, 10, 20), (0, 255, 0))  # x0 == x1 invalid
    assert out is frame  # passthrough for invalid bbox


def test_draw_bbox_draws_rectangle_and_clips():
    frame = _frame(60, 60, 3, value=0)
    color = (0, 0, 255)  # red in BGR

    # bbox partially outside original; with scale/offset it will be shifted/clipped
    out = draw_bbox(
        frame,
        bbox_xyxy=(-10, -10, 50, 50),
        color_bgr=color,
        offset=(5, 5),
        scale=(1.0, 1.0),
    )

    assert out is not frame
    # Should have drawn something
    assert np.any(out != frame)
    # At least some red pixels should exist (allowing for thickness)
    assert np.any((out[:, :, 2] > 0) & (out[:, :, 0] == 0) & (out[:, :, 1] == 0))


def test_draw_keypoints_filters_by_cutoff_and_nans_and_draws():
    overlay = _frame(80, 80, 3, value=0).copy()
    cmap = __import__("matplotlib.pyplot").pyplot.get_cmap("viridis")

    # keypoints: (x, y, conf)
    kpts = np.array(
        [
            [10.0, 10.0, 0.2],  # below cutoff -> ignored
            [np.nan, 15.0, 0.99],  # NaN -> ignored
            [20.0, np.nan, 0.99],  # NaN -> ignored
            [30.0, 30.0, 0.99],  # should draw
        ],
        dtype=float,
    )

    draw_keypoints(
        overlay=overlay,
        p_cutoff=0.9,
        sx=1.0,
        ox=0,
        sy=1.0,
        oy=0,
        radius=3,
        cmap=cmap,
        keypoints=kpts,
        marker=None,  # circle
    )

    assert np.any(overlay != 0)  # something drawn


def test_draw_pose_single_animal_draws_when_conf_above_cutoff():
    frame = _frame(100, 100, 3, value=0)
    pose = np.array(
        [
            [10.0, 10.0, 0.95],
            [20.0, 20.0, 0.95],
        ],
        dtype=float,
    )
    out = draw_pose(frame, pose, p_cutoff=0.9, colormap="viridis", offset=(0, 0), scale=(1.0, 1.0))
    assert out is not frame
    assert np.any(out != frame)


def test_draw_pose_single_animal_no_draw_below_cutoff():
    frame = _frame(100, 100, 3, value=0)
    pose = np.array([[10.0, 10.0, 0.1]], dtype=float)
    out = draw_pose(frame, pose, p_cutoff=0.9, colormap="viridis", offset=(0, 0), scale=(1.0, 1.0))
    # overlay returned, but should be identical if nothing is drawn
    assert np.array_equal(out, frame)


def test_draw_pose_multi_animal_draws_distinct_markers():
    frame = _frame(120, 120, 3, value=0)
    # A x N x 3 : 2 animals, 1 keypoint each
    pose = np.array(
        [
            [[30.0, 30.0, 0.99]],
            [[60.0, 60.0, 0.99]],
        ],
        dtype=float,
    )
    out = draw_pose(frame, pose, p_cutoff=0.9, colormap="viridis", offset=(0, 0), scale=(1.0, 1.0))
    assert out is not frame
    assert np.any(out != frame)


@pytest.fixture
def resolved_skeleton() -> ResolvedSkeleton:
    definition = SkeletonDefinition(
        identifier="test.line",
        display_name="Line",
        edges=(SkeletonEdge("a", "b"),),
    )

    return ResolvedSkeleton(
        definition=definition,
        keypoint_names=("a", "b"),
        edges=((0, 1),),
    )


@pytest.fixture
def solid_style() -> SkeletonStyle:
    return SkeletonStyle(
        color_mode=SkeletonColorMode.SOLID,
        color_bgr=(0, 255, 0),
        thickness=2,
        scale_with_zoom=False,
    )


def test_draw_skeleton_renders_single_pose(
    resolved_skeleton: ResolvedSkeleton,
    solid_style: SkeletonStyle,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pose = np.array(
        [
            [10.0, 10.0, 0.9],
            [40.0, 40.0, 0.9],
        ],
        dtype=np.float32,
    )

    result = draw_skeleton(
        frame,
        pose,
        resolved_skeleton,
        solid_style,
        p_cutoff=0.5,
    )

    assert result.code == SkeletonRenderCode.RENDERED
    assert result.edges_drawn == 1
    assert np.any(frame != 0)


def test_draw_skeleton_renders_multiple_people(
    resolved_skeleton: ResolvedSkeleton,
    solid_style: SkeletonStyle,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    poses = np.array(
        [
            [
                [5.0, 5.0, 0.9],
                [20.0, 20.0, 0.9],
            ],
            [
                [25.0, 5.0, 0.9],
                [40.0, 20.0, 0.9],
            ],
        ],
        dtype=np.float32,
    )

    result = draw_skeleton(
        frame,
        poses,
        resolved_skeleton,
        solid_style,
        p_cutoff=0.5,
    )

    assert result.code == SkeletonRenderCode.RENDERED
    assert result.edges_drawn == 2
    assert np.any(frame != 0)


def test_draw_skeleton_skips_low_confidence_edge(
    resolved_skeleton: ResolvedSkeleton,
    solid_style: SkeletonStyle,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pose = np.array(
        [
            [10.0, 10.0, 0.9],
            [40.0, 40.0, 0.1],
        ],
        dtype=np.float32,
    )

    result = draw_skeleton(
        frame,
        pose,
        resolved_skeleton,
        solid_style,
        p_cutoff=0.5,
    )

    assert result.code == SkeletonRenderCode.RENDERED
    assert result.edges_drawn == 0
    assert not np.any(frame)


def test_draw_skeleton_skips_non_finite_edge(
    resolved_skeleton: ResolvedSkeleton,
    solid_style: SkeletonStyle,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pose = np.array(
        [
            [np.nan, 10.0, 0.9],
            [40.0, 40.0, 0.9],
        ],
        dtype=np.float32,
    )

    result = draw_skeleton(
        frame,
        pose,
        resolved_skeleton,
        solid_style,
        p_cutoff=0.5,
    )

    assert result.code == SkeletonRenderCode.RENDERED
    assert result.edges_drawn == 0
    assert not np.any(frame)


@pytest.mark.parametrize(
    "pose",
    [
        np.zeros((2,), dtype=np.float32),
        np.zeros((1, 2, 3, 4), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ],
)
def test_draw_skeleton_rejects_invalid_pose_shape(
    pose: np.ndarray,
    resolved_skeleton: ResolvedSkeleton,
    solid_style: SkeletonStyle,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    result = draw_skeleton(
        frame,
        pose,
        resolved_skeleton,
        solid_style,
        p_cutoff=0.5,
    )

    assert result.code == SkeletonRenderCode.INVALID_POSE


def test_draw_skeleton_reports_keypoint_count_mismatch(
    resolved_skeleton: ResolvedSkeleton,
    solid_style: SkeletonStyle,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pose = np.zeros((3, 3), dtype=np.float32)

    result = draw_skeleton(
        frame,
        pose,
        resolved_skeleton,
        solid_style,
        p_cutoff=0.5,
    )

    assert result.code == (SkeletonRenderCode.KEYPOINT_COUNT_MISMATCH)


def test_draw_skeleton_requires_gradient_colors(
    resolved_skeleton: ResolvedSkeleton,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pose = np.array(
        [
            [10.0, 10.0, 0.9],
            [40.0, 40.0, 0.9],
        ],
        dtype=np.float32,
    )
    style = SkeletonStyle(
        color_mode=SkeletonColorMode.GRADIENT_KEYPOINTS,
    )

    result = draw_skeleton(
        frame,
        pose,
        resolved_skeleton,
        style,
        p_cutoff=0.5,
        keypoint_colors=None,
    )

    assert result.code == (SkeletonRenderCode.COLOR_COUNT_MISMATCH)


def test_draw_skeleton_renders_gradient(
    resolved_skeleton: ResolvedSkeleton,
) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pose = np.array(
        [
            [10.0, 25.0, 0.9],
            [40.0, 25.0, 0.9],
        ],
        dtype=np.float32,
    )
    style = SkeletonStyle(
        color_mode=SkeletonColorMode.GRADIENT_KEYPOINTS,
        thickness=2,
        gradient_steps=8,
        scale_with_zoom=False,
    )

    result = draw_skeleton(
        frame,
        pose,
        resolved_skeleton,
        style,
        p_cutoff=0.5,
        keypoint_colors=(
            (255, 0, 0),
            (0, 0, 255),
        ),
    )

    assert result.code == SkeletonRenderCode.RENDERED
    assert result.edges_drawn == 1
    assert np.any(frame != 0)
