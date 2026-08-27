from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlclivegui.config import (
    SkeletonColorMode,
    SkeletonStyle,
    VisualizationSettings,
)


def test_visualization_settings_apply_skeleton_defaults() -> None:
    settings = VisualizationSettings(
        p_cutoff=0.6,
        colormap="hot",
        bbox_color=(0, 0, 255),
    )

    assert settings.show_pose is True
    assert settings.show_skeleton is False
    assert settings.skeleton_style == SkeletonStyle()


def test_visualization_settings_accept_legacy_data() -> None:
    settings = VisualizationSettings.model_validate(
        {
            "p_cutoff": 0.6,
            "colormap": "hot",
            "bbox_color": [0, 0, 255],
        }
    )

    assert settings.show_pose is True
    assert settings.show_skeleton is False
    assert settings.skeleton_style.color_mode == SkeletonColorMode.SOLID
    assert settings.skeleton_style.color_bgr == (0, 255, 255)


def test_visualization_settings_round_trip() -> None:
    original = VisualizationSettings(
        p_cutoff=0.7,
        colormap="viridis",
        bbox_color=(255, 0, 0),
        show_pose=False,
        show_skeleton=True,
        skeleton_style=SkeletonStyle(
            color_mode=SkeletonColorMode.GRADIENT_KEYPOINTS,
            color_bgr=(10, 20, 30),
            thickness=5,
            gradient_steps=24,
            scale_with_zoom=False,
        ),
    )

    restored = VisualizationSettings.model_validate(original.model_dump())

    assert restored == original


@pytest.mark.parametrize("thickness", [0, 21])
def test_skeleton_thickness_rejects_out_of_range_values(
    thickness: int,
) -> None:
    with pytest.raises(ValidationError):
        SkeletonStyle(thickness=thickness)


@pytest.mark.parametrize("steps", [0, 1, 129])
def test_skeleton_gradient_steps_reject_out_of_range_values(
    steps: int,
) -> None:
    with pytest.raises(ValidationError):
        SkeletonStyle(gradient_steps=steps)


def test_skeleton_thickness_scales_with_smallest_axis() -> None:
    style = SkeletonStyle(
        thickness=4,
        scale_with_zoom=True,
    )

    assert style.effective_thickness(0.5, 0.25) == 1
    assert style.effective_thickness(2.0, 1.5) == 6


def test_skeleton_thickness_can_ignore_scale() -> None:
    style = SkeletonStyle(
        thickness=4,
        scale_with_zoom=False,
    )

    assert style.effective_thickness(0.1, 10.0) == 4
