from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlclivegui.config import (
    SkeletonStyle,
)
from dlclivegui.display.overlays import (
    BoundingBoxOverlaySettings,
    OverlayRenderer,
    OverlaySettings,
    PoseOverlaySettings,
    SkeletonOverlaySettings,
)


@dataclass
class PacketStub:
    keypoint_names: list[str] | None
    skeleton_id: str | None
    skeleton_edges: tuple[tuple[str, str], ...] | None


def make_settings(
    *,
    show_pose: bool = False,
    show_skeleton: bool = False,
    show_bbox: bool = False,
    style: SkeletonStyle | None = None,
) -> OverlaySettings:
    return OverlaySettings(
        pose=PoseOverlaySettings(
            visible=show_pose,
            p_cutoff=0.5,
            colormap="viridis",
        ),
        skeleton=SkeletonOverlaySettings(
            visible=show_skeleton,
            style=style or SkeletonStyle(),
        ),
        bounding_box=BoundingBoxOverlaySettings(
            visible=show_bbox,
            coordinates=(1, 1, 8, 8),
            color_bgr=(0, 0, 255),
        ),
    )


def make_pose() -> np.ndarray:
    return np.array(
        [
            [2.0, 2.0, 0.9],
            [8.0, 8.0, 0.9],
        ],
        dtype=np.float32,
    )


def make_packet() -> PacketStub:
    return PacketStub(
        keypoint_names=["a", "b"],
        skeleton_id="test.line",
        skeleton_edges=(("a", "b"),),
    )


def test_renderer_does_not_modify_input_frame() -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    original = frame.copy()

    result = renderer.render(
        frame,
        pose=None,
        packet=None,
        overlay_settings=make_settings(),
    )

    assert np.array_equal(frame, original)
    assert result.frame is not frame


def test_renderer_warns_once_for_missing_skeleton_metadata() -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    first = renderer.render(
        frame,
        pose=make_pose(),
        packet=None,
        overlay_settings=make_settings(
            show_skeleton=True,
        ),
    )
    second = renderer.render(
        frame,
        pose=make_pose(),
        packet=None,
        overlay_settings=make_settings(
            show_skeleton=True,
        ),
    )

    assert first.warning is not None
    assert second.warning is None


def test_renderer_does_not_warn_when_skeleton_is_hidden() -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = renderer.render(
        frame,
        pose=make_pose(),
        packet=None,
        overlay_settings=make_settings(
            show_skeleton=False,
        ),
    )

    assert result.warning is None


def test_renderer_renders_skeleton_without_pose_markers() -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    result = renderer.render(
        frame,
        pose=make_pose(),
        packet=make_packet(),
        overlay_settings=make_settings(
            show_pose=False,
            show_skeleton=True,
        ),
    )

    assert result.warning is None
    assert np.any(result.frame != 0)


def test_renderer_clear_runtime_state_resets_warning_deduplication() -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    settings = make_settings(show_skeleton=True)
    pose = make_pose()

    first = renderer.render(
        frame,
        pose=pose,
        packet=None,
        overlay_settings=settings,
    )
    second = renderer.render(
        frame,
        pose=pose,
        packet=None,
        overlay_settings=settings,
    )

    renderer.clear_runtime_state()

    third = renderer.render(
        frame,
        pose=pose,
        packet=None,
        overlay_settings=settings,
    )

    expected_warning = "Skeleton metadata is unavailable for this pose output."

    assert first.warning == expected_warning
    assert second.warning is None
    assert third.warning == expected_warning


def test_renderer_caches_skeleton_resolution(
    monkeypatch,
) -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    packet = make_packet()
    calls = 0

    from dlclivegui.display import overlays

    original = overlays.resolve_packet_skeleton

    def counting_resolver(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(
        overlays,
        "resolve_packet_skeleton",
        counting_resolver,
    )

    for _ in range(3):
        renderer.render(
            frame,
            pose=make_pose(),
            packet=packet,
            overlay_settings=make_settings(
                show_skeleton=True,
            ),
        )

    assert calls == 1
