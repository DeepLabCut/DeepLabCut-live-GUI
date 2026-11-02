from __future__ import annotations

import pytest

from dlclivegui.display.skeleton import (
    SkeletonDefinition,
    SkeletonEdge,
    SkeletonResolutionError,
    resolve_packet_skeleton,
    resolve_skeleton,
)


class PacketStub:
    def __init__(
        self,
        *,
        keypoint_names: list[str] | None,
        skeleton_id: str | None,
        skeleton_edges: tuple[tuple[str, str], ...] | None,
    ) -> None:
        self.keypoint_names = keypoint_names
        self.skeleton_id = skeleton_id
        self.skeleton_edges = skeleton_edges


@pytest.fixture
def simple_definition() -> SkeletonDefinition:
    return SkeletonDefinition(
        identifier="test.simple",
        display_name="Simple",
        edges=(
            SkeletonEdge("nose", "shoulder"),
            SkeletonEdge("shoulder", "hip"),
        ),
    )


def test_resolve_skeleton_maps_names_to_pose_order(
    simple_definition: SkeletonDefinition,
) -> None:
    resolved = resolve_skeleton(
        simple_definition,
        ["hip", "nose", "shoulder"],
    )

    assert resolved.keypoint_names == (
        "hip",
        "nose",
        "shoulder",
    )
    assert resolved.edges == (
        (1, 2),
        (2, 0),
    )


def test_resolve_skeleton_rejects_empty_keypoint_names(
    simple_definition: SkeletonDefinition,
) -> None:
    with pytest.raises(
        SkeletonResolutionError,
        match="without keypoint names",
    ):
        resolve_skeleton(simple_definition, [])


def test_resolve_skeleton_rejects_duplicate_names(
    simple_definition: SkeletonDefinition,
) -> None:
    with pytest.raises(
        SkeletonResolutionError,
        match="duplicate",
    ):
        resolve_skeleton(
            simple_definition,
            ["nose", "nose", "hip"],
        )


def test_resolve_skeleton_reports_all_missing_names(
    simple_definition: SkeletonDefinition,
) -> None:
    with pytest.raises(SkeletonResolutionError) as exc_info:
        resolve_skeleton(
            simple_definition,
            ["nose"],
        )

    message = str(exc_info.value)
    assert "shoulder" in message
    assert "hip" in message


def test_resolve_packet_skeleton_returns_none_without_names() -> None:
    packet = PacketStub(
        keypoint_names=None,
        skeleton_id="test.simple",
        skeleton_edges=(("nose", "shoulder"),),
    )

    assert resolve_packet_skeleton(packet) is None


def test_resolve_packet_skeleton_returns_none_without_edges() -> None:
    packet = PacketStub(
        keypoint_names=["nose", "shoulder"],
        skeleton_id="test.simple",
        skeleton_edges=None,
    )

    assert resolve_packet_skeleton(packet) is None


def test_resolve_packet_skeleton_constructs_topology() -> None:
    packet = PacketStub(
        keypoint_names=["shoulder", "nose"],
        skeleton_id="test.simple",
        skeleton_edges=(("nose", "shoulder"),),
    )

    resolved = resolve_packet_skeleton(packet)

    assert resolved is not None
    assert resolved.edges == ((1, 0),)
    assert resolved.keypoint_names == (
        "shoulder",
        "nose",
    )
