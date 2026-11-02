import numpy as np
import pytest

from dlclivegui.display import (
    compute_tile_info,
    compute_tiling_geometry,
    create_tiled_frame,
)
from dlclivegui.display.display import keypoint_colors_bgr

pytestmark = pytest.mark.unit


def test_compute_tiling_geometry_empty():
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry({})
    assert cam_ids == []
    assert (rows, cols) == (1, 1)
    assert (tile_w, tile_h) == (640, 480)


def test_compute_tiling_geometry_single_frame_respects_max_canvas_and_min_tile(test_frame):
    frames = {"camA": test_frame(480, 640, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))
    assert cam_ids == ["camA"]
    assert (rows, cols) == (1, 1)
    assert tile_w >= 160
    assert tile_h >= 120
    assert tile_w <= 1200
    assert tile_h <= 800


def test_compute_tiling_geometry_two_frames_is_1x2(test_frame):
    frames = {"camB": test_frame(480, 640, 3), "camA": test_frame(480, 640, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))

    # Preserve insertion/display order, do not sort by camera ID.
    assert cam_ids == ["camB", "camA"]
    assert (rows, cols) == (1, 2)
    assert tile_w >= 160 and tile_h >= 120


def test_compute_tiling_geometry_three_frames_is_2x2(test_frame):
    frames = {"c3": test_frame(480, 640, 3), "c1": test_frame(480, 640, 3), "c2": test_frame(480, 640, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))

    # Preserve insertion/display order.
    assert cam_ids == ["c3", "c1", "c2"]
    assert (rows, cols) == (2, 2)
    assert tile_w >= 160 and tile_h >= 120


def test_compute_tiling_geometry_reference_aspect_is_first_display_order_cam(test_frame):
    # camB is first in insertion/display order and has aspect 0.5.
    # camA has aspect 2.0.
    frames = {
        "camB": test_frame(400, 200, 3),  # aspect = 200 / 400 = 0.5
        "camA": test_frame(200, 400, 3),  # aspect = 400 / 200 = 2.0
    }

    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))

    assert cam_ids == ["camB", "camA"]

    # For 2 cams, rows=1 cols=2 => initial tile_w=600 tile_h=800 => tile_aspect=0.75
    # frame_aspect for camB = 0.5 <= 0.75 => tile_w adjusted to tile_h * frame_aspect = 800 * 0.5 = 400
    assert (rows, cols) == (1, 2)
    assert tile_w == 400
    assert tile_h == 800


def test_compute_tiling_geometry_preserves_frame_insertion_order():
    frames = {
        "gentl:serial:30220469": np.zeros((10, 20, 3), dtype=np.uint8),
        "gentl:serial:10620051": np.zeros((10, 20, 3), dtype=np.uint8),
    }

    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames)

    assert cam_ids == ["gentl:serial:30220469", "gentl:serial:10620051"]
    assert rows == 1
    assert cols == 2
    assert tile_w > 0
    assert tile_h > 0


def test_compute_tiling_geometry_preserves_reversed_insertion_order():
    frames = {
        "gentl:serial:10620051": np.zeros((10, 20, 3), dtype=np.uint8),
        "gentl:serial:30220469": np.zeros((10, 20, 3), dtype=np.uint8),
    }

    cam_ids, *_ = compute_tiling_geometry(frames)

    assert cam_ids == ["gentl:serial:10620051", "gentl:serial:30220469"]


def test_compute_tile_info_uses_display_order_for_offsets():
    cam0 = "gentl:serial:30220469"
    cam1 = "gentl:serial:10620051"

    frames = {
        cam0: np.zeros((100, 200, 3), dtype=np.uint8),
        cam1: np.zeros((100, 200, 3), dtype=np.uint8),
    }

    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames)

    offset0, scale0 = compute_tile_info(cam0, frames[cam0], frames)
    offset1, scale1 = compute_tile_info(cam1, frames[cam1], frames)

    assert cam_ids == [cam0, cam1]
    assert offset0 == (0, 0)
    assert offset1 == (tile_w, 0)
    assert scale0[0] > 0
    assert scale0[1] > 0
    assert scale1[0] > 0
    assert scale1[1] > 0


def test_create_tiled_frame_preserves_display_order_by_tile_content():
    # First frame is blue-ish, second is red-ish.
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    first[:, :] = (255, 0, 0)  # BGR blue

    second = np.zeros((100, 100, 3), dtype=np.uint8)
    second[:, :] = (0, 0, 255)  # BGR red

    frames = {
        "gentl:serial:30220469": first,
        "gentl:serial:10620051": second,
    }

    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(400, 200))
    out = create_tiled_frame(frames, max_canvas=(400, 200))

    assert cam_ids == ["gentl:serial:30220469", "gentl:serial:10620051"]
    assert (rows, cols) == (1, 2)

    # Sample away from text label area.
    left_sample = out[tile_h // 2, tile_w // 2]
    right_sample = out[tile_h // 2, tile_w + tile_w // 2]

    assert left_sample[0] > left_sample[2]  # blue tile first
    assert right_sample[2] > right_sample[0]  # red tile second


def test_create_tiled_frame_empty_returns_default_canvas():
    out = create_tiled_frame({})
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_create_tiled_frame_grayscale_converted_and_labeled(test_frame):
    # Use a zero grayscale frame; any nonzero in output likely comes from putText label
    frames = {"camA": test_frame(120, 160, c=1, value=0)}
    out = create_tiled_frame(frames, max_canvas=(320, 240))

    assert out.ndim == 3 and out.shape[2] == 3
    # Label should introduce some nonzero (green) pixels
    assert np.any(out != 0)


def test_create_tiled_frame_bgra_converted_and_labeled(test_frame):
    # BGRA frame
    bgra = test_frame(120, 160, c=4, value=0)
    frames = {"camA": bgra}
    out = create_tiled_frame(frames, max_canvas=(320, 240))

    assert out.ndim == 3 and out.shape[2] == 3
    assert np.any(out != 0)


def test_create_tiled_frame_canvas_shape_matches_geometry(test_frame):
    frames = {
        "camA": test_frame(200, 400, 3, value=0),
        "camB": test_frame(200, 400, 3, value=0),
    }
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(800, 400))
    out = create_tiled_frame(frames, max_canvas=(800, 400))
    assert out.shape == (rows * tile_h, cols * tile_w, 3)
    # both tiles should get labels (nonzero pixels)
    assert np.any(out != 0)


def test_compute_tile_info_offset_and_scale_matches_tiling(test_frame):
    # 2 frames => 1x2 tiling, preserving insertion/display order: ["cam2", "cam1"]
    frames = {"cam2": test_frame(200, 400, 3), "cam1": test_frame(200, 400, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))

    original = test_frame(200, 400, 3)
    (ox, oy), (sx, sy) = compute_tile_info("cam2", original, frames, max_canvas=(1200, 800))

    assert cam_ids == ["cam2", "cam1"]
    assert (rows, cols) == (1, 2)

    # cam2 is first in display order => row 0 col 0
    assert ox == 0
    assert oy == 0
    assert sx == pytest.approx(tile_w / 400)
    assert sy == pytest.approx(tile_h / 200)


def test_keypoint_colors_returns_requested_count() -> None:
    colors = keypoint_colors_bgr("viridis", 5)

    assert len(colors) == 5
    assert all(len(color) == 3 for color in colors)
    assert all(0 <= channel <= 255 for color in colors for channel in color)


def test_keypoint_colors_zero_count_returns_empty_tuple() -> None:
    assert keypoint_colors_bgr("viridis", 0) == ()


def test_keypoint_colors_one_count_returns_one_color() -> None:
    colors = keypoint_colors_bgr("viridis", 1)

    assert len(colors) == 1


def test_keypoint_colors_are_deterministic() -> None:
    first = keypoint_colors_bgr("viridis", 4)
    second = keypoint_colors_bgr("viridis", 4)

    assert first == second


def test_keypoint_colors_reject_negative_count() -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        keypoint_colors_bgr("viridis", -1)
