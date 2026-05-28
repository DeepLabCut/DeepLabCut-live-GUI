import numpy as np
import pytest

from dlclivegui.utils.display import (  # noqa: E402
    compute_tile_info,
    compute_tiling_geometry,
    create_tiled_frame,
    draw_bbox,
    draw_keypoints,
    draw_pose,
)

pytestmark = pytest.mark.unit


def _frame(h, w, c=3, value=0, dtype=np.uint8):
    """Helper to create test frames with predictable content."""
    if c == 1:
        return (np.ones((h, w), dtype=dtype) * value).astype(dtype)
    return (np.ones((h, w, c), dtype=dtype) * value).astype(dtype)


def test_compute_tiling_geometry_empty():
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry({})
    assert cam_ids == []
    assert (rows, cols) == (1, 1)
    assert (tile_w, tile_h) == (640, 480)


def test_compute_tiling_geometry_single_frame_respects_max_canvas_and_min_tile():
    frames = {"camA": _frame(480, 640, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))
    assert cam_ids == ["camA"]
    assert (rows, cols) == (1, 1)
    assert tile_w >= 160
    assert tile_h >= 120
    assert tile_w <= 1200
    assert tile_h <= 800


def test_compute_tiling_geometry_two_frames_is_1x2():
    frames = {"camB": _frame(480, 640, 3), "camA": _frame(480, 640, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))

    # Preserve insertion/display order, do not sort by camera ID.
    assert cam_ids == ["camB", "camA"]
    assert (rows, cols) == (1, 2)
    assert tile_w >= 160 and tile_h >= 120


def test_compute_tiling_geometry_three_frames_is_2x2():
    frames = {"c3": _frame(480, 640, 3), "c1": _frame(480, 640, 3), "c2": _frame(480, 640, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))

    # Preserve insertion/display order.
    assert cam_ids == ["c3", "c1", "c2"]
    assert (rows, cols) == (2, 2)
    assert tile_w >= 160 and tile_h >= 120


def test_compute_tiling_geometry_reference_aspect_is_first_display_order_cam():
    # camB is first in insertion/display order and has aspect 0.5.
    # camA has aspect 2.0.
    frames = {
        "camB": _frame(400, 200, 3),  # aspect = 200 / 400 = 0.5
        "camA": _frame(200, 400, 3),  # aspect = 400 / 200 = 2.0
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


def test_create_tiled_frame_grayscale_converted_and_labeled():
    # Use a zero grayscale frame; any nonzero in output likely comes from putText label
    frames = {"camA": _frame(120, 160, c=1, value=0)}
    out = create_tiled_frame(frames, max_canvas=(320, 240))

    assert out.ndim == 3 and out.shape[2] == 3
    # Label should introduce some nonzero (green) pixels
    assert np.any(out != 0)


def test_create_tiled_frame_bgra_converted_and_labeled():
    # BGRA frame
    bgra = _frame(120, 160, c=4, value=0)
    frames = {"camA": bgra}
    out = create_tiled_frame(frames, max_canvas=(320, 240))

    assert out.ndim == 3 and out.shape[2] == 3
    assert np.any(out != 0)


def test_create_tiled_frame_canvas_shape_matches_geometry():
    frames = {
        "camA": _frame(200, 400, 3, value=0),
        "camB": _frame(200, 400, 3, value=0),
    }
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(800, 400))
    out = create_tiled_frame(frames, max_canvas=(800, 400))
    assert out.shape == (rows * tile_h, cols * tile_w, 3)
    # both tiles should get labels (nonzero pixels)
    assert np.any(out != 0)


def test_compute_tile_info_offset_and_scale_matches_tiling():
    # 2 frames => 1x2 tiling, preserving insertion/display order: ["cam2", "cam1"]
    frames = {"cam2": _frame(200, 400, 3), "cam1": _frame(200, 400, 3)}
    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=(1200, 800))

    original = _frame(200, 400, 3)
    (ox, oy), (sx, sy) = compute_tile_info("cam2", original, frames, max_canvas=(1200, 800))

    assert cam_ids == ["cam2", "cam1"]
    assert (rows, cols) == (1, 2)

    # cam2 is first in display order => row 0 col 0
    assert ox == 0
    assert oy == 0
    assert sx == pytest.approx(tile_w / 400)
    assert sy == pytest.approx(tile_h / 200)


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
