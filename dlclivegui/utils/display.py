# dlclivegui/utils/display.py
from __future__ import annotations

import enum

import cv2
import matplotlib.pyplot as plt
import numpy as np


class BBoxColors(enum.Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def get_all_display_names() -> list[str]:
        return [color.name.capitalize() for color in BBoxColors]


def color_to_rgb(color_name: str) -> tuple[int, int, int]:
    """Convert a color name to an RGB tuple."""
    try:
        return BBoxColors[color_name.upper()].value
    except KeyError:
        raise ValueError(f"Unknown color name: {color_name}") from None


def compute_tiling_geometry(
    frames: dict[str, np.ndarray],
    max_canvas: tuple[int, int] = (1200, 800),
) -> tuple[list[str], int, int, int, int]:
    """Compute consistent tiling geometry for both tiling and overlay transforms.

    Returns:
        (sorted_cam_ids, rows, cols, tile_w, tile_h)

    Notes:
    - We intentionally base tile aspect on the first frame in sorted_cam_ids,
      because create_tiled_frame uses the same ordering. This guarantees that
      compute_tile_info() and create_tiled_frame() agree on tile_w/tile_h.
    - If frames have different aspect ratios, they will be resized (possibly distorted)
      to the same tile size. Overlay scale then matches that same resize.
    """
    if not frames:
        return ([], 1, 1, 640, 480)

    cam_ids = sorted(frames.keys())
    frames_list = [frames[cid] for cid in cam_ids]
    num_frames = len(frames_list)

    if num_frames == 1:
        rows, cols = 1, 1
    elif num_frames == 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2

    max_w, max_h = max_canvas

    # Reference aspect is based on the first frame in sorted order (matches tiler).
    h0, w0 = frames_list[0].shape[:2]
    frame_aspect = (w0 / h0) if h0 > 0 else 1.0

    tile_w = max_w // cols
    tile_h = max_h // rows

    # Adjust tile size to keep the *reference* aspect ratio.
    tile_aspect = (tile_w / tile_h) if tile_h > 0 else 1.0
    if frame_aspect > tile_aspect:
        tile_h = int(tile_w / frame_aspect)
    else:
        tile_w = int(tile_h * frame_aspect)

    tile_w = max(160, int(tile_w))
    tile_h = max(120, int(tile_h))

    return cam_ids, rows, cols, tile_w, tile_h


def create_tiled_frame(frames: dict[str, np.ndarray], max_canvas: tuple[int, int] = (1200, 800)) -> np.ndarray:
    """Create a tiled canvas (1x1, 1x2, or 2x2) with camera-id labels.

    Uses compute_tiling_geometry() so tile_w/tile_h are consistent with compute_tile_info().
    """
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=max_canvas)

    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    # Only show up to rows*cols cameras
    for idx, cam_id in enumerate(cam_ids[: rows * cols]):
        frame = frames[cam_id]

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        resized = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        cv2.putText(
            resized,
            cam_id,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        row = idx // cols
        col = idx % cols
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = resized

    return canvas


def compute_tile_info(
    dlc_cam_id: str,
    original_frame: np.ndarray,
    frames: dict[str, np.ndarray],
    max_canvas: tuple[int, int] = (1200, 800),
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Return ((offset_x, offset_y), (scale_x, scale_y)) for overlaying on the tiled view.

    Critical robustness fix:
    - Tile dimensions are computed from the same reference used by create_tiled_frame()
      (first frame in sorted order), so offsets/scales match the actual tiling.
    """
    if not frames:
        return (0, 0), (1.0, 1.0)

    cam_ids, rows, cols, tile_w, tile_h = compute_tiling_geometry(frames, max_canvas=max_canvas)

    # Which tile contains the DLC camera?
    try:
        dlc_cam_idx = cam_ids.index(dlc_cam_id)
    except ValueError:
        dlc_cam_idx = 0

    row = dlc_cam_idx // cols
    col = dlc_cam_idx % cols
    offset_x = col * tile_w
    offset_y = row * tile_h

    orig_h, orig_w = original_frame.shape[:2]
    scale_x = (tile_w / orig_w) if orig_w > 0 else 1.0
    scale_y = (tile_h / orig_h) if orig_h > 0 else 1.0

    return (offset_x, offset_y), (scale_x, scale_y)


def draw_bbox(
    frame: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    color_bgr: tuple[int, int, int],
    offset: tuple[int, int] = (0, 0),
    scale: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Draw a bbox on the frame, transformed by offset/scale for tiled views."""
    x0, y0, x1, y1 = bbox_xyxy
    if x0 >= x1 or y0 >= y1:
        return frame

    ox, oy = offset
    sx, sy = scale
    x0s = int(x0 * sx + ox)
    y0s = int(y0 * sy + oy)
    x1s = int(x1 * sx + ox)
    y1s = int(y1 * sy + oy)

    h, w = frame.shape[:2]
    x0s = max(0, min(x0s, w - 1))
    y0s = max(0, min(y0s, h - 1))
    x1s = max(x0s + 1, min(x1s, w))
    y1s = max(y0s + 1, min(y1s, h))

    out = frame.copy()
    cv2.rectangle(out, (x0s, y0s), (x1s, y1s), color_bgr, 2)
    return out


def draw_keypoints(overlay, p_cutoff, sx, ox, sy, oy, radius, cmap, keypoints: np.ndarray, marker: int | None) -> None:
    num_kpts = len(keypoints)
    for idx, kpt in enumerate(keypoints):
        if len(kpt) < 2:
            continue
        x, y = kpt[:2]
        conf = kpt[2] if len(kpt) > 2 else 1.0
        if np.isnan(x) or np.isnan(y) or conf < p_cutoff:
            continue

        xs = int(x * sx + ox)
        ys = int(y * sy + oy)

        t = idx / max(num_kpts - 1, 1)
        rgba = cmap(t)
        bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        if marker is None:
            cv2.circle(overlay, (xs, ys), radius, bgr, -1)
        else:
            cv2.drawMarker(overlay, (xs, ys), bgr, marker, radius * 2, 2)


def draw_pose(
    frame: np.ndarray,
    pose: np.ndarray,
    p_cutoff: float,
    colormap: str,
    offset: tuple[int, int],
    scale: tuple[float, float],
    base_radius: int = 4,
) -> np.ndarray:
    """Draw single- or multi-animal pose (N x 3 or A x N x 3) on the frame."""
    overlay = frame.copy()
    pose_arr = np.asarray(pose)
    ox, oy = offset
    sx, sy = scale
    radius = max(2, int(base_radius * min(sx, sy)))
    cmap = plt.get_cmap(colormap)

    if pose_arr.ndim == 3:
        markers = [
            cv2.MARKER_CROSS,
            cv2.MARKER_TILTED_CROSS,
            cv2.MARKER_STAR,
            cv2.MARKER_DIAMOND,
            cv2.MARKER_SQUARE,
            cv2.MARKER_TRIANGLE_UP,
            cv2.MARKER_TRIANGLE_DOWN,
        ]
        for i, animal_pose in enumerate(pose_arr):
            draw_keypoints(
                overlay,
                p_cutoff,
                sx,
                ox,
                sy,
                oy,
                radius,
                cmap,
                animal_pose,
                markers[i % len(markers)],
            )
    else:
        draw_keypoints(overlay, p_cutoff, sx, ox, sy, oy, radius, cmap, pose_arr, marker=None)

    return overlay
