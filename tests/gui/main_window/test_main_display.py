from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtGui import QPixmap

from dlclivegui.config import (
    SkeletonColorMode,
    SkeletonStyle,
)
from dlclivegui.display.overlays import (
    BoundingBoxOverlaySettings,
    OverlayRenderer,
    OverlayRenderResult,
    OverlaySettings,
    PoseOverlaySettings,
    SkeletonOverlaySettings,
)
from dlclivegui.gui.misc import color_dropdowns as color_ui


def test_skeleton_controls_use_default_settings(window) -> None:
    assert not window.show_skeleton_checkbox.isChecked()
    assert window.skeleton_thickness_spin.value() == 2

    mode, color = (
        window._skeleton_style_from_ui().color_mode,
        window._skeleton_style_from_ui().color_bgr,
    )

    assert mode == SkeletonColorMode.SOLID
    assert color == (0, 255, 255)


def test_apply_config_restores_skeleton_settings(
    window,
) -> None:
    config = window._config.model_copy(deep=True)
    config.visualization.show_pose = False
    config.visualization.show_skeleton = True
    config.visualization.skeleton_style = SkeletonStyle(
        color_mode=SkeletonColorMode.GRADIENT_KEYPOINTS,
        thickness=6,
        gradient_steps=24,
        scale_with_zoom=False,
    )

    window._apply_config(config)

    assert not window.show_predictions_checkbox.isChecked()
    assert window.show_skeleton_checkbox.isChecked()
    assert window.skeleton_thickness_spin.value() == 6

    style = window._skeleton_style_from_ui()
    assert style.color_mode == SkeletonColorMode.GRADIENT_KEYPOINTS
    assert style.gradient_steps == 24
    assert style.scale_with_zoom is False


def test_visualization_settings_from_ui_include_skeleton(
    window,
) -> None:
    window.show_predictions_checkbox.setChecked(False)
    window.show_skeleton_checkbox.setChecked(True)
    window.skeleton_thickness_spin.setValue(5)

    color_ui.set_skeleton_combo_from_style(
        window.skeleton_color_combo,
        mode="solid",
        color=(0, 255, 0),
    )

    settings = window._visualization_settings_from_ui()

    assert settings.show_pose is False
    assert settings.show_skeleton is True
    assert settings.skeleton_style.thickness == 5
    assert settings.skeleton_style.color_bgr == (0, 255, 0)


def test_update_video_display_delegates_to_overlay_renderer(
    window,
    monkeypatch,
) -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    calls = []

    def fake_render(
        rendered_frame,
        **kwargs,
    ):
        calls.append((rendered_frame, kwargs))
        return OverlayRenderResult(
            frame=rendered_frame.copy(),
        )

    monkeypatch.setattr(
        window._overlay_renderer,
        "render",
        fake_render,
    )
    monkeypatch.setattr(
        "dlclivegui.gui.main_window.frame_to_pixmap",
        lambda _frame, _size: QPixmap(10, 10),
    )

    window._update_video_display(frame)

    assert len(calls) == 1
    assert calls[0][1]["offset"] == window._dlc_tile_offset
    assert calls[0][1]["scale"] == window._dlc_tile_scale


def test_start_inference_clears_overlay_runtime_state(
    window,
    monkeypatch,
) -> None:
    cleared = False

    def fake_clear() -> None:
        nonlocal cleared
        cleared = True

    monkeypatch.setattr(
        window._overlay_renderer,
        "clear_runtime_state",
        fake_clear,
    )
    monkeypatch.setattr(
        window.multi_camera_controller,
        "is_running",
        lambda: True,
    )
    monkeypatch.setattr(
        window,
        "_configure_dlc",
        lambda: True,
    )
    monkeypatch.setattr(
        window._dlc,
        "reset",
        lambda: None,
    )

    window._start_inference()

    assert cleared


@dataclass
class PacketStub:
    keypoint_names: list[str] | None
    skeleton_id: str | None
    skeleton_edges: tuple[tuple[str, str], ...] | None


def test_end_to_end_skeleton_overlay() -> None:
    renderer = OverlayRenderer()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    pose = np.array(
        [
            [10.0, 10.0, 0.9],
            [50.0, 50.0, 0.9],
        ],
        dtype=np.float32,
    )
    packet = PacketStub(
        keypoint_names=["start", "end"],
        skeleton_id="test.line",
        skeleton_edges=(("start", "end"),),
    )
    settings = OverlaySettings(
        pose=PoseOverlaySettings(
            visible=False,
            p_cutoff=0.5,
            colormap="viridis",
        ),
        skeleton=SkeletonOverlaySettings(
            visible=True,
            style=SkeletonStyle(
                color_bgr=(0, 255, 0),
                thickness=2,
                scale_with_zoom=False,
            ),
        ),
        bounding_box=BoundingBoxOverlaySettings(
            visible=False,
            coordinates=(0, 0, 0, 0),
            color_bgr=(0, 0, 255),
        ),
    )

    result = renderer.render(
        frame,
        pose=pose,
        packet=packet,
        overlay_settings=settings,
    )

    assert result.warning is None
    assert np.any(result.frame != 0)
    assert not np.any(frame)
