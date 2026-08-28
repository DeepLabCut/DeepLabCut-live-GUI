from __future__ import annotations

from dlclivegui.display import BBoxColors
from dlclivegui.gui.misc.color_dropdowns import (
    get_skeleton_style_from_combo,
    make_skeleton_color_combo,
    set_skeleton_combo_from_style,
)


def test_skeleton_combo_contains_gradient_and_all_colors(
    qtbot,
) -> None:
    combo = make_skeleton_color_combo(
        BBoxColors,
        include_icons=False,
    )
    qtbot.addWidget(combo)

    assert combo.count() == 1 + len(BBoxColors)

    gradient = combo.itemData(0)
    assert gradient == {
        "mode": "gradient_keypoints",
        "color": None,
    }


def test_skeleton_combo_round_trips_gradient_style(
    qtbot,
) -> None:
    combo = make_skeleton_color_combo(
        BBoxColors,
        include_icons=False,
    )
    qtbot.addWidget(combo)

    set_skeleton_combo_from_style(
        combo,
        mode="gradient_keypoints",
        color=None,
    )

    assert get_skeleton_style_from_combo(combo) == (
        "gradient_keypoints",
        None,
    )


def test_skeleton_combo_round_trips_solid_color(
    qtbot,
) -> None:
    combo = make_skeleton_color_combo(
        BBoxColors,
        include_icons=False,
    )
    qtbot.addWidget(combo)

    set_skeleton_combo_from_style(
        combo,
        mode="solid",
        color=(0, 255, 0),
    )

    assert get_skeleton_style_from_combo(combo) == (
        "solid",
        (0, 255, 0),
    )


def test_skeleton_combo_falls_back_to_first_solid_color(
    qtbot,
) -> None:
    combo = make_skeleton_color_combo(
        BBoxColors,
        include_icons=False,
    )
    qtbot.addWidget(combo)

    set_skeleton_combo_from_style(
        combo,
        mode="solid",
        color=(1, 2, 3),
    )

    mode, color = get_skeleton_style_from_combo(combo)

    assert mode == "solid"
    assert color == BBoxColors.RED.value


def test_skeleton_combo_can_create_icons(qtbot) -> None:
    combo = make_skeleton_color_combo(
        BBoxColors,
        include_icons=True,
    )
    qtbot.addWidget(combo)

    assert not combo.itemIcon(0).isNull()
