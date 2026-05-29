import pytest

from dlclivegui.config import ApplicationSettings, CameraSettings, CameraTriggerSettings, MultiCameraSettings


@pytest.mark.unit
def test_save_applies_gentl_trigger_defaults_to_top_level_camera():
    cam = CameraSettings(
        backend="gentl",
        properties={"gentl": {}},
    )

    settings = ApplicationSettings(
        camera=cam,
        multi_camera=MultiCameraSettings(cameras=[cam]),
    )

    data = settings.to_dict()

    assert "trigger" in data["camera"]["properties"]["gentl"]


@pytest.mark.unit
def test_save_applies_gentl_trigger_defaults_to_multi_camera():
    cam = CameraSettings(
        backend="gentl",
        properties={"gentl": {}},
    )

    settings = ApplicationSettings(
        multi_camera=MultiCameraSettings(cameras=[cam]),
    )

    data = settings.to_dict()

    assert "trigger" in data["multi_camera"]["cameras"][0]["properties"]["gentl"]


@pytest.mark.unit
def test_trigger_source_defaults_to_auto():
    trigger = CameraTriggerSettings()

    assert trigger.source == "auto"
