import pytest

from dlclivegui.config import (
    ApplicationSettings,
    CameraSettings,
    CameraTriggerSettings,
    MultiCameraSettings,
    RecordingSettings,
)


@pytest.mark.unit
def test_missing_trigger_config_defaults_to_off():
    cam = CameraSettings(
        backend="gentl",
        properties={"gentl": {}},
    )

    trigger = cam.get_trigger_settings()

    assert trigger.role == "off"
    assert trigger.source == "auto"


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["gentl", "basler"])
def test_explicit_trigger_config_roundtrips_through_application_settings(backend):
    cam = CameraSettings(
        backend=backend,
        properties={},
    )
    cam.set_trigger_settings(
        CameraTriggerSettings(
            role="follower",
            source="Line1",
            strict=True,
        )
    )

    settings = ApplicationSettings(
        camera=cam,
        multi_camera=MultiCameraSettings(cameras=[cam]),
    )
    restored = ApplicationSettings.from_dict(settings.to_dict())

    top_level_trigger = restored.camera.get_trigger_settings()
    multi_camera_trigger = restored.multi_camera.cameras[0].get_trigger_settings()

    for trigger in (top_level_trigger, multi_camera_trigger):
        assert trigger.role == "follower"
        assert trigger.source == "Line1"
        assert trigger.strict is True


@pytest.mark.unit
def test_save_does_not_insert_implicit_trigger_config():
    cam = CameraSettings(
        backend="gentl",
        properties={"gentl": {}},
    )
    settings = ApplicationSettings(
        camera=cam,
        multi_camera=MultiCameraSettings(cameras=[cam]),
    )

    data = settings.to_dict()

    assert data["camera"]["properties"]["gentl"] == {}
    assert data["multi_camera"]["cameras"][0]["properties"]["gentl"] == {}


@pytest.mark.unit
def test_trigger_source_defaults_to_auto():
    trigger = CameraTriggerSettings()

    assert trigger.source == "auto"


def test_recording_settings_writegear_options_default():
    settings = RecordingSettings(codec="libx264", crf=23, fast_encoding=False)

    opts = settings.writegear_options(100.0)

    assert opts["-input_framerate"] == "100.000000"
    assert opts["-vcodec"] == "libx264"
    assert opts["-crf"] == "23"
    assert "-preset" not in opts
    assert "-tune" not in opts


def test_recording_settings_writegear_options_fast_encoding_x264():
    settings = RecordingSettings(codec="libx264", crf=23, fast_encoding=True)

    opts = settings.writegear_options(100.0)

    assert opts["-input_framerate"] == "100.000000"
    assert opts["-vcodec"] == "libx264"
    assert opts["-crf"] == "23"
    assert opts["-preset"] == "ultrafast"
    assert opts["-tune"] == "zerolatency"


def test_recording_settings_writegear_options_fast_encoding_nvenc_no_x264_options():
    settings = RecordingSettings(codec="h264_nvenc", crf=23, fast_encoding=True)

    opts = settings.writegear_options(100.0)

    assert opts["-vcodec"] == "h264_nvenc"
    assert "-preset" not in opts
    assert "-tune" not in opts


def test_recording_settings_writegear_options_invalid_fps_falls_back_to_30():
    settings = RecordingSettings(codec="libx264", crf=23)

    opts = settings.writegear_options(None)

    assert opts["-input_framerate"] == "30.000000"
