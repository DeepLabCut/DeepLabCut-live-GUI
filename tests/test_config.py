import pytest

from dlclivegui.config import (
    DEFAULT_RECORDING_FPS,
    ApplicationSettings,
    CameraSettings,
    CameraTriggerSettings,
    MultiCameraSettings,
    RecordingSettings,
    WebStreamSettings,
)
from dlclivegui.services.video_recorder import build_writegear_options


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


@pytest.mark.unit
def test_recording_settings_writegear_overrides_default():
    settings = RecordingSettings(
        codec="libx264",
        crf=23,
        fast_encoding=False,
    )

    assert settings.writegear_overrides() == {}


@pytest.mark.unit
@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_recording_settings_writegear_overrides_fast_encoding(codec):
    settings = RecordingSettings(
        codec=codec,
        crf=23,
        fast_encoding=True,
    )

    assert settings.writegear_overrides() == {
        "-preset": "ultrafast",
        "-tune": "zerolatency",
    }


@pytest.mark.unit
def test_recording_settings_writegear_overrides_nvenc():
    settings = RecordingSettings(
        codec="h264_nvenc",
        crf=23,
        fast_encoding=True,
    )

    assert settings.writegear_overrides() == {}


@pytest.mark.unit
def test_build_writegear_options_default():
    settings = RecordingSettings(
        codec="libx264",
        crf=23,
        fast_encoding=False,
    )

    opts = build_writegear_options(
        frame_rate=100.0,
        codec=settings.codec,
        crf=settings.crf,
        overrides=settings.writegear_overrides(),
    )

    assert opts == {
        "-input_framerate": 100.0,
        "-vcodec": "libx264",
        "-crf": 23,
    }


@pytest.mark.unit
def test_build_writegear_options_fast_encoding_x264():
    settings = RecordingSettings(
        codec="libx264",
        crf=23,
        fast_encoding=True,
    )

    opts = build_writegear_options(
        frame_rate=100.0,
        codec=settings.codec,
        crf=settings.crf,
        overrides=settings.writegear_overrides(),
    )

    assert opts == {
        "-input_framerate": 100.0,
        "-vcodec": "libx264",
        "-crf": 23,
        "-preset": "ultrafast",
        "-tune": "zerolatency",
    }


@pytest.mark.unit
def test_build_writegear_options_fast_encoding_nvenc():
    settings = RecordingSettings(
        codec="h264_nvenc",
        crf=23,
        fast_encoding=True,
    )

    opts = build_writegear_options(
        frame_rate=100.0,
        codec=settings.codec,
        crf=settings.crf,
        overrides=settings.writegear_overrides(),
    )

    assert opts == {
        "-input_framerate": 100.0,
        "-vcodec": "h264_nvenc",
        "-crf": 23,
    }


@pytest.mark.unit
def test_build_writegear_options_invalid_fps_uses_default():
    settings = RecordingSettings(
        codec="libx264",
        crf=23,
    )

    opts = build_writegear_options(
        frame_rate=None,
        codec=settings.codec,
        crf=settings.crf,
        overrides=settings.writegear_overrides(),
    )

    assert opts["-input_framerate"] == DEFAULT_RECORDING_FPS


@pytest.mark.unit
def test_web_stream_defaults_when_missing_from_config():
    settings = ApplicationSettings.from_dict({})
    assert settings.web_stream == WebStreamSettings()


@pytest.mark.unit
def test_web_stream_roundtrips_through_json(tmp_path):
    ws = WebStreamSettings(port=9000, fps=2.0, max_width=320, max_height=240, jpeg_quality=50)
    path = tmp_path / "cfg.json"
    ApplicationSettings(web_stream=ws).save(path)
    assert ApplicationSettings.load(path).web_stream == ws


@pytest.mark.unit
def test_web_stream_exposure_is_not_in_shareable_config():
    # On/off and network exposure live in local QSettings; a config file must never carry or apply them.
    assert set(ApplicationSettings().to_dict()["web_stream"]) == {
        "port",
        "fps",
        "max_width",
        "max_height",
        "jpeg_quality",
    }
    loaded = ApplicationSettings.from_dict({"web_stream": {"enabled": True, "host": "0.0.0.0", "port": 9000}})
    assert loaded.web_stream == WebStreamSettings(port=9000)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field, value",
    [("port", 0), ("port", 70000), ("fps", 0.0), ("fps", 60.0), ("jpeg_quality", 0), ("max_width", 0)],
)
def test_web_stream_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        WebStreamSettings(**{field: value})
