from __future__ import annotations

import numpy as np
import pytest

from dlclivegui.cameras.base import CapturedFrame
from dlclivegui.utils.timestamps import FrameTimestampMetadata

# ---------------------------------------------------------------------
# Core lifecycle
# ---------------------------------------------------------------------


def test_basler_open_starts_grabbing_and_read_returns_frame(patched_basler_sdk, basler_settings_factory):
    bb = patched_basler_sdk

    settings = basler_settings_factory()
    be = bb.BaslerCameraBackend(settings)

    be.open()

    assert be._camera is not None
    assert be._camera.IsOpen()
    assert be._camera.IsGrabbing()
    assert be._converter is not None

    payload = be.read()
    frame, ts = payload.frame, payload.software_timestamp
    assert isinstance(ts, float)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (10, 10, 3)

    be.close()
    assert be._camera is None
    assert be._converter is None


def test_basler_fast_start_does_not_start_grabbing_and_read_raises(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(properties={"basler": {"fast_start": True}})
    be = bb.BaslerCameraBackend(settings)

    be.open()

    assert be._camera is not None
    assert be._camera.IsOpen()
    assert not be._camera.IsGrabbing()
    assert be._converter is None

    with pytest.raises(RuntimeError, match="fast-start"):
        be.read()

    be.close()


def test_basler_close_is_idempotent(patched_basler_sdk, basler_settings_factory):
    bb = patched_basler_sdk

    be = bb.BaslerCameraBackend(basler_settings_factory())
    be.open()
    be.close()
    be.close()


def test_basler_stop_before_open_and_after_close_is_safe(patched_basler_sdk, basler_settings_factory):
    bb = patched_basler_sdk

    be = bb.BaslerCameraBackend(basler_settings_factory())

    be.stop()

    be.open()
    be.stop()

    assert be._camera is not None
    assert not be._camera.IsGrabbing()

    be.close()
    be.stop()


def test_basler_read_before_open_raises_runtimeerror(patched_basler_sdk, basler_settings_factory):
    bb = patched_basler_sdk

    be = bb.BaslerCameraBackend(basler_settings_factory())

    with pytest.raises(RuntimeError, match="not opened"):
        be.read()


# ---------------------------------------------------------------------
# Discovery / identity / rebind
# ---------------------------------------------------------------------


def test_basler_discover_devices_returns_serial_identity_and_label(
    patched_basler_sdk,
):
    bb = patched_basler_sdk

    cams = bb.BaslerCameraBackend.discover_devices(max_devices=10)

    assert len(cams) == 2
    assert cams[0].device_id == "FAKE-BASLER-0"
    assert "Basler" in cams[0].label
    assert "FAKE-BASLER-0" in cams[0].label
    assert cams[0].path


def test_basler_quick_ping_true_for_existing_false_for_missing(patched_basler_sdk):
    bb = patched_basler_sdk

    assert bb.BaslerCameraBackend.quick_ping(0) is True
    assert bb.BaslerCameraBackend.quick_ping(1) is True
    assert bb.BaslerCameraBackend.quick_ping(2) is False


def test_basler_rebind_settings_uses_serial_device_id(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        index=0,
        properties={"basler": {"device_id": "FAKE-BASLER-1"}},
    )

    out = bb.BaslerCameraBackend.rebind_settings(settings)

    assert int(out.index) == 1
    ns = out.properties["basler"]
    assert ns["device_id"] == "FAKE-BASLER-1"
    assert ns["device_name"]


def test_basler_open_selects_device_id_and_persists_identity(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        index=0,
        properties={"basler": {"device_id": "FAKE-BASLER-1"}},
    )

    be = bb.BaslerCameraBackend(settings)
    be.open()

    ns = settings.properties["basler"]
    assert ns["device_id"] == "FAKE-BASLER-1"
    assert ns["device_name"]

    be.close()


def test_basler_open_index_out_of_range_raises(patched_basler_sdk, basler_settings_factory):
    bb = patched_basler_sdk

    settings = basler_settings_factory(index=99)
    be = bb.BaslerCameraBackend(settings)

    with pytest.raises(RuntimeError, match="out of range"):
        be.open()


# ---------------------------------------------------------------------
# Camera controls
# ---------------------------------------------------------------------


def test_basler_resolution_auto_does_not_modify_dimensions(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(width=0, height=0)
    be = bb.BaslerCameraBackend(settings)

    be.open()

    assert be._camera.Width.GetValue() == 1920
    assert be._camera.Height.GetValue() == 1080
    assert be.actual_resolution == (1920, 1080)

    be.close()


def test_basler_resolution_request_snaps_to_increment(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(width=641, height=481)
    be = bb.BaslerCameraBackend(settings)

    be.open()

    assert be._camera.Width.GetValue() == 640
    assert be._camera.Height.GetValue() == 480
    assert be.actual_resolution == (640, 480)

    be.close()


def test_basler_exposure_gain_fps_are_applied_when_nonzero(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(exposure=20000, gain=2.5, fps=50.0)
    be = bb.BaslerCameraBackend(settings)

    be.open()

    assert be._camera.ExposureAuto.GetValue() == "Off"
    assert be._camera.ExposureTime.GetValue() == pytest.approx(20000.0)
    assert be._camera.GainAuto.GetValue() == "Off"
    assert be._camera.Gain.GetValue() == pytest.approx(2.5)
    assert be._camera.AcquisitionFrameRateEnable.GetValue() is True
    assert be._camera.AcquisitionFrameRate.GetValue() == pytest.approx(50.0)

    be.close()


# ---------------------------------------------------------------------
# Basler trigger behavior
# ---------------------------------------------------------------------


def test_basler_static_capabilities_advertises_hardware_trigger_best_effort_and_mono(
    patched_basler_sdk,
):
    bb = patched_basler_sdk
    from dlclivegui.cameras.base import SupportLevel

    caps = bb.BaslerCameraBackend.static_capabilities()
    assert caps["hardware_trigger"] == SupportLevel.BEST_EFFORT
    assert caps["preserve_mono"] == SupportLevel.SUPPORTED


def test_basler_default_trigger_is_off_and_free_runs(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory()
    be = bb.BaslerCameraBackend(settings)

    be.open()

    assert be._camera.TriggerMode.GetValue() == "Off"
    assert be.waits_for_hardware_trigger is False

    payload = be.read()
    frame = payload.frame
    assert frame.shape == (10, 10, 3)

    be.close()


def test_basler_follower_auto_selects_line1_and_times_out_waiting_for_trigger(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        properties={
            "basler": {
                "trigger": {
                    "role": "follower",
                    "selector": "FrameStart",
                    "source": "auto",
                    "activation": "RisingEdge",
                    "timeout": 5.0,
                    "strict": False,
                }
            }
        }
    )

    be = bb.BaslerCameraBackend(settings)

    # Timeout is configured in seconds but pypylon RetrieveResult uses ms;
    # hardware-trigger waits should be capped for responsive shutdown.
    assert be._retrieve_timeout_ms == 1000

    be.open()

    assert be.waits_for_hardware_trigger is True
    assert be._camera.TriggerSelector.GetValue() == "FrameStart"
    assert be._camera.TriggerSource.GetValue() == "Line1"
    assert be._camera.TriggerActivation.GetValue() == "RisingEdge"
    assert be._camera.TriggerMode.GetValue() == "On"

    with pytest.raises(TimeoutError, match="waiting for hardware trigger"):
        be.read()

    assert be._camera.retrieve_calls[-1] == 1000

    be.close()


def test_basler_follower_strict_invalid_source_raises(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        properties={
            "basler": {
                "trigger": {
                    "role": "follower",
                    "selector": "FrameStart",
                    "source": "NotARealSource",
                    "activation": "RisingEdge",
                    "strict": True,
                }
            }
        }
    )

    be = bb.BaslerCameraBackend(settings)

    with pytest.raises(RuntimeError, match="TriggerSource"):
        be.open()


def test_basler_follower_non_strict_invalid_source_disables_trigger(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        properties={
            "basler": {
                "trigger": {
                    "role": "follower",
                    "source": "NotARealSource",
                    "strict": False,
                }
            }
        }
    )

    be = bb.BaslerCameraBackend(settings)
    be.open()

    assert be._camera.TriggerMode.GetValue() == "Off"
    assert be.waits_for_hardware_trigger is False

    payload = be.read()
    frame = payload.frame
    assert frame.shape == (10, 10, 3)

    be.close()


def test_basler_master_configures_generic_line_output_and_restores_on_close(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        properties={
            "basler": {
                "trigger": {
                    "role": "master",
                    "output_line": "Line2",
                    "output_source": "ExposureActive",
                    "strict": False,
                }
            }
        }
    )

    be = bb.BaslerCameraBackend(settings)
    be.open()

    cam = be._camera
    assert cam.LineSelector.GetValue() == "Line2"
    assert cam.LineMode.GetValue() == "Output"
    assert cam.LineSource.GetValue() == "ExposureActive"
    assert be.waits_for_hardware_trigger is False

    be.close()

    # Local reference remains valid after backend clears self._camera.
    assert cam.LineSource.GetValue() == "Off"
    assert cam.LineMode.GetValue() == "Input"


@pytest.mark.xfail(reason="Software trigger support is not implemented yet.")
def test_basler_software_trigger_requires_trigger_once_before_read(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        properties={
            "basler": {
                "trigger": {
                    "role": "software",
                    "selector": "FrameStart",
                    "strict": False,
                }
            }
        }
    )

    be = bb.BaslerCameraBackend(settings)
    be.open()

    assert be._camera.TriggerMode.GetValue() == "On"
    assert be._camera.TriggerSource.GetValue() == "Software"
    assert be.waits_for_hardware_trigger is False

    # No software trigger has been fired yet.
    with pytest.raises(RuntimeError, match="Failed to retrieve image"):
        be.read()

    be.trigger_once()
    assert be._camera.software_trigger_calls == 1

    frame = be.read().frame
    assert frame.shape == (10, 10, 3)

    be.close()


def test_basler_close_turns_input_trigger_off(
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    settings = basler_settings_factory(
        properties={
            "basler": {
                "trigger": {
                    "role": "external",
                    "source": "Line1",
                    "activation": "RisingEdge",
                }
            }
        }
    )

    be = bb.BaslerCameraBackend(settings)
    be.open()

    cam = be._camera
    assert cam.TriggerMode.GetValue() == "On"

    be.close()

    assert cam.TriggerMode.GetValue() == "Off"


def test_basler_hardware_trigger_maps_pylon_timeout_to_timeout_error(
    monkeypatch,
    patched_basler_sdk,
    basler_settings_factory,
):
    bb = patched_basler_sdk

    class FakePylonTimeout(Exception):
        pass

    settings = basler_settings_factory(
        properties={
            "basler": {
                "trigger": {
                    "role": "follower",
                    "source": "Line1",
                }
            }
        }
    )
    backend = bb.BaslerCameraBackend(settings)
    backend.open()

    pylon_to = bb.genicam.TimeoutException

    def raise_timeout(*_args, **_kwargs):
        raise pylon_to("Simulated timeout")

    monkeypatch.setattr(backend._camera, "RetrieveResult", raise_timeout)

    try:
        with pytest.raises(
            TimeoutError,
            match="waiting for hardware trigger",
        ) as exc_info:
            backend.read()

        assert isinstance(exc_info.value.__cause__, bb.genicam.TimeoutException)

    finally:
        backend.close()


class TestBaslerFrameTimestamps:
    @pytest.mark.unit
    def test_read_returns_captured_frame_with_hardware_timestamp_metadata(
        self,
        patched_basler_sdk,
        basler_settings_factory,
    ):
        bb = patched_basler_sdk

        settings = basler_settings_factory()
        be = bb.BaslerCameraBackend(settings)
        be.open()

        captured = be.read()

        assert isinstance(captured, CapturedFrame)
        assert captured.frame is not None
        assert isinstance(captured.software_timestamp, float)

        meta = captured.timestamp_metadata
        assert isinstance(meta, FrameTimestampMetadata)

        assert meta.backend == "basler"
        assert meta.source == "grab_result.GetTimeStamp"
        assert meta.kind == "camera_clock"
        assert meta.raw_unit == "ticks"
        assert meta.raw_value == 123456789
        assert meta.tick_frequency_hz == pytest.approx(1_000_000_000.0)
        assert meta.seconds == pytest.approx(0.123456789)
        assert meta.default_reported == "seconds"

        source_dict = meta.to_source_dict()
        assert source_dict["backend"] == "basler"
        assert source_dict["source"] == "grab_result.GetTimeStamp"

        frame_dict = meta.to_frame_dict()
        assert frame_dict["seconds"] == pytest.approx(0.123456789)
        assert frame_dict["raw_value"] == 123456789

        be.close()

    @pytest.mark.unit
    def test_make_timestamp_metadata_returns_none_for_zero_ticks(
        self,
        patched_basler_sdk,
        basler_settings_factory,
    ):
        bb = patched_basler_sdk
        backend = bb.BaslerCameraBackend(basler_settings_factory())

        class GrabResult:
            @staticmethod
            def GetTimeStamp():
                return 0

        metadata = backend._make_timestamp_metadata(GrabResult())

        assert metadata is None

    import pytest

    @pytest.mark.parametrize("frequency", [None, 0.0, -1.0])
    def test_make_timestamp_metadata_uses_raw_ticks_for_invalid_frequency(
        self,
        patched_basler_sdk,
        basler_settings_factory,
        frequency,
    ):
        bb = patched_basler_sdk
        backend = bb.BaslerCameraBackend(basler_settings_factory())
        backend._timestamp_tick_frequency_hz = frequency
        backend._timestamp_tick_frequency_source = None

        class GrabResult:
            @staticmethod
            def GetTimeStamp():
                return 12_345

        metadata = backend._make_timestamp_metadata(GrabResult())

        assert metadata is not None
        assert metadata.default_reported == "raw_value"
        assert metadata.seconds is None
        assert metadata.raw_value == 12_345
        assert metadata.tick_frequency_hz == frequency
