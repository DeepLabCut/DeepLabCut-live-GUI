# tests/cameras/backends/test_gentl_trigger.py
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------
# GenTL hardware trigger configuration
# ---------------------------------------------------------------------


def _gentl_trigger_settings(gentl_settings_factory, trigger: dict, **kwargs):
    """Build CameraSettings with a GenTL trigger block."""
    return gentl_settings_factory(properties={"gentl": {"trigger": trigger}}, **kwargs)


def test_gentl_capabilities_advertise_hardware_trigger_best_effort(patch_gentl_sdk):
    gb = patch_gentl_sdk

    caps = gb.GenTLCameraBackend.static_capabilities()

    assert caps.get("hardware_trigger") == gb.SupportLevel.BEST_EFFORT


def test_trigger_default_off_configures_trigger_mode_off(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerMode.value == "Off"

    ns = settings.properties.get("gentl", {})
    assert ns.get("trigger_actual", {}).get("role") == "off"

    be.close()


def test_trigger_explicit_off_configures_trigger_mode_off(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(gentl_settings_factory, {"role": "off"})
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerMode.value == "Off"
    assert settings.properties["gentl"]["trigger_actual"]["role"] == "off"

    be.close()


def test_trigger_external_configures_input_line_and_timeout(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "selector": "FrameStart",
            "source": "Line0",
            "activation": "RisingEdge",
            "timeout": 10.0,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerSelector.value == "FrameStart"
    assert nm.TriggerSource.value == "Line0"
    assert nm.TriggerActivation.value == "RisingEdge"
    assert nm.TriggerMode.value == "On"
    assert be._timeout == pytest.approx(10.0)

    ns = settings.properties["gentl"]
    assert ns["trigger_actual"]["role"] == "external"
    assert ns["trigger_actual"]["source"] == "Line0"

    be.close()


def test_trigger_follower_configures_input_line(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "follower",
            "selector": "FrameStart",
            "source": "Line1",
            "activation": "FallingEdge",
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerSelector.value == "FrameStart"
    assert nm.TriggerSource.value == "Line1"
    assert nm.TriggerActivation.value == "FallingEdge"
    assert nm.TriggerMode.value == "On"

    ns = settings.properties["gentl"]
    assert ns["trigger_actual"]["role"] == "follower"

    be.close()


def test_trigger_master_configures_output_line_and_keeps_trigger_off(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "master",
            "output_line": "Line2",
            "output_source": "ExposureActive",
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerMode.value == "Off"
    assert nm.LineSelector.value == "Line2"
    assert nm.LineMode.value == "Output"
    assert nm.LineSource.value == "ExposureActive"

    ns = settings.properties["gentl"]
    assert ns["trigger_actual"]["role"] == "master"
    assert ns["trigger_actual"]["output_line"] == "Line2"

    be.close()


def test_trigger_invalid_source_non_strict_does_not_crash(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "source": "LineDoesNotExist",
            "strict": False,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    # Source was unsupported, so the fake node should retain its default.
    assert nm.TriggerSource.value == "Line0"
    # Non-strict mode should still allow opening; TriggerMode may be enabled
    # because TriggerSource failure is best-effort in this mode.
    assert be._acquirer is not None

    be.close()


def test_trigger_invalid_source_strict_raises_and_cleans_up(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "source": "LineDoesNotExist",
            "strict": True,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    with pytest.raises(RuntimeError):
        be.open()

    assert be._harvester is None
    assert be._shared_entry is None
    assert be._acquirer is None


def test_trigger_invalid_master_output_source_non_strict_does_not_crash(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "master",
            "output_line": "Line2",
            "output_source": "NotARealLineSource",
            "strict": False,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerMode.value == "Off"
    assert nm.LineSelector.value == "Line2"
    assert nm.LineMode.value == "Output"
    # Unsupported source should not be applied in non-strict mode.
    assert nm.LineSource.value == "Off"

    be.close()


def test_trigger_invalid_master_output_source_strict_raises_and_cleans_up(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "master",
            "output_line": "Line2",
            "output_source": "NotARealLineSource",
            "strict": True,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    with pytest.raises(RuntimeError):
        be.open()

    assert be._harvester is None
    assert be._shared_entry is None
    assert be._acquirer is None


def test_trigger_alias_on_maps_to_external(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "on",
            "source": "Line1",
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerMode.value == "On"
    assert nm.TriggerSource.value == "Line1"
    assert settings.properties["gentl"]["trigger_actual"]["role"] == "external"

    be.close()


def test_trigger_timeout_overrides_default_fetch_timeout(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "timeout": 7.5,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    assert be._timeout == pytest.approx(7.5)

    # Fake acquisition is started, so read should pass and record the timeout.
    frame, _ = be.read()
    assert frame is not None
    assert be._acquirer.fetch_calls[-1] == pytest.approx(7.5)

    be.close()


def test_trigger_timeout_error_mentions_hardware_trigger_when_waiting(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "timeout": 3.0,
        },
        # fast_start keeps acquisition stopped; fake fetch then raises timeout.
        # This lets us assert the backend timeout message without hardware.
    )
    settings.properties["gentl"]["fast_start"] = True
    be = gb.GenTLCameraBackend(settings)

    be.open()

    with pytest.raises(TimeoutError) as ei:
        be.read()

    msg = str(ei.value).lower()
    assert "gentl timeout" in msg
    assert "hardware trigger" in msg or "trigger" in msg

    be.close()


def test_trigger_actual_is_persisted_for_debugging(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "follower",
            "source": "Line1",
            "activation": "FallingEdge",
            "timeout": 9.0,
            "strict": False,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()

    actual = settings.properties["gentl"].get("trigger_actual")
    assert isinstance(actual, dict)
    assert actual["role"] == "follower"
    assert actual["source"] == "Line1"
    assert actual["activation"] == "FallingEdge"
    assert actual["timeout"] == pytest.approx(9.0)

    be.close()
