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
            "timeout": gb.GenTLCameraBackend._MAX_HARDWARE_TRIGGER_FETCH_TIMEOUT,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    assert nm.TriggerSelector.value == "FrameStart"
    assert nm.TriggerSource.value == "Line0"
    assert nm.TriggerActivation.value == "RisingEdge"
    assert nm.TriggerMode.value == "On"
    assert be.waits_for_hardware_trigger is True
    assert be._timeout == pytest.approx(gb.GenTLCameraBackend._MAX_HARDWARE_TRIGGER_FETCH_TIMEOUT)

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


def test_trigger_invalid_source_non_strict_disables_trigger(patch_gentl_sdk, gentl_settings_factory):
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
    assert nm.TriggerSource.value == "Line1"

    # Safety behavior: do not arm TriggerMode on the previous/default source.
    assert nm.TriggerMode.value == "Off"

    # Controller should not treat timeouts as expected trigger waits.
    assert be.waits_for_hardware_trigger is False

    # trigger_actual is persisted after _configure_trigger(); since we reset
    # self._trigger to off, the effective trigger state is off.
    actual = settings.properties["gentl"]["trigger_actual"]
    assert actual["role"] == "off"

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


def test_trigger_timeout_is_capped_for_hardware_trigger_fetch_polling(
    patch_gentl_sdk,
    gentl_settings_factory,
):
    gb = patch_gentl_sdk
    expected_fetch_timeout = gb.GenTLCameraBackend._MAX_HARDWARE_TRIGGER_FETCH_TIMEOUT

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "timeout": 7.5,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    try:
        be.open()

        # Hardware-trigger fetch calls are intentionally capped so stop(wait=True)
        # is not blocked by a long user trigger timeout.
        assert be._timeout == pytest.approx(expected_fetch_timeout)

        # Fake acquisition is started, so read should pass and record the capped timeout.
        frame = be.read().frame
        assert frame is not None
        assert be._acquirer.fetch_calls[-1] == pytest.approx(expected_fetch_timeout)

        # The requested trigger timeout is still preserved in persisted trigger_actual.
        actual = settings.properties["gentl"]["trigger_actual"]
        assert actual["timeout"] == pytest.approx(7.5)

    finally:
        be.close()


def test_trigger_timeout_error_mentions_hardware_trigger_when_waiting(
    patch_gentl_sdk,
    gentl_settings_factory,
):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "timeout": 3.0,
        },
    )
    # fast_start keeps acquisition stopped; fake fetch then raises timeout.
    # This lets us assert the backend timeout message without hardware.
    settings.properties["gentl"]["fast_start"] = True

    be = gb.GenTLCameraBackend(settings)

    try:
        be.open()

        assert be._timeout == pytest.approx(gb.GenTLCameraBackend._MAX_HARDWARE_TRIGGER_FETCH_TIMEOUT)

        with pytest.raises(TimeoutError) as ei:
            be.read()

        msg = str(ei.value).lower()
        assert "gentl timeout" in msg
        assert "hardware trigger" in msg or "trigger" in msg

    finally:
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

    try:
        be.open()

        # Requested timeout remains in trigger_actual for debugging/config visibility.
        actual = settings.properties["gentl"].get("trigger_actual")
        assert isinstance(actual, dict)
        assert actual["role"] == "follower"
        assert actual["source"] == "Line1"
        assert actual["activation"] == "FallingEdge"
        assert actual["timeout"] == pytest.approx(9.0)

        # But each blocking Harvester.fetch() call is capped for responsive shutdown.
        assert be._timeout == pytest.approx(gb.GenTLCameraBackend._MAX_HARDWARE_TRIGGER_FETCH_TIMEOUT)

    finally:
        be.close()


def test_trigger_invalid_selector_non_strict_disables_trigger(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "external",
            "selector": "NotARealSelector",
            "source": "Line1",
            "strict": False,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    be.open()
    nm = be._acquirer.remote_device.node_map

    # Selector was unsupported, so the fake node should retain its default.
    assert nm.TriggerSelector.value == "FrameStart"

    # Source may have been applied, but trigger must not be armed because
    # the required selector routing failed.
    assert nm.TriggerSource.value == "Line1"
    assert nm.TriggerMode.value == "Off"
    assert be.waits_for_hardware_trigger is False

    actual = settings.properties["gentl"]["trigger_actual"]
    assert actual["role"] == "off"

    be.close()


def test_trigger_timeout_not_capped_for_master_mode(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = _gentl_trigger_settings(
        gentl_settings_factory,
        {
            "role": "master",
            "timeout": 7.5,
        },
    )
    be = gb.GenTLCameraBackend(settings)

    try:
        be.open()

        # Master is free-running / trigger-generating, not waiting for hardware input.
        assert be.waits_for_hardware_trigger is False
        assert be._timeout == pytest.approx(7.5)

    finally:
        be.close()


def test_resolve_trigger_source_auto_selects_supported_line(
    patch_gentl_sdk,
    gentl_settings_factory,
):
    gb = patch_gentl_sdk
    be = gb.GenTLCameraBackend(gentl_settings_factory())

    class Node:
        symbolics = ["Line1", "Software", "Any"]

    class NodeMap:
        TriggerSource = Node()

    source, ok = be._resolve_trigger_source(NodeMap(), "auto", strict=False)

    assert ok is True
    assert source == "Line1"


def test_resolve_trigger_source_strict_raises_for_unsupported_explicit_line(
    patch_gentl_sdk,
    gentl_settings_factory,
):
    gb = patch_gentl_sdk
    be = gb.GenTLCameraBackend(gentl_settings_factory())

    class Node:
        symbolics = ["Line1", "Software", "Any"]

    class NodeMap:
        TriggerSource = Node()

    with pytest.raises(RuntimeError, match="TriggerSource.*Line0"):
        be._resolve_trigger_source(NodeMap(), "Line0", strict=True)
