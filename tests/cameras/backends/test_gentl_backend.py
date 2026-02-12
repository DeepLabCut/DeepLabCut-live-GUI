# tests/cameras/backends/test_gentl_backend.py
from __future__ import annotations

import types

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Core lifecycle + strict transaction model
# ---------------------------------------------------------------------
def test_open_starts_stream_and_read_returns_frame(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    be.open()
    assert be._harvester is not None
    assert be._acquirer is not None

    # Strict model validated via behavior: read must succeed after normal open()
    frame, ts = be.read()
    assert isinstance(ts, float)
    assert isinstance(frame, np.ndarray)
    assert frame.size > 0
    # Backend converts to BGR; ensure 3-channel output
    assert frame.ndim == 3 and frame.shape[2] == 3

    be.close()
    assert be._harvester is None
    assert be._acquirer is None
    assert be._device_label is None


def test_fast_start_does_not_start_stream_and_read_times_out(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory(properties={"gentl": {"fast_start": True}})
    be = gb.GenTLCameraBackend(settings)

    be.open()
    assert be._acquirer is not None

    # Strict model: fast_start -> open() does NOT start acquisition -> read must fail
    with pytest.raises(TimeoutError):
        be.read()

    be.close()


def test_close_is_idempotent(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    be = gb.GenTLCameraBackend(gentl_settings_factory())
    be.open()
    be.close()
    # Must not raise
    be.close()


def test_stop_is_safe_before_open_and_after_close(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk
    be = gb.GenTLCameraBackend(gentl_settings_factory())

    # stop before open should not raise
    be.stop()

    be.open()

    # stop should make acquisition unusable for strict fetch/read
    be.stop()
    with pytest.raises(TimeoutError):
        be.read()

    be.close()

    # stop after close should not raise
    be.stop()


def test_read_before_open_raises_runtimeerror(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk
    be = gb.GenTLCameraBackend(gentl_settings_factory())

    with pytest.raises(RuntimeError):
        be.read()


# ---------------------------------------------------------------------
# Device selection + robustness (behavior/state based, minimal circularity)
# ---------------------------------------------------------------------
def test_device_id_exact_match_selects_correct_device_and_updates_index(
    patch_gentl_sdk, gentl_settings_factory, gentl_inventory
):
    gb = patch_gentl_sdk

    # Two devices; device_id targets SER1 => should bind to index 1
    gentl_inventory[:] = [
        {
            "display_name": "Dev0 (SER0)",
            "model": "M0",
            "vendor": "V",
            "serial_number": "SER0",
            "id_": "ID0",
            "tl_type": "Custom",
            "user_defined_name": "U0",
            "version": "1.0.0",
            "access_status": 1000,
        },
        {
            "display_name": "Dev1 (SER1)",
            "model": "M1",
            "vendor": "V",
            "serial_number": "SER1",
            "id_": "ID1",
            "tl_type": "Custom",
            "user_defined_name": "U1",
            "version": "1.0.0",
            "access_status": 1000,
        },
    ]

    settings = gentl_settings_factory(index=0, properties={"gentl": {"device_id": "serial:SER1"}})
    be = gb.GenTLCameraBackend(settings)
    be.open()

    # Backend observable outcome: settings.index updated
    assert int(be.settings.index) == 1

    # Backend observable outcome: persisted identity and serial
    ns = settings.properties.get("gentl", {})
    assert ns.get("device_id") == "serial:SER1"
    assert ns.get("serial_number") == "SER1"

    be.close()


def test_ambiguous_serial_prefix_raises(patch_gentl_sdk, gentl_settings_factory, gentl_inventory):
    gb = patch_gentl_sdk

    gentl_inventory[:] = [
        {"display_name": "DevA", "serial_number": "ABC-1"},
        {"display_name": "DevB", "serial_number": "ABC-2"},
    ]

    settings = gentl_settings_factory(properties={"gentl": {"device_id": "serial:ABC"}})
    be = gb.GenTLCameraBackend(settings)

    with pytest.raises(RuntimeError):
        be.open()


def test_open_index_out_of_range_raises(patch_gentl_sdk, gentl_settings_factory, gentl_inventory):
    gb = patch_gentl_sdk

    gentl_inventory[:] = [{"display_name": "OnlyDev", "serial_number": "SER0"}]
    settings = gentl_settings_factory(index=5)
    be = gb.GenTLCameraBackend(settings)

    with pytest.raises(RuntimeError):
        be.open()


def test_missing_serial_produces_fingerprint_device_id(patch_gentl_sdk, gentl_settings_factory, gentl_inventory):
    gb = patch_gentl_sdk

    gentl_inventory[:] = [
        {
            "display_name": "DevNoSerial",
            "vendor": "V",
            "model": "M",
            "serial_number": "",  # missing/blank
            "id_": "ID-NO-SERIAL",
            "tl_type": "Custom",
            "user_defined_name": "U",
            "version": "1.0.0",
            "access_status": 1000,
        }
    ]

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)
    be.open()

    ns = settings.properties["gentl"]
    assert isinstance(ns.get("device_id"), str)
    assert ns["device_id"].startswith("fp:")

    # Rich metadata should still be persisted
    assert ns.get("device_info_id") == "ID-NO-SERIAL"
    assert ns.get("device_display_name") == "DevNoSerial"

    be.close()


# ---------------------------------------------------------------------
# Persistence contract (UI relies on these keys)
# ---------------------------------------------------------------------
def test_open_persists_rich_metadata_in_namespace(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)
    be.open()

    ns = settings.properties.get("gentl", {})
    assert isinstance(ns, dict)

    # Identity basics
    assert "device_id" in ns and str(ns["device_id"])
    assert "cti_file" in ns and str(ns["cti_file"])

    # Rich info keys (minimum contract)
    for k in (
        "device_display_name",
        "device_info_id",
        "device_vendor",
        "device_model",
        "device_tl_type",
        "device_user_defined_name",
        "device_version",
        "device_access_status",
    ):
        assert k in ns, f"Missing persisted key: {k}"

    # If serial exists, it should be persisted
    assert ns.get("serial_number")
    assert ns.get("device_serial_number")

    # device_name derived from node_map label resolution
    assert ns.get("device_name")

    be.close()


def test_open_persists_cti_file_even_when_provided_in_props(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory(properties={"cti_file": "from-props.cti", "gentl": {}})
    be = gb.GenTLCameraBackend(settings)
    be.open()

    ns = settings.properties["gentl"]
    assert isinstance(ns.get("cti_file"), str) and ns["cti_file"]

    be.close()


# ---------------------------------------------------------------------
# Discovery / ping / rebind (still unit-only via patched SDK)
# ---------------------------------------------------------------------
def test_discover_devices_returns_device_id_and_label(patch_gentl_sdk):
    gb = patch_gentl_sdk

    cams = gb.GenTLCameraBackend.discover_devices(max_devices=10)
    assert isinstance(cams, list)
    assert cams

    cam0 = cams[0]
    assert getattr(cam0, "label", "")
    assert getattr(cam0, "device_id", None) is not None
    assert str(cam0.device_id).startswith(("serial:", "fp:"))


def test_discover_devices_prefers_display_name_for_label(patch_gentl_sdk, gentl_inventory):
    gb = patch_gentl_sdk

    gentl_inventory[:] = [
        {"display_name": "Pretty Name (SERX)", "vendor": "V", "model": "M", "serial_number": "SERX", "id_": "IDX"}
    ]

    cams = gb.GenTLCameraBackend.discover_devices(max_devices=10)
    assert cams and cams[0].label == "Pretty Name (SERX)"


def test_quick_ping_true_for_existing_false_for_missing(patch_gentl_sdk, gentl_inventory):
    gb = patch_gentl_sdk

    gentl_inventory[:] = [{"display_name": "Dev0", "serial_number": "SER0"}]
    assert gb.GenTLCameraBackend.quick_ping(0) is True
    assert gb.GenTLCameraBackend.quick_ping(1) is False


def test_rebind_settings_updates_index_using_device_id_with_attribute_entries(
    patch_gentl_sdk, gentl_settings_factory, gentl_inventory
):
    """
    rebind_settings has some getattr(...) usage; feed attribute-like entries to match that path.
    """
    gb = patch_gentl_sdk

    gentl_inventory[:] = [
        types.SimpleNamespace(
            display_name="Dev0", serial_number="SER0", vendor="V", model="M0", id_="ID0", tl_type="T"
        ),
        types.SimpleNamespace(
            display_name="Dev1", serial_number="SER1", vendor="V", model="M1", id_="ID1", tl_type="T"
        ),
    ]

    settings = gentl_settings_factory(index=0, properties={"gentl": {"device_id": "serial:SER1"}})
    out = gb.GenTLCameraBackend.rebind_settings(settings)

    assert int(out.index) == 1
    ns = out.properties.get("gentl", {})
    assert ns.get("device_id") == "serial:SER1"


def test_rebind_settings_no_device_id_no_change(patch_gentl_sdk, gentl_settings_factory, gentl_inventory):
    gb = patch_gentl_sdk

    gentl_inventory[:] = [
        {"display_name": "Dev0", "serial_number": "SER0"},
        {"display_name": "Dev1", "serial_number": "SER1"},
    ]
    settings = gentl_settings_factory(index=1, properties={"gentl": {}})
    out = gb.GenTLCameraBackend.rebind_settings(settings)

    assert int(out.index) == 1


# ---------------------------------------------------------------------
# _configure_* coverage (assert on node_map side effects, not logs)
# ---------------------------------------------------------------------
def test_resolution_auto_does_not_modify_node_dimensions(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory(width=0, height=0)  # Auto
    be = gb.GenTLCameraBackend(settings)
    be.open()

    nm = be._acquirer.remote_device.node_map
    assert int(nm.Width.value) == 1920
    assert int(nm.Height.value) == 1080

    be.close()


def test_resolution_request_is_aligned_to_increment(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory(width=641, height=481)  # odd -> should align
    be = gb.GenTLCameraBackend(settings)
    be.open()

    nm = be._acquirer.remote_device.node_map
    assert int(nm.Width.value) % 2 == 0
    assert int(nm.Height.value) % 2 == 0

    assert be.actual_resolution is not None
    w, h = be.actual_resolution
    assert w == int(nm.Width.value)
    assert h == int(nm.Height.value)

    be.close()


def test_manual_exposure_gain_fps_are_applied_when_nonzero(patch_gentl_sdk, gentl_settings_factory):
    """
    Covers _configure_exposure/_configure_gain/_configure_frame_rate success path.
    """
    gb = patch_gentl_sdk

    settings = gentl_settings_factory(exposure=20000, gain=3.0, fps=50.0)
    be = gb.GenTLCameraBackend(settings)
    be.open()

    nm = be._acquirer.remote_device.node_map
    assert float(nm.ExposureTime.value) == pytest.approx(20000.0)
    assert float(nm.Gain.value) == pytest.approx(3.0)
    assert float(nm.AcquisitionFrameRate.value) == pytest.approx(50.0)

    be.close()


def test_pixel_format_unavailable_does_not_crash_open_and_streams(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory(properties={"gentl": {"pixel_format": "NotAFormat"}})
    be = gb.GenTLCameraBackend(settings)
    be.open()

    # No fake-internal checks; just verify it can read
    frame, _ = be.read()
    assert frame is not None and frame.size > 0

    be.close()


# ---------------------------------------------------------------------
# Direct unit tests for _create_acquirer (fallback paths + error aggregation)
# ---------------------------------------------------------------------
def test__create_acquirer_prefers_create_serial_dict(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    class H:
        device_info_list = [{"serial_number": "SER0"}]

        def create(self, selector=None, index=None):
            # Return different sentinels depending on call form
            if isinstance(selector, dict) and selector.get("serial_number") == "SERX":
                return "ACQ_SERIAL_DICT"
            raise RuntimeError("unexpected call")

    be._harvester = H()
    acq = be._create_acquirer("SERX", 0)
    assert acq == "ACQ_SERIAL_DICT"


def test__create_acquirer_index_kw_typeerror_falls_back_to_positional(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    class H:
        device_info_list = [{"serial_number": "SER0"}]

        def create(self, *args, **kwargs):
            # Simulate a Harvester that does NOT accept keyword index
            if "index" in kwargs:
                raise TypeError("index kw not supported")
            # Positional index works
            if len(args) == 1 and args[0] == 2:
                return "ACQ_POS_INDEX"
            raise RuntimeError("unexpected call")

    be._harvester = H()
    acq = be._create_acquirer(None, 2)
    assert acq == "ACQ_POS_INDEX"


def test__create_acquirer_falls_back_to_create_image_acquirer_when_create_fails(
    patch_gentl_sdk, gentl_settings_factory
):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    class H:
        device_info_list = [{"serial_number": "SER0"}]

        def create(self, *args, **kwargs):
            raise RuntimeError("create fails")

        def create_image_acquirer(self, selector=None, index=None):
            # Succeeds here
            if isinstance(selector, dict) and selector.get("serial_number") == "SERX":
                return "ACQ_CIA_SERIAL"
            if index == 1:
                return "ACQ_CIA_INDEX"
            return "ACQ_CIA_OTHER"

    be._harvester = H()
    acq = be._create_acquirer("SERX", 1)
    assert acq == "ACQ_CIA_SERIAL"


def test__create_acquirer_uses_device_info_fallback_when_available(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    device_info_obj = {"serial_number": "SER0", "id_": "ID0"}

    class H:
        device_info_list = [device_info_obj]

        def create(self, *args, **kwargs):
            # Fail index, succeed if given device_info object
            if "index" in kwargs or (len(args) == 1 and isinstance(args[0], int)):
                raise RuntimeError("index path fails")
            if len(args) == 1 and args[0] is device_info_obj:
                return "ACQ_DEVICE_INFO"
            raise RuntimeError("unexpected call")

    be._harvester = H()
    acq = be._create_acquirer(None, 0)
    assert acq == "ACQ_DEVICE_INFO"


def test__create_acquirer_tries_default_create_when_index0_and_no_serial(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    class H:
        device_info_list = [{"serial_number": "SER0"}]

        def create(self, *args, **kwargs):
            # Fail index attempts; succeed only on no-arg create()
            if args or kwargs:
                raise RuntimeError("only no-arg create works")
            return "ACQ_DEFAULT"

    be._harvester = H()
    acq = be._create_acquirer(None, 0)
    assert acq == "ACQ_DEFAULT"


def test__create_acquirer_raises_runtimeerror_with_joined_errors(patch_gentl_sdk, gentl_settings_factory):
    gb = patch_gentl_sdk

    settings = gentl_settings_factory()
    be = gb.GenTLCameraBackend(settings)

    class H:
        device_info_list = [{"serial_number": "SER0"}]

        def create(self, *args, **kwargs):
            raise RuntimeError("create boom")

        def create_image_acquirer(self, *args, **kwargs):
            raise RuntimeError("cia boom")

    be._harvester = H()

    with pytest.raises(RuntimeError) as ei:
        be._create_acquirer("SERX", 0)

    # Error message should include some context about attempted creation methods
    msg = str(ei.value).lower()
    assert "failed to initialise gentl image acquirer" in msg
