# tests/gui/camera_config/test_trigger_config_dialog.py
from __future__ import annotations

import pytest

from dlclivegui.config import CameraSettings
from dlclivegui.gui.camera_config.trigger_config_dialog import (
    TriggerConfigDialog,
    _backend_namespace,
    trigger_ui_profile_for_backend,
)


class TestTriggerUiProfiles:
    @pytest.mark.parametrize(
        ("backend", "supports_input", "supports_master", "show_strobe", "show_line"),
        [
            ("gentl", True, True, True, True),
            ("basler", True, True, False, True),
            ("opencv", False, False, False, False),
            ("fake", False, False, False, False),
        ],
    )
    def test_profile_capabilities_by_backend(
        self,
        backend: str,
        supports_input: bool,
        supports_master: bool,
        show_strobe: bool,
        show_line: bool,
    ):
        profile = trigger_ui_profile_for_backend(backend)

        assert profile.supports_input is supports_input
        assert profile.supports_master is supports_master
        assert profile.show_strobe_fields is show_strobe
        assert profile.show_line_output_fields is show_line

    def test_profile_backend_is_case_insensitive(self):
        upper = trigger_ui_profile_for_backend("GeNtL")
        lower = trigger_ui_profile_for_backend("gentl")

        assert upper == lower


class TestBackendNamespace:
    def test_backend_namespace_creates_backend_dict(self):
        cam = CameraSettings(backend="gentl", index=0, properties={})

        ns = _backend_namespace(cam)

        assert ns == {}
        assert cam.properties == {"gentl": {}}

    def test_backend_namespace_replaces_non_dict_properties(self):
        cam = CameraSettings(backend="gentl", index=0, properties={})
        cam.properties = None

        ns = _backend_namespace(cam)
        ns["trigger"] = {"role": "external"}

        assert cam.properties == {"gentl": {"trigger": {"role": "external"}}}

    def test_backend_namespace_replaces_non_dict_namespace(self):
        cam = CameraSettings(backend="gentl", index=0, properties={"gentl": "bad"})

        ns = _backend_namespace(cam)

        assert ns == {}
        assert cam.properties == {"gentl": {}}


class TestTriggerConfigDialogPresentation:
    @pytest.mark.gui
    def test_unknown_backend_exposes_only_off_role_and_disables_trigger_fields(self, qtbot):
        cam = CameraSettings(backend="opencv", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        roles = [dlg.role_combo.itemData(i) for i in range(dlg.role_combo.count())]

        assert roles == ["off"]
        assert not dlg.selector_edit.isVisible()
        assert not dlg.source_combo.isVisible()
        assert not dlg.activation_combo.isVisible()
        assert not dlg.output_line_edit.isVisible()
        assert not dlg.strobe_polarity_combo.isVisible()

    @pytest.mark.gui
    def test_gentl_profile_shows_input_master_line_and_strobe_fields(self, qtbot):
        cam = CameraSettings(backend="gentl", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        roles = [dlg.role_combo.itemData(i) for i in range(dlg.role_combo.count())]

        assert roles == ["off", "external", "follower", "master"]
        assert not dlg.selector_edit.isHidden()
        assert not dlg.source_combo.isHidden()
        assert not dlg.activation_combo.isHidden()
        assert not dlg.output_line_edit.isHidden()
        assert not dlg.output_source_edit.isHidden()
        assert not dlg.strobe_polarity_combo.isHidden()
        assert not dlg.strobe_operation_combo.isHidden()
        assert not dlg.strobe_duration_spin.isHidden()
        assert not dlg.strobe_delay_spin.isHidden()

    @pytest.mark.gui
    def test_basler_profile_shows_line_fields_but_hides_strobe_fields(self, qtbot):
        cam = CameraSettings(backend="basler", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        roles = [dlg.role_combo.itemData(i) for i in range(dlg.role_combo.count())]

        assert roles == ["off", "external", "follower", "master"]
        assert not dlg.output_line_edit.isHidden()
        assert not dlg.output_source_edit.isHidden()
        assert dlg.strobe_polarity_combo.isHidden()
        assert dlg.strobe_operation_combo.isHidden()
        assert dlg.strobe_duration_spin.isHidden()
        assert dlg.strobe_delay_spin.isHidden()

    @pytest.mark.gui
    def test_role_changes_enable_input_and_output_fields(self, qtbot):
        cam = CameraSettings(backend="gentl", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("off"))
        assert not dlg.selector_edit.isEnabled()
        assert not dlg.source_combo.isEnabled()
        assert not dlg.output_line_edit.isEnabled()

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("external"))
        assert dlg.selector_edit.isEnabled()
        assert dlg.source_combo.isEnabled()
        assert dlg.activation_combo.isEnabled()
        assert not dlg.output_line_edit.isEnabled()

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("master"))
        assert not dlg.selector_edit.isEnabled()
        assert not dlg.source_combo.isEnabled()
        assert dlg.output_line_edit.isEnabled()
        assert dlg.output_source_edit.isEnabled()
        assert dlg.strobe_polarity_combo.isEnabled()
        assert dlg.strobe_delay_spin.isEnabled()

    @pytest.mark.gui
    def test_fixed_duration_enables_strobe_duration_only_for_master(self, qtbot):
        cam = CameraSettings(backend="gentl", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("master"))
        dlg.strobe_operation_combo.setCurrentIndex(dlg.strobe_operation_combo.findData("FixedDuration"))

        assert dlg.strobe_duration_spin.isEnabled()

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("external"))

        assert not dlg.strobe_duration_spin.isEnabled()


class TestTriggerConfigDialogModelRoundtrip:
    @pytest.mark.gui
    def test_loads_existing_trigger_settings(self, qtbot):
        cam = CameraSettings(
            backend="gentl",
            index=0,
            properties={
                "gentl": {
                    "trigger": {
                        "role": "external",
                        "selector": "FrameStart",
                        "source": "Line1",
                        "activation": "FallingEdge",
                        "timeout": 0.25,
                        "strict": True,
                    }
                }
            },
        )

        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        assert dlg.role_combo.currentData() == "external"
        assert dlg.selector_edit.text() == "FrameStart"
        assert dlg.source_combo.currentText() == "Line1"
        assert dlg.activation_combo.currentData() == "FallingEdge"
        assert dlg.timeout_spin.value() == pytest.approx(0.25)
        assert dlg.strict_checkbox.isChecked()

    @pytest.mark.gui
    def test_accept_external_writes_trigger_payload_to_backend_namespace(self, qtbot):
        cam = CameraSettings(backend="gentl", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("external"))
        dlg.selector_edit.setText("FrameStart")
        dlg.source_combo.setCurrentText("Line1")
        dlg.activation_combo.setCurrentIndex(dlg.activation_combo.findData("FallingEdge"))
        dlg.timeout_spin.setValue(0.5)
        dlg.strict_checkbox.setChecked(True)

        with qtbot.waitSignal(dlg.accepted, timeout=1000):
            dlg._accept()

        trigger = dlg.camera_settings.properties["gentl"]["trigger"]

        assert trigger["role"] == "external"
        assert trigger["selector"] == "FrameStart"
        assert trigger["source"] == "Line1"
        assert trigger["activation"] == "FallingEdge"
        assert trigger["timeout"] == pytest.approx(0.5)
        assert trigger["strict"] is True

    @pytest.mark.gui
    def test_accept_off_clears_timeout(self, qtbot):
        cam = CameraSettings(
            backend="gentl",
            index=0,
            properties={"gentl": {"trigger": {"role": "external", "timeout": 2.0}}},
        )
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("off"))
        dlg.timeout_spin.setValue(1.25)

        with qtbot.waitSignal(dlg.accepted, timeout=1000):
            dlg._accept()

        trigger = dlg.camera_settings.properties["gentl"]["trigger"]

        assert trigger["role"] == "off"
        assert trigger.get("timeout") is None

    @pytest.mark.gui
    def test_accept_master_gentl_includes_strobe_values(self, qtbot):
        cam = CameraSettings(backend="gentl", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("master"))
        dlg.output_line_edit.setText("Line2")
        dlg.output_source_edit.setText("ExposureActive")
        dlg.strobe_polarity_combo.setCurrentIndex(dlg.strobe_polarity_combo.findData("ActiveLow"))
        dlg.strobe_operation_combo.setCurrentIndex(dlg.strobe_operation_combo.findData("FixedDuration"))
        dlg.strobe_duration_spin.setValue(1200)
        dlg.strobe_delay_spin.setValue(300)

        with qtbot.waitSignal(dlg.accepted, timeout=1000):
            dlg._accept()

        trigger = dlg.camera_settings.properties["gentl"]["trigger"]

        assert trigger["role"] == "master"
        assert trigger["output_line"] == "Line2"
        assert trigger["output_source"] == "ExposureActive"
        assert trigger["strobe_polarity"] == "ActiveLow"
        assert trigger["strobe_operation"] == "FixedDuration"
        assert trigger["strobe_duration"] == 1200
        assert trigger["strobe_delay"] == 300

    @pytest.mark.gui
    def test_accept_master_basler_does_not_add_strobe_values(self, qtbot):
        cam = CameraSettings(backend="basler", index=0, properties={})
        dlg = TriggerConfigDialog(cam)
        qtbot.addWidget(dlg)

        dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("master"))
        dlg.strobe_duration_spin.setValue(1200)
        dlg.strobe_delay_spin.setValue(300)

        with qtbot.waitSignal(dlg.accepted, timeout=1000):
            dlg._accept()

        trigger = dlg.camera_settings.properties["basler"]["trigger"]

        assert trigger["role"] == "master"
        assert "strobe_duration" not in trigger
        assert "strobe_delay" not in trigger
        assert "strobe_polarity" not in trigger
        assert "strobe_operation" not in trigger
