# dlclivegui/gui/camera_config/trigger_config_dialog.py
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...config import CameraSettings, CameraTriggerSettings


def _backend_namespace(cam: CameraSettings) -> dict:
    backend = (cam.backend or "").lower()
    if not isinstance(cam.properties, dict):
        cam.properties = {}
    ns = cam.properties.setdefault(backend, {})
    if not isinstance(ns, dict):
        ns = {}
        cam.properties[backend] = ns
    return ns


@dataclass(frozen=True)
class TriggerUiProfile:
    supports_input: bool = True
    supports_master: bool = False
    supports_software: bool = False

    show_strobe_fields: bool = False
    show_line_output_fields: bool = False

    source_suggestions: tuple[str, ...] = ("auto",)
    default_source: str = "auto"

    default_output_line: str = "Line2"
    default_output_source: str = "ExposureActive"

    help_text: str = ""


def trigger_ui_profile_for_backend(backend: str) -> TriggerUiProfile:
    """Return GUI-only trigger presentation profile for a backend.

    This intentionally does not perform backend/runtime validation.
    Backends still own actual GenICam/pypylon/Harvesters configuration.
    """
    backend = (backend or "").lower()

    if backend == "gentl":
        return TriggerUiProfile(
            supports_input=True,
            supports_master=True,
            supports_software=False,
            show_strobe_fields=True,
            show_line_output_fields=True,
            source_suggestions=("auto", "Line0", "Line1", "Line2", "Any", "Software"),
            default_source="auto",
            default_output_line="Line2",
            default_output_source="ExposureActive",
            help_text=(
                "GenTL trigger support is best-effort and depends on the camera's GenICam nodes. "
                "Some cameras expose generic Line* output nodes; TIS/DMK 37U cameras may expose Strobe* nodes."
            ),
        )

    if backend == "basler":
        return TriggerUiProfile(
            supports_input=True,
            supports_master=True,
            supports_software=False,  # enable later when controller supports trigger_once()
            show_strobe_fields=False,
            show_line_output_fields=True,
            source_suggestions=("auto", "Line1", "Line2", "Line3", "Line4", "Software"),
            default_source="auto",
            default_output_line="Line2",
            default_output_source="ExposureActive",
            help_text=(
                "Basler trigger support uses pylon camera features when available. "
                "The available trigger sources and output lines depend on the camera model."
            ),
        )

    return TriggerUiProfile(
        supports_input=False,
        supports_master=False,
        supports_software=False,
        show_strobe_fields=False,
        show_line_output_fields=False,
        source_suggestions=("auto",),
        help_text="This backend does not expose trigger configuration.",
    )


class TriggerConfigDialog(QDialog):
    """Dialog for editing per-camera trigger settings.

    The dialog is backend-aware only for presentation.
    Actual trigger configuration remains backend-owned.
    """

    def __init__(self, cam: CameraSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Configure trigger mode")
        self.setMinimumWidth(460)

        self._cam = cam.model_copy(deep=True)
        self._backend = (self._cam.backend or "").lower()
        self._profile = trigger_ui_profile_for_backend(self._backend)

        ns = _backend_namespace(self._cam)
        try:
            self._trigger = CameraTriggerSettings.from_any(ns.get("trigger"))
        except Exception:
            self._trigger = CameraTriggerSettings()

        self._setup_ui()
        self._load_from_trigger(self._trigger)
        self._sync_role_ui()

    @property
    def camera_settings(self) -> CameraSettings:
        return self._cam

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        info_text = (
            "Configure per-camera trigger settings.\n"
            "External/follower mode arms the camera and waits for trigger pulses on a selected input source.\n"
            "Master mode configures an output signal if the backend/camera exposes compatible output-line features.\n"
            "Some fields are backend- or camera-model-specific and may be ignored unless strict mode is enabled.\n"
            "In strict mode, unsupported trigger nodes fail camera open."
        )
        if self._profile.help_text:
            info_text += f"\n\n{self._profile.help_text}"

        self.info_label = QLabel(info_text)
        self.info_label.setWordWrap(True)
        root.addWidget(self.info_label)

        group_title = f"Trigger Settings ({self._backend or 'unknown'})"
        group = QGroupBox(group_title)
        self.form = QFormLayout(group)

        # ----------------------------
        # Role
        # ----------------------------
        self.role_combo = QComboBox()
        self.role_combo.addItem("Off / Free-run", "off")

        if self._profile.supports_input:
            self.role_combo.addItem("External trigger", "external")
            self.role_combo.addItem("Follower", "follower")

        if self._profile.supports_master:
            self.role_combo.addItem("Master output", "master")

        if self._profile.supports_software:
            self.role_combo.addItem("Software trigger", "software")

        self.form.addRow("Role:", self.role_combo)

        # ----------------------------
        # Input trigger fields
        # ----------------------------
        self.selector_edit = QLineEdit()
        self.selector_edit.setPlaceholderText("FrameStart")
        self.selector_edit.setToolTip("TriggerSelector value. Most area-scan cameras use FrameStart.")
        self.form.addRow("Trigger selector:", self.selector_edit)

        self.source_combo = QComboBox()
        self.source_combo.setEditable(True)
        for value in self._profile.source_suggestions:
            self.source_combo.addItem(value, value)
        self.source_combo.setToolTip(
            "TriggerSource value. Suggestions are backend defaults only; "
            "the backend validates the actual camera-supported values when opening."
        )
        if self.source_combo.lineEdit() is not None:
            self.source_combo.lineEdit().setPlaceholderText("auto, Line1, Software, ...")
        self.form.addRow("Trigger source:", self.source_combo)

        self.activation_combo = QComboBox()
        for value in ("RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"):
            self.activation_combo.addItem(value, value)
        self.activation_combo.setToolTip(
            "TriggerActivation value. Some software/internal trigger sources may ignore this."
        )
        self.form.addRow("Activation:", self.activation_combo)

        # ----------------------------
        # Generic output line fields
        # ----------------------------
        self.output_line_edit = QLineEdit()
        self.output_line_edit.setPlaceholderText(self._profile.default_output_line)
        self.output_line_edit.setToolTip(
            "Generic LineSelector value for cameras exposing LineSelector/LineSource. "
            "Ignored if the backend/camera does not support generic line output."
        )
        self.form.addRow("Output line:", self.output_line_edit)

        self.output_source_edit = QLineEdit()
        self.output_source_edit.setPlaceholderText(self._profile.default_output_source)
        self.output_source_edit.setToolTip(
            "Generic LineSource value for cameras exposing LineSource, e.g. ExposureActive."
        )
        self.form.addRow("Output source:", self.output_source_edit)

        # ----------------------------
        # Strobe fields, mainly useful for specific GenTL/TIS devices
        # ----------------------------
        self.strobe_polarity_combo = QComboBox()
        self.strobe_polarity_combo.addItem("Active high", "ActiveHigh")
        self.strobe_polarity_combo.addItem("Active low", "ActiveLow")
        self.strobe_polarity_combo.setToolTip(
            "Strobe output polarity. Only used by backends/cameras exposing compatible Strobe* nodes."
        )
        self.form.addRow("Strobe polarity:", self.strobe_polarity_combo)

        self.strobe_operation_combo = QComboBox()
        self.strobe_operation_combo.addItem("Exposure duration", "Exposure")
        self.strobe_operation_combo.addItem("Fixed duration", "FixedDuration")
        self.strobe_operation_combo.setToolTip(
            "Strobe operation. Only used by backends/cameras exposing compatible Strobe* nodes."
        )
        self.form.addRow("Strobe operation:", self.strobe_operation_combo)

        self.strobe_duration_spin = QSpinBox()
        self.strobe_duration_spin.setRange(0, 32767)
        self.strobe_duration_spin.setSingleStep(100)
        self.strobe_duration_spin.setSuffix(" µs")
        self.strobe_duration_spin.setSpecialValueText("Default")
        self.strobe_duration_spin.setToolTip(
            "Used only when strobe operation is FixedDuration. 0 means backend/device default."
        )
        self.form.addRow("Strobe duration:", self.strobe_duration_spin)

        self.strobe_delay_spin = QSpinBox()
        self.strobe_delay_spin.setRange(0, 32767)
        self.strobe_delay_spin.setSingleStep(100)
        self.strobe_delay_spin.setSuffix(" µs")
        self.strobe_delay_spin.setSpecialValueText("Default")
        self.strobe_delay_spin.setToolTip("Delay before strobe output. 0 means no explicit delay/device default.")
        self.form.addRow("Strobe delay:", self.strobe_delay_spin)

        # ----------------------------
        # Common options
        # ----------------------------
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.0, 3600.0)
        self.timeout_spin.setDecimals(3)
        self.timeout_spin.setSingleStep(0.1)
        self.timeout_spin.setSpecialValueText("Default")
        self.timeout_spin.setToolTip(
            "Read/fetch timeout in seconds. The backend may cap individual waits to keep preview shutdown responsive."
        )
        self.form.addRow("Read timeout:", self.timeout_spin)

        self.strict_checkbox = QCheckBox("Strict mode")
        self.strict_checkbox.setToolTip("If enabled, missing/unsupported trigger features fail camera open.")
        self.form.addRow(self.strict_checkbox)

        root.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.role_combo.currentIndexChanged.connect(self._sync_role_ui)
        self.strobe_operation_combo.currentIndexChanged.connect(self._sync_role_ui)

        # Hide backend-irrelevant rows immediately.
        self._apply_profile_visibility()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_form_row_visible(self, widget: QWidget, visible: bool) -> None:
        """Hide/show a QFormLayout field and its label."""
        widget.setVisible(visible)
        try:
            label = self.form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        except Exception:
            pass

    def _set_combo_text(self, combo: QComboBox, text: str) -> None:
        text = str(text or "")
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText(text)

    def _combo_text(self, combo: QComboBox, fallback: str) -> str:
        text = str(combo.currentText() or "").strip()
        return text or fallback

    def _apply_profile_visibility(self) -> None:
        """Apply static backend-profile visibility.

        Role-specific enablement is handled separately by _sync_role_ui().
        """
        # Input trigger fields are only meaningful for input/software roles.
        self._set_form_row_visible(self.selector_edit, self._profile.supports_input or self._profile.supports_software)
        self._set_form_row_visible(self.source_combo, self._profile.supports_input or self._profile.supports_software)
        self._set_form_row_visible(
            self.activation_combo,
            self._profile.supports_input,
        )

        # Output fields depend on backend presentation profile.
        self._set_form_row_visible(self.output_line_edit, self._profile.show_line_output_fields)
        self._set_form_row_visible(self.output_source_edit, self._profile.show_line_output_fields)

        # Strobe fields should not appear for Basler.
        self._set_form_row_visible(self.strobe_polarity_combo, self._profile.show_strobe_fields)
        self._set_form_row_visible(self.strobe_operation_combo, self._profile.show_strobe_fields)
        self._set_form_row_visible(self.strobe_duration_spin, self._profile.show_strobe_fields)
        self._set_form_row_visible(self.strobe_delay_spin, self._profile.show_strobe_fields)

    # ------------------------------------------------------------------
    # Model <-> UI
    # ------------------------------------------------------------------

    def _load_from_trigger(self, trigger: CameraTriggerSettings) -> None:
        role = str(getattr(trigger, "role", "off") or "off").lower()
        idx = self.role_combo.findData(role)
        self.role_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.selector_edit.setText(str(getattr(trigger, "selector", "FrameStart") or "FrameStart"))

        source = str(getattr(trigger, "source", self._profile.default_source) or self._profile.default_source)
        self._set_combo_text(self.source_combo, source)

        activation = str(getattr(trigger, "activation", "RisingEdge") or "RisingEdge")
        idx = self.activation_combo.findData(activation)
        self.activation_combo.setCurrentIndex(idx if idx >= 0 else 0)

        output_line = str(
            getattr(trigger, "output_line", self._profile.default_output_line) or self._profile.default_output_line
        )
        self.output_line_edit.setText(output_line)

        output_source = str(
            getattr(trigger, "output_source", self._profile.default_output_source)
            or self._profile.default_output_source
        )
        self.output_source_edit.setText(output_source)

        strobe_polarity = str(getattr(trigger, "strobe_polarity", "ActiveHigh") or "ActiveHigh")
        idx = self.strobe_polarity_combo.findData(strobe_polarity)
        self.strobe_polarity_combo.setCurrentIndex(idx if idx >= 0 else 0)

        strobe_operation = str(getattr(trigger, "strobe_operation", "Exposure") or "Exposure")
        idx = self.strobe_operation_combo.findData(strobe_operation)
        self.strobe_operation_combo.setCurrentIndex(idx if idx >= 0 else 0)

        strobe_duration = getattr(trigger, "strobe_duration", None)
        self.strobe_duration_spin.setValue(int(strobe_duration) if strobe_duration is not None else 0)

        strobe_delay = getattr(trigger, "strobe_delay", None)
        self.strobe_delay_spin.setValue(int(strobe_delay) if strobe_delay is not None else 0)

        timeout = getattr(trigger, "timeout", None)
        self.timeout_spin.setValue(float(timeout) if timeout else 0.0)

        self.strict_checkbox.setChecked(bool(getattr(trigger, "strict", False)))

    def _sync_role_ui(self) -> None:
        role = str(self.role_combo.currentData() or "off")

        input_enabled = role in {"external", "follower", "software"}
        hw_input_enabled = role in {"external", "follower"}
        output_enabled = role == "master"

        # Input fields.
        self.selector_edit.setEnabled(input_enabled)
        self.source_combo.setEnabled(input_enabled)
        self.activation_combo.setEnabled(hw_input_enabled)

        # Generic Line* output fields.
        line_output_active = output_enabled and self._profile.show_line_output_fields
        self.output_line_edit.setEnabled(line_output_active)
        self.output_source_edit.setEnabled(line_output_active)

        # Strobe fields.
        strobe_active = output_enabled and self._profile.show_strobe_fields
        self.strobe_polarity_combo.setEnabled(strobe_active)
        self.strobe_operation_combo.setEnabled(strobe_active)

        fixed_duration = (
            strobe_active and str(self.strobe_operation_combo.currentData() or "Exposure") == "FixedDuration"
        )
        self.strobe_duration_spin.setEnabled(fixed_duration)
        self.strobe_delay_spin.setEnabled(strobe_active)

        # Timeout is useful for trigger-waiting modes. Keep it available for
        # software too if software support is later enabled.
        self.timeout_spin.setEnabled(role in {"external", "follower", "software"})

    def _accept(self) -> None:
        role = str(self.role_combo.currentData() or "off")

        payload = {
            "role": role,
            "selector": self.selector_edit.text().strip() or "FrameStart",
            "source": self._combo_text(self.source_combo, self._profile.default_source),
            "activation": str(self.activation_combo.currentData() or "RisingEdge"),
            "output_line": self.output_line_edit.text().strip() or self._profile.default_output_line,
            "output_source": self.output_source_edit.text().strip() or self._profile.default_output_source,
            "strict": bool(self.strict_checkbox.isChecked()),
        }

        timeout = float(self.timeout_spin.value())
        if role in {"external", "follower", "software"} and timeout > 0:
            payload["timeout"] = timeout
        elif role == "off":
            payload["timeout"] = None

        # Only include strobe-specific settings for profiles that expose them.
        # This avoids cluttering Basler trigger configs with TIS-specific fields.
        if self._profile.show_strobe_fields:
            payload["strobe_polarity"] = str(self.strobe_polarity_combo.currentData() or "ActiveHigh")
            payload["strobe_operation"] = str(self.strobe_operation_combo.currentData() or "Exposure")

            strobe_duration = int(self.strobe_duration_spin.value())
            if role == "master" and strobe_duration > 0:
                payload["strobe_duration"] = strobe_duration

            strobe_delay = int(self.strobe_delay_spin.value())
            if role == "master" and strobe_delay > 0:
                payload["strobe_delay"] = strobe_delay

        try:
            trigger = CameraTriggerSettings.from_any(payload)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to apply trigger settings: {exc}")
            return

        ns = _backend_namespace(self._cam)
        ns["trigger"] = trigger.to_properties()

        self.accept()
