# dlclivegui/gui/camera_config/trigger_config_dialog.py
from __future__ import annotations

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


class TriggerConfigDialog(QDialog):
    """Small dialog for editing per-camera hardware trigger settings."""

    def __init__(self, cam: CameraSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Configure trigger mode")
        self.setMinimumWidth(420)

        self._cam = cam.model_copy(deep=True)

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

        info = QLabel(
            "Configure hardware trigger settings for this camera.\n"
            "Follower/external mode arms the camera and waits for electrical pulses on TRIGGER_IN.\n"
            "Master mode enables STROBE_OUT pulses. For TIS/DMK 37U cameras this uses Strobe settings; "
            "Line output settings are kept as a generic fallback.\n"
            "In strict mode, unsupported trigger nodes fail camera open."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        group = QGroupBox("Hardware Trigger")
        form = QFormLayout(group)

        self.role_combo = QComboBox()
        self.role_combo.addItem("Off / Free-run", "off")
        self.role_combo.addItem("External trigger", "external")
        self.role_combo.addItem("Follower", "follower")
        self.role_combo.addItem("Master", "master")
        form.addRow("Role:", self.role_combo)

        self.selector_edit = QLineEdit()
        self.selector_edit.setPlaceholderText("FrameStart")
        form.addRow("Trigger selector:", self.selector_edit)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("auto, Line0, Software, ...")
        form.addRow("Trigger source:", self.source_edit)

        self.activation_combo = QComboBox()
        for value in ("RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"):
            self.activation_combo.addItem(value, value)
        form.addRow("Activation:", self.activation_combo)

        self.output_line_edit = QLineEdit()
        self.output_line_edit.setPlaceholderText("Line2")
        self.output_line_edit.setToolTip(
            "Generic Line* output selector for cameras exposing LineSelector/LineSource. "
            "Ignored by TIS/DMK 37U strobe-based output."
        )
        form.addRow("Output line:", self.output_line_edit)

        self.output_source_edit = QLineEdit()
        self.output_source_edit.setPlaceholderText("ExposureActive")
        self.output_source_edit.setToolTip(
            "Generic LineSource value for cameras exposing LineSource. "
            "For TIS/DMK 37U cameras, use Strobe operation instead."
        )
        form.addRow("Output source:", self.output_source_edit)

        self.strobe_polarity_combo = QComboBox()
        self.strobe_polarity_combo.addItem("Active high", "ActiveHigh")
        self.strobe_polarity_combo.addItem("Active low", "ActiveLow")
        self.strobe_polarity_combo.setToolTip(
            "Polarity of STROBE_OUT. If the follower does not trigger, also try changing the follower activation edge."
        )
        form.addRow("Strobe polarity:", self.strobe_polarity_combo)

        self.strobe_operation_combo = QComboBox()
        self.strobe_operation_combo.addItem("Exposure duration", "Exposure")
        self.strobe_operation_combo.addItem("Fixed duration", "FixedDuration")
        self.strobe_operation_combo.setToolTip(
            "Exposure: strobe pulse length follows exposure time. "
            "FixedDuration: strobe pulse length is set by Strobe duration."
        )
        form.addRow("Strobe operation:", self.strobe_operation_combo)

        self.strobe_duration_spin = QSpinBox()
        self.strobe_duration_spin.setRange(0, 32767)
        self.strobe_duration_spin.setSingleStep(100)
        self.strobe_duration_spin.setSuffix(" µs")
        self.strobe_duration_spin.setSpecialValueText("Default")
        self.strobe_duration_spin.setToolTip(
            "Used only when Strobe operation is FixedDuration. 0 means backend/device default."
        )
        form.addRow("Strobe duration:", self.strobe_duration_spin)

        self.strobe_delay_spin = QSpinBox()
        self.strobe_delay_spin.setRange(0, 32767)
        self.strobe_delay_spin.setSingleStep(100)
        self.strobe_delay_spin.setSuffix(" µs")
        self.strobe_delay_spin.setSpecialValueText("Default")
        self.strobe_delay_spin.setToolTip(
            "Delay between start of exposure and STROBE_OUT pulse. 0 means no delay/device default."
        )
        form.addRow("Strobe delay:", self.strobe_delay_spin)

        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.0, 3600.0)
        self.timeout_spin.setDecimals(3)
        self.timeout_spin.setSingleStep(0.1)
        self.timeout_spin.setSpecialValueText("Default")
        self.timeout_spin.setToolTip(
            "Fetch poll timeout in seconds. The backend may cap individual fetches to keep preview shutdown responsive."
        )
        form.addRow("Read timeout:", self.timeout_spin)

        self.strict_checkbox = QCheckBox("Strict mode")
        self.strict_checkbox.setToolTip("If enabled, missing/unsupported GenICam trigger nodes fail camera open.")
        form.addRow(self.strict_checkbox)

        root.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.role_combo.currentIndexChanged.connect(self._sync_role_ui)
        self.strobe_operation_combo.currentIndexChanged.connect(self._sync_role_ui)

    def _load_from_trigger(self, trigger: CameraTriggerSettings) -> None:
        role = str(getattr(trigger, "role", "off") or "off").lower()
        idx = self.role_combo.findData(role)
        self.role_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.selector_edit.setText(str(getattr(trigger, "selector", "FrameStart") or "FrameStart"))
        self.source_edit.setText(str(getattr(trigger, "source", "auto") or "auto"))

        activation = str(getattr(trigger, "activation", "RisingEdge") or "RisingEdge")
        idx = self.activation_combo.findData(activation)
        self.activation_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.output_line_edit.setText(str(getattr(trigger, "output_line", "Line2") or "Line2"))
        self.output_source_edit.setText(str(getattr(trigger, "output_source", "ExposureActive") or "ExposureActive"))

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

        input_enabled = role in {"external", "follower"}

        self.selector_edit.setEnabled(input_enabled)
        self.source_edit.setEnabled(input_enabled)
        self.activation_combo.setEnabled(input_enabled)

        output_enabled = role == "master"
        # Generic Line* fallback fields.
        self.output_line_edit.setEnabled(output_enabled)
        self.output_source_edit.setEnabled(output_enabled)

        # TIS/DMK 37U Strobe* fields.
        self.strobe_polarity_combo.setEnabled(output_enabled)
        self.strobe_operation_combo.setEnabled(output_enabled)

        fixed_duration = (
            output_enabled and str(self.strobe_operation_combo.currentData() or "Exposure") == "FixedDuration"
        )
        self.strobe_duration_spin.setEnabled(fixed_duration)
        self.strobe_delay_spin.setEnabled(output_enabled)

        # Timeout is mostly useful for external/follower, but harmless for any role.
        self.timeout_spin.setEnabled(role in {"external", "follower"})

    def _accept(self) -> None:
        role = str(self.role_combo.currentData() or "off")

        payload = {
            "role": role,
            "selector": self.selector_edit.text().strip() or "FrameStart",
            "source": self.source_edit.text().strip() or "auto",
            "activation": str(self.activation_combo.currentData() or "RisingEdge"),
            # Generic/SFNC Line* fallback output settings.
            "output_line": self.output_line_edit.text().strip() or "Line2",
            "output_source": self.output_source_edit.text().strip() or "ExposureActive",
            # Strobe output settings used by TIS/DMK 37U cameras.
            "strobe_polarity": str(self.strobe_polarity_combo.currentData() or "ActiveHigh"),
            "strobe_operation": str(self.strobe_operation_combo.currentData() or "Exposure"),
            "strict": bool(self.strict_checkbox.isChecked()),
        }

        timeout = float(self.timeout_spin.value())
        if role in {"external", "follower"} and timeout > 0:
            payload["timeout"] = timeout
        elif role == "off":
            payload["timeout"] = None  # ensure timeout is cleared when disabling trigger
        strobe_duration = int(self.strobe_duration_spin.value())
        if role == "master" and strobe_duration > 0:
            payload["strobe_duration"] = strobe_duration

        strobe_delay = int(self.strobe_delay_spin.value())
        if role == "master" and strobe_delay > 0:
            payload["strobe_delay"] = strobe_delay

        try:
            trigger = CameraTriggerSettings.from_any(payload)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply trigger settings: {e}")
            return

        ns = _backend_namespace(self._cam)
        ns["trigger"] = trigger.to_properties()

        self.accept()
