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
        self._trigger = CameraTriggerSettings.from_any(ns.get("trigger"))

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
            "Unsupported fields are ignored by the backend unless strict mode is enabled."
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
        self.source_edit.setPlaceholderText("Line0")
        form.addRow("Trigger source:", self.source_edit)

        self.activation_combo = QComboBox()
        for value in ("RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"):
            self.activation_combo.addItem(value, value)
        form.addRow("Activation:", self.activation_combo)

        self.output_line_edit = QLineEdit()
        self.output_line_edit.setPlaceholderText("Line2")
        form.addRow("Output line:", self.output_line_edit)

        self.output_source_edit = QLineEdit()
        self.output_source_edit.setPlaceholderText("ExposureActive")
        form.addRow("Output source:", self.output_source_edit)

        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.0, 3600.0)
        self.timeout_spin.setDecimals(3)
        self.timeout_spin.setSingleStep(0.1)
        self.timeout_spin.setSpecialValueText("Default")
        self.timeout_spin.setToolTip(
            "Fetch poll timeout in seconds. For triggered cameras, 0.2–0.5s is usually responsive."
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

    def _load_from_trigger(self, trigger: CameraTriggerSettings) -> None:
        role = str(getattr(trigger, "role", "off") or "off").lower()
        idx = self.role_combo.findData(role)
        self.role_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.selector_edit.setText(str(getattr(trigger, "selector", "FrameStart") or "FrameStart"))
        self.source_edit.setText(str(getattr(trigger, "source", "Line0") or "Line0"))

        activation = str(getattr(trigger, "activation", "RisingEdge") or "RisingEdge")
        idx = self.activation_combo.findData(activation)
        self.activation_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.output_line_edit.setText(str(getattr(trigger, "output_line", "Line2") or "Line2"))
        self.output_source_edit.setText(str(getattr(trigger, "output_source", "ExposureActive") or "ExposureActive"))

        timeout = getattr(trigger, "timeout", None)
        self.timeout_spin.setValue(float(timeout) if timeout else 0.0)

        self.strict_checkbox.setChecked(bool(getattr(trigger, "strict", False)))

    def _sync_role_ui(self) -> None:
        role = str(self.role_combo.currentData() or "off")

        input_enabled = role in {"external", "follower"}
        output_enabled = role == "master"

        self.selector_edit.setEnabled(input_enabled)
        self.source_edit.setEnabled(input_enabled)
        self.activation_combo.setEnabled(input_enabled)

        self.output_line_edit.setEnabled(output_enabled)
        self.output_source_edit.setEnabled(output_enabled)

        # Timeout is mostly useful for external/follower, but harmless for any role.
        self.timeout_spin.setEnabled(role in {"external", "follower"})

    def _accept(self) -> None:
        role = str(self.role_combo.currentData() or "off")

        payload = {
            "role": role,
            "selector": self.selector_edit.text().strip() or "FrameStart",
            "source": self.source_edit.text().strip() or "Line0",
            "activation": str(self.activation_combo.currentData() or "RisingEdge"),
            "output_line": self.output_line_edit.text().strip() or "Line2",
            "output_source": self.output_source_edit.text().strip() or "ExposureActive",
            "strict": bool(self.strict_checkbox.isChecked()),
        }

        timeout = float(self.timeout_spin.value())
        if timeout > 0:
            payload["timeout"] = timeout

        trigger = CameraTriggerSettings.from_any(payload)

        ns = _backend_namespace(self._cam)
        ns["trigger"] = trigger.model_dump(exclude_none=True)

        self.accept()
