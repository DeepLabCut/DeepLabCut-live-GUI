"""Camera configuration dialog for multi-camera setup."""

from __future__ import annotations

import logging
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.config import CameraSettings, MultiCameraSettings

LOGGER = logging.getLogger(__name__)


class CameraConfigDialog(QDialog):
    """Dialog for configuring multiple cameras."""

    MAX_CAMERAS = 4
    settings_changed = pyqtSignal(object)  # MultiCameraSettings

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        multi_camera_settings: Optional[MultiCameraSettings] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Configure Cameras")
        self.setMinimumSize(800, 600)

        self._multi_camera_settings = (
            multi_camera_settings if multi_camera_settings else MultiCameraSettings()
        )
        self._detected_cameras: List[DetectedCamera] = []
        self._current_edit_index: Optional[int] = None

        self._setup_ui()
        self._populate_from_settings()
        self._connect_signals()

    def _setup_ui(self) -> None:
        # Main layout for the dialog
        main_layout = QVBoxLayout(self)

        # Horizontal layout for left and right panels
        panels_layout = QHBoxLayout()

        # Left panel: Camera list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Active cameras list
        active_group = QGroupBox("Active Cameras")
        active_layout = QVBoxLayout(active_group)

        self.active_cameras_list = QListWidget()
        self.active_cameras_list.setMinimumWidth(250)
        active_layout.addWidget(self.active_cameras_list)

        # Buttons for managing active cameras
        list_buttons = QHBoxLayout()
        self.remove_camera_btn = QPushButton("Remove")
        self.remove_camera_btn.setEnabled(False)
        self.move_up_btn = QPushButton("↑")
        self.move_up_btn.setEnabled(False)
        self.move_down_btn = QPushButton("↓")
        self.move_down_btn.setEnabled(False)
        list_buttons.addWidget(self.remove_camera_btn)
        list_buttons.addWidget(self.move_up_btn)
        list_buttons.addWidget(self.move_down_btn)
        active_layout.addLayout(list_buttons)

        left_layout.addWidget(active_group)

        # Available cameras section
        available_group = QGroupBox("Available Cameras")
        available_layout = QVBoxLayout(available_group)

        # Backend selection
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        availability = CameraFactory.available_backends()
        for backend in CameraFactory.backend_names():
            label = backend
            if not availability.get(backend, True):
                label = f"{backend} (unavailable)"
            self.backend_combo.addItem(label, backend)
        backend_layout.addWidget(self.backend_combo)
        self.refresh_btn = QPushButton("Refresh")
        backend_layout.addWidget(self.refresh_btn)
        available_layout.addLayout(backend_layout)

        self.available_cameras_list = QListWidget()
        available_layout.addWidget(self.available_cameras_list)

        self.add_camera_btn = QPushButton("Add Selected Camera →")
        self.add_camera_btn.setEnabled(False)
        available_layout.addWidget(self.add_camera_btn)

        left_layout.addWidget(available_group)

        # Right panel: Camera settings editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        settings_group = QGroupBox("Camera Settings")
        self.settings_form = QFormLayout(settings_group)

        self.cam_enabled_checkbox = QCheckBox("Enabled")
        self.cam_enabled_checkbox.setChecked(True)
        self.settings_form.addRow(self.cam_enabled_checkbox)

        self.cam_name_label = QLabel("Camera 0")
        self.cam_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.settings_form.addRow("Name:", self.cam_name_label)

        self.cam_index_label = QLabel("0")
        self.settings_form.addRow("Index:", self.cam_index_label)

        self.cam_backend_label = QLabel("opencv")
        self.settings_form.addRow("Backend:", self.cam_backend_label)

        self.cam_fps = QDoubleSpinBox()
        self.cam_fps.setRange(1.0, 240.0)
        self.cam_fps.setDecimals(2)
        self.cam_fps.setValue(30.0)
        self.settings_form.addRow("Frame Rate:", self.cam_fps)

        self.cam_exposure = QSpinBox()
        self.cam_exposure.setRange(0, 1000000)
        self.cam_exposure.setValue(0)
        self.cam_exposure.setSpecialValueText("Auto")
        self.cam_exposure.setSuffix(" μs")
        self.settings_form.addRow("Exposure:", self.cam_exposure)

        self.cam_gain = QDoubleSpinBox()
        self.cam_gain.setRange(0.0, 100.0)
        self.cam_gain.setValue(0.0)
        self.cam_gain.setSpecialValueText("Auto")
        self.cam_gain.setDecimals(2)
        self.settings_form.addRow("Gain:", self.cam_gain)

        # Rotation
        self.cam_rotation = QComboBox()
        self.cam_rotation.addItem("0° (default)", 0)
        self.cam_rotation.addItem("90°", 90)
        self.cam_rotation.addItem("180°", 180)
        self.cam_rotation.addItem("270°", 270)
        self.settings_form.addRow("Rotation:", self.cam_rotation)

        # Crop settings
        crop_widget = QWidget()
        crop_layout = QHBoxLayout(crop_widget)
        crop_layout.setContentsMargins(0, 0, 0, 0)

        self.cam_crop_x0 = QSpinBox()
        self.cam_crop_x0.setRange(0, 7680)
        self.cam_crop_x0.setPrefix("x0:")
        self.cam_crop_x0.setSpecialValueText("x0:None")
        crop_layout.addWidget(self.cam_crop_x0)

        self.cam_crop_y0 = QSpinBox()
        self.cam_crop_y0.setRange(0, 4320)
        self.cam_crop_y0.setPrefix("y0:")
        self.cam_crop_y0.setSpecialValueText("y0:None")
        crop_layout.addWidget(self.cam_crop_y0)

        self.cam_crop_x1 = QSpinBox()
        self.cam_crop_x1.setRange(0, 7680)
        self.cam_crop_x1.setPrefix("x1:")
        self.cam_crop_x1.setSpecialValueText("x1:None")
        crop_layout.addWidget(self.cam_crop_x1)

        self.cam_crop_y1 = QSpinBox()
        self.cam_crop_y1.setRange(0, 4320)
        self.cam_crop_y1.setPrefix("y1:")
        self.cam_crop_y1.setSpecialValueText("y1:None")
        crop_layout.addWidget(self.cam_crop_y1)

        self.settings_form.addRow("Crop (x0,y0,x1,y1):", crop_widget)

        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.setEnabled(False)
        self.settings_form.addRow(self.apply_settings_btn)

        right_layout.addWidget(settings_group)
        right_layout.addStretch(1)

        # Dialog buttons
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        button_layout.addStretch(1)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)

        # Add panels to horizontal layout
        panels_layout.addWidget(left_panel, stretch=1)
        panels_layout.addWidget(right_panel, stretch=1)

        # Add everything to main layout
        main_layout.addLayout(panels_layout)
        main_layout.addLayout(button_layout)

    def _connect_signals(self) -> None:
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.refresh_btn.clicked.connect(self._refresh_available_cameras)
        self.add_camera_btn.clicked.connect(self._add_selected_camera)
        self.remove_camera_btn.clicked.connect(self._remove_selected_camera)
        self.move_up_btn.clicked.connect(self._move_camera_up)
        self.move_down_btn.clicked.connect(self._move_camera_down)
        self.active_cameras_list.currentRowChanged.connect(self._on_active_camera_selected)
        self.available_cameras_list.currentRowChanged.connect(self._on_available_camera_selected)
        self.apply_settings_btn.clicked.connect(self._apply_camera_settings)
        self.ok_btn.clicked.connect(self._on_ok_clicked)
        self.cancel_btn.clicked.connect(self.reject)

    def _populate_from_settings(self) -> None:
        """Populate the dialog from existing settings."""
        self.active_cameras_list.clear()
        for cam in self._multi_camera_settings.cameras:
            item = QListWidgetItem(self._format_camera_label(cam))
            item.setData(Qt.ItemDataRole.UserRole, cam)
            if not cam.enabled:
                item.setForeground(Qt.GlobalColor.gray)
            self.active_cameras_list.addItem(item)

        self._refresh_available_cameras()
        self._update_button_states()

    def _format_camera_label(self, cam: CameraSettings) -> str:
        """Format camera label for display."""
        status = "✓" if cam.enabled else "○"
        return f"{status} {cam.name} [{cam.backend}:{cam.index}]"

    def _on_backend_changed(self, _index: int) -> None:
        self._refresh_available_cameras()

    def _refresh_available_cameras(self) -> None:
        """Refresh the list of available cameras."""
        backend = self.backend_combo.currentData()
        if not backend:
            backend = self.backend_combo.currentText().split()[0]

        self.available_cameras_list.clear()
        self._detected_cameras = CameraFactory.detect_cameras(backend, max_devices=10)

        for cam in self._detected_cameras:
            item = QListWidgetItem(f"{cam.label} (index {cam.index})")
            item.setData(Qt.ItemDataRole.UserRole, cam)
            self.available_cameras_list.addItem(item)

        self._update_button_states()

    def _on_available_camera_selected(self, row: int) -> None:
        self.add_camera_btn.setEnabled(row >= 0)

    def _on_active_camera_selected(self, row: int) -> None:
        """Handle selection of an active camera."""
        self._current_edit_index = row
        self._update_button_states()

        if row < 0 or row >= self.active_cameras_list.count():
            self._clear_settings_form()
            return

        item = self.active_cameras_list.item(row)
        cam = item.data(Qt.ItemDataRole.UserRole)
        if cam:
            self._load_camera_to_form(cam)

    def _load_camera_to_form(self, cam: CameraSettings) -> None:
        """Load camera settings into the form."""
        self.cam_enabled_checkbox.setChecked(cam.enabled)
        self.cam_name_label.setText(cam.name)
        self.cam_index_label.setText(str(cam.index))
        self.cam_backend_label.setText(cam.backend)
        self.cam_fps.setValue(cam.fps)
        self.cam_exposure.setValue(cam.exposure)
        self.cam_gain.setValue(cam.gain)

        # Set rotation
        rot_index = self.cam_rotation.findData(cam.rotation)
        if rot_index >= 0:
            self.cam_rotation.setCurrentIndex(rot_index)

        self.cam_crop_x0.setValue(cam.crop_x0)
        self.cam_crop_y0.setValue(cam.crop_y0)
        self.cam_crop_x1.setValue(cam.crop_x1)
        self.cam_crop_y1.setValue(cam.crop_y1)

        self.apply_settings_btn.setEnabled(True)

    def _clear_settings_form(self) -> None:
        """Clear the settings form."""
        self.cam_enabled_checkbox.setChecked(True)
        self.cam_name_label.setText("")
        self.cam_index_label.setText("")
        self.cam_backend_label.setText("")
        self.cam_fps.setValue(30.0)
        self.cam_exposure.setValue(0)
        self.cam_gain.setValue(0.0)
        self.cam_rotation.setCurrentIndex(0)
        self.cam_crop_x0.setValue(0)
        self.cam_crop_y0.setValue(0)
        self.cam_crop_x1.setValue(0)
        self.cam_crop_y1.setValue(0)
        self.apply_settings_btn.setEnabled(False)

    def _add_selected_camera(self) -> None:
        """Add the selected available camera to active cameras."""
        row = self.available_cameras_list.currentRow()
        if row < 0:
            return

        # Check limit
        active_count = len(
            [
                i
                for i in range(self.active_cameras_list.count())
                if self.active_cameras_list.item(i).data(Qt.ItemDataRole.UserRole).enabled
            ]
        )
        if active_count >= self.MAX_CAMERAS:
            QMessageBox.warning(
                self,
                "Maximum Cameras",
                f"Maximum of {self.MAX_CAMERAS} active cameras allowed.",
            )
            return

        item = self.available_cameras_list.item(row)
        detected = item.data(Qt.ItemDataRole.UserRole)
        backend = self.backend_combo.currentData() or "opencv"

        # Create new camera settings
        new_cam = CameraSettings(
            name=detected.label,
            index=detected.index,
            fps=30.0,
            backend=backend,
            exposure=0,
            gain=0.0,
            enabled=True,
        )

        self._multi_camera_settings.cameras.append(new_cam)

        # Add to list
        new_item = QListWidgetItem(self._format_camera_label(new_cam))
        new_item.setData(Qt.ItemDataRole.UserRole, new_cam)
        self.active_cameras_list.addItem(new_item)
        self.active_cameras_list.setCurrentItem(new_item)

        self._update_button_states()

    def _remove_selected_camera(self) -> None:
        """Remove the selected camera from active cameras."""
        row = self.active_cameras_list.currentRow()
        if row < 0:
            return

        self.active_cameras_list.takeItem(row)
        if row < len(self._multi_camera_settings.cameras):
            del self._multi_camera_settings.cameras[row]

        self._current_edit_index = None
        self._clear_settings_form()
        self._update_button_states()

    def _move_camera_up(self) -> None:
        """Move selected camera up in the list."""
        row = self.active_cameras_list.currentRow()
        if row <= 0:
            return

        item = self.active_cameras_list.takeItem(row)
        self.active_cameras_list.insertItem(row - 1, item)
        self.active_cameras_list.setCurrentRow(row - 1)

        # Update settings list
        cams = self._multi_camera_settings.cameras
        cams[row], cams[row - 1] = cams[row - 1], cams[row]

    def _move_camera_down(self) -> None:
        """Move selected camera down in the list."""
        row = self.active_cameras_list.currentRow()
        if row < 0 or row >= self.active_cameras_list.count() - 1:
            return

        item = self.active_cameras_list.takeItem(row)
        self.active_cameras_list.insertItem(row + 1, item)
        self.active_cameras_list.setCurrentRow(row + 1)

        # Update settings list
        cams = self._multi_camera_settings.cameras
        cams[row], cams[row + 1] = cams[row + 1], cams[row]

    def _apply_camera_settings(self) -> None:
        """Apply current form settings to the selected camera."""
        if self._current_edit_index is None:
            return

        row = self._current_edit_index
        if row < 0 or row >= len(self._multi_camera_settings.cameras):
            return

        cam = self._multi_camera_settings.cameras[row]
        cam.enabled = self.cam_enabled_checkbox.isChecked()
        cam.fps = self.cam_fps.value()
        cam.exposure = self.cam_exposure.value()
        cam.gain = self.cam_gain.value()
        cam.rotation = self.cam_rotation.currentData() or 0
        cam.crop_x0 = self.cam_crop_x0.value()
        cam.crop_y0 = self.cam_crop_y0.value()
        cam.crop_x1 = self.cam_crop_x1.value()
        cam.crop_y1 = self.cam_crop_y1.value()

        # Update list item
        item = self.active_cameras_list.item(row)
        item.setText(self._format_camera_label(cam))
        item.setData(Qt.ItemDataRole.UserRole, cam)
        if not cam.enabled:
            item.setForeground(Qt.GlobalColor.gray)
        else:
            item.setForeground(Qt.GlobalColor.black)

        self._update_button_states()

    def _update_button_states(self) -> None:
        """Update button enabled states."""
        active_row = self.active_cameras_list.currentRow()
        has_active_selection = active_row >= 0

        self.remove_camera_btn.setEnabled(has_active_selection)
        self.move_up_btn.setEnabled(has_active_selection and active_row > 0)
        self.move_down_btn.setEnabled(
            has_active_selection and active_row < self.active_cameras_list.count() - 1
        )

        available_row = self.available_cameras_list.currentRow()
        self.add_camera_btn.setEnabled(available_row >= 0)

    def _on_ok_clicked(self) -> None:
        """Handle OK button click."""
        # Validate that we have at least one enabled camera if any cameras are configured
        if self._multi_camera_settings.cameras:
            active = self._multi_camera_settings.get_active_cameras()
            if not active:
                QMessageBox.warning(
                    self,
                    "No Active Cameras",
                    "Please enable at least one camera or remove all cameras.",
                )
                return

        self.settings_changed.emit(self._multi_camera_settings)
        self.accept()

    def get_settings(self) -> MultiCameraSettings:
        """Get the current multi-camera settings."""
        return self._multi_camera_settings
