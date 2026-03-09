"""UI building blocks for camera configuration dialog."""

# dlclivegui/gui/camera_config/ui_blocks.py

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...cameras import CameraFactory
from ..misc.drag_spinbox import ScrubSpinBox
from ..misc.eliding_label import ElidingPathLabel
from ..misc.layouts import make_two_field_row

if TYPE_CHECKING:
    from camera_config_dialog import CameraConfigDialog


__all__ = [
    "setup_camera_config_dialog_ui",
    "build_left_panel",
    "build_right_panel",
    "build_dialog_buttons_row",
    "build_active_cameras_group",
    "build_available_cameras_group",
    "build_settings_group",
    "build_preview_group",
    "build_right_scroll_container",
]


# ---------------------------------------------------------------------
# Public high-level entry point
# ---------------------------------------------------------------------
def setup_camera_config_dialog_ui(dlg: CameraConfigDialog) -> None:
    """
    Build the full dialog UI on `dlg`.

    This mirrors the original _setup_ui() structure:
      - main vertical layout
      - left/right panels row
      - OK/Cancel buttons row

    All widgets are attached to `dlg` using the same attribute names
    as the original monolithic implementation.
    """
    main_layout = QVBoxLayout(dlg)

    panels_layout = QHBoxLayout()
    left_panel = build_left_panel(dlg)
    right_panel = build_right_panel(dlg)

    panels_layout.addWidget(left_panel, stretch=1)
    panels_layout.addWidget(right_panel, stretch=1)

    buttons_layout = build_dialog_buttons_row(dlg)

    main_layout.addLayout(panels_layout)
    main_layout.addLayout(buttons_layout)


# ---------------------------------------------------------------------
# Left side: Active + Available panels
# ---------------------------------------------------------------------
def build_left_panel(dlg: CameraConfigDialog) -> QWidget:
    """Build the entire left panel (Active Cameras + Available Cameras)."""
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)

    active_group = build_active_cameras_group(dlg)
    available_group = build_available_cameras_group(dlg)

    left_layout.addWidget(active_group)
    left_layout.addWidget(available_group)

    return left_panel


def build_active_cameras_group(dlg: CameraConfigDialog) -> QGroupBox:
    """Build the 'Active Cameras' group box."""
    active_group = QGroupBox("Active Cameras")
    active_layout = QVBoxLayout(active_group)

    dlg.active_cameras_list = QListWidget()
    dlg.active_cameras_list.setMinimumWidth(250)
    active_layout.addWidget(dlg.active_cameras_list)

    # Buttons for managing active cameras
    list_buttons = QHBoxLayout()

    dlg.remove_camera_btn = QPushButton("Remove")
    dlg.remove_camera_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
    dlg.remove_camera_btn.setEnabled(False)

    dlg.move_up_btn = QPushButton("↑")
    dlg.move_up_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
    dlg.move_up_btn.setEnabled(False)

    dlg.move_down_btn = QPushButton("↓")
    dlg.move_down_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
    dlg.move_down_btn.setEnabled(False)

    list_buttons.addWidget(dlg.remove_camera_btn)
    list_buttons.addWidget(dlg.move_up_btn)
    list_buttons.addWidget(dlg.move_down_btn)

    active_layout.addLayout(list_buttons)
    return active_group


def build_available_cameras_group(dlg: CameraConfigDialog) -> QGroupBox:
    """Build the 'Available Cameras' group box, including scan UI widgets."""
    available_group = QGroupBox("Available Cameras")
    available_layout = QVBoxLayout(available_group)

    # Backend selection row
    backend_layout = QHBoxLayout()
    backend_layout.addWidget(QLabel("Backend:"))

    dlg.backend_combo = QComboBox()

    availability = CameraFactory.available_backends()
    for backend in CameraFactory.backend_names():
        label = backend
        if not availability.get(backend, True):
            label = f"{backend} (unavailable)"
        dlg.backend_combo.addItem(label, backend)

    if dlg.backend_combo.count() == 0:
        raise RuntimeError("No camera backends are registered!")

    # Switch to first available backend
    for i in range(dlg.backend_combo.count()):
        backend = dlg.backend_combo.itemData(i)
        if availability.get(backend, False):
            dlg.backend_combo.setCurrentIndex(i)
            break

    backend_layout.addWidget(dlg.backend_combo)

    dlg.refresh_btn = QPushButton("Refresh")
    dlg.refresh_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
    backend_layout.addWidget(dlg.refresh_btn)

    available_layout.addLayout(backend_layout)

    # Available list
    dlg.available_cameras_list = QListWidget()
    available_layout.addWidget(dlg.available_cameras_list)

    # Scan overlay (covers the available list area)
    dlg._scan_overlay = QLabel(available_group)
    dlg._scan_overlay.setVisible(False)
    dlg._scan_overlay.setAlignment(Qt.AlignCenter)
    dlg._scan_overlay.setWordWrap(True)
    dlg._scan_overlay.setStyleSheet(
        "background-color: rgba(0, 0, 0, 140);color: white;padding: 12px;border: 1px solid #333;font-size: 12px;"
    )
    dlg._scan_overlay.setText("Discovering cameras…")

    # Keep existing event filter behavior
    dlg.available_cameras_list.installEventFilter(dlg)

    # Indeterminate progress bar + status text for async scan
    dlg.scan_progress = QProgressBar()
    dlg.scan_progress.setRange(0, 0)
    dlg.scan_progress.setVisible(False)
    available_layout.addWidget(dlg.scan_progress)

    # Scan cancel button
    dlg.scan_cancel_btn = QPushButton("Cancel Scan")
    dlg.scan_cancel_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
    dlg.scan_cancel_btn.setVisible(False)

    # dlg must provide request_scan_cancel()
    if hasattr(dlg, "request_scan_cancel"):
        dlg.scan_cancel_btn.clicked.connect(dlg.request_scan_cancel)  # type: ignore[attr-defined]

    available_layout.addWidget(dlg.scan_cancel_btn)

    # Add camera button
    dlg.add_camera_btn = QPushButton("Add Selected Camera →")
    dlg.add_camera_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
    dlg.add_camera_btn.setEnabled(False)
    available_layout.addWidget(dlg.add_camera_btn)

    return available_group


# ---------------------------------------------------------------------
# Right side: Settings + Preview + Scroll container
# ---------------------------------------------------------------------
def build_right_panel(dlg: CameraConfigDialog) -> QWidget:
    """Build the entire right panel (settings + preview inside a scroll area)."""
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)

    settings_group = build_settings_group(dlg)
    preview_group = build_preview_group(dlg)

    scroll = build_right_scroll_container(dlg, settings_group, preview_group)
    right_layout.addWidget(scroll)

    return right_panel


def build_settings_group(dlg: CameraConfigDialog) -> QGroupBox:
    """Build the 'Camera Settings' group box and its form widgets."""
    settings_group = QGroupBox("Camera Settings")
    dlg.settings_form = QFormLayout(settings_group)
    dlg.settings_form.setVerticalSpacing(6)
    dlg.settings_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    # --- Basic toggles/labels ---
    dlg.cam_enabled_checkbox = QCheckBox("Enabled")
    dlg.cam_enabled_checkbox.setChecked(True)
    dlg.settings_form.addRow(dlg.cam_enabled_checkbox)

    dlg.cam_name_label = QLabel("Camera 0")
    dlg.cam_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
    dlg.settings_form.addRow("Name:", dlg.cam_name_label)

    dlg.cam_device_name_label = ElidingPathLabel("")
    dlg.cam_device_name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    dlg.cam_device_name_label.setWordWrap(True)
    dlg.settings_form.addRow("Device ID:", dlg.cam_device_name_label)

    dlg.cam_index_label = QLabel("0")

    dlg.cam_backend_label = QLabel("opencv")
    id_backend_row = make_two_field_row(
        "Index:",
        dlg.cam_index_label,
        "Backend:",
        dlg.cam_backend_label,
        key_width=120,
        gap=15,
    )
    dlg.settings_form.addRow(id_backend_row)

    # --- Detected read-only labels ---
    dlg.detected_resolution_label = QLabel("—")
    dlg.detected_resolution_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

    dlg.detected_fps_label = QLabel("—")
    dlg.detected_fps_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

    detected_row = make_two_field_row(
        "Detected resolution:",
        dlg.detected_resolution_label,
        "Detected FPS:",
        dlg.detected_fps_label,
        key_width=120,
        gap=10,
    )
    dlg.settings_form.addRow(detected_row)

    # --- Requested resolution controls (Auto = 0) ---
    dlg.cam_width = QSpinBox()
    dlg.cam_width.setRange(0, 10000)
    dlg.cam_width.setValue(0)
    dlg.cam_width.setSpecialValueText("Auto")

    dlg.cam_height = QSpinBox()
    dlg.cam_height.setRange(0, 10000)
    dlg.cam_height.setValue(0)
    dlg.cam_height.setSpecialValueText("Auto")

    res_row = make_two_field_row("W", dlg.cam_width, "H", dlg.cam_height, key_width=30)
    dlg.settings_form.addRow("Resolution:", res_row)

    # --- FPS + Rotation grouped ---
    dlg.cam_fps = QDoubleSpinBox()
    dlg.cam_fps.setRange(0.0, 240.0)
    dlg.cam_fps.setDecimals(2)
    dlg.cam_fps.setSingleStep(1.0)
    dlg.cam_fps.setValue(0.0)
    dlg.cam_fps.setSpecialValueText("Auto")

    dlg.cam_rotation = QComboBox()
    dlg.cam_rotation.addItem("0°", 0)
    dlg.cam_rotation.addItem("90°", 90)
    dlg.cam_rotation.addItem("180°", 180)
    dlg.cam_rotation.addItem("270°", 270)

    fps_rot_row = make_two_field_row("FPS", dlg.cam_fps, "Rot", dlg.cam_rotation, key_width=30)
    dlg.settings_form.addRow("Capture:", fps_rot_row)

    # --- Exposure + Gain grouped ---
    dlg.cam_exposure = QSpinBox()
    dlg.cam_exposure.setRange(0, 1000000)
    dlg.cam_exposure.setValue(0)
    dlg.cam_exposure.setSpecialValueText("Auto")
    dlg.cam_exposure.setSuffix(" μs")

    dlg.cam_gain = QDoubleSpinBox()
    dlg.cam_gain.setRange(0.0, 100.0)
    dlg.cam_gain.setValue(0.0)
    dlg.cam_gain.setSpecialValueText("Auto")
    dlg.cam_gain.setDecimals(2)

    exp_gain_row = make_two_field_row("Exp", dlg.cam_exposure, "Gain", dlg.cam_gain, key_width=30)
    dlg.settings_form.addRow("Analog:", exp_gain_row)

    # --- Crop row ---
    crop_widget = QWidget()
    crop_layout = QHBoxLayout(crop_widget)
    crop_layout.setContentsMargins(0, 0, 0, 0)

    dlg.cam_crop_x0 = ScrubSpinBox()
    dlg.cam_crop_x0.setRange(0, 7680)
    dlg.cam_crop_x0.setPrefix("x0:")
    dlg.cam_crop_x0.setSpecialValueText("x0:None")
    crop_layout.addWidget(dlg.cam_crop_x0)

    dlg.cam_crop_y0 = ScrubSpinBox()
    dlg.cam_crop_y0.setRange(0, 4320)
    dlg.cam_crop_y0.setPrefix("y0:")
    dlg.cam_crop_y0.setSpecialValueText("y0:None")
    crop_layout.addWidget(dlg.cam_crop_y0)

    dlg.cam_crop_x1 = ScrubSpinBox()
    dlg.cam_crop_x1.setRange(0, 7680)
    dlg.cam_crop_x1.setPrefix("x1:")
    dlg.cam_crop_x1.setSpecialValueText("x1:None")
    crop_layout.addWidget(dlg.cam_crop_x1)

    dlg.cam_crop_y1 = ScrubSpinBox()
    dlg.cam_crop_y1.setRange(0, 4320)
    dlg.cam_crop_y1.setPrefix("y1:")
    dlg.cam_crop_y1.setSpecialValueText("y1:None")
    crop_layout.addWidget(dlg.cam_crop_y1)

    dlg.settings_form.addRow("Crop:", crop_widget)

    # Apply/Reset buttons row
    dlg.apply_settings_btn = QPushButton("Apply Settings")
    dlg.apply_settings_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
    dlg.apply_settings_btn.setEnabled(False)

    dlg.reset_settings_btn = QPushButton("Reset Settings")
    dlg.reset_settings_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
    dlg.reset_settings_btn.setEnabled(False)

    sttgs_buttons_row = QWidget()
    sttgs_button_layout = QHBoxLayout(sttgs_buttons_row)
    sttgs_button_layout.setContentsMargins(0, 0, 0, 0)
    sttgs_button_layout.setSpacing(8)
    sttgs_button_layout.addWidget(dlg.apply_settings_btn)
    sttgs_button_layout.addWidget(dlg.reset_settings_btn)

    dlg.settings_form.addRow(sttgs_buttons_row)

    # Preview button
    dlg.preview_btn = QPushButton("Start Preview")
    dlg.preview_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    dlg.preview_btn.setEnabled(False)
    dlg.settings_form.addRow(dlg.preview_btn)

    # Pressing enter on any settings field applies settings
    dlg.cam_fps.setKeyboardTracking(False)

    fields = [
        dlg.cam_enabled_checkbox,
        dlg.cam_width,
        dlg.cam_height,
        dlg.cam_fps,
        dlg.cam_exposure,
        dlg.cam_gain,
        dlg.cam_crop_x0,
        dlg.cam_crop_y0,
        dlg.cam_crop_x1,
        dlg.cam_crop_y1,
    ]
    for field in fields:
        if hasattr(field, "lineEdit"):
            le = field.lineEdit()  # type: ignore[call-arg]
            if hasattr(le, "returnPressed") and hasattr(dlg, "_apply_camera_settings"):
                le.returnPressed.connect(dlg._apply_camera_settings)  # type: ignore[attr-defined]
        if hasattr(field, "installEventFilter"):
            field.installEventFilter(dlg)

    return settings_group


def build_preview_group(dlg: CameraConfigDialog) -> QGroupBox:
    """Build the 'Camera Preview' group box."""
    dlg.preview_group = QGroupBox("Camera Preview")
    preview_layout = QVBoxLayout(dlg.preview_group)

    dlg.preview_label = QLabel("No preview")
    dlg.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    dlg.preview_label.setMinimumSize(320, 240)
    dlg.preview_label.setMaximumSize(400, 300)
    dlg.preview_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
    preview_layout.addWidget(dlg.preview_label)
    dlg.preview_label.installEventFilter(dlg)

    dlg.preview_status = QTextEdit()
    dlg.preview_status.setReadOnly(True)
    dlg.preview_status.setFixedHeight(45)
    dlg.preview_status.setStyleSheet("QTextEdit { background: #141414; color: #bdbdbd; border: 1px solid #2a2a2a; }")
    font = QFont("Consolas")
    font.setPointSize(9)
    dlg.preview_status.setFont(font)
    preview_layout.addWidget(dlg.preview_status)

    dlg._loading_overlay = QLabel(dlg.preview_group)
    dlg._loading_overlay.setVisible(False)
    dlg._loading_overlay.setAlignment(Qt.AlignCenter)
    dlg._loading_overlay.setStyleSheet("background-color: rgba(0,0,0,140); color: white; border: 1px solid #333;")
    dlg._loading_overlay.setText("Loading camera…")

    dlg.preview_group.setVisible(False)
    return dlg.preview_group


def build_right_scroll_container(
    dlg: CameraConfigDialog, settings_group: QGroupBox, preview_group: QGroupBox
) -> QScrollArea:
    """Wrap the settings and preview groups in a scroll area to prevent squishing."""
    scroll = QScrollArea()
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)

    scroll_contents = QWidget()
    scroll_contents.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    dlg._settings_scroll = scroll
    dlg._settings_scroll_contents = scroll_contents

    scroll_contents.setMinimumWidth(scroll.viewport().width())
    scroll.viewport().installEventFilter(dlg)

    scroll_layout = QVBoxLayout(scroll_contents)
    scroll_layout.setContentsMargins(0, 0, 0, 10)
    scroll_layout.setSpacing(10)

    settings_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    preview_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    scroll_layout.addWidget(settings_group)
    scroll_layout.addWidget(preview_group)
    scroll_layout.addStretch(1)

    scroll.setWidget(scroll_contents)
    return scroll


# ---------------------------------------------------------------------
# Bottom row: OK / Cancel buttons
# ---------------------------------------------------------------------
def build_dialog_buttons_row(dlg: CameraConfigDialog) -> QHBoxLayout:
    """Build the bottom OK/Cancel button row."""
    sttgs_button_layout = QHBoxLayout()

    dlg.ok_btn = QPushButton("OK")
    dlg.ok_btn.setAutoDefault(False)
    dlg.ok_btn.setDefault(False)
    dlg.ok_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_DialogOkButton))

    dlg.cancel_btn = QPushButton("Cancel")
    dlg.cancel_btn.setAutoDefault(False)
    dlg.cancel_btn.setDefault(False)
    dlg.cancel_btn.setIcon(dlg.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))

    sttgs_button_layout.addStretch(1)
    sttgs_button_layout.addWidget(dlg.ok_btn)
    sttgs_button_layout.addWidget(dlg.cancel_btn)

    return sttgs_button_layout
