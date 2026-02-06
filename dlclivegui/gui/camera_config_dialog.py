"""Camera configuration dialog for multi-camera setup (with async preview loading)."""

# dlclivegui/gui/camera_config_dialog.py
from __future__ import annotations

import copy
import logging

import cv2
from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QKeyEvent, QPixmap, QTextCursor
from PySide6.QtWidgets import (
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.base import CameraBackend
from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.config import CameraSettings, MultiCameraSettings

LOGGER = logging.getLogger(__name__)


def _apply_detected_identity(cam: CameraSettings, detected: DetectedCamera, backend: str) -> None:
    """Persist stable identity from a detected camera into cam.properties under backend namespace."""
    if not isinstance(cam.properties, dict):
        cam.properties = {}

    ns = cam.properties.get(backend.lower())
    if not isinstance(ns, dict):
        ns = {}
        cam.properties[backend.lower()] = ns

    # Store whatever we have (backend-specific but written generically)
    if getattr(detected, "device_id", None):
        ns["device_id"] = detected.device_id
    if getattr(detected, "vid", None) is not None:
        ns["device_vid"] = int(detected.vid)
    if getattr(detected, "pid", None) is not None:
        ns["device_pid"] = int(detected.pid)
    if getattr(detected, "path", None):
        ns["device_path"] = detected.path

    # Optional: store human name for matching fallback
    if getattr(detected, "label", None):
        ns["device_name"] = detected.label

    # Optional: store backend_hint if you expose it (e.g., CAP_DSHOW)
    if getattr(detected, "backend_hint", None) is not None:
        ns["backend_hint"] = int(detected.backend_hint)


# -------------------------------
# Background worker to detect cameras
# -------------------------------
class DetectCamerasWorker(QThread):
    """Background worker to detect cameras for the selected backend."""

    progress = Signal(str)  # human-readable text
    result = Signal(list)  # list[DetectedCamera]
    error = Signal(str)
    finished = Signal()

    def __init__(self, backend: str, max_devices: int = 10, parent: QWidget | None = None):
        super().__init__(parent)
        self.backend = backend
        self.max_devices = max_devices

    def run(self):
        try:
            # Initial message
            self.progress.emit(f"Scanning {self.backend} cameras…")

            cams = CameraFactory.detect_cameras(
                self.backend,
                max_devices=self.max_devices,
                should_cancel=self.isInterruptionRequested,
                progress_cb=self.progress.emit,
            )
            self.result.emit(cams)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


# -------------------------------
# Singleton camera preview loader worker
# -------------------------------
class CameraLoadWorker(QThread):
    """Open/configure a camera backend off the UI thread with progress and cancel support."""

    progress = Signal(str)  # Human-readable status updates
    success = Signal(object)  # Emits the ready backend (CameraBackend)
    error = Signal(str)  # Emits error message
    canceled = Signal()  # Emits when canceled before success

    def __init__(self, cam: CameraSettings, parent: QWidget | None = None):
        super().__init__(parent)
        # Work on a defensive copy so we never mutate the original settings
        self._cam = copy.deepcopy(cam)
        # Make first-time opening snappier by allowing backend fast-path if supported
        if isinstance(self._cam.properties, dict):
            self._cam.properties.setdefault("fast_start", True)
        self._cancel = False
        self._backend: CameraBackend | None = None

    def request_cancel(self):
        self._cancel = True

    def _check_cancel(self) -> bool:
        if self._cancel:
            self.progress.emit("Canceled by user.")
            return True
        return False

    def run(self):
        try:
            self.progress.emit("Creating backend…")
            if self._check_cancel():
                self.canceled.emit()
                return

            LOGGER.debug("Creating camera backend for %s:%d", self._cam.backend, self._cam.index)
            self.progress.emit("Opening device…")
            # Open only in GUI thread to avoid simultaneous opens
            self.success.emit(self._cam)

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            try:
                if self._backend:
                    self._backend.close()
            except Exception:
                pass
            self.error.emit(msg)


class CameraConfigDialog(QDialog):
    """Dialog for configuring multiple cameras with async preview loading."""

    MAX_CAMERAS = 4
    settings_changed = Signal(object)  # MultiCameraSettingsModel
    # Camera discovery signals
    scan_started = Signal(str)
    scan_finished = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        multi_camera_settings: MultiCameraSettings | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Configure Cameras")
        self.setMinimumSize(960, 720)

        self._dlc_camera_id = None
        self.dlc_camera_id: str | None = None
        # Actual/working camera settings
        self._multi_camera_settings = multi_camera_settings
        self._working_settings = self._multi_camera_settings.model_copy(deep=True)
        self._detected_cameras: list[DetectedCamera] = []
        self._current_edit_index: int | None = None

        # Preview state
        self._preview_backend: CameraBackend | None = None
        self._preview_timer: QTimer | None = None
        self._preview_active: bool = False

        # Camera detection worker
        self._scan_worker: DetectCamerasWorker | None = None

        # Singleton loader per dialog
        self._loader: CameraLoadWorker | None = None
        self._loading_active: bool = False

        self._setup_ui()
        self._populate_from_settings()
        self._connect_signals()

    @property
    def dlc_camera_id(self) -> str | None:
        """Get the currently selected DLC camera ID."""
        return self._dlc_camera_id

    @dlc_camera_id.setter
    def dlc_camera_id(self, value: str | None) -> None:
        """Set the currently selected DLC camera ID."""
        self._dlc_camera_id = value
        self._refresh_camera_labels()

    # -------------------------------
    # Config helpers
    # ------------------------------

    def _build_model_from_form(self, base: CameraSettings) -> CameraSettings:
        # construct a dict from form widgets; Pydantic will coerce/validate
        payload = base.model_dump()
        payload.update(
            {
                "enabled": bool(self.cam_enabled_checkbox.isChecked()),
                "fps": float(self.cam_fps.value()),
                "exposure": int(self.cam_exposure.value()),
                "gain": float(self.cam_gain.value()),
                "rotation": int(self.cam_rotation.currentData() or 0),
                "crop_x0": int(self.cam_crop_x0.value()),
                "crop_y0": int(self.cam_crop_y0.value()),
                "crop_x1": int(self.cam_crop_x1.value()),
                "crop_y1": int(self.cam_crop_y1.value()),
            }
        )
        #  Validate and coerce; if invalid, Pydantic will raise
        return CameraSettings.model_validate(payload)

    def _merge_backend_settings_back(self, opened_settings: CameraSettings) -> None:
        """Merge identity/index changes learned during preview open back into the working settings."""
        if self._current_edit_index is None:
            return
        row = self._current_edit_index
        if row < 0 or row >= len(self._working_settings.cameras):
            return

        target = self._working_settings.cameras[row]

        # Update index if backend rebinding occurred
        try:
            target.index = int(opened_settings.index)
        except Exception:
            pass

        # Merge properties (especially stable IDs) back
        if isinstance(opened_settings.properties, dict):
            if not isinstance(target.properties, dict):
                target.properties = {}
            # shallow merge is ok; backend namespaces are nested dicts
            for k, v in opened_settings.properties.items():
                if isinstance(v, dict) and isinstance(target.properties.get(k), dict):
                    target.properties[k].update(v)
                else:
                    target.properties[k] = v

        # Update UI list item text to reflect any changes
        self._update_active_list_item(row, target)
        self._load_camera_to_form(target)

    # -------------------------------
    # UI setup
    # -------------------------------
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
        self.remove_camera_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.remove_camera_btn.setEnabled(False)
        self.move_up_btn = QPushButton("↑")
        self.move_up_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        self.move_up_btn.setEnabled(False)
        self.move_down_btn = QPushButton("↓")
        self.move_down_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
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
        if self.backend_combo.count() == 0:
            raise RuntimeError("No camera backends are registered!")
        # Switch to first available backend
        for i in range(self.backend_combo.count()):
            backend = self.backend_combo.itemData(i)
            if availability.get(backend, False):
                self.backend_combo.setCurrentIndex(i)
                break
        backend_layout.addWidget(self.backend_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        backend_layout.addWidget(self.refresh_btn)
        available_layout.addLayout(backend_layout)

        self.available_cameras_list = QListWidget()
        available_layout.addWidget(self.available_cameras_list)

        # Show status overlay during scan
        self._scan_overlay = QLabel(available_group)
        self._scan_overlay.setVisible(False)
        self._scan_overlay.setAlignment(Qt.AlignCenter)
        self._scan_overlay.setWordWrap(True)
        self._scan_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 140);color: white;padding: 12px;border: 1px solid #333;font-size: 12px;"
        )
        self._scan_overlay.setText("Discovering cameras…")
        self.available_cameras_list.installEventFilter(self)

        # Indeterminate progress bar + status text for async scan
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 0)
        self.scan_progress.setVisible(False)

        available_layout.addWidget(self.scan_progress)

        self.scan_cancel_btn = QPushButton("Cancel Scan")
        self.scan_cancel_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        self.scan_cancel_btn.setVisible(False)
        self.scan_cancel_btn.clicked.connect(self._on_scan_cancel)
        available_layout.addWidget(self.scan_cancel_btn)

        self.add_camera_btn = QPushButton("Add Selected Camera →")
        self.add_camera_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
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

        self.cam_device_id_label = QLabel("")
        self.cam_device_id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.settings_form.addRow("Device ID:", self.cam_device_id_label)

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
        self.apply_settings_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.apply_settings_btn.setEnabled(False)
        self.settings_form.addRow(self.apply_settings_btn)

        # Preview button
        self.preview_btn = QPushButton("Start Preview")
        self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.preview_btn.setEnabled(False)
        self.settings_form.addRow(self.preview_btn)

        right_layout.addWidget(settings_group)

        # Preview widget
        self.preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(self.preview_group)

        self.preview_label = QLabel("No preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setMaximumSize(400, 300)
        self.preview_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        preview_layout.addWidget(self.preview_label)
        self.preview_label.installEventFilter(self)  # For resize events

        # Small, read-only status console for loader messages
        self.preview_status = QTextEdit()
        self.preview_status.setReadOnly(True)
        self.preview_status.setFixedHeight(45)
        self.preview_status.setStyleSheet(
            "QTextEdit { background: #141414; color: #bdbdbd; border: 1px solid #2a2a2a; }"
        )
        font = QFont("Consolas")
        font.setPointSize(9)
        self.preview_status.setFont(font)
        preview_layout.addWidget(self.preview_status)

        # Overlay label for loading glass pane
        self._loading_overlay = QLabel(self.preview_group)
        self._loading_overlay.setVisible(False)
        self._loading_overlay.setAlignment(Qt.AlignCenter)
        self._loading_overlay.setStyleSheet("background-color: rgba(0,0,0,140); color: white; border: 1px solid #333;")
        self._loading_overlay.setText("Loading camera…")

        self.preview_group.setVisible(False)
        right_layout.addWidget(self.preview_group)

        right_layout.addStretch(1)

        # Dialog buttons
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setAutoDefault(False)
        self.ok_btn.setDefault(False)
        self.ok_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOkButton))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.setDefault(False)
        self.cancel_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
        button_layout.addStretch(1)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)

        # Add panels to horizontal layout
        panels_layout.addWidget(left_panel, stretch=1)
        panels_layout.addWidget(right_panel, stretch=1)

        # Add everything to main layout
        main_layout.addLayout(panels_layout)
        main_layout.addLayout(button_layout)

        # Pressing enter on any settings field applies settings
        self.cam_fps.setKeyboardTracking(False)
        fields = [
            self.cam_enabled_checkbox,
            self.cam_fps,
            self.cam_exposure,
            self.cam_gain,
            self.cam_crop_x0,
            self.cam_crop_y0,
            self.cam_crop_x1,
            self.cam_crop_y1,
        ]
        for field in fields:
            if hasattr(field, "lineEdit"):
                if hasattr(field.lineEdit(), "returnPressed"):
                    field.lineEdit().returnPressed.connect(self._apply_camera_settings)
            if hasattr(field, "installEventFilter"):
                field.installEventFilter(self)

    # Maintain overlay geometry when resizing
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay") and self._loading_overlay.isVisible():
            self._position_loading_overlay()

    def eventFilter(self, obj, event):
        # Keep your existing overlay resize handling
        if obj is self.available_cameras_list and event.type() == event.Type.Resize:
            if self._scan_overlay and self._scan_overlay.isVisible():
                self._position_scan_overlay()
            return super().eventFilter(obj, event)

        # Intercept Enter in FPS and crop spinboxes
        if event.type() == QEvent.KeyPress and isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if obj in (self.cam_fps, self.cam_crop_x0, self.cam_crop_y0, self.cam_crop_x1, self.cam_crop_y1):
                    # Commit any pending text → value
                    try:
                        obj.interpretText()
                    except Exception:
                        pass
                    # Apply settings to persist crop/FPS to CameraSettings
                    self._apply_camera_settings()
                    # Consume so OK isn't triggered
                    return True

        return super().eventFilter(obj, event)

    def _position_scan_overlay(self) -> None:
        """Position scan overlay to cover the available_cameras_list area."""
        if not self._scan_overlay or not self.available_cameras_list:
            return
        parent = self._scan_overlay.parent()  # available_group
        top_left = self.available_cameras_list.mapTo(parent, self.available_cameras_list.rect().topLeft())
        rect = self.available_cameras_list.rect()
        self._scan_overlay.setGeometry(top_left.x(), top_left.y(), rect.width(), rect.height())

    def _show_scan_overlay(self, message: str = "Discovering cameras…") -> None:
        self._scan_overlay.setText(message)
        self._scan_overlay.setVisible(True)
        self._position_scan_overlay()

    def _hide_scan_overlay(self) -> None:
        self._scan_overlay.setVisible(False)

    def _position_loading_overlay(self):
        # Cover just the preview image area (label), not the whole group
        if not self.preview_label:
            return
        gp = self.preview_label.mapTo(self.preview_group, self.preview_label.rect().topLeft())
        rect = self.preview_label.rect()
        self._loading_overlay.setGeometry(gp.x(), gp.y(), rect.width(), rect.height())

    def _camera_identity_key(self, cam: CameraSettings) -> tuple:
        backend = (cam.backend or "").lower()
        props = cam.properties if isinstance(cam.properties, dict) else {}
        ns = props.get(backend, {}) if isinstance(props, dict) else {}
        device_id = ns.get("device_id")

        # Prefer stable identity if present, otherwise fallback
        if device_id:
            return (backend, "device_id", device_id)
        return (backend, "index", int(cam.index))

    # -------------------------------
    # Signals / population
    # -------------------------------
    def _connect_signals(self) -> None:
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.refresh_btn.clicked.connect(self._refresh_available_cameras)
        self.add_camera_btn.clicked.connect(self._add_selected_camera)
        self.remove_camera_btn.clicked.connect(self._remove_selected_camera)
        self.move_up_btn.clicked.connect(self._move_camera_up)
        self.move_down_btn.clicked.connect(self._move_camera_down)
        self.active_cameras_list.currentRowChanged.connect(self._on_active_camera_selected)
        self.available_cameras_list.currentRowChanged.connect(self._on_available_camera_selected)
        self.available_cameras_list.itemDoubleClicked.connect(self._on_available_camera_double_clicked)
        self.apply_settings_btn.clicked.connect(self._apply_camera_settings)
        self.preview_btn.clicked.connect(self._toggle_preview)
        self.ok_btn.clicked.connect(self._on_ok_clicked)
        self.cancel_btn.clicked.connect(self.reject)
        self.scan_started.connect(lambda _: setattr(self, "_dialog_active", True))
        self.scan_finished.connect(lambda: setattr(self, "_dialog_active", False))
        for sb in (self.cam_fps, self.cam_crop_x0, self.cam_crop_y0, self.cam_crop_x1, self.cam_crop_y1):
            if hasattr(sb, "valueChanged"):
                sb.valueChanged.connect(lambda _=None: self.apply_settings_btn.setEnabled(True))
        self.cam_rotation.currentIndexChanged.connect(lambda _: self.apply_settings_btn.setEnabled(True))

    def _populate_from_settings(self) -> None:
        """Populate the dialog from existing settings."""
        self.active_cameras_list.clear()
        for i, cam in enumerate(self._working_settings.cameras):
            item = QListWidgetItem(self._format_camera_label(cam, i))
            item.setData(Qt.ItemDataRole.UserRole, cam)
            if not cam.enabled:
                item.setForeground(Qt.GlobalColor.gray)
            self.active_cameras_list.addItem(item)

        self._refresh_available_cameras()
        self._update_button_states()

    def _format_camera_label(self, cam: CameraSettings, index: int = -1) -> str:
        status = "✓" if cam.enabled else "○"
        this_id = f"{cam.backend}:{cam.index}"
        dlc_indicator = " [DLC]" if this_id == self._dlc_camera_id and cam.enabled else ""
        return f"{status} {cam.name} [{cam.backend}:{cam.index}]{dlc_indicator}"

    def _refresh_camera_labels(self) -> None:
        cam_list = getattr(self, "active_cameras_list", None)
        if cam_list:
            for i in range(cam_list.count()):
                item = cam_list.item(i)
                cam = item.data(Qt.ItemDataRole.UserRole)
                if cam:
                    item.setText(self._format_camera_label(cam, i))

    def _on_backend_changed(self, _index: int) -> None:
        self._refresh_available_cameras()

    def _is_backend_opencv(self, backend_name: str) -> bool:
        return backend_name.lower() == "opencv"

    def _update_controls_for_backend(self, backend_name: str) -> None:
        # FIXME in camera backend ABC, we should have a method to query supported features
        is_opencv = self._is_backend_opencv(backend_name)
        self.cam_exposure.setEnabled(not is_opencv)
        self.cam_gain.setEnabled(not is_opencv)

        tip = ""
        if is_opencv:
            tip = (
                "Exposure/Gain are not configurable via the generic OpenCV backend and "
                "will be ignored by most UVC devices."
            )
        self.cam_exposure.setToolTip(tip)
        self.cam_gain.setToolTip(tip)

    def _refresh_available_cameras(self) -> None:
        """Refresh the list of available cameras asynchronously."""
        backend = self.backend_combo.currentData()
        if not backend:
            backend = self.backend_combo.currentText().split()[0]

        # If already scanning, ignore new requests to avoid races
        if getattr(self, "_scan_worker", None) and self._scan_worker.isRunning():
            self._show_scan_overlay("Already discovering cameras…")
            return

        # Reset list UI and show progress
        self.available_cameras_list.clear()
        self._detected_cameras = []
        msg = f"Discovering {backend} cameras…"
        self._show_scan_overlay(msg)
        self.scan_progress.setRange(0, 0)
        self.scan_progress.setVisible(True)
        self.scan_cancel_btn.setVisible(True)
        self.add_camera_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.backend_combo.setEnabled(False)

        # Start worker
        self._scan_worker = DetectCamerasWorker(backend, max_devices=10, parent=self)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.result.connect(self._on_scan_result)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self.scan_started.emit(f"Scanning {backend} cameras…")
        self._scan_worker.start()

    def _on_scan_progress(self, msg: str) -> None:
        self._show_scan_overlay(msg or "Discovering cameras…")

    def _on_scan_result(self, cams: list) -> None:
        self._detected_cameras = cams or []
        self.available_cameras_list.clear()  # replace list contents

        if not self._detected_cameras:
            placeholder = QListWidgetItem("No cameras detected.")
            placeholder.setFlags(Qt.ItemIsEnabled)
            self.available_cameras_list.addItem(placeholder)
            return

        for cam in self._detected_cameras:
            item = QListWidgetItem(f"{cam.label} (index {cam.index})")
            item.setData(Qt.ItemDataRole.UserRole, cam)
            self.available_cameras_list.addItem(item)

        self.available_cameras_list.setCurrentRow(0)

    def _on_scan_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Camera Scan", f"Failed to detect cameras:\n{msg}")

    def _on_scan_finished(self) -> None:
        self._hide_scan_overlay()
        self.scan_progress.setVisible(False)
        self._scan_worker = None

        self.scan_cancel_btn.setVisible(False)
        self.scan_cancel_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.backend_combo.setEnabled(True)

        self._update_button_states()
        self.scan_finished.emit()

    def _on_scan_cancel(self) -> None:
        """User requested to cancel discovery."""
        if self._scan_worker and self._scan_worker.isRunning():
            try:
                self._scan_worker.requestInterruption()
            except Exception:
                pass
            # Keep the busy bar, update texts
            self._show_scan_overlay("Canceling discovery…")
            self.scan_progress.setVisible(True)  # stay visible as indeterminate
            self.scan_cancel_btn.setEnabled(False)

    def _on_available_camera_selected(self, row: int) -> None:
        self.add_camera_btn.setEnabled(row >= 0)

    def _on_available_camera_double_clicked(self, item: QListWidgetItem) -> None:
        self._add_selected_camera()

    def _on_active_camera_selected(self, row: int) -> None:
        # Stop any running preview when selection changes
        if self._preview_active:
            self._stop_preview()
        self._current_edit_index = row
        self._update_button_states()
        if row < 0 or row >= self.active_cameras_list.count():
            self._clear_settings_form()
            return
        item = self.active_cameras_list.item(row)
        cam = item.data(Qt.ItemDataRole.UserRole)
        if cam:
            self.apply_settings_btn.setEnabled(True)
            self._load_camera_to_form(cam)

    # -------------------------------
    # UI helpers/actions
    # -------------------------------

    def _needs_preview_reopen(self, cam: CameraSettings) -> bool:
        if not (self._preview_active and self._preview_backend):
            return False

        # FPS: for OpenCV, treat FPS changes as requiring reopen.
        if self._is_backend_opencv(cam.backend):
            prev_fps = getattr(self._preview_backend.settings, "fps", None)
            if isinstance(prev_fps, (int, float)) and abs(cam.fps - float(prev_fps)) > 0.1:
                return True

        return any(
            [
                cam.exposure != getattr(self._preview_backend.settings, "exposure", cam.exposure),
                cam.gain != getattr(self._preview_backend.settings, "gain", cam.gain),
                cam.rotation != getattr(self._preview_backend.settings, "rotation", cam.rotation),
                (cam.crop_x0, cam.crop_y0, cam.crop_x1, cam.crop_y1)
                != (
                    getattr(self._preview_backend.settings, "crop_x0", cam.crop_x0),
                    getattr(self._preview_backend.settings, "crop_y0", cam.crop_y0),
                    getattr(self._preview_backend.settings, "crop_x1", cam.crop_x1),
                    getattr(self._preview_backend.settings, "crop_y1", cam.crop_y1),
                ),
            ]
        )

    def _backend_actual_fps(self) -> float | None:
        """Return backend's actual FPS if known; for OpenCV do NOT fall back to settings.fps."""
        if not self._preview_backend:
            return None
        try:
            actual = getattr(self._preview_backend, "actual_fps", None)
            if isinstance(actual, (int, float)) and actual > 0:
                return float(actual)
            return None
        except Exception:
            return None

    def _adjust_preview_timer_for_fps(self, fps: float | None) -> None:
        """Adjust preview cadence to match actual FPS (bounded for CPU)."""
        if not self._preview_timer or not fps or fps <= 0:
            return
        interval_ms = max(15, int(1000.0 / min(max(fps, 1.0), 60.0)))
        self._preview_timer.start(interval_ms)

    def _reconcile_fps_from_backend(self, cam: CameraSettings) -> None:
        """Clamp UI/settings to measured device FPS when we can actually measure it."""
        if not self._is_backend_opencv(cam.backend):
            return

        actual = self._backend_actual_fps()
        if actual is None:
            # OpenCV can't reliably report FPS; do not overwrite user's requested value.
            self._append_status("[Info] OpenCV can't reliably report actual FPS; keeping requested value.")
            return

        if abs(cam.fps - actual) > 0.5:
            cam.fps = actual
            self.cam_fps.setValue(actual)
            self._append_status(f"[Info] FPS adjusted to device-supported ~{actual:.2f}.")
            self._adjust_preview_timer_for_fps(actual)

    def _update_active_list_item(self, row: int, cam: CameraSettings) -> None:
        """Refresh the active camera list row text and color."""
        item = self.active_cameras_list.item(row)
        if not item:
            return
        item.setText(self._format_camera_label(cam, row))
        item.setData(Qt.ItemDataRole.UserRole, cam)
        item.setForeground(Qt.GlobalColor.gray if not cam.enabled else Qt.GlobalColor.black)
        self._refresh_camera_labels()
        self._update_button_states()

    def _load_camera_to_form(self, cam: CameraSettings) -> None:
        backend = (cam.backend or "").lower()
        props = cam.properties if isinstance(cam.properties, dict) else {}
        ns = props.get(backend, {}) if isinstance(props, dict) else {}
        self.cam_enabled_checkbox.setChecked(cam.enabled)
        self.cam_name_label.setText(cam.name)
        self.cam_device_id_label.setText(str(ns.get("device_id", "")))
        self.cam_index_label.setText(str(cam.index))
        self.cam_backend_label.setText(cam.backend)
        self._update_controls_for_backend(cam.backend)
        self.cam_fps.setValue(cam.fps)
        self.cam_exposure.setValue(cam.exposure)
        self.cam_gain.setValue(cam.gain)
        rot_index = self.cam_rotation.findData(cam.rotation)
        if rot_index >= 0:
            self.cam_rotation.setCurrentIndex(rot_index)
        self.cam_crop_x0.setValue(cam.crop_x0)
        self.cam_crop_y0.setValue(cam.crop_y0)
        self.cam_crop_x1.setValue(cam.crop_x1)
        self.cam_crop_y1.setValue(cam.crop_y1)
        self.apply_settings_btn.setEnabled(True)

    def _write_form_to_cam(self, cam: CameraSettings) -> None:
        cam.enabled = bool(self.cam_enabled_checkbox.isChecked())
        cam.fps = float(self.cam_fps.value())
        cam.exposure = int(self.cam_exposure.value())
        cam.gain = float(self.cam_gain.value())
        cam.rotation = int(self.cam_rotation.currentData() or 0)
        cam.crop_x0 = int(self.cam_crop_x0.value())
        cam.crop_y0 = int(self.cam_crop_y0.value())
        cam.crop_x1 = int(self.cam_crop_x1.value())
        cam.crop_y1 = int(self.cam_crop_y1.value())

    def _clear_settings_form(self) -> None:
        self.cam_enabled_checkbox.setChecked(True)
        self.cam_name_label.setText("")
        self.cam_device_id_label.setText("")
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
        row = self.available_cameras_list.currentRow()
        if row < 0:
            return
        # limit check
        active_count = len(
            [
                i
                for i in range(self.active_cameras_list.count())
                if self.active_cameras_list.item(i).data(Qt.ItemDataRole.UserRole).enabled
            ]
        )
        if active_count >= self.MAX_CAMERAS:
            QMessageBox.warning(self, "Maximum Cameras", f"Maximum of {self.MAX_CAMERAS} active cameras allowed.")
            return
        item = self.available_cameras_list.item(row)
        detected = item.data(Qt.ItemDataRole.UserRole)
        # make sure this is to lower for comparison against camera_identity_key
        backend = (self.backend_combo.currentData() or "opencv").lower()

        det_key = None
        if getattr(detected, "device_id", None):
            det_key = (backend, "device_id", detected.device_id)
        else:
            det_key = (backend, "index", int(detected.index))

        for i in range(self.active_cameras_list.count()):
            existing_cam = self.active_cameras_list.item(i).data(Qt.ItemDataRole.UserRole)
            if self._camera_identity_key(existing_cam) == det_key:
                QMessageBox.warning(self, "Duplicate Camera", "This camera is already in the active list.")
                return

        new_cam = CameraSettings(
            name=detected.label,
            index=detected.index,
            fps=30.0,
            backend=backend,
            exposure=0,
            gain=0.0,
            enabled=True,
            properties={},
        )
        _apply_detected_identity(new_cam, detected, backend)
        self._working_settings.cameras.append(new_cam)
        new_index = len(self._working_settings.cameras) - 1
        new_item = QListWidgetItem(self._format_camera_label(new_cam, new_index))
        new_item.setData(Qt.ItemDataRole.UserRole, new_cam)
        self.active_cameras_list.addItem(new_item)
        self.active_cameras_list.setCurrentItem(new_item)
        self._refresh_camera_labels()
        self._update_button_states()

    def _remove_selected_camera(self) -> None:
        row = self.active_cameras_list.currentRow()
        if row < 0:
            return
        self.active_cameras_list.takeItem(row)
        if row < len(self._working_settings.cameras):
            del self._working_settings.cameras[row]
        self._current_edit_index = None
        self._clear_settings_form()
        self._refresh_camera_labels()
        self._update_button_states()

    def _move_camera_up(self) -> None:
        row = self.active_cameras_list.currentRow()
        if row <= 0:
            return
        item = self.active_cameras_list.takeItem(row)
        self.active_cameras_list.insertItem(row - 1, item)
        self.active_cameras_list.setCurrentRow(row - 1)
        cams = self._working_settings.cameras
        cams[row], cams[row - 1] = cams[row - 1], cams[row]
        self._refresh_camera_labels()

    def _move_camera_down(self) -> None:
        row = self.active_cameras_list.currentRow()
        if row < 0 or row >= self.active_cameras_list.count() - 1:
            return
        item = self.active_cameras_list.takeItem(row)
        self.active_cameras_list.insertItem(row + 1, item)
        self.active_cameras_list.setCurrentRow(row + 1)
        cams = self._working_settings.cameras
        cams[row], cams[row + 1] = cams[row + 1], cams[row]
        self._refresh_camera_labels()

    def _apply_camera_settings(self) -> None:
        try:
            for sb in (self.cam_fps, self.cam_crop_x0, self.cam_crop_y0, self.cam_crop_x1, self.cam_crop_y1):
                try:
                    sb.interpretText()
                except Exception:
                    pass
            if self._current_edit_index is None:
                return
            row = self._current_edit_index
            if row < 0 or row >= len(self._working_settings.cameras):
                return

            current_model = self._working_settings.cameras[row]
            new_model = self._build_model_from_form(current_model)

            cam = self._working_settings.cameras[row]
            self._write_form_to_cam(cam)

            must_reopen = False
            if self._preview_active and self._preview_backend:
                prev_model = getattr(self._preview_backend, "settings", None)
                if prev_model:
                    must_reopen = self._needs_preview_reopen(new_model)

            if self._preview_active:
                if must_reopen:
                    self._stop_preview()
                    self._start_preview()
                else:
                    self._reconcile_fps_from_backend(new_model)
                    if not self._backend_actual_fps():
                        self._append_status("[Info] FPS will reconcile automatically during preview.")

            # Persist validated model back
            self._working_settings.cameras[row] = new_model
            self._update_active_list_item(row, new_model)

        except Exception as exc:
            LOGGER.exception("Apply camera settings failed")
            QMessageBox.warning(self, "Apply Settings Error", str(exc))

    def _update_button_states(self) -> None:
        active_row = self.active_cameras_list.currentRow()
        has_active_selection = active_row >= 0
        self.remove_camera_btn.setEnabled(has_active_selection)
        self.move_up_btn.setEnabled(has_active_selection and active_row > 0)
        self.move_down_btn.setEnabled(has_active_selection and active_row < self.active_cameras_list.count() - 1)
        # During loading, preview button becomes "Cancel Loading"
        self.preview_btn.setEnabled(has_active_selection or self._loading_active)
        available_row = self.available_cameras_list.currentRow()
        self.add_camera_btn.setEnabled(available_row >= 0)

    def _on_ok_clicked(self) -> None:
        self._stop_preview()
        active = self._working_settings.get_active_cameras()
        if self._working_settings.cameras and not active:
            QMessageBox.warning(self, "No Active Cameras", "Please enable at least one camera or remove all cameras.")
            return
        self.settings_changed.emit(copy.deepcopy(self._working_settings))
        self.accept()

    def reject(self) -> None:
        """Handle dialog rejection (Cancel or close)."""
        self._stop_preview()

        if getattr(self, "_scan_worker", None) and self._scan_worker.isRunning():
            try:
                self._scan_worker.requestInterruption()
            except Exception:
                pass
            self._scan_worker.wait(1500)
            self._scan_worker = None

        self._hide_scan_overlay()
        self.scan_progress.setVisible(False)
        self.scan_cancel_btn.setVisible(False)
        self.scan_cancel_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.backend_combo.setEnabled(True)

        super().reject()

    # -------------------------------
    # Preview start/stop (ASYNC)
    # -------------------------------
    def _toggle_preview(self) -> None:
        if self._loading_active:
            self._cancel_loading()
            return
        if self._preview_active:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self) -> None:
        """Start camera preview asynchronously (no UI freeze)."""
        if self._current_edit_index is None or self._current_edit_index < 0:
            return
        item = self.active_cameras_list.item(self._current_edit_index)
        if not item:
            return
        cam = item.data(Qt.ItemDataRole.UserRole)
        if not cam:
            return

        # Ensure any existing preview or loader is stopped/canceled
        self._stop_preview()
        # if self._loader and self._loader.isRunning():
        # self._loader.request_cancel()
        # Create worker
        self._loader = CameraLoadWorker(cam, self)
        self._loader.progress.connect(self._on_loader_progress)
        self._loader.success.connect(self._on_loader_success)
        self._loader.error.connect(self._on_loader_error)
        self._loader.canceled.connect(self._on_loader_canceled)
        self._loader.finished.connect(self._on_loader_finished)
        self._loading_active = True
        self._update_button_states()

        # Prepare UI
        self.preview_group.setVisible(True)
        self.preview_label.setText("No preview")
        self.preview_status.clear()
        self._show_loading_overlay("Loading camera…")
        self._set_preview_button_loading(True)

        self._loader.start()

    def _stop_preview(self) -> None:
        """Stop camera preview and cancel any ongoing loading."""
        # Cancel loader if running
        if self._loader and self._loader.isRunning():
            self._loader.request_cancel()
            self._loader.wait(1500)
            self._loader = None
        # Stop timer
        if self._preview_timer:
            self._preview_timer.stop()
            self._preview_timer = None
        # Close backend
        if self._preview_backend:
            try:
                self._preview_backend.close()
            except Exception:
                pass
            self._preview_backend = None
        # Reset UI
        self._loading_active = False
        self._preview_active = False
        self._set_preview_button_loading(False)
        self.preview_btn.setText("Start Preview")
        self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.preview_group.setVisible(False)
        self.preview_label.setText("No preview")
        self.preview_label.setPixmap(QPixmap())
        self._hide_loading_overlay()
        self._update_button_states()

    # -------------------------------
    # Loader UI helpers / slots
    # -------------------------------
    def _set_preview_button_loading(self, loading: bool) -> None:
        if loading:
            self.preview_btn.setText("Cancel Loading")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        else:
            self.preview_btn.setText("Start Preview")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def _show_loading_overlay(self, message: str) -> None:
        self._loading_overlay.setText(message)
        self._loading_overlay.setVisible(True)
        self._position_loading_overlay()

    def _hide_loading_overlay(self) -> None:
        self._loading_overlay.setVisible(False)

    def _append_status(self, text: str) -> None:
        self.preview_status.append(text)
        self.preview_status.moveCursor(QTextCursor.End)
        self.preview_status.ensureCursorVisible()

    def _cancel_loading(self) -> None:
        if self._loader and self._loader.isRunning():
            self._append_status("Cancel requested…")
            self._loader.request_cancel()
            # UI will flip back on finished -> _on_loader_finished
        else:
            self._loading_active = False
            self._set_preview_button_loading(False)
            self._hide_loading_overlay()
            self._update_button_states()

    # Loader signal handlers
    def _on_loader_progress(self, message: str) -> None:
        self._show_loading_overlay(message)
        self._append_status(message)

    def _on_loader_success(self, payload) -> None:
        try:
            if isinstance(payload, CameraSettings):
                cam_settings = payload
                self._append_status("Opening camera…")
                self._preview_backend = CameraFactory.create(cam_settings)
                self._preview_backend.open()

                opened_sttngs = getattr(self._preview_backend, "settings", None)
                if isinstance(opened_sttngs, CameraSettings):
                    backend = opened_sttngs.backend
                    index = opened_sttngs.index
                    device_id = (opened_sttngs.properties or {}).get(backend.lower(), {}).get("device_id", "")
                    self._append_status(f"Opened {backend}:{index} device_id={device_id}")
                    self._merge_backend_settings_back(opened_sttngs)
                    if self._current_edit_index is not None and 0 <= self._current_edit_index < len(
                        self._working_settings.cameras
                    ):
                        self._load_camera_to_form(self._working_settings.cameras[self._current_edit_index])
            else:
                raise TypeError(f"Unexpected success payload type: {type(payload)}")

            # Start preview UX
            self._append_status("Starting preview…")
            self._preview_active = True
            self.preview_btn.setText("Stop Preview")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.preview_group.setVisible(True)
            self.preview_label.setText("Starting…")
            self._hide_loading_overlay()

            # Timer @ ~25 fps default; cadence may be overridden above
            self._preview_timer = QTimer(self)
            self._preview_timer.timeout.connect(self._update_preview)
            self._preview_timer.start(40)

            # FPS reconciliation + cadence (single source of truth)
            actual_fps = self._backend_actual_fps()
            if actual_fps:
                self._adjust_preview_timer_for_fps(actual_fps)

            self.apply_settings_btn.setEnabled(True)
        except Exception as exc:
            self._on_loader_error(str(exc))

    def _on_loader_error(self, error: str) -> None:
        self._append_status(f"Error: {error}")
        LOGGER.exception("Failed to start preview")
        self._preview_active = False
        self._loading_active = False
        self._hide_loading_overlay()
        self.preview_group.setVisible(False)
        self._set_preview_button_loading(False)
        self._update_button_states()
        QMessageBox.warning(self, "Preview Error", f"Failed to start camera preview:\n{error}")

    def _on_loader_canceled(self) -> None:
        self._append_status("Loading canceled.")
        self._hide_loading_overlay()

    def _on_loader_finished(self):
        self._loading_active = False
        self._loader = None

        # If preview ended successfully, ensure Stop Preview is shown
        if self._preview_active:
            self.preview_btn.setText("Stop Preview")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        else:
            # Otherwise show Start Preview
            self.preview_btn.setText("Start Preview")
            self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        # ALWAYS refresh button states
        self._update_button_states()

    # -------------------------------
    # Preview frame update
    # -------------------------------
    def _update_preview(self) -> None:
        """Update preview frame."""
        if not self._preview_backend or not self._preview_active:
            return

        try:
            frame, _ = self._preview_backend.read()
            if frame is None or frame.size == 0:
                return

            # Apply rotation if set in the form (real-time from UI)
            rotation = self.cam_rotation.currentData()
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Apply crop if set in the form (real-time from UI)
            h, w = frame.shape[:2]
            x0 = self.cam_crop_x0.value()
            y0 = self.cam_crop_y0.value()
            x1 = self.cam_crop_x1.value() or w
            y1 = self.cam_crop_y1.value() or h
            # Clamp to frame bounds
            x0 = max(0, min(x0, w))
            y0 = max(0, min(y0, h))
            x1 = max(x0, min(x1, w))
            y1 = max(y0, min(y1, h))
            if x1 > x0 and y1 > y0:
                frame = frame[y0:y1, x0:x1]

            # Resize to fit preview label
            h, w = frame.shape[:2]
            max_w, max_h = 400, 300
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            # Convert to QImage and display
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(q_img))

        except Exception as exc:
            LOGGER.debug(f"Preview frame skipped: {exc}")
