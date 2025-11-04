"""PyQt6 based GUI for DeepLabCut Live."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from dlclivegui.camera_controller import CameraController, FrameData
from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.config import (
    DEFAULT_CONFIG,
    ApplicationSettings,
    BoundingBoxSettings,
    CameraSettings,
    DLCProcessorSettings,
    RecordingSettings,
)
from dlclivegui.dlc_processor import DLCLiveProcessor, PoseResult, ProcessorStats
from dlclivegui.processors.processor_utils import instantiate_from_scan, scan_processor_folder
from dlclivegui.video_recorder import RecorderStats, VideoRecorder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO)

PATH2MODELS = "C:\\Users\\User\\Repos\\DeepLabCut-live-GUI\\dlc_training\\dlclive"


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config: Optional[ApplicationSettings] = None):
        super().__init__()
        self.setWindowTitle("DeepLabCut Live GUI")
        self._config = config or DEFAULT_CONFIG
        self._config_path: Optional[Path] = None
        self._current_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._last_pose: Optional[PoseResult] = None
        self._dlc_active: bool = False
        self._video_recorder: Optional[VideoRecorder] = None
        self._rotation_degrees: int = 0
        self._detected_cameras: list[DetectedCamera] = []
        self._active_camera_settings: Optional[CameraSettings] = None
        self._camera_frame_times: deque[float] = deque(maxlen=240)
        self._last_drop_warning = 0.0
        self._last_recorder_summary = "Recorder idle"
        self._display_interval = 1.0 / 25.0
        self._last_display_time = 0.0
        self._dlc_initialized = False
        self._scanned_processors: dict = {}
        self._processor_keys: list = []
        self._last_processor_vid_recording = False
        self._auto_record_session_name: Optional[str] = None
        self._bbox_x0 = 0
        self._bbox_y0 = 0
        self._bbox_x1 = 0
        self._bbox_y1 = 0
        self._bbox_enabled = False

        self.camera_controller = CameraController()
        self.dlc_processor = DLCLiveProcessor()

        self._setup_ui()
        self._connect_signals()
        self._apply_config(self._config)
        self._refresh_processors()  # Scan and populate processor dropdown
        self._update_inference_buttons()
        self._update_camera_controls_enabled()
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(500)
        self._metrics_timer.timeout.connect(self._update_metrics)
        self._metrics_timer.start()
        self._update_metrics()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)

        # Video display widget
        self.video_label = QLabel("Camera preview not started")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Controls panel with fixed width to prevent shifting
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(500)
        controls_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.addWidget(self._build_camera_group())
        controls_layout.addWidget(self._build_dlc_group())
        controls_layout.addWidget(self._build_recording_group())
        controls_layout.addWidget(self._build_bbox_group())

        # Preview/Stop buttons at bottom of controls - wrap in widget
        button_bar_widget = QWidget()
        button_bar = QHBoxLayout(button_bar_widget)
        button_bar.setContentsMargins(0, 5, 0, 5)
        self.preview_button = QPushButton("Start Preview")
        self.preview_button.setMinimumWidth(150)
        self.stop_preview_button = QPushButton("Stop Preview")
        self.stop_preview_button.setEnabled(False)
        self.stop_preview_button.setMinimumWidth(150)
        button_bar.addWidget(self.preview_button)
        button_bar.addWidget(self.stop_preview_button)
        controls_layout.addWidget(button_bar_widget)
        controls_layout.addStretch(1)

        # Add controls and video to main layout
        layout.addWidget(controls_widget, stretch=0)
        layout.addWidget(self.video_label, stretch=1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._build_menus()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        load_action = QAction("Load configuration…", self)
        load_action.triggered.connect(self._action_load_config)
        file_menu.addAction(load_action)

        save_action = QAction("Save configuration", self)
        save_action.triggered.connect(self._action_save_config)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save configuration as…", self)
        save_as_action.triggered.connect(self._action_save_config_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Camera settings")
        form = QFormLayout(group)

        self.camera_backend = QComboBox()
        availability = CameraFactory.available_backends()
        for backend in CameraFactory.backend_names():
            label = backend
            if not availability.get(backend, True):
                label = f"{backend} (unavailable)"
            self.camera_backend.addItem(label, backend)
        form.addRow("Backend", self.camera_backend)

        index_layout = QHBoxLayout()
        self.camera_index = QComboBox()
        self.camera_index.setEditable(True)
        index_layout.addWidget(self.camera_index)
        self.refresh_cameras_button = QPushButton("Refresh")
        index_layout.addWidget(self.refresh_cameras_button)
        form.addRow("Camera", index_layout)

        self.camera_fps = QDoubleSpinBox()
        self.camera_fps.setRange(1.0, 240.0)
        self.camera_fps.setDecimals(2)
        form.addRow("Frame rate", self.camera_fps)

        self.camera_exposure = QSpinBox()
        self.camera_exposure.setRange(0, 1000000)
        self.camera_exposure.setValue(0)
        self.camera_exposure.setSpecialValueText("Auto")
        self.camera_exposure.setSuffix(" μs")
        form.addRow("Exposure", self.camera_exposure)

        self.camera_gain = QDoubleSpinBox()
        self.camera_gain.setRange(0.0, 100.0)
        self.camera_gain.setValue(0.0)
        self.camera_gain.setSpecialValueText("Auto")
        self.camera_gain.setDecimals(2)
        form.addRow("Gain", self.camera_gain)

        # Crop settings
        crop_layout = QHBoxLayout()
        self.crop_x0 = QSpinBox()
        self.crop_x0.setRange(0, 7680)
        self.crop_x0.setPrefix("x0:")
        self.crop_x0.setSpecialValueText("x0:None")
        crop_layout.addWidget(self.crop_x0)

        self.crop_y0 = QSpinBox()
        self.crop_y0.setRange(0, 4320)
        self.crop_y0.setPrefix("y0:")
        self.crop_y0.setSpecialValueText("y0:None")
        crop_layout.addWidget(self.crop_y0)

        self.crop_x1 = QSpinBox()
        self.crop_x1.setRange(0, 7680)
        self.crop_x1.setPrefix("x1:")
        self.crop_x1.setSpecialValueText("x1:None")
        crop_layout.addWidget(self.crop_x1)

        self.crop_y1 = QSpinBox()
        self.crop_y1.setRange(0, 4320)
        self.crop_y1.setPrefix("y1:")
        self.crop_y1.setSpecialValueText("y1:None")
        crop_layout.addWidget(self.crop_y1)

        form.addRow("Crop (x0,y0,x1,y1)", crop_layout)

        self.rotation_combo = QComboBox()
        self.rotation_combo.addItem("0° (default)", 0)
        self.rotation_combo.addItem("90°", 90)
        self.rotation_combo.addItem("180°", 180)
        self.rotation_combo.addItem("270°", 270)
        form.addRow("Rotation", self.rotation_combo)

        self.camera_stats_label = QLabel("Camera idle")
        form.addRow("Throughput", self.camera_stats_label)

        return group

    def _build_dlc_group(self) -> QGroupBox:
        group = QGroupBox("DLCLive settings")
        form = QFormLayout(group)

        path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("/path/to/exported/model")
        path_layout.addWidget(self.model_path_edit)
        self.browse_model_button = QPushButton("Browse…")
        self.browse_model_button.clicked.connect(self._action_browse_model)
        path_layout.addWidget(self.browse_model_button)
        form.addRow("Model directory", path_layout)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("Base (TensorFlow)", "base")
        self.model_type_combo.addItem("PyTorch", "pytorch")
        self.model_type_combo.setCurrentIndex(1)  # Default to PyTorch
        form.addRow("Model type", self.model_type_combo)

        # Processor selection
        processor_path_layout = QHBoxLayout()
        self.processor_folder_edit = QLineEdit()
        self.processor_folder_edit.setText(str(Path(__file__).parent.joinpath("processors")))
        processor_path_layout.addWidget(self.processor_folder_edit)

        self.browse_processor_folder_button = QPushButton("Browse...")
        self.browse_processor_folder_button.clicked.connect(self._action_browse_processor_folder)
        processor_path_layout.addWidget(self.browse_processor_folder_button)

        self.refresh_processors_button = QPushButton("Refresh")
        self.refresh_processors_button.clicked.connect(self._refresh_processors)
        processor_path_layout.addWidget(self.refresh_processors_button)
        form.addRow("Processor folder", processor_path_layout)

        self.processor_combo = QComboBox()
        self.processor_combo.addItem("No Processor", None)
        form.addRow("Processor", self.processor_combo)

        self.additional_options_edit = QPlainTextEdit()
        self.additional_options_edit.setPlaceholderText("")
        self.additional_options_edit.setFixedHeight(40)
        form.addRow("Additional options", self.additional_options_edit)

        # Wrap inference buttons in a widget to prevent shifting
        inference_button_widget = QWidget()
        inference_buttons = QHBoxLayout(inference_button_widget)
        inference_buttons.setContentsMargins(0, 0, 0, 0)
        self.start_inference_button = QPushButton("Start pose inference")
        self.start_inference_button.setEnabled(False)
        self.start_inference_button.setMinimumWidth(150)
        inference_buttons.addWidget(self.start_inference_button)
        self.stop_inference_button = QPushButton("Stop pose inference")
        self.stop_inference_button.setEnabled(False)
        self.stop_inference_button.setMinimumWidth(150)
        inference_buttons.addWidget(self.stop_inference_button)
        form.addRow(inference_button_widget)

        self.show_predictions_checkbox = QCheckBox("Display pose predictions")
        self.show_predictions_checkbox.setChecked(True)
        form.addRow(self.show_predictions_checkbox)

        self.auto_record_checkbox = QCheckBox("Auto-record video on processor command")
        self.auto_record_checkbox.setChecked(False)
        self.auto_record_checkbox.setToolTip(
            "Automatically start/stop video recording when processor receives video recording commands"
        )
        form.addRow(self.auto_record_checkbox)

        self.processor_status_label = QLabel("Processor: No clients | Recording: No")
        self.processor_status_label.setWordWrap(True)
        form.addRow("Processor Status", self.processor_status_label)

        self.dlc_stats_label = QLabel("DLC processor idle")
        self.dlc_stats_label.setWordWrap(True)
        form.addRow("Performance", self.dlc_stats_label)

        return group

    def _build_recording_group(self) -> QGroupBox:
        group = QGroupBox("Recording")
        form = QFormLayout(group)

        dir_layout = QHBoxLayout()
        self.output_directory_edit = QLineEdit()
        dir_layout.addWidget(self.output_directory_edit)
        browse_dir = QPushButton("Browse…")
        browse_dir.clicked.connect(self._action_browse_directory)
        dir_layout.addWidget(browse_dir)
        form.addRow("Output directory", dir_layout)

        self.filename_edit = QLineEdit()
        form.addRow("Filename", self.filename_edit)

        self.container_combo = QComboBox()
        self.container_combo.setEditable(True)
        self.container_combo.addItems(["mp4", "avi", "mov"])
        form.addRow("Container", self.container_combo)

        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["h264_nvenc", "libx264"])
        self.codec_combo.setCurrentText("h264_nvenc")
        form.addRow("Codec", self.codec_combo)

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(23)
        form.addRow("CRF", self.crf_spin)

        # Wrap recording buttons in a widget to prevent shifting
        recording_button_widget = QWidget()
        buttons = QHBoxLayout(recording_button_widget)
        buttons.setContentsMargins(0, 0, 0, 0)
        self.start_record_button = QPushButton("Start recording")
        self.start_record_button.setMinimumWidth(150)
        buttons.addWidget(self.start_record_button)
        self.stop_record_button = QPushButton("Stop recording")
        self.stop_record_button.setEnabled(False)
        self.stop_record_button.setMinimumWidth(150)
        buttons.addWidget(self.stop_record_button)
        form.addRow(recording_button_widget)

        self.recording_stats_label = QLabel(self._last_recorder_summary)
        self.recording_stats_label.setWordWrap(True)
        form.addRow("Performance", self.recording_stats_label)

        return group

    def _build_bbox_group(self) -> QGroupBox:
        """Build bounding box visualization controls."""
        group = QGroupBox("Bounding Box Visualization")
        form = QFormLayout(group)

        self.bbox_enabled_checkbox = QCheckBox("Show bounding box")
        self.bbox_enabled_checkbox.setChecked(False)
        form.addRow(self.bbox_enabled_checkbox)

        bbox_layout = QHBoxLayout()

        self.bbox_x0_spin = QSpinBox()
        self.bbox_x0_spin.setRange(0, 7680)
        self.bbox_x0_spin.setPrefix("x0:")
        self.bbox_x0_spin.setValue(0)
        bbox_layout.addWidget(self.bbox_x0_spin)

        self.bbox_y0_spin = QSpinBox()
        self.bbox_y0_spin.setRange(0, 4320)
        self.bbox_y0_spin.setPrefix("y0:")
        self.bbox_y0_spin.setValue(0)
        bbox_layout.addWidget(self.bbox_y0_spin)

        self.bbox_x1_spin = QSpinBox()
        self.bbox_x1_spin.setRange(0, 7680)
        self.bbox_x1_spin.setPrefix("x1:")
        self.bbox_x1_spin.setValue(100)
        bbox_layout.addWidget(self.bbox_x1_spin)

        self.bbox_y1_spin = QSpinBox()
        self.bbox_y1_spin.setRange(0, 4320)
        self.bbox_y1_spin.setPrefix("y1:")
        self.bbox_y1_spin.setValue(100)
        bbox_layout.addWidget(self.bbox_y1_spin)

        form.addRow("Coordinates", bbox_layout)

        return group

    # ------------------------------------------------------------------ signals
    def _connect_signals(self) -> None:
        self.preview_button.clicked.connect(self._start_preview)
        self.stop_preview_button.clicked.connect(self._stop_preview)
        self.start_record_button.clicked.connect(self._start_recording)
        self.stop_record_button.clicked.connect(self._stop_recording)
        self.refresh_cameras_button.clicked.connect(
            lambda: self._refresh_camera_indices(keep_current=True)
        )
        self.camera_backend.currentIndexChanged.connect(self._on_backend_changed)
        self.camera_backend.currentIndexChanged.connect(self._update_backend_specific_controls)
        self.rotation_combo.currentIndexChanged.connect(self._on_rotation_changed)
        self.start_inference_button.clicked.connect(self._start_inference)
        self.stop_inference_button.clicked.connect(lambda: self._stop_inference())
        self.show_predictions_checkbox.stateChanged.connect(self._on_show_predictions_changed)

        # Connect bounding box controls
        self.bbox_enabled_checkbox.stateChanged.connect(self._on_bbox_changed)
        self.bbox_x0_spin.valueChanged.connect(self._on_bbox_changed)
        self.bbox_y0_spin.valueChanged.connect(self._on_bbox_changed)
        self.bbox_x1_spin.valueChanged.connect(self._on_bbox_changed)
        self.bbox_y1_spin.valueChanged.connect(self._on_bbox_changed)

        self.camera_controller.frame_ready.connect(self._on_frame_ready)
        self.camera_controller.started.connect(self._on_camera_started)
        self.camera_controller.error.connect(self._show_error)
        self.camera_controller.warning.connect(self._show_warning)
        self.camera_controller.stopped.connect(self._on_camera_stopped)

        self.dlc_processor.pose_ready.connect(self._on_pose_ready)
        self.dlc_processor.error.connect(self._on_dlc_error)
        self.dlc_processor.initialized.connect(self._on_dlc_initialised)

    # ------------------------------------------------------------------ config
    def _apply_config(self, config: ApplicationSettings) -> None:
        camera = config.camera
        self.camera_fps.setValue(float(camera.fps))

        # Set exposure and gain from config
        self.camera_exposure.setValue(int(camera.exposure))
        self.camera_gain.setValue(float(camera.gain))

        # Set crop settings from config
        self.crop_x0.setValue(int(camera.crop_x0) if hasattr(camera, "crop_x0") else 0)
        self.crop_y0.setValue(int(camera.crop_y0) if hasattr(camera, "crop_y0") else 0)
        self.crop_x1.setValue(int(camera.crop_x1) if hasattr(camera, "crop_x1") else 0)
        self.crop_y1.setValue(int(camera.crop_y1) if hasattr(camera, "crop_y1") else 0)

        backend_name = camera.backend or "opencv"
        self.camera_backend.blockSignals(True)
        index = self.camera_backend.findData(backend_name)
        if index >= 0:
            self.camera_backend.setCurrentIndex(index)
        else:
            self.camera_backend.setEditText(backend_name)
        self.camera_backend.blockSignals(False)
        self._refresh_camera_indices(keep_current=False)
        self._select_camera_by_index(camera.index, fallback_text=camera.name or str(camera.index))

        self._active_camera_settings = None
        self._update_backend_specific_controls()

        dlc = config.dlc
        self.model_path_edit.setText(dlc.model_path)

        # Set model type
        model_type = dlc.model_type or "base"
        model_type_index = self.model_type_combo.findData(model_type)
        if model_type_index >= 0:
            self.model_type_combo.setCurrentIndex(model_type_index)

        self.additional_options_edit.setPlainText(json.dumps(dlc.additional_options, indent=2))

        recording = config.recording
        self.output_directory_edit.setText(recording.directory)
        self.filename_edit.setText(recording.filename)
        self.container_combo.setCurrentText(recording.container)
        codec_index = self.codec_combo.findText(recording.codec)
        if codec_index >= 0:
            self.codec_combo.setCurrentIndex(codec_index)
        else:
            self.codec_combo.addItem(recording.codec)
            self.codec_combo.setCurrentIndex(self.codec_combo.count() - 1)
        self.crf_spin.setValue(int(recording.crf))

        # Set bounding box settings from config
        bbox = config.bbox
        self.bbox_enabled_checkbox.setChecked(bbox.enabled)
        self.bbox_x0_spin.setValue(bbox.x0)
        self.bbox_y0_spin.setValue(bbox.y0)
        self.bbox_x1_spin.setValue(bbox.x1)
        self.bbox_y1_spin.setValue(bbox.y1)

    def _current_config(self) -> ApplicationSettings:
        return ApplicationSettings(
            camera=self._camera_settings_from_ui(),
            dlc=self._dlc_settings_from_ui(),
            recording=self._recording_settings_from_ui(),
            bbox=self._bbox_settings_from_ui(),
        )

    def _camera_settings_from_ui(self) -> CameraSettings:
        index = self._current_camera_index_value()
        if index is None:
            raise ValueError("Camera selection must provide a numeric index")
        backend_text = self._current_backend_name()

        # Get exposure and gain from explicit UI fields
        exposure = self.camera_exposure.value()
        gain = self.camera_gain.value()

        # Get crop settings from UI
        crop_x0 = self.crop_x0.value()
        crop_y0 = self.crop_y0.value()
        crop_x1 = self.crop_x1.value()
        crop_y1 = self.crop_y1.value()

        name_text = self.camera_index.currentText().strip()
        settings = CameraSettings(
            name=name_text or f"Camera {index}",
            index=index,
            fps=self.camera_fps.value(),
            backend=backend_text or "opencv",
            exposure=exposure,
            gain=gain,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            crop_x1=crop_x1,
            crop_y1=crop_y1,
            properties={},
        )
        return settings.apply_defaults()

    def _current_backend_name(self) -> str:
        backend_data = self.camera_backend.currentData()
        if isinstance(backend_data, str) and backend_data:
            return backend_data
        text = self.camera_backend.currentText().strip()
        return text or "opencv"

    def _refresh_camera_indices(self, *_args: object, keep_current: bool = True) -> None:
        backend = self._current_backend_name()
        # Get max_devices from config, default to 3
        max_devices = (
            self._config.camera.max_devices if hasattr(self._config.camera, "max_devices") else 3
        )
        detected = CameraFactory.detect_cameras(backend, max_devices=max_devices)
        debug_info = [f"{camera.index}:{camera.label}" for camera in detected]
        logging.info(f"[CameraDetection] Available cameras for backend '{backend}': {debug_info}")
        self._detected_cameras = detected
        previous_index = self._current_camera_index_value()
        previous_text = self.camera_index.currentText()
        self.camera_index.blockSignals(True)
        self.camera_index.clear()
        for camera in detected:
            self.camera_index.addItem(camera.label, camera.index)
        if keep_current and previous_index is not None:
            self._select_camera_by_index(previous_index, fallback_text=previous_text)
        elif detected:
            self.camera_index.setCurrentIndex(0)
        else:
            if keep_current and previous_text:
                self.camera_index.setEditText(previous_text)
            else:
                self.camera_index.setEditText("")
        self.camera_index.blockSignals(False)

    def _select_camera_by_index(self, index: int, fallback_text: Optional[str] = None) -> None:
        self.camera_index.blockSignals(True)
        for row in range(self.camera_index.count()):
            if self.camera_index.itemData(row) == index:
                self.camera_index.setCurrentIndex(row)
                break
        else:
            text = fallback_text if fallback_text is not None else str(index)
            self.camera_index.setEditText(text)
        self.camera_index.blockSignals(False)

    def _current_camera_index_value(self) -> Optional[int]:
        data = self.camera_index.currentData()
        if isinstance(data, int):
            return data
        text = self.camera_index.currentText().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
        debug_info = [f"{camera.index}:{camera.label}" for camera in detected]
        logging.info(f"[CameraDetection] Available cameras for backend '{backend}': {debug_info}")
        self._detected_cameras = detected
        previous_index = self._current_camera_index_value()
        previous_text = self.camera_index.currentText()
        self.camera_index.blockSignals(True)
        self.camera_index.clear()
        for camera in detected:
            self.camera_index.addItem(camera.label, camera.index)
        if keep_current and previous_index is not None:
            self._select_camera_by_index(previous_index, fallback_text=previous_text)
        elif detected:
            self.camera_index.setCurrentIndex(0)
        else:
            if keep_current and previous_text:
                self.camera_index.setEditText(previous_text)
            else:
                self.camera_index.setEditText("")
        self.camera_index.blockSignals(False)

    def _select_camera_by_index(self, index: int, fallback_text: Optional[str] = None) -> None:
        self.camera_index.blockSignals(True)
        for row in range(self.camera_index.count()):
            if self.camera_index.itemData(row) == index:
                self.camera_index.setCurrentIndex(row)
                break
        else:
            text = fallback_text if fallback_text is not None else str(index)
            self.camera_index.setEditText(text)
        self.camera_index.blockSignals(False)

    def _current_camera_index_value(self) -> Optional[int]:
        data = self.camera_index.currentData()
        if isinstance(data, int):
            return data
        text = self.camera_index.currentText().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _parse_json(self, value: str) -> dict:
        text = value.strip()
        if not text:
            return {}
        return json.loads(text)

    def _dlc_settings_from_ui(self) -> DLCProcessorSettings:
        model_type = self.model_type_combo.currentData()
        if not isinstance(model_type, str):
            model_type = "base"

        return DLCProcessorSettings(
            model_path=self.model_path_edit.text().strip(),
            model_type=model_type,
            additional_options=self._parse_json(self.additional_options_edit.toPlainText()),
        )

    def _recording_settings_from_ui(self) -> RecordingSettings:
        return RecordingSettings(
            enabled=True,  # Always enabled - recording controlled by button
            directory=self.output_directory_edit.text().strip(),
            filename=self.filename_edit.text().strip() or "session.mp4",
            container=self.container_combo.currentText().strip() or "mp4",
            codec=self.codec_combo.currentText().strip() or "libx264",
            crf=int(self.crf_spin.value()),
        )

    def _bbox_settings_from_ui(self) -> BoundingBoxSettings:
        return BoundingBoxSettings(
            enabled=self.bbox_enabled_checkbox.isChecked(),
            x0=self.bbox_x0_spin.value(),
            y0=self.bbox_y0_spin.value(),
            x1=self.bbox_x1_spin.value(),
            y1=self.bbox_y1_spin.value(),
        )

    # ------------------------------------------------------------------ actions
    def _action_load_config(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load configuration", str(Path.home()), "JSON files (*.json)"
        )
        if not file_name:
            return
        try:
            config = ApplicationSettings.load(file_name)
        except Exception as exc:  # pragma: no cover - GUI interaction
            self._show_error(str(exc))
            return
        self._config = config
        self._config_path = Path(file_name)
        self._apply_config(config)
        self.statusBar().showMessage(f"Loaded configuration: {file_name}", 5000)

    def _action_save_config(self) -> None:
        if self._config_path is None:
            self._action_save_config_as()
            return
        self._save_config_to_path(self._config_path)

    def _action_save_config_as(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", str(Path.home()), "JSON files (*.json)"
        )
        if not file_name:
            return
        path = Path(file_name)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        self._config_path = path
        self._save_config_to_path(path)

    def _save_config_to_path(self, path: Path) -> None:
        try:
            config = self._current_config()
            config.save(path)
        except Exception as exc:  # pragma: no cover - GUI interaction
            self._show_error(str(exc))
            return
        self.statusBar().showMessage(f"Saved configuration to {path}", 5000)

    def _action_browse_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DLCLive model file", PATH2MODELS, "Model files (*.pt *.pb);;All files (*.*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def _action_browse_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select output directory", str(Path.home())
        )
        if directory:
            self.output_directory_edit.setText(directory)

    def _action_browse_processor_folder(self) -> None:
        """Browse for processor folder."""
        current_path = self.processor_folder_edit.text() or "./processors"
        directory = QFileDialog.getExistingDirectory(self, "Select processor folder", current_path)
        if directory:
            self.processor_folder_edit.setText(directory)
            self._refresh_processors()

    def _refresh_processors(self) -> None:
        """Scan processor folder and populate dropdown."""
        folder_path = self.processor_folder_edit.text() or "./processors"

        # Clear existing items (keep "No Processor")
        self.processor_combo.clear()
        self.processor_combo.addItem("No Processor", None)

        # Scan folder
        try:
            self._scanned_processors = scan_processor_folder(folder_path)
            self._processor_keys = list(self._scanned_processors.keys())

            # Populate dropdown
            for key in self._processor_keys:
                info = self._scanned_processors[key]
                display_name = f"{info['name']} ({info['file']})"
                self.processor_combo.addItem(display_name, key)

            status_msg = f"Found {len(self._processor_keys)} processor(s) in {folder_path}"
            self.statusBar().showMessage(status_msg, 3000)

        except Exception as e:
            error_msg = f"Error scanning processors: {e}"
            self.statusBar().showMessage(error_msg, 5000)
            logging.error(error_msg)
            self._scanned_processors = {}
            self._processor_keys = []

    def _on_backend_changed(self, *_args: object) -> None:
        self._refresh_camera_indices(keep_current=False)

    def _update_backend_specific_controls(self) -> None:
        """Enable/disable controls based on selected backend."""
        backend = self._current_backend_name()
        is_opencv = backend.lower() == "opencv"

        # Disable exposure and gain controls for OpenCV backend
        self.camera_exposure.setEnabled(not is_opencv)
        self.camera_gain.setEnabled(not is_opencv)

        # Set tooltip to explain why controls are disabled
        if is_opencv:
            tooltip = "Exposure and gain control not supported with OpenCV backend"
            self.camera_exposure.setToolTip(tooltip)
            self.camera_gain.setToolTip(tooltip)
        else:
            self.camera_exposure.setToolTip("")
            self.camera_gain.setToolTip("")

    def _on_rotation_changed(self, _index: int) -> None:
        data = self.rotation_combo.currentData()
        self._rotation_degrees = int(data) if isinstance(data, int) else 0
        if self._raw_frame is not None:
            rotated = self._apply_rotation(self._raw_frame)
            self._current_frame = rotated
            self._last_pose = None
            self._display_frame(rotated, force=False)

    # ------------------------------------------------------------------ camera control
    def _start_preview(self) -> None:
        try:
            settings = self._camera_settings_from_ui()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._active_camera_settings = settings
        self.camera_controller.start(settings)
        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(True)
        self._current_frame = None
        self._raw_frame = None
        self._last_pose = None
        self._dlc_active = False
        self._camera_frame_times.clear()
        self._last_display_time = 0.0
        if hasattr(self, "camera_stats_label"):
            self.camera_stats_label.setText("Camera starting…")
        self.statusBar().showMessage("Starting camera preview…", 3000)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    def _stop_preview(self) -> None:
        if not self.camera_controller.is_running():
            return
        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)
        self.statusBar().showMessage("Stopping camera preview…", 3000)
        self.camera_controller.stop()
        self._stop_inference(show_message=False)
        self._camera_frame_times.clear()
        self._last_display_time = 0.0
        if hasattr(self, "camera_stats_label"):
            self.camera_stats_label.setText("Camera idle")

    def _on_camera_started(self, settings: CameraSettings) -> None:
        self._active_camera_settings = settings
        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(True)
        if getattr(settings, "fps", None):
            self.camera_fps.blockSignals(True)
            self.camera_fps.setValue(float(settings.fps))
            self.camera_fps.blockSignals(False)
        # Resolution will be determined from actual camera frames
        if getattr(settings, "fps", None):
            fps_text = f"{float(settings.fps):.2f} FPS"
        else:
            fps_text = "unknown FPS"
        self.statusBar().showMessage(f"Camera preview started @ {fps_text}", 5000)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    def _on_camera_stopped(self) -> None:
        if self._video_recorder and self._video_recorder.is_running:
            self._stop_recording()
        self.preview_button.setEnabled(True)
        self.stop_preview_button.setEnabled(False)
        self._stop_inference(show_message=False)
        self._current_frame = None
        self._raw_frame = None
        self._last_pose = None
        self._active_camera_settings = None
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Camera preview not started")
        self.statusBar().showMessage("Camera preview stopped", 3000)
        self._camera_frame_times.clear()
        self._last_display_time = 0.0
        if hasattr(self, "camera_stats_label"):
            self.camera_stats_label.setText("Camera idle")
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    def _configure_dlc(self) -> bool:
        try:
            settings = self._dlc_settings_from_ui()
        except (ValueError, json.JSONDecodeError) as exc:
            self._show_error(f"Invalid DLCLive settings: {exc}")
            return False
        if not settings.model_path:
            self._show_error("Please select a DLCLive model before starting inference.")
            return False

        # Instantiate processor if selected
        processor = None
        selected_key = self.processor_combo.currentData()
        if selected_key is not None and self._scanned_processors:
            try:
                # For now, instantiate with no parameters
                # TODO: Add parameter dialog for processors that need params
                # or pass kwargs from config ?
                processor = instantiate_from_scan(self._scanned_processors, selected_key)
                processor_name = self._scanned_processors[selected_key]["name"]
                self.statusBar().showMessage(f"Loaded processor: {processor_name}", 3000)
            except Exception as e:
                error_msg = f"Failed to instantiate processor: {e}"
                self._show_error(error_msg)
                logging.error(error_msg)
                return False

        self.dlc_processor.configure(settings, processor=processor)
        return True

    def _update_inference_buttons(self) -> None:
        preview_running = self.camera_controller.is_running()
        self.start_inference_button.setEnabled(preview_running and not self._dlc_active)
        self.stop_inference_button.setEnabled(preview_running and self._dlc_active)

    def _update_dlc_controls_enabled(self) -> None:
        """Enable/disable DLC settings based on inference state."""
        allow_changes = not self._dlc_active
        widgets = [
            self.model_path_edit,
            self.browse_model_button,
            self.model_type_combo,
            self.processor_folder_edit,
            self.browse_processor_folder_button,
            self.refresh_processors_button,
            self.processor_combo,
            self.additional_options_edit,
        ]
        for widget in widgets:
            widget.setEnabled(allow_changes)

    def _update_camera_controls_enabled(self) -> None:
        recording_active = self._video_recorder is not None and self._video_recorder.is_running
        allow_changes = (
            not self.camera_controller.is_running()
            and not self._dlc_active
            and not recording_active
        )
        widgets = [
            self.camera_backend,
            self.camera_index,
            self.refresh_cameras_button,
            self.camera_fps,
            self.camera_exposure,
            self.camera_gain,
            self.crop_x0,
            self.crop_y0,
            self.crop_x1,
            self.crop_y1,
            self.rotation_combo,
            self.codec_combo,
            self.crf_spin,
        ]
        for widget in widgets:
            widget.setEnabled(allow_changes)

    def _track_camera_frame(self) -> None:
        now = time.perf_counter()
        self._camera_frame_times.append(now)
        window_seconds = 5.0
        while self._camera_frame_times and now - self._camera_frame_times[0] > window_seconds:
            self._camera_frame_times.popleft()

    def _display_frame(self, frame: np.ndarray, *, force: bool = False) -> None:
        if frame is None:
            return
        now = time.perf_counter()
        if not force and (now - self._last_display_time) < self._display_interval:
            return
        self._last_display_time = now
        self._update_video_display(frame)

    def _compute_fps(self, times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        if duration <= 0:
            return 0.0
        return (len(times) - 1) / duration

    def _format_recorder_stats(self, stats: RecorderStats) -> str:
        latency_ms = stats.last_latency * 1000.0
        avg_ms = stats.average_latency * 1000.0
        buffer_ms = stats.buffer_seconds * 1000.0
        write_fps = stats.write_fps
        enqueue = stats.frames_enqueued
        written = stats.frames_written
        dropped = stats.dropped_frames
        return (
            f"{written}/{enqueue} frames | write {write_fps:.1f} fps | "
            f"latency {latency_ms:.1f} ms (avg {avg_ms:.1f} ms) | "
            f"queue {stats.queue_size} (~{buffer_ms:.0f} ms) | dropped {dropped}"
        )

    def _format_dlc_stats(self, stats: ProcessorStats) -> str:
        """Format DLC processor statistics for display."""
        latency_ms = stats.last_latency * 1000.0
        avg_ms = stats.average_latency * 1000.0
        processing_fps = stats.processing_fps
        enqueue = stats.frames_enqueued
        processed = stats.frames_processed
        dropped = stats.frames_dropped
        return (
            f"{processed}/{enqueue} frames | inference {processing_fps:.1f} fps | "
            f"latency {latency_ms:.1f} ms (avg {avg_ms:.1f} ms) | "
            f"queue {stats.queue_size} | dropped {dropped}"
        )

    def _update_metrics(self) -> None:
        if hasattr(self, "camera_stats_label"):
            if self.camera_controller.is_running():
                fps = self._compute_fps(self._camera_frame_times)
                if fps > 0:
                    self.camera_stats_label.setText(f"{fps:.1f} fps (last 5 s)")
                else:
                    self.camera_stats_label.setText("Measuring…")
            else:
                self.camera_stats_label.setText("Camera idle")

        if hasattr(self, "dlc_stats_label"):
            if self._dlc_active and self._dlc_initialized:
                stats = self.dlc_processor.get_stats()
                summary = self._format_dlc_stats(stats)
                self.dlc_stats_label.setText(summary)
            else:
                self.dlc_stats_label.setText("DLC processor idle")

        # Update processor status (connection and recording state)
        if hasattr(self, "processor_status_label"):
            self._update_processor_status()

        if hasattr(self, "recording_stats_label"):
            if self._video_recorder is not None:
                stats = self._video_recorder.get_stats()
                if stats is not None:
                    summary = self._format_recorder_stats(stats)
                    self._last_recorder_summary = summary
                    self.recording_stats_label.setText(summary)
                elif not self._video_recorder.is_running:
                    self._last_recorder_summary = "Recorder idle"
                    self.recording_stats_label.setText(self._last_recorder_summary)
            else:
                self.recording_stats_label.setText(self._last_recorder_summary)

    def _update_processor_status(self) -> None:
        """Update processor connection and recording status, handle auto-recording."""
        if not self._dlc_active or not self._dlc_initialized:
            self.processor_status_label.setText("Processor: Not active")
            return

        # Get processor instance from dlc_processor
        processor = self.dlc_processor._processor

        if processor is None:
            self.processor_status_label.setText("Processor: None loaded")
            return

        # Check if processor has the required attributes (socket-based processors)
        if not hasattr(processor, "conns") or not hasattr(processor, "_recording"):
            self.processor_status_label.setText("Processor: No status info")
            return

        # Get connection count and recording state
        num_clients = len(processor.conns)
        is_recording = processor.recording if hasattr(processor, "recording") else False

        # Format status message
        client_str = f"{num_clients} client{'s' if num_clients != 1 else ''}"
        recording_str = "Yes" if is_recording else "No"
        self.processor_status_label.setText(f"Clients: {client_str} | Recording: {recording_str}")

        # Handle auto-recording based on processor's video recording flag
        if hasattr(processor, "_vid_recording") and self.auto_record_checkbox.isChecked():
            current_vid_recording = processor.video_recording

            # Check if video recording state changed
            if current_vid_recording != self._last_processor_vid_recording:
                if current_vid_recording:
                    # Start video recording
                    if not self._video_recorder or not self._video_recorder.is_running:
                        # Get session name from processor
                        session_name = getattr(processor, "session_name", "auto_session")
                        self._auto_record_session_name = session_name

                        # Update filename with session name
                        original_filename = self.filename_edit.text()
                        self.filename_edit.setText(f"{session_name}.mp4")

                        self._start_recording()
                        self.statusBar().showMessage(
                            f"Auto-started recording: {session_name}", 3000
                        )
                        logging.info(f"Auto-recording started for session: {session_name}")
                else:
                    # Stop video recording
                    if self._video_recorder and self._video_recorder.is_running:
                        self._stop_recording()
                        self.statusBar().showMessage("Auto-stopped recording", 3000)
                        logging.info("Auto-recording stopped")

                self._last_processor_vid_recording = current_vid_recording

    def _start_inference(self) -> None:
        if self._dlc_active:
            self.statusBar().showMessage("Pose inference already running", 3000)
            return
        if not self.camera_controller.is_running():
            self._show_error("Start the camera preview before running pose inference.")
            return
        if not self._configure_dlc():
            self._update_inference_buttons()
            return
        self.dlc_processor.reset()
        self._last_pose = None
        self._dlc_active = True
        self._dlc_initialized = False

        # Update button to show initializing state
        self.start_inference_button.setText("Initializing DLCLive!")
        self.start_inference_button.setStyleSheet("background-color: #4A90E2; color: white;")
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(True)

        self.statusBar().showMessage("Initializing DLCLive…", 3000)
        self._update_camera_controls_enabled()
        self._update_dlc_controls_enabled()

    def _stop_inference(self, show_message: bool = True) -> None:
        was_active = self._dlc_active
        self._dlc_active = False
        self._dlc_initialized = False
        self.dlc_processor.reset()
        self._last_pose = None
        self._last_processor_vid_recording = False
        self._auto_record_session_name = None

        # Reset button appearance
        self.start_inference_button.setText("Start pose inference")
        self.start_inference_button.setStyleSheet("")

        if self._current_frame is not None:
            self._display_frame(self._current_frame, force=True)
        if was_active and show_message:
            self.statusBar().showMessage("Pose inference stopped", 3000)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()
        self._update_dlc_controls_enabled()

    # ------------------------------------------------------------------ recording
    def _start_recording(self) -> None:
        if self._video_recorder and self._video_recorder.is_running:
            return
        if not self.camera_controller.is_running():
            self._show_error("Start the camera preview before recording.")
            return
        if self._current_frame is None:
            self._show_error("Wait for the first preview frame before recording.")
            return
        recording = self._recording_settings_from_ui()
        if not recording.enabled:
            self._show_error("Recording is disabled in the configuration.")
            return
        frame = self._current_frame
        assert frame is not None
        height, width = frame.shape[:2]
        frame_rate = (
            self._active_camera_settings.fps
            if self._active_camera_settings is not None
            else self.camera_fps.value()
        )
        output_path = recording.output_path()
        self._video_recorder = VideoRecorder(
            output_path,
            frame_size=(height, width),  # Use numpy convention: (height, width)
            frame_rate=float(frame_rate),
            codec=recording.codec,
            crf=recording.crf,
        )
        self._last_drop_warning = 0.0
        try:
            self._video_recorder.start()
        except Exception as exc:  # pragma: no cover - runtime error
            self._show_error(str(exc))
            self._video_recorder = None
            return
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        if hasattr(self, "recording_stats_label"):
            self._last_recorder_summary = "Recorder running…"
            self.recording_stats_label.setText(self._last_recorder_summary)
        self.statusBar().showMessage(f"Recording to {output_path}", 5000)
        self._update_camera_controls_enabled()

    def _stop_recording(self) -> None:
        if not self._video_recorder:
            return
        recorder = self._video_recorder
        recorder.stop()
        stats = recorder.get_stats() if recorder is not None else None
        self._video_recorder = None
        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        if hasattr(self, "recording_stats_label"):
            if stats is not None:
                summary = self._format_recorder_stats(stats)
            else:
                summary = "Recorder idle"
            self._last_recorder_summary = summary
            self.recording_stats_label.setText(summary)
        else:
            self._last_recorder_summary = (
                self._format_recorder_stats(stats) if stats is not None else "Recorder idle"
            )
        self._last_drop_warning = 0.0
        self.statusBar().showMessage("Recording stopped", 3000)
        self._update_camera_controls_enabled()

    # ------------------------------------------------------------------ frame handling
    def _on_frame_ready(self, frame_data: FrameData) -> None:
        raw_frame = frame_data.image
        self._raw_frame = raw_frame

        # Apply cropping before rotation
        frame = self._apply_crop(raw_frame)

        # Apply rotation
        frame = self._apply_rotation(frame)
        frame = np.ascontiguousarray(frame)
        self._current_frame = frame
        self._track_camera_frame()
        if self._video_recorder and self._video_recorder.is_running:
            try:
                success = self._video_recorder.write(frame, timestamp=frame_data.timestamp)
                if not success:
                    now = time.perf_counter()
                    if now - self._last_drop_warning > 1.0:
                        self.statusBar().showMessage("Recorder backlog full; dropping frames", 2000)
                        self._last_drop_warning = now
            except RuntimeError as exc:
                # Check if it's a frame size error
                if "Frame size changed" in str(exc):
                    self._show_warning(f"Camera resolution changed - restarting recording: {exc}")
                    was_recording = self._video_recorder and self._video_recorder.is_running
                    self._stop_recording()
                    # Restart recording with new resolution if it was already running
                    if was_recording:
                        try:
                            self._start_recording()
                        except Exception as restart_exc:
                            self._show_error(f"Failed to restart recording: {restart_exc}")
                else:
                    self._show_error(str(exc))
                    self._stop_recording()
        if self._dlc_active:
            self.dlc_processor.enqueue_frame(frame, frame_data.timestamp)
        self._display_frame(frame)

    def _on_pose_ready(self, result: PoseResult) -> None:
        if not self._dlc_active:
            return
        self._last_pose = result
        # logging.info(f"Pose result: {result.pose}, Timestamp: {result.timestamp}")
        if self._current_frame is not None:
            self._display_frame(self._current_frame, force=True)

    def _on_dlc_error(self, message: str) -> None:
        self._stop_inference(show_message=False)
        self._show_error(message)

    def _update_video_display(self, frame: np.ndarray) -> None:
        display_frame = frame
        if (
            self.show_predictions_checkbox.isChecked()
            and self._last_pose
            and self._last_pose.pose is not None
        ):
            display_frame = self._draw_pose(frame, self._last_pose.pose)

        # Draw bounding box if enabled
        if self._bbox_enabled:
            display_frame = self._draw_bbox(display_frame)

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        """Apply cropping to the frame based on settings."""
        if self._active_camera_settings is None:
            return frame

        crop_region = self._active_camera_settings.get_crop_region()
        if crop_region is None:
            return frame

        x0, y0, x1, y1 = crop_region
        height, width = frame.shape[:2]

        # Validate and constrain crop coordinates
        x0 = max(0, min(x0, width))
        y0 = max(0, min(y0, height))
        x1 = max(x0, min(x1, width)) if x1 > 0 else width
        y1 = max(y0, min(y1, height)) if y1 > 0 else height

        # Apply crop
        if x0 < x1 and y0 < y1:
            return frame[y0:y1, x0:x1]
        else:
            # Invalid crop region, return original frame
            return frame

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        if self._rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self._rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self._rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _on_show_predictions_changed(self, _state: int) -> None:
        if self._current_frame is not None:
            self._display_frame(self._current_frame, force=True)

    def _on_bbox_changed(self, _value: int = 0) -> None:
        """Handle bounding box parameter changes."""
        self._bbox_enabled = self.bbox_enabled_checkbox.isChecked()
        self._bbox_x0 = self.bbox_x0_spin.value()
        self._bbox_y0 = self.bbox_y0_spin.value()
        self._bbox_x1 = self.bbox_x1_spin.value()
        self._bbox_y1 = self.bbox_y1_spin.value()

        # Force redraw if preview is running
        if self._current_frame is not None:
            self._display_frame(self._current_frame, force=True)

    def _draw_bbox(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding box on frame with red lines."""
        overlay = frame.copy()
        x0 = self._bbox_x0
        y0 = self._bbox_y0
        x1 = self._bbox_x1
        y1 = self._bbox_y1

        # Validate coordinates
        if x0 >= x1 or y0 >= y1:
            return overlay

        height, width = frame.shape[:2]
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))

        # Draw red rectangle (BGR format: red is (0, 0, 255))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)

        return overlay

    def _draw_pose(self, frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        for keypoint in np.asarray(pose):
            if len(keypoint) < 2:
                continue
            x, y = keypoint[:2]
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(overlay, (int(x), int(y)), 4, (0, 255, 0), -1)
        return overlay

    def _on_dlc_initialised(self, success: bool) -> None:
        if success:
            self._dlc_initialized = True
            # Update button to show running state
            self.start_inference_button.setText("DLCLive running!")
            self.start_inference_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.statusBar().showMessage("DLCLive initialized successfully", 3000)
        else:
            self._dlc_initialized = False
            # Reset button on failure
            self.start_inference_button.setText("Start pose inference")
            self.start_inference_button.setStyleSheet("")
            self.statusBar().showMessage("DLCLive initialization failed", 5000)
            # Stop inference since initialization failed
            self._stop_inference(show_message=False)

    # ------------------------------------------------------------------ helpers
    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        QMessageBox.critical(self, "Error", message)

    def _show_warning(self, message: str) -> None:
        """Display a warning message in the status bar without blocking."""
        self.statusBar().showMessage(f"⚠ {message}", 3000)

    # ------------------------------------------------------------------ Qt overrides
    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - GUI behaviour
        if self.camera_controller.is_running():
            self.camera_controller.stop(wait=True)
        if self._video_recorder and self._video_recorder.is_running:
            self._video_recorder.stop()
        self.dlc_processor.shutdown()
        if hasattr(self, "_metrics_timer"):
            self._metrics_timer.stop()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - manual start
    main()
