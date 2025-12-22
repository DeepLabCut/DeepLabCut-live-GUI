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
import matplotlib.pyplot as plt
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
    QStyle,
    QVBoxLayout,
    QWidget,
)

from dlclivegui.camera_config_dialog import CameraConfigDialog
from dlclivegui.config import (
    DEFAULT_CONFIG,
    ApplicationSettings,
    BoundingBoxSettings,
    CameraSettings,
    DLCProcessorSettings,
    MultiCameraSettings,
    RecordingSettings,
    VisualizationSettings,
)
from dlclivegui.dlc_processor import DLCLiveProcessor, PoseResult, ProcessorStats
from dlclivegui.multi_camera_controller import MultiCameraController, MultiFrameData, get_camera_id
from dlclivegui.processors.processor_utils import instantiate_from_scan, scan_processor_folder
from dlclivegui.video_recorder import RecorderStats, VideoRecorder

logging.basicConfig(level=logging.INFO)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config: Optional[ApplicationSettings] = None):
        super().__init__()
        self.setWindowTitle("DeepLabCut Live GUI")

        # Try to load myconfig.json from the application directory if no config provided
        if config is None:
            myconfig_path = Path(__file__).parent.parent / "myconfig.json"
            if myconfig_path.exists():
                try:
                    config = ApplicationSettings.load(str(myconfig_path))
                    self._config_path = myconfig_path
                    logging.info(f"Loaded configuration from {myconfig_path}")
                except Exception as exc:
                    logging.warning(f"Failed to load myconfig.json: {exc}. Using default config.")
                    config = DEFAULT_CONFIG
                    self._config_path = None
            else:
                config = DEFAULT_CONFIG
                self._config_path = None
        else:
            self._config_path = None

        self._config = config
        self._current_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._last_pose: Optional[PoseResult] = None
        self._dlc_active: bool = False
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

        # Visualization settings (will be updated from config)
        self._p_cutoff = 0.6
        self._colormap = "hot"
        self._bbox_color = (0, 0, 255)  # BGR: red

        self.multi_camera_controller = MultiCameraController()
        self.dlc_processor = DLCLiveProcessor()

        # Multi-camera state
        self._multi_camera_mode = False
        self._multi_camera_recorders: dict[str, VideoRecorder] = {}
        self._multi_camera_frames: dict[str, np.ndarray] = {}

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

        # Show status message if myconfig.json was loaded
        if self._config_path and self._config_path.name == "myconfig.json":
            self.statusBar().showMessage(
                f"Auto-loaded configuration from {self._config_path}", 5000
            )

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)

        # Video panel with display and performance stats
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # Video display widget
        self.video_label = QLabel("Camera preview not started")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_label)

        # Stats panel below video with clear labels
        stats_widget = QWidget()
        stats_widget.setStyleSheet("padding: 5px;")
        stats_widget.setMinimumWidth(800)  # Prevent excessive line breaks
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        stats_layout.setSpacing(3)

        # Camera throughput stats
        camera_stats_container = QHBoxLayout()
        camera_stats_label_title = QLabel("<b>Camera:</b>")
        camera_stats_container.addWidget(camera_stats_label_title)
        self.camera_stats_label = QLabel("Camera idle")
        self.camera_stats_label.setWordWrap(True)
        camera_stats_container.addWidget(self.camera_stats_label)
        camera_stats_container.addStretch(1)
        stats_layout.addLayout(camera_stats_container)

        # DLC processor stats
        dlc_stats_container = QHBoxLayout()
        dlc_stats_label_title = QLabel("<b>DLC Processor:</b>")
        dlc_stats_container.addWidget(dlc_stats_label_title)
        self.dlc_stats_label = QLabel("DLC processor idle")
        self.dlc_stats_label.setWordWrap(True)
        dlc_stats_container.addWidget(self.dlc_stats_label)
        dlc_stats_container.addStretch(1)
        stats_layout.addLayout(dlc_stats_container)

        # Video recorder stats
        recorder_stats_container = QHBoxLayout()
        recorder_stats_label_title = QLabel("<b>Recorder:</b>")
        recorder_stats_container.addWidget(recorder_stats_label_title)
        self.recording_stats_label = QLabel("Recorder idle")
        self.recording_stats_label.setWordWrap(True)
        recorder_stats_container.addWidget(self.recording_stats_label)
        recorder_stats_container.addStretch(1)
        stats_layout.addLayout(recorder_stats_container)

        video_layout.addWidget(stats_widget)

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
        self.preview_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.preview_button.setMinimumWidth(150)
        self.stop_preview_button = QPushButton("Stop Preview")
        self.stop_preview_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.stop_preview_button.setEnabled(False)
        self.stop_preview_button.setMinimumWidth(150)
        button_bar.addWidget(self.preview_button)
        button_bar.addWidget(self.stop_preview_button)
        controls_layout.addWidget(button_bar_widget)
        controls_layout.addStretch(1)

        # Add controls and video panel to main layout
        layout.addWidget(controls_widget, stretch=0)
        layout.addWidget(video_panel, stretch=1)

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

        # Camera config button - opens dialog for all camera configuration
        config_layout = QHBoxLayout()
        self.config_cameras_button = QPushButton("Configure Cameras...")
        self.config_cameras_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        )
        self.config_cameras_button.setToolTip("Configure camera settings (single or multi-camera)")
        config_layout.addWidget(self.config_cameras_button)
        form.addRow(config_layout)

        # Active cameras display label
        self.active_cameras_label = QLabel("No cameras configured")
        self.active_cameras_label.setWordWrap(True)
        form.addRow("Active:", self.active_cameras_label)

        return group

    def _build_dlc_group(self) -> QGroupBox:
        group = QGroupBox("DLCLive settings")
        form = QFormLayout(group)

        path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("/path/to/exported/model")
        path_layout.addWidget(self.model_path_edit)
        self.browse_model_button = QPushButton("Browse…")
        self.browse_model_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self.browse_model_button.clicked.connect(self._action_browse_model)
        path_layout.addWidget(self.browse_model_button)
        form.addRow("Model file", path_layout)

        # Processor selection
        processor_path_layout = QHBoxLayout()
        self.processor_folder_edit = QLineEdit()
        self.processor_folder_edit.setText(str(Path(__file__).parent.joinpath("processors")))
        processor_path_layout.addWidget(self.processor_folder_edit)

        self.browse_processor_folder_button = QPushButton("Browse...")
        self.browse_processor_folder_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self.browse_processor_folder_button.clicked.connect(self._action_browse_processor_folder)
        processor_path_layout.addWidget(self.browse_processor_folder_button)

        self.refresh_processors_button = QPushButton("Refresh")
        self.refresh_processors_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
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
        self.start_inference_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight)
        )
        self.start_inference_button.setEnabled(False)
        self.start_inference_button.setMinimumWidth(150)
        inference_buttons.addWidget(self.start_inference_button)
        self.stop_inference_button = QPushButton("Stop pose inference")
        self.stop_inference_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop)
        )
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

        return group

    def _build_recording_group(self) -> QGroupBox:
        group = QGroupBox("Recording")
        form = QFormLayout(group)

        dir_layout = QHBoxLayout()
        self.output_directory_edit = QLineEdit()
        dir_layout.addWidget(self.output_directory_edit)
        browse_dir = QPushButton("Browse…")
        browse_dir.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
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
        if os.sys.platform == "darwin":
            self.codec_combo.addItems(["h264_videotoolbox", "libx264", "hevc_videotoolbox"])
        else:
            self.codec_combo.addItems(["h264_nvenc", "libx264", "hevc_nvenc"])
        self.codec_combo.setCurrentText("libx264")
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
        self.start_record_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogYesButton)
        )
        self.start_record_button.setMinimumWidth(150)
        buttons.addWidget(self.start_record_button)
        self.stop_record_button = QPushButton("Stop recording")
        self.stop_record_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogNoButton)
        )
        self.stop_record_button.setEnabled(False)
        self.stop_record_button.setMinimumWidth(150)
        buttons.addWidget(self.stop_record_button)
        form.addRow(recording_button_widget)

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
        self.start_inference_button.clicked.connect(self._start_inference)
        self.stop_inference_button.clicked.connect(lambda: self._stop_inference())
        self.show_predictions_checkbox.stateChanged.connect(self._on_show_predictions_changed)

        # Camera config dialog
        self.config_cameras_button.clicked.connect(self._open_camera_config_dialog)

        # Connect bounding box controls
        self.bbox_enabled_checkbox.stateChanged.connect(self._on_bbox_changed)
        self.bbox_x0_spin.valueChanged.connect(self._on_bbox_changed)
        self.bbox_y0_spin.valueChanged.connect(self._on_bbox_changed)
        self.bbox_x1_spin.valueChanged.connect(self._on_bbox_changed)
        self.bbox_y1_spin.valueChanged.connect(self._on_bbox_changed)

        # Multi-camera controller signals (used for both single and multi-camera modes)
        self.multi_camera_controller.frame_ready.connect(self._on_multi_frame_ready)
        self.multi_camera_controller.all_started.connect(self._on_multi_camera_started)
        self.multi_camera_controller.all_stopped.connect(self._on_multi_camera_stopped)
        self.multi_camera_controller.camera_error.connect(self._on_multi_camera_error)

        self.dlc_processor.pose_ready.connect(self._on_pose_ready)
        self.dlc_processor.error.connect(self._on_dlc_error)
        self.dlc_processor.initialized.connect(self._on_dlc_initialised)

    # ------------------------------------------------------------------ config
    def _apply_config(self, config: ApplicationSettings) -> None:
        # Update active cameras label
        self._update_active_cameras_label()

        dlc = config.dlc
        self.model_path_edit.setText(dlc.model_path)

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

        # Set visualization settings from config
        viz = config.visualization
        self._p_cutoff = viz.p_cutoff
        self._colormap = viz.colormap
        self._bbox_color = viz.get_bbox_color_bgr()

    def _current_config(self) -> ApplicationSettings:
        # Get the first camera from multi-camera config for backward compatibility
        active_cameras = self._config.multi_camera.get_active_cameras()
        camera = active_cameras[0] if active_cameras else CameraSettings()

        return ApplicationSettings(
            camera=camera,
            multi_camera=self._config.multi_camera,
            dlc=self._dlc_settings_from_ui(),
            recording=self._recording_settings_from_ui(),
            bbox=self._bbox_settings_from_ui(),
            visualization=self._visualization_settings_from_ui(),
        )

    def _parse_json(self, value: str) -> dict:
        text = value.strip()
        if not text:
            return {}
        return json.loads(text)

    def _dlc_settings_from_ui(self) -> DLCProcessorSettings:
        return DLCProcessorSettings(
            model_path=self.model_path_edit.text().strip(),
            model_directory=self._config.dlc.model_directory,  # Preserve from config
            device=self._config.dlc.device,  # Preserve from config
            dynamic=self._config.dlc.dynamic,  # Preserve from config
            resize=self._config.dlc.resize,  # Preserve from config
            precision=self._config.dlc.precision,  # Preserve from config
            model_type="pytorch",
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

    def _visualization_settings_from_ui(self) -> VisualizationSettings:
        return VisualizationSettings(
            p_cutoff=self._p_cutoff,
            colormap=self._colormap,
            bbox_color=self._bbox_color,
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
        # Use model_directory from config, default to current directory
        start_dir = self._config.dlc.model_directory or "."
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DLCLive model file",
            start_dir,
            "Model files (*.pt *.pb);;All files (*.*)",
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

    # ------------------------------------------------------------------ multi-camera
    def _open_camera_config_dialog(self) -> None:
        """Open the camera configuration dialog."""
        dialog = CameraConfigDialog(self, self._config.multi_camera)
        dialog.settings_changed.connect(self._on_multi_camera_settings_changed)
        dialog.exec()

    def _on_multi_camera_settings_changed(self, settings: MultiCameraSettings) -> None:
        """Handle changes to multi-camera settings."""
        self._config.multi_camera = settings
        self._update_active_cameras_label()
        active_count = len(settings.get_active_cameras())
        self.statusBar().showMessage(
            f"Camera configuration updated: {active_count} active camera(s)", 3000
        )

    def _update_active_cameras_label(self) -> None:
        """Update the label showing active cameras."""
        active_cams = self._config.multi_camera.get_active_cameras()
        if not active_cams:
            self.active_cameras_label.setText("No cameras configured")
        elif len(active_cams) == 1:
            cam = active_cams[0]
            self.active_cameras_label.setText(
                f"{cam.name} [{cam.backend}:{cam.index}] @ {cam.fps:.1f} fps"
            )
        else:
            cam_names = [f"{c.name}" for c in active_cams]
            self.active_cameras_label.setText(f"{len(active_cams)} cameras: {', '.join(cam_names)}")

    def _on_multi_frame_ready(self, frame_data: MultiFrameData) -> None:
        """Handle frames from multiple cameras."""
        self._multi_camera_frames = frame_data.frames
        self._track_camera_frame()  # Track FPS

        # For single camera mode, also set raw_frame for DLC processing
        if len(frame_data.frames) == 1:
            cam_id = next(iter(frame_data.frames.keys()))
            self._raw_frame = frame_data.frames[cam_id]

        # Record individual camera feeds if recording is active
        if self._multi_camera_recorders:
            for cam_id, frame in frame_data.frames.items():
                if cam_id in self._multi_camera_recorders:
                    recorder = self._multi_camera_recorders[cam_id]
                    if recorder.is_running:
                        timestamp = frame_data.timestamps.get(cam_id, time.time())
                        try:
                            recorder.write(frame, timestamp=timestamp)
                        except Exception as exc:
                            logging.warning(f"Failed to write frame for camera {cam_id}: {exc}")

        # Display tiled frame (or single frame for 1 camera)
        if frame_data.tiled_frame is not None:
            self._current_frame = frame_data.tiled_frame
            self._display_frame(frame_data.tiled_frame)

        # For DLC processing, use single frame if only one camera
        if self._dlc_active and len(frame_data.frames) == 1:
            cam_id = next(iter(frame_data.frames.keys()))
            frame = frame_data.frames[cam_id]
            timestamp = frame_data.timestamps.get(cam_id, time.time())
            self.dlc_processor.enqueue_frame(frame, timestamp)

    def _on_multi_camera_started(self) -> None:
        """Handle all cameras started event."""
        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(True)
        active_count = self.multi_camera_controller.get_active_count()
        self.statusBar().showMessage(
            f"Multi-camera preview started: {active_count} camera(s)", 5000
        )
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    def _on_multi_camera_stopped(self) -> None:
        """Handle all cameras stopped event."""
        # Stop all multi-camera recorders
        self._stop_multi_camera_recording()

        self.preview_button.setEnabled(True)
        self.stop_preview_button.setEnabled(False)
        self._current_frame = None
        self._multi_camera_frames.clear()
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Camera preview not started")
        self.statusBar().showMessage("Multi-camera preview stopped", 3000)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    def _on_multi_camera_error(self, camera_id: str, message: str) -> None:
        """Handle error from a camera in multi-camera mode."""
        self._show_warning(f"Camera {camera_id} error: {message}")

    def _start_multi_camera_recording(self) -> None:
        """Start recording from all active cameras."""
        if self._multi_camera_recorders:
            return  # Already recording

        recording = self._recording_settings_from_ui()
        if not recording.enabled:
            self._show_error("Recording is disabled in the configuration.")
            return

        active_cams = self._config.multi_camera.get_active_cameras()
        if not active_cams:
            self._show_error("No active cameras configured.")
            return

        base_path = recording.output_path()
        base_stem = base_path.stem

        for cam in active_cams:
            cam_id = get_camera_id(cam)
            # Create unique filename for each camera
            cam_filename = f"{base_stem}_{cam.backend}_cam{cam.index}{base_path.suffix}"
            cam_path = base_path.parent / cam_filename

            # Get frame from current frames if available
            frame = self._multi_camera_frames.get(cam_id)
            frame_size = (frame.shape[0], frame.shape[1]) if frame is not None else None

            recorder = VideoRecorder(
                cam_path,
                frame_size=frame_size,
                frame_rate=float(cam.fps),
                codec=recording.codec,
                crf=recording.crf,
            )

            try:
                recorder.start()
                self._multi_camera_recorders[cam_id] = recorder
                logging.info(f"Started recording camera {cam_id} to {cam_path}")
            except Exception as exc:
                self._show_error(f"Failed to start recording for camera {cam_id}: {exc}")

        if self._multi_camera_recorders:
            self.start_record_button.setEnabled(False)
            self.stop_record_button.setEnabled(True)
            self.statusBar().showMessage(
                f"Recording {len(self._multi_camera_recorders)} camera(s) to {recording.directory}",
                5000,
            )
            self._update_camera_controls_enabled()

    def _stop_multi_camera_recording(self) -> None:
        """Stop recording from all cameras."""
        if not self._multi_camera_recorders:
            return

        for cam_id, recorder in self._multi_camera_recorders.items():
            try:
                recorder.stop()
                logging.info(f"Stopped recording camera {cam_id}")
            except Exception as exc:
                logging.warning(f"Error stopping recorder for camera {cam_id}: {exc}")

        self._multi_camera_recorders.clear()
        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.statusBar().showMessage("Multi-camera recording stopped", 3000)
        self._update_camera_controls_enabled()

    # ------------------------------------------------------------------ camera control
    def _start_preview(self) -> None:
        """Start camera preview - uses multi-camera controller for all configurations."""
        active_cams = self._config.multi_camera.get_active_cameras()
        if not active_cams:
            self._show_error("No cameras configured. Use 'Configure Cameras...' to add cameras.")
            return

        # Determine if we're in single or multi-camera mode
        self._multi_camera_mode = len(active_cams) > 1

        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(True)
        self._current_frame = None
        self._raw_frame = None
        self._last_pose = None
        self._multi_camera_frames.clear()
        self._camera_frame_times.clear()
        self._last_display_time = 0.0

        if hasattr(self, "camera_stats_label"):
            self.camera_stats_label.setText(f"Starting {len(active_cams)} camera(s)…")
        self.statusBar().showMessage(f"Starting preview ({len(active_cams)} camera(s))…", 3000)

        # Store active settings for single camera mode (for DLC, recording frame rate, etc.)
        self._active_camera_settings = active_cams[0] if active_cams else None

        self.multi_camera_controller.start(active_cams)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    def _stop_preview(self) -> None:
        """Stop camera preview."""
        if not self.multi_camera_controller.is_running():
            return

        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)
        self.statusBar().showMessage("Stopping preview…", 3000)

        # Stop any active recording first
        self._stop_multi_camera_recording()

        self.multi_camera_controller.stop()
        self._stop_inference(show_message=False)
        self._camera_frame_times.clear()
        self._last_display_time = 0.0
        if hasattr(self, "camera_stats_label"):
            self.camera_stats_label.setText("Camera idle")

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
        preview_running = self.multi_camera_controller.is_running()
        self.start_inference_button.setEnabled(preview_running and not self._dlc_active)
        self.stop_inference_button.setEnabled(preview_running and self._dlc_active)

    def _update_dlc_controls_enabled(self) -> None:
        """Enable/disable DLC settings based on inference state."""
        allow_changes = not self._dlc_active
        widgets = [
            self.model_path_edit,
            self.browse_model_button,
            self.processor_folder_edit,
            self.browse_processor_folder_button,
            self.refresh_processors_button,
            self.processor_combo,
            self.additional_options_edit,
        ]
        for widget in widgets:
            widget.setEnabled(allow_changes)

    def _update_camera_controls_enabled(self) -> None:
        multi_cam_recording = bool(self._multi_camera_recorders)

        # Check if preview is running
        preview_running = self.multi_camera_controller.is_running()

        allow_changes = not preview_running and not self._dlc_active and not multi_cam_recording

        # Recording settings (codec, crf) should be editable when not recording
        recording_editable = not multi_cam_recording
        self.codec_combo.setEnabled(recording_editable)
        self.crf_spin.setEnabled(recording_editable)

        # Config cameras button should be available when not in preview/recording
        self.config_cameras_button.setEnabled(allow_changes)

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

        # Add profiling info if available
        profile_info = ""
        if stats.avg_inference_time > 0:
            inf_ms = stats.avg_inference_time * 1000.0
            queue_ms = stats.avg_queue_wait * 1000.0
            signal_ms = stats.avg_signal_emit_time * 1000.0
            total_ms = stats.avg_total_process_time * 1000.0

            # Add GPU vs processor breakdown if available
            gpu_breakdown = ""
            if stats.avg_gpu_inference_time > 0 or stats.avg_processor_overhead > 0:
                gpu_ms = stats.avg_gpu_inference_time * 1000.0
                proc_ms = stats.avg_processor_overhead * 1000.0
                gpu_breakdown = f" (GPU:{gpu_ms:.1f}ms+proc:{proc_ms:.1f}ms)"

            profile_info = (
                f"\n[Profile] inf:{inf_ms:.1f}ms{gpu_breakdown} queue:{queue_ms:.1f}ms "
                f"signal:{signal_ms:.1f}ms total:{total_ms:.1f}ms"
            )

        return (
            f"{processed}/{enqueue} frames | inference {processing_fps:.1f} fps | "
            f"latency {latency_ms:.1f} ms (avg {avg_ms:.1f} ms) | "
            f"queue {stats.queue_size} | dropped {dropped}{profile_info}"
        )

    def _update_metrics(self) -> None:
        if hasattr(self, "camera_stats_label"):
            running = self.multi_camera_controller.is_running()

            if running:
                active_count = self.multi_camera_controller.get_active_count()
                fps = self._compute_fps(self._camera_frame_times)
                if fps > 0:
                    if active_count > 1:
                        self.camera_stats_label.setText(
                            f"{active_count} cameras | {fps:.1f} fps (last 5 s)"
                        )
                    else:
                        self.camera_stats_label.setText(f"{fps:.1f} fps (last 5 s)")
                else:
                    if active_count > 1:
                        self.camera_stats_label.setText(f"{active_count} cameras | Measuring…")
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
            # Handle multi-camera recording stats
            if self._multi_camera_recorders:
                num_recorders = len(self._multi_camera_recorders)
                if num_recorders == 1:
                    # Single camera - show detailed stats
                    recorder = next(iter(self._multi_camera_recorders.values()))
                    stats = recorder.get_stats()
                    if stats:
                        summary = self._format_recorder_stats(stats)
                    else:
                        summary = "Recording..."
                else:
                    # Multiple cameras - show aggregated stats with per-camera details
                    total_written = 0
                    total_dropped = 0
                    total_queue = 0
                    max_latency = 0.0
                    avg_latencies = []
                    for recorder in self._multi_camera_recorders.values():
                        stats = recorder.get_stats()
                        if stats:
                            total_written += stats.frames_written
                            total_dropped += stats.dropped_frames
                            total_queue += stats.queue_size
                            max_latency = max(max_latency, stats.last_latency)
                            avg_latencies.append(stats.average_latency)
                    avg_latency = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0
                    summary = (
                        f"{num_recorders} cams | {total_written} frames | "
                        f"latency {max_latency*1000:.1f}ms (avg {avg_latency*1000:.1f}ms) | "
                        f"queue {total_queue} | dropped {total_dropped}"
                    )
                self._last_recorder_summary = summary
                self.recording_stats_label.setText(summary)
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
                    if not self._multi_camera_recorders:
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
                    if self._multi_camera_recorders:
                        self._stop_recording()
                        self.statusBar().showMessage("Auto-stopped recording", 3000)
                        logging.info("Auto-recording stopped")

                self._last_processor_vid_recording = current_vid_recording

    def _start_inference(self) -> None:
        if self._dlc_active:
            self.statusBar().showMessage("Pose inference already running", 3000)
            return
        if not self.multi_camera_controller.is_running():
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
        """Start recording from all active cameras."""
        # Auto-start preview if not running
        if not self.multi_camera_controller.is_running():
            self._start_preview()
            # Wait a moment for cameras to initialize before recording
            # The recording will start after preview is confirmed running
            self.statusBar().showMessage("Starting preview before recording...", 3000)
            # Use a single-shot timer to start recording after preview starts
            QTimer.singleShot(500, self._start_multi_camera_recording)
            return

        # Preview already running, start recording immediately
        self._start_multi_camera_recording()

    def _stop_recording(self) -> None:
        """Stop recording from all cameras."""
        self._stop_multi_camera_recording()

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

        # Draw rectangle with configured color
        cv2.rectangle(overlay, (x0, y0), (x1, y1), self._bbox_color, 2)

        return overlay

    def _draw_pose(self, frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
        overlay = frame.copy()

        # Get the colormap from config
        cmap = plt.get_cmap(self._colormap)
        num_keypoints = len(np.asarray(pose))

        for idx, keypoint in enumerate(np.asarray(pose)):
            if len(keypoint) < 2:
                continue
            x, y = keypoint[:2]
            confidence = keypoint[2] if len(keypoint) > 2 else 1.0
            if np.isnan(x) or np.isnan(y):
                continue
            if confidence < self._p_cutoff:
                continue

            # Get color from colormap (cycle through 0 to 1)
            color_normalized = idx / max(num_keypoints - 1, 1)
            rgba = cmap(color_normalized)
            # Convert from RGBA [0, 1] to BGR [0, 255] for OpenCV
            bgr_color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

            cv2.circle(overlay, (int(x), int(y)), 4, bgr_color, -1)
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
        if self.multi_camera_controller.is_running():
            self.multi_camera_controller.stop(wait=True)
        # Stop all multi-camera recorders
        for recorder in self._multi_camera_recorders.values():
            if recorder.is_running:
                recorder.stop()
        self._multi_camera_recorders.clear()
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
