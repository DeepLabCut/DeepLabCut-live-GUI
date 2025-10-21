"""PyQt6 based GUI for DeepLabCut Live."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QDoubleSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from dlclivegui.camera_controller import CameraController, FrameData
from dlclivegui.cameras import CameraFactory
from dlclivegui.cameras.factory import DetectedCamera
from dlclivegui.config import (
    ApplicationSettings,
    CameraSettings,
    DLCProcessorSettings,
    RecordingSettings,
    DEFAULT_CONFIG,
)
from dlclivegui.dlc_processor import DLCLiveProcessor, PoseResult
from dlclivegui.video_recorder import VideoRecorder


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

        self.camera_controller = CameraController()
        self.dlc_processor = DLCLiveProcessor()

        self._setup_ui()
        self._connect_signals()
        self._apply_config(self._config)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)

        self.video_label = QLabel("Camera preview not started")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.addWidget(self._build_camera_group())
        controls_layout.addWidget(self._build_dlc_group())
        controls_layout.addWidget(self._build_recording_group())

        button_bar = QHBoxLayout()
        self.preview_button = QPushButton("Start Preview")
        self.stop_preview_button = QPushButton("Stop Preview")
        self.stop_preview_button.setEnabled(False)
        button_bar.addWidget(self.preview_button)
        button_bar.addWidget(self.stop_preview_button)
        controls_layout.addLayout(button_bar)
        controls_layout.addStretch(1)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.video_label)
        preview_layout.addStretch(1)

        layout.addWidget(controls_widget)
        layout.addLayout(preview_layout, stretch=1)

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
        self.camera_backend.setEditable(True)
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

        self.camera_width = QSpinBox()
        self.camera_width.setRange(1, 7680)
        form.addRow("Width", self.camera_width)

        self.camera_height = QSpinBox()
        self.camera_height.setRange(1, 4320)
        form.addRow("Height", self.camera_height)

        self.camera_fps = QDoubleSpinBox()
        self.camera_fps.setRange(1.0, 240.0)
        self.camera_fps.setDecimals(2)
        form.addRow("Frame rate", self.camera_fps)

        self.camera_properties_edit = QPlainTextEdit()
        self.camera_properties_edit.setPlaceholderText(
            '{"exposure": 15000, "gain": 0.5, "serial": "123456"}'
        )
        self.camera_properties_edit.setFixedHeight(60)
        form.addRow("Advanced properties", self.camera_properties_edit)

        self.rotation_combo = QComboBox()
        self.rotation_combo.addItem("0° (default)", 0)
        self.rotation_combo.addItem("90°", 90)
        self.rotation_combo.addItem("180°", 180)
        self.rotation_combo.addItem("270°", 270)
        form.addRow("Rotation", self.rotation_combo)

        return group

    def _build_dlc_group(self) -> QGroupBox:
        group = QGroupBox("DLCLive settings")
        form = QFormLayout(group)

        path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        path_layout.addWidget(self.model_path_edit)
        browse_model = QPushButton("Browse…")
        browse_model.clicked.connect(self._action_browse_model)
        path_layout.addWidget(browse_model)
        form.addRow("Model path", path_layout)

        self.shuffle_edit = QLineEdit()
        self.shuffle_edit.setPlaceholderText("Optional integer")
        form.addRow("Shuffle", self.shuffle_edit)

        self.training_edit = QLineEdit()
        self.training_edit.setPlaceholderText("Optional integer")
        form.addRow("Training set index", self.training_edit)

        self.processor_combo = QComboBox()
        self.processor_combo.setEditable(True)
        self.processor_combo.addItems(["cpu", "gpu", "tensorrt"])
        form.addRow("Processor", self.processor_combo)

        self.processor_args_edit = QPlainTextEdit()
        self.processor_args_edit.setPlaceholderText('{"device": 0}')
        self.processor_args_edit.setFixedHeight(60)
        form.addRow("Processor args", self.processor_args_edit)

        self.additional_options_edit = QPlainTextEdit()
        self.additional_options_edit.setPlaceholderText('{"allow_growth": true}')
        self.additional_options_edit.setFixedHeight(60)
        form.addRow("Additional options", self.additional_options_edit)

        inference_buttons = QHBoxLayout()
        self.start_inference_button = QPushButton("Start pose inference")
        self.start_inference_button.setEnabled(False)
        inference_buttons.addWidget(self.start_inference_button)
        self.stop_inference_button = QPushButton("Stop pose inference")
        self.stop_inference_button.setEnabled(False)
        inference_buttons.addWidget(self.stop_inference_button)
        form.addRow(inference_buttons)

        self.show_predictions_checkbox = QCheckBox("Display pose predictions")
        self.show_predictions_checkbox.setChecked(True)
        form.addRow(self.show_predictions_checkbox)

        return group

    def _build_recording_group(self) -> QGroupBox:
        group = QGroupBox("Recording")
        form = QFormLayout(group)

        self.recording_enabled_checkbox = QCheckBox("Record video while running")
        form.addRow(self.recording_enabled_checkbox)

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

        self.recording_options_edit = QPlainTextEdit()
        self.recording_options_edit.setPlaceholderText('{"compression_mode": "mp4"}')
        self.recording_options_edit.setFixedHeight(60)
        form.addRow("WriteGear options", self.recording_options_edit)

        self.start_record_button = QPushButton("Start recording")
        self.stop_record_button = QPushButton("Stop recording")
        self.stop_record_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_record_button)
        buttons.addWidget(self.stop_record_button)
        form.addRow(buttons)

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
        self.camera_backend.editTextChanged.connect(self._on_backend_changed)
        self.rotation_combo.currentIndexChanged.connect(self._on_rotation_changed)
        self.start_inference_button.clicked.connect(self._start_inference)
        self.stop_inference_button.clicked.connect(lambda: self._stop_inference())
        self.show_predictions_checkbox.stateChanged.connect(
            self._on_show_predictions_changed
        )

        self.camera_controller.frame_ready.connect(self._on_frame_ready)
        self.camera_controller.started.connect(self._on_camera_started)
        self.camera_controller.error.connect(self._show_error)
        self.camera_controller.stopped.connect(self._on_camera_stopped)

        self.dlc_processor.pose_ready.connect(self._on_pose_ready)
        self.dlc_processor.error.connect(self._on_dlc_error)
        self.dlc_processor.initialized.connect(self._on_dlc_initialised)

    # ------------------------------------------------------------------ config
    def _apply_config(self, config: ApplicationSettings) -> None:
        camera = config.camera
        self.camera_width.setValue(int(camera.width))
        self.camera_height.setValue(int(camera.height))
        self.camera_fps.setValue(float(camera.fps))
        backend_name = camera.backend or "opencv"
        index = self.camera_backend.findData(backend_name)
        if index >= 0:
            self.camera_backend.setCurrentIndex(index)
        else:
            self.camera_backend.setEditText(backend_name)
        self._refresh_camera_indices(keep_current=False)
        self._select_camera_by_index(
            camera.index, fallback_text=camera.name or str(camera.index)
        )
        self.camera_properties_edit.setPlainText(
            json.dumps(camera.properties, indent=2) if camera.properties else ""
        )
        self._active_camera_settings = None

        dlc = config.dlc
        self.model_path_edit.setText(dlc.model_path)
        self.shuffle_edit.setText("" if dlc.shuffle is None else str(dlc.shuffle))
        self.training_edit.setText(
            "" if dlc.trainingsetindex is None else str(dlc.trainingsetindex)
        )
        self.processor_combo.setCurrentText(dlc.processor or "cpu")
        self.processor_args_edit.setPlainText(json.dumps(dlc.processor_args, indent=2))
        self.additional_options_edit.setPlainText(
            json.dumps(dlc.additional_options, indent=2)
        )

        recording = config.recording
        self.recording_enabled_checkbox.setChecked(recording.enabled)
        self.output_directory_edit.setText(recording.directory)
        self.filename_edit.setText(recording.filename)
        self.container_combo.setCurrentText(recording.container)
        self.recording_options_edit.setPlainText(json.dumps(recording.options, indent=2))

    def _current_config(self) -> ApplicationSettings:
        return ApplicationSettings(
            camera=self._camera_settings_from_ui(),
            dlc=self._dlc_settings_from_ui(),
            recording=self._recording_settings_from_ui(),
        )

    def _camera_settings_from_ui(self) -> CameraSettings:
        index = self._current_camera_index_value()
        if index is None:
            raise ValueError("Camera selection must provide a numeric index")
        backend_text = self._current_backend_name()
        properties = self._parse_json(self.camera_properties_edit.toPlainText())
        name_text = self.camera_index.currentText().strip()
        settings = CameraSettings(
            name=name_text or f"Camera {index}",
            index=index,
            width=self.camera_width.value(),
            height=self.camera_height.value(),
            fps=self.camera_fps.value(),
            backend=backend_text or "opencv",
            properties=properties,
        )
        return settings.apply_defaults()

    def _current_backend_name(self) -> str:
        backend_data = self.camera_backend.currentData()
        if isinstance(backend_data, str) and backend_data:
            return backend_data
        text = self.camera_backend.currentText().strip()
        return text or "opencv"

    def _refresh_camera_indices(
        self, *_args: object, keep_current: bool = True
    ) -> None:
        backend = self._current_backend_name()
        detected = CameraFactory.detect_cameras(backend)
        debug_info = [f"{camera.index}:{camera.label}" for camera in detected]
        print(
            f"[CameraDetection] Available cameras for backend '{backend}': {debug_info}"
        )
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

    def _select_camera_by_index(
        self, index: int, fallback_text: Optional[str] = None
    ) -> None:
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

    def _current_backend_name(self) -> str:
        backend_data = self.camera_backend.currentData()
        if isinstance(backend_data, str) and backend_data:
            return backend_data
        text = self.camera_backend.currentText().strip()
        return text or "opencv"

    def _refresh_camera_indices(
        self, *_args: object, keep_current: bool = True
    ) -> None:
        backend = self._current_backend_name()
        detected = CameraFactory.detect_cameras(backend)
        debug_info = [f"{camera.index}:{camera.label}" for camera in detected]
        print(
            f"[CameraDetection] Available cameras for backend '{backend}': {debug_info}"
        )
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

    def _select_camera_by_index(
        self, index: int, fallback_text: Optional[str] = None
    ) -> None:
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

    def _parse_optional_int(self, value: str) -> Optional[int]:
        text = value.strip()
        if not text:
            return None
        return int(text)

    def _parse_json(self, value: str) -> dict:
        text = value.strip()
        if not text:
            return {}
        return json.loads(text)

    def _dlc_settings_from_ui(self) -> DLCProcessorSettings:
        return DLCProcessorSettings(
            model_path=self.model_path_edit.text().strip(),
            shuffle=self._parse_optional_int(self.shuffle_edit.text()),
            trainingsetindex=self._parse_optional_int(self.training_edit.text()),
            processor=self.processor_combo.currentText().strip() or "cpu",
            processor_args=self._parse_json(self.processor_args_edit.toPlainText()),
            additional_options=self._parse_json(
                self.additional_options_edit.toPlainText()
            ),
        )

    def _recording_settings_from_ui(self) -> RecordingSettings:
        return RecordingSettings(
            enabled=self.recording_enabled_checkbox.isChecked(),
            directory=self.output_directory_edit.text().strip(),
            filename=self.filename_edit.text().strip() or "session.mp4",
            container=self.container_combo.currentText().strip() or "mp4",
            options=self._parse_json(self.recording_options_edit.toPlainText()),
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
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select DLCLive model", str(Path.home()), "All files (*.*)"
        )
        if file_name:
            self.model_path_edit.setText(file_name)

    def _action_browse_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select output directory", str(Path.home())
        )
        if directory:
            self.output_directory_edit.setText(directory)

    def _on_backend_changed(self, *_args: object) -> None:
        self._refresh_camera_indices(keep_current=False)

    def _on_rotation_changed(self, _index: int) -> None:
        data = self.rotation_combo.currentData()
        self._rotation_degrees = int(data) if isinstance(data, int) else 0
        if self._raw_frame is not None:
            rotated = self._apply_rotation(self._raw_frame)
            self._current_frame = rotated
            self._last_pose = None
            self._update_video_display(rotated)

    def _on_backend_changed(self, *_args: object) -> None:
        self._refresh_camera_indices(keep_current=False)

    def _on_rotation_changed(self, _index: int) -> None:
        data = self.rotation_combo.currentData()
        self._rotation_degrees = int(data) if isinstance(data, int) else 0
        if self._raw_frame is not None:
            rotated = self._apply_rotation(self._raw_frame)
            self._current_frame = rotated
            self._last_pose = None
            self._update_video_display(rotated)

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

    def _on_camera_started(self, settings: CameraSettings) -> None:
        self._active_camera_settings = settings
        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(True)
        self.camera_width.blockSignals(True)
        self.camera_width.setValue(int(settings.width))
        self.camera_width.blockSignals(False)
        self.camera_height.blockSignals(True)
        self.camera_height.setValue(int(settings.height))
        self.camera_height.blockSignals(False)
        if getattr(settings, "fps", None):
            self.camera_fps.blockSignals(True)
            self.camera_fps.setValue(float(settings.fps))
            self.camera_fps.blockSignals(False)
        resolution = f"{int(settings.width)}×{int(settings.height)}"
        if getattr(settings, "fps", None):
            fps_text = f"{float(settings.fps):.2f} FPS"
        else:
            fps_text = "unknown FPS"
        self.statusBar().showMessage(
            f"Camera preview started: {resolution} @ {fps_text}", 5000
        )
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
        self.dlc_processor.configure(settings)
        return True

    def _update_inference_buttons(self) -> None:
        preview_running = self.camera_controller.is_running()
        self.start_inference_button.setEnabled(preview_running and not self._dlc_active)
        self.stop_inference_button.setEnabled(preview_running and self._dlc_active)

    def _update_camera_controls_enabled(self) -> None:
        recording_active = (
            self._video_recorder is not None and self._video_recorder.is_running
        )
        allow_changes = (
            not self.camera_controller.is_running()
            and not self._dlc_active
            and not recording_active
        )
        widgets = [
            self.camera_backend,
            self.camera_index,
            self.refresh_cameras_button,
            self.camera_width,
            self.camera_height,
            self.camera_fps,
            self.camera_properties_edit,
            self.rotation_combo,
        ]
        for widget in widgets:
            widget.setEnabled(allow_changes)

    def _start_inference(self) -> None:
        if self._dlc_active:
            self.statusBar().showMessage("Pose inference already running", 3000)
            return
        if not self.camera_controller.is_running():
            self._show_error(
                "Start the camera preview before running pose inference."
            )
            return
        if not self._configure_dlc():
            self._update_inference_buttons()
            return
        self.dlc_processor.reset()
        self._last_pose = None
        self._dlc_active = True
        self.statusBar().showMessage("Starting pose inference…", 3000)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

    def _stop_inference(self, show_message: bool = True) -> None:
        was_active = self._dlc_active
        self._dlc_active = False
        self.dlc_processor.reset()
        self._last_pose = None
        if self._current_frame is not None:
            self._update_video_display(self._current_frame)
        if was_active and show_message:
            self.statusBar().showMessage("Pose inference stopped", 3000)
        self._update_inference_buttons()
        self._update_camera_controls_enabled()

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
        try:
            recording = self._recording_settings_from_ui()
        except json.JSONDecodeError as exc:
            self._show_error(f"Invalid recording options: {exc}")
            return
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
            recording.options,
            frame_size=(int(width), int(height)),
            frame_rate=float(frame_rate),
        )
        try:
            self._video_recorder.start()
        except Exception as exc:  # pragma: no cover - runtime error
            self._show_error(str(exc))
            self._video_recorder = None
            return
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.statusBar().showMessage(f"Recording to {output_path}", 5000)
        self._update_camera_controls_enabled()

    def _stop_recording(self) -> None:
        if not self._video_recorder:
            return
        self._video_recorder.stop()
        self._video_recorder = None
        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.statusBar().showMessage("Recording stopped", 3000)
        self._update_camera_controls_enabled()

    # ------------------------------------------------------------------ frame handling
    def _on_frame_ready(self, frame_data: FrameData) -> None:
        raw_frame = frame_data.image
        self._raw_frame = raw_frame
        frame = self._apply_rotation(raw_frame)
        self._current_frame = frame
        if self._active_camera_settings is not None:
            height, width = frame.shape[:2]
            self._active_camera_settings.width = int(width)
            self._active_camera_settings.height = int(height)
        if self._video_recorder and self._video_recorder.is_running:
            self._video_recorder.write(frame)
        if self._dlc_active:
            self.dlc_processor.enqueue_frame(frame, frame_data.timestamp)
        self._update_video_display(frame)

    def _on_pose_ready(self, result: PoseResult) -> None:
        if not self._dlc_active:
            return
        self._last_pose = result
        if self._current_frame is not None:
            self._update_video_display(self._current_frame)

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
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

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
            self._update_video_display(self._current_frame)

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
            self.statusBar().showMessage("DLCLive initialised", 3000)
        else:
            self.statusBar().showMessage("DLCLive initialisation failed", 3000)

    # ------------------------------------------------------------------ helpers
    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        QMessageBox.critical(self, "Error", message)

    # ------------------------------------------------------------------ Qt overrides
    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - GUI behaviour
        if self.camera_controller.is_running():
            self.camera_controller.stop(wait=True)
        if self._video_recorder and self._video_recorder.is_running:
            self._video_recorder.stop()
        self.dlc_processor.shutdown()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - manual start
    main()
