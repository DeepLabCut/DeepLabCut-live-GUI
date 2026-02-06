"""PySide6 based GUI for DeepLabCut Live."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import time
from pathlib import Path

os.environ["PYLON_CAMEMU"] = "2"

import cv2
import numpy as np
from PySide6.QtCore import QRect, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from dlclivegui.cameras import CameraFactory
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

from ..processors.processor_utils import (
    default_processors_dir,
    instantiate_from_scan,
    scan_processor_folder,
    scan_processor_package,
)
from ..services.dlc_processor import DLCLiveProcessor, PoseResult
from ..services.multi_camera_controller import MultiCameraController, MultiFrameData, get_camera_id
from ..utils.display import BBoxColors, compute_tile_info, create_tiled_frame, draw_bbox, draw_pose
from ..utils.settings_store import DLCLiveGUISettingsStore, ModelPathStore
from ..utils.stats import format_dlc_stats
from ..utils.utils import FPSTracker
from .camera_config_dialog import CameraConfigDialog
from .misc.elidinglabel import ElidingPathLabel
from .recording_manager import RecordingManager
from .theme import LOGO, LOGO_ALPHA, AppStyle, apply_theme

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # FIXME @C-Achard set back to INFO for release
logger = logging.getLogger("DLCLiveGUI")


class DLCLiveMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config: ApplicationSettings | None = None):
        super().__init__()
        self.setWindowTitle("DeepLabCut Live GUI")

        self.settings = QSettings("DeepLabCut", "DLCLiveGUI")
        self._model_path_store = ModelPathStore(self.settings)
        self._settings_store = DLCLiveGUISettingsStore(self.settings)

        if config is None:
            # 1) snapshot
            cfg = self._settings_store.load_full_config_snapshot()
            if cfg is not None:
                config = cfg
                self._config_path = None
                logger.info("Loaded configuration from QSettings snapshot.")
            else:
                # 2) last config file path
                last_cfg_path = self._settings_store.get_last_config_path()
                if last_cfg_path:
                    try:
                        p = Path(last_cfg_path)
                        if p.exists() and p.is_file():
                            config = ApplicationSettings.load(str(p))
                            self._config_path = p
                            logger.info(f"Loaded configuration from last config path: {p}")
                        else:
                            config = DEFAULT_CONFIG
                            self._config_path = None
                    except Exception as exc:
                        logger.warning(
                            f"Failed to load last config path ({last_cfg_path}): {exc}. Using default config."
                        )
                        config = DEFAULT_CONFIG
                        self._config_path = None
                else:
                    # 3) default
                    config = DEFAULT_CONFIG
                    self._config_path = None
        else:
            self._config_path = None

        self._fps_tracker = FPSTracker()
        self._rec_manager = RecordingManager()
        self._dlc = DLCLiveProcessor()
        self.multi_camera_controller = MultiCameraController()

        self._config = config
        self._inference_camera_id: str | None = None  # Camera ID used for inference
        self._running_cams_ids: set[str] = set()
        self._current_frame: np.ndarray | None = None
        self._raw_frame: np.ndarray | None = None
        self._last_pose: PoseResult | None = None
        self._dlc_active: bool = False
        self._active_camera_settings: CameraSettings | None = None
        self._last_drop_warning = 0.0
        self._last_recorder_summary = "Recorder idle"
        self._display_interval = 1.0 / 25.0
        self._last_display_time = 0.0
        self._dlc_initialized = False
        self._scanned_processors: dict = {}
        self._processor_keys: list = []
        self._last_processor_vid_recording = False
        self._auto_record_session_name: str | None = None
        self._bbox_x0 = 0
        self._bbox_y0 = 0
        self._bbox_x1 = 0
        self._bbox_y1 = 0
        self._bbox_enabled = False
        # UI elements
        self._current_style: AppStyle = AppStyle.DARK
        self._cam_dialog: CameraConfigDialog | None = None

        # Visualization settings (will be updated from config)
        self._p_cutoff = 0.6
        self._colormap = "hot"
        self._bbox_color = (0, 0, 255)  # BGR: red

        # Multi-camera state
        self._multi_camera_mode = False
        self._multi_camera_frames: dict[str, np.ndarray] = {}
        # DLC pose rendering info for tiled view
        self._dlc_tile_offset: tuple[int, int] = (0, 0)  # (x, y) offset in tiled frame
        self._dlc_tile_scale: tuple[float, float] = (1.0, 1.0)  # (scale_x, scale_y)
        # Display flag (decoupled from frame capture for performance)
        self._display_dirty: bool = False

        self._load_icons()
        self._preview_pixmap = QPixmap(LOGO_ALPHA)
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

        # Display timer - decoupled from frame capture for performance
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(33)  # ~30 fps display rate
        self._display_timer.timeout.connect(self._update_display_from_pending)
        self._display_timer.start()

        # Show status message if myconfig.json was loaded
        if self._config_path and self._config_path.name == "myconfig.json":
            self.statusBar().showMessage(f"Auto-loaded configuration from {self._config_path}", 5000)

        # Validate cameras from loaded config (deferred to allow window to show first)
        # NOTE IMPORTANT (tests/CI): This is scheduled via a QTimer and may fire during pytest-qt teardown.
        QTimer.singleShot(100, self._validate_configured_cameras)
        # If validation triggers a modal QMessageBox (warning/error) while the parent window is closing,
        # it can cause errors with unpredictable timing (heap corruption / access violations).
        #
        # Mitigations for tests/CI:
        #   - Disable this timer by monkeypatching _validate_configured_cameras in GUI tests
        #   - OR monkeypatch/override _show_warning/_show_error to no-op in GUI tests (easiest)
        #   - OR use a cancellable QTimer attribute and stop() it in closeEven

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.multi_camera_controller.is_running():
            self._show_logo_and_text()

    # ------------------------------------------------------------------ UI
    def _init_theme_actions(self) -> None:
        """Set initial checked state for theme actions based on current app stylesheet."""
        self.action_dark_mode.setChecked(self._current_style == AppStyle.DARK)
        self.action_light_mode.setChecked(self._current_style == AppStyle.SYS_DEFAULT)

    def _apply_theme(self, mode: AppStyle) -> None:
        """Apply the selected theme and update menu action states."""
        apply_theme(mode, self.action_dark_mode, self.action_light_mode)
        self._current_style = mode

    def _load_icons(self):
        self.setWindowIcon(QIcon(LOGO))

    def _setup_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)

        # Video panel with display and performance stats
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # Video display widget
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_label, stretch=1)

        # Stats panel below video with clear labels
        stats_widget = QWidget()
        stats_widget.setStyleSheet("padding: 5px;")
        # stats_widget.setMinimumWidth(800)  # Prevent excessive line breaks
        stats_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        stats_widget.setMinimumHeight(80)

        stats_layout = QGridLayout(stats_widget)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        stats_layout.setHorizontalSpacing(8)  # tighten horizontal gap between title and value
        stats_layout.setVerticalSpacing(3)

        row = 0

        # Camera
        title_camera = QLabel("<b>Camera:</b>")
        title_camera.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        stats_layout.addWidget(title_camera, row, 0, alignment=Qt.AlignTop)

        self.camera_stats_label = QLabel("Camera idle")
        self.camera_stats_label.setWordWrap(True)
        self.camera_stats_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        stats_layout.addWidget(self.camera_stats_label, row, 1, alignment=Qt.AlignTop)
        row += 1

        # DLC
        title_dlc = QLabel("<b>DLC Processor:</b>")
        title_dlc.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        stats_layout.addWidget(title_dlc, row, 0, alignment=Qt.AlignTop)

        self.dlc_stats_label = QLabel("DLC processor idle")
        self.dlc_stats_label.setWordWrap(True)
        self.dlc_stats_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        stats_layout.addWidget(self.dlc_stats_label, row, 1, alignment=Qt.AlignTop)
        row += 1

        # Recorder
        title_rec = QLabel("<b>Recorder:</b>")
        title_rec.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        stats_layout.addWidget(title_rec, row, 0, alignment=Qt.AlignTop)

        self.recording_stats_label = QLabel("Recorder idle")
        self.recording_stats_label.setWordWrap(True)
        self.recording_stats_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        stats_layout.addWidget(self.recording_stats_label, row, 1, alignment=Qt.AlignTop)

        # Critical: make column 1 (values) eat the width, keep column 0 tight
        stats_layout.setColumnStretch(0, 0)
        stats_layout.setColumnStretch(1, 1)
        video_layout.addWidget(stats_widget, stretch=0)

        # Allow user to select stats text
        for lbl in (self.camera_stats_label, self.dlc_stats_label, self.recording_stats_label):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

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
        self.stop_preview_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_preview_button.setEnabled(False)
        self.stop_preview_button.setMinimumWidth(150)
        button_bar.addWidget(self.preview_button)
        button_bar.addWidget(self.stop_preview_button)
        controls_layout.addWidget(button_bar_widget)
        controls_layout.addStretch(1)

        # Add controls and video panel to main layout
        layout.addWidget(controls_widget, stretch=0)
        layout.addWidget(video_panel, stretch=1)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._build_menus()
        QTimer.singleShot(0, self._show_logo_and_text)

    def _build_menus(self) -> None:
        # File menu
        file_menu = self.menuBar().addMenu("&File")

        ## Save/Load config
        self.load_config_action = QAction("Load configuration…", self)
        self.load_config_action.triggered.connect(self._action_load_config)
        file_menu.addAction(self.load_config_action)
        save_action = QAction("Save configuration", self)
        save_action.triggered.connect(self._action_save_config)
        file_menu.addAction(save_action)
        save_as_action = QAction("Save configuration as…", self)
        save_as_action.triggered.connect(self._action_save_config_as)
        file_menu.addAction(save_as_action)
        ## Open recording folder
        open_rec_folder_action = QAction("Open recording folder", self)
        open_rec_folder_action.triggered.connect(self._action_open_recording_folder)
        file_menu.addAction(open_rec_folder_action)
        ## Close
        file_menu.addSeparator()
        exit_action = QAction("Close window", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = self.menuBar().addMenu("&View")
        appearance_menu = view_menu.addMenu("Appearance")
        ## Style actions
        self.action_dark_mode = QAction("Dark theme", self, checkable=True)
        self.action_light_mode = QAction("System theme", self, checkable=True)
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        theme_group.addAction(self.action_dark_mode)
        theme_group.addAction(self.action_light_mode)
        self.action_dark_mode.triggered.connect(lambda: self._apply_theme(AppStyle.DARK))
        self.action_light_mode.triggered.connect(lambda: self._apply_theme(AppStyle.SYS_DEFAULT))

        appearance_menu.addAction(self.action_light_mode)
        appearance_menu.addAction(self.action_dark_mode)
        self._apply_theme(self._current_style)
        self._init_theme_actions()

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Camera settings")
        form = QFormLayout(group)

        # Camera config button - opens dialog for all camera configuration
        config_layout = QHBoxLayout()
        self.config_cameras_button = QPushButton("Configure Cameras...")
        self.config_cameras_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
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
        self.browse_model_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.browse_model_button.clicked.connect(self._action_browse_model)
        path_layout.addWidget(self.browse_model_button)
        form.addRow("Model file", path_layout)

        # Processor selection
        processor_path_layout = QHBoxLayout()
        self.processor_folder_edit = QLineEdit()
        self.processor_folder_edit.setText(default_processors_dir())
        processor_path_layout.addWidget(self.processor_folder_edit)

        self.browse_processor_folder_button = QPushButton("Browse...")
        self.browse_processor_folder_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.browse_processor_folder_button.clicked.connect(self._action_browse_processor_folder)
        processor_path_layout.addWidget(self.browse_processor_folder_button)

        self.refresh_processors_button = QPushButton("Refresh")
        self.refresh_processors_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_processors_button.clicked.connect(self._refresh_processors)
        processor_path_layout.addWidget(self.refresh_processors_button)
        form.addRow("Processor folder", processor_path_layout)

        self.processor_combo = QComboBox()
        self.processor_combo.addItem("No Processor", None)
        form.addRow("Processor", self.processor_combo)

        # self.additional_options_edit = QPlainTextEdit()
        # self.additional_options_edit.setPlaceholderText("")
        # self.additional_options_edit.setFixedHeight(40)
        # form.addRow("Additional options", self.additional_options_edit)
        self.dlc_camera_combo = QComboBox()
        self.dlc_camera_combo.setToolTip("Select which camera to use for pose inference")
        form.addRow("Inference Camera", self.dlc_camera_combo)

        # Wrap inference buttons in a widget to prevent shifting
        inference_button_widget = QWidget()
        inference_buttons = QHBoxLayout(inference_button_widget)
        inference_buttons.setContentsMargins(0, 0, 0, 0)
        self.start_inference_button = QPushButton("Start pose inference")
        self.start_inference_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self.start_inference_button.setEnabled(False)
        self.start_inference_button.setMinimumWidth(150)
        inference_buttons.addWidget(self.start_inference_button)
        self.stop_inference_button = QPushButton("Stop pose inference")
        self.stop_inference_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
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
        """Build recording controls group."""
        group = QGroupBox("Recording")
        form = QFormLayout(group)
        # Output directory selection
        dir_layout = QHBoxLayout()
        self.output_directory_edit = QLineEdit()
        dir_layout.addWidget(self.output_directory_edit)
        browse_dir = QPushButton("Browse…")
        browse_dir.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        browse_dir.clicked.connect(self._action_browse_directory)
        dir_layout.addWidget(browse_dir)
        form.addRow("Output directory", dir_layout)

        # Session + run name
        self.session_name_edit = QLineEdit()
        self.session_name_edit.setPlaceholderText("e.g. mouseA_day1")
        form.addRow("Session name", self.session_name_edit)

        self.use_timestamp_checkbox = QCheckBox("Use timestamp for run folder name")
        self.use_timestamp_checkbox.setChecked(True)
        self.use_timestamp_checkbox.setToolTip(
            "If checked, run folder will be run_YYYYMMDD_HHMMSS_mmm.\n"
            "If unchecked, run folder will be run_0001, run_0002, ..."
        )
        form.addRow("", self.use_timestamp_checkbox)

        # Show recording path preview

        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.recording_path_preview = ElidingPathLabel("")
        # No need to assign mouseReleaseEvent: the label handles click-to-copy internally
        form.addRow("Will save to", self.recording_path_preview)
        # self.recording_path_preview = QLabel("")
        # # Ensure it never gets squished vertically
        # self.recording_path_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # self.recording_path_preview.setWordWrap(False)
        # self.recording_path_preview.setCursor(Qt.PointingHandCursor)
        # self.recording_path_preview.setToolTip("")  # will show the preview path
        # self.recording_path_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # self.recording_path_preview.mouseReleaseEvent = self._copy_path_on_click
        # form.addRow("Will save to", self.recording_path_preview)

        self.filename_edit = QLineEdit()
        form.addRow("Filename", self.filename_edit)

        # Container + codec + CRF in a single row
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(8)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 3)
        grid.setColumnStretch(4, 0)
        grid.setColumnStretch(5, 2)

        ## Container
        container_label = QLabel("Container")
        container_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        grid.addWidget(container_label, 0, 0)

        self.container_combo = QComboBox()
        self.container_combo.setToolTip("Select the video container/format")
        self.container_combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.container_combo.setEditable(True)
        self.container_combo.addItems(["mp4", "avi", "mov"])
        # Ensure it never becomes unreadable:
        self.container_combo.setMinimumContentsLength(8)
        self.container_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        grid.addWidget(self.container_combo, 0, 1)

        ## Codec
        codec_label = QLabel("Codec")
        codec_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        grid.addWidget(codec_label, 0, 2)

        self.codec_combo = QComboBox()
        self.codec_combo.setToolTip("Select the video codec to use for recording")
        self.codec_combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        if os.sys.platform == "darwin":
            self.codec_combo.addItems(["h264_videotoolbox", "libx264", "hevc_videotoolbox"])
        else:
            self.codec_combo.addItems(["h264_nvenc", "libx264", "hevc_nvenc"])

        self.codec_combo.setCurrentText("libx264")
        # Optional: a modest minimum content length helps prevent jitter
        self.codec_combo.setMinimumContentsLength(6)
        self.codec_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        grid.addWidget(self.codec_combo, 0, 3)

        ## CRF
        crf_label = QLabel("CRF")
        crf_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        grid.addWidget(crf_label, 0, 4)

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)  # FFmpeg CRF range for x264/x265
        self.crf_spin.setValue(RecordingSettings().crf)
        self.crf_spin.setToolTip("Constant Rate Factor (0 = lossless, 51 = worst)")
        self.crf_spin.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        grid.addWidget(self.crf_spin, 0, 5)

        form.addRow(grid)

        # Record with overlays
        self.record_with_overlays_checkbox = QCheckBox("Record video with overlays")
        self.record_with_overlays_checkbox.setToolTip(
            "Enable to include pose overlays in recorded video (keypoints & bounding boxes)"
        )
        self.record_with_overlays_checkbox.setChecked(False)
        form.addRow(self.record_with_overlays_checkbox)

        # Wrap recording buttons in a widget to prevent shifting
        recording_button_widget = QWidget()
        buttons = QHBoxLayout(recording_button_widget)
        buttons.setContentsMargins(0, 0, 0, 0)
        self.start_record_button = QPushButton("Start recording")
        self.start_record_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogYesButton))
        self.start_record_button.setMinimumWidth(150)
        buttons.addWidget(self.start_record_button)
        self.stop_record_button = QPushButton("Stop recording")
        self.stop_record_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogNoButton))
        self.stop_record_button.setEnabled(False)
        self.stop_record_button.setMinimumWidth(150)
        buttons.addWidget(self.stop_record_button)
        form.addRow(recording_button_widget)

        # Add "Open folder" button
        self.open_rec_folder_button = QPushButton("Open recording folder")
        self.open_rec_folder_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_rec_folder_button.clicked.connect(self._action_open_recording_folder)
        form.addRow(self.open_rec_folder_button)

        return group

    def _build_bbox_group(self) -> QGroupBox:
        group = QGroupBox("Bounding Box Visualization")
        form = QFormLayout(group)

        row_widget = QWidget()
        checkbox_layout = QHBoxLayout(row_widget)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.bbox_enabled_checkbox = QCheckBox("Show bounding box")
        self.bbox_enabled_checkbox.setChecked(False)
        checkbox_layout.addWidget(self.bbox_enabled_checkbox)
        checkbox_layout.addWidget(QLabel("Color:"))

        self.bbox_color_combo = QComboBox()
        self._populate_bbox_color_combo_with_swatches()
        self.bbox_color_combo.setCurrentIndex(0)
        checkbox_layout.addWidget(self.bbox_color_combo)
        checkbox_layout.addStretch(1)
        form.addRow(row_widget)

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
        self.bbox_color_combo.currentIndexChanged.connect(self._on_bbox_color_changed)

        # Multi-camera controller signals (used for both single and multi-camera modes)
        self.multi_camera_controller.frame_ready.connect(self._on_multi_frame_ready)
        self.multi_camera_controller.all_started.connect(self._on_multi_camera_started)
        self.multi_camera_controller.all_stopped.connect(self._on_multi_camera_stopped)
        self.multi_camera_controller.camera_error.connect(self._on_multi_camera_error)
        self.multi_camera_controller.initialization_failed.connect(self._on_multi_camera_initialization_failed)

        # DLC processor signals
        self._dlc.pose_ready.connect(self._on_pose_ready)
        self._dlc.error.connect(self._on_dlc_error)
        self._dlc.initialized.connect(self._on_dlc_initialised)
        self.dlc_camera_combo.currentIndexChanged.connect(self._on_dlc_camera_changed)

        # Recording settings
        ## Session name persistence + preview updates
        if hasattr(self, "session_name_edit"):
            self.session_name_edit.editingFinished.connect(self._on_session_name_editing_finished)
        if hasattr(self, "use_timestamp_checkbox"):
            self.use_timestamp_checkbox.stateChanged.connect(self._on_use_timestamp_changed)
        if hasattr(self, "output_directory_edit"):
            self.output_directory_edit.textChanged.connect(lambda _t: self._update_recording_path_preview())
        if hasattr(self, "filename_edit"):
            self.filename_edit.textChanged.connect(lambda _t: self._update_recording_path_preview())
        if hasattr(self, "container_combo"):
            self.container_combo.currentTextChanged.connect(lambda _t: self._update_recording_path_preview())

    # ------------------------------------------------------------------
    # Config
    def _apply_config(self, config: ApplicationSettings) -> None:
        # Update active cameras label
        self._update_active_cameras_label()

        # Set DLC settings from config
        dlc = config.dlc
        resolved_model_path = self._model_path_store.resolve(dlc.model_path)
        self.model_path_edit.setText(resolved_model_path)

        # self.additional_options_edit.setPlainText(json.dumps(dlc.additional_options, indent=2))

        # Set recording settings from config
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
        ## Restore persisted session name if empty
        if hasattr(self, "session_name_edit"):
            if not self.session_name_edit.text().strip():
                persisted = self._settings_store.get_session_name()
                if persisted:
                    self.session_name_edit.setText(persisted)
        ## Restore "Use timestamp" checkbox state
        if hasattr(self, "use_timestamp_checkbox"):
            self.use_timestamp_checkbox.setChecked(self._settings_store.get_use_timestamp(default=True))

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
        if hasattr(self, "bbox_color_combo"):
            self._set_combo_from_color(self._bbox_color)

        # Update DLC camera list
        self._refresh_dlc_camera_list()

        # Update recording path preview
        self._update_recording_path_preview()

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
            model_type="pytorch",  # FIXME @C-Achard hardcoded for now, we should allow tf models too
            # additional_options=self._parse_json(self.additional_options_edit.toPlainText()),
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

    # ------------------------------------------------------------------
    # Actions
    def _action_load_config(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Load configuration", str(Path.home()), "JSON files (*.json)")
        if not file_name:
            return
        try:
            config = ApplicationSettings.load(file_name)
        except Exception as exc:  # pragma: no cover - GUI interaction
            self._show_error(str(exc))
            return
        self._settings_store.set_last_config_path(file_name)
        self._settings_store.save_full_config_snapshot(config)
        self._config = config
        self._config_path = Path(file_name)
        self._apply_config(config)
        self.statusBar().showMessage(f"Loaded configuration: {file_name}", 5000)
        # Validate cameras after loading
        self._validate_configured_cameras()

    def _action_save_config(self) -> None:
        if self._config_path is None:
            self._action_save_config_as()
            return
        self._save_config_to_path(self._config_path)

    def _action_save_config_as(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(self, "Save configuration", str(Path.home()), "JSON files (*.json)")
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
            self._settings_store.set_last_config_path(str(path))
            self._settings_store.save_full_config_snapshot(config)
        except Exception as exc:  # pragma: no cover - GUI interaction
            self._show_error(str(exc))
            return
        self.statusBar().showMessage(f"Saved configuration to {path}", 5000)

    def _action_browse_model(self) -> None:
        # Prefer persisted last-used directory, then config.dlc.model_directory, then home
        start_dir = self._model_path_store.suggest_start_dir(self._config.dlc.model_directory)
        preselect = self._model_path_store.suggest_selected_file()

        dlg = QFileDialog(self, "Select DLCLive model file")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters(
            [
                "Model files (*.pt *.pth *.pb)",
                "PyTorch models (*.pt *.pth)",
                "TensorFlow models (*.pb)",
                "All files (*.*)",
            ]
        )
        dlg.setDirectory(start_dir)

        # Preselect last used model if it exists (optional but nice)
        if preselect:
            dlg.selectFile(preselect)

        if dlg.exec():
            selected = dlg.selectedFiles()
            if not selected:
                return
            file_path = selected[0]
            self.model_path_edit.setText(file_path)

            # Persist model path + directory
            self._model_path_store.save_if_valid(file_path)

            # Optional: update config so next startup uses this directory too
            try:
                self._config.dlc.model_directory = str(Path(file_path).parent)
            except Exception:
                pass

    def _action_browse_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory", str(Path.home()))
        if directory:
            self.output_directory_edit.setText(directory)

    def _action_browse_processor_folder(self) -> None:
        """Browse for processor folder."""
        current_path = self.processor_folder_edit.text() or default_processors_dir()
        directory = QFileDialog.getExistingDirectory(self, "Select processor folder", current_path)
        if directory:
            self.processor_folder_edit.setText(directory)
            self._refresh_processors()

    def _action_open_recording_folder(self) -> None:
        """
        Open the recording folder in the system file explorer.
        Priority:
            1. If a run directory exists (during/after recording), open it.
            2. Else if the session directory exists, open it.
            3. Else if the base output directory exists, open it.
            4. Otherwise: show warning.
        """
        try:
            # 1. Real run directory if available (RecordingManager)
            run_dir = getattr(self._rec_manager, "run_dir", None)
            if run_dir and Path(run_dir).exists():
                target = Path(run_dir)
            else:
                # 2. Session folder
                out_dir = Path(self.output_directory_edit.text().strip()).expanduser()
                sess_name = self.session_name_edit.text().strip() or "session"
                sess_dir = out_dir / sess_name

                if sess_dir.exists():
                    target = sess_dir
                elif out_dir.exists():
                    target = out_dir
                else:
                    self.statusBar().showMessage("Recording folder does not exist yet.", 5000)
                    return

            # --- Use Qt's built-in cross-platform folder opener ---
            url = QUrl.fromLocalFile(str(target))
            ok = QDesktopServices.openUrl(url)

            if ok:
                self.statusBar().showMessage(f"Opened folder: {target}", 3000)
            else:
                self.statusBar().showMessage("Could not open folder.", 5000)

        except Exception as exc:
            logger.error(f"Failed to open folder: {exc}")
            self.statusBar().showMessage("Could not open recording folder.", 5000)

    def _refresh_processors(self) -> None:
        self.processor_combo.clear()
        self.processor_combo.addItem("No Processor", None)

        selected_folder = self.processor_folder_edit.text().strip()
        if Path(selected_folder).exists():
            self._scanned_processors = scan_processor_folder(selected_folder)
        else:
            self._scanned_processors = scan_processor_package("dlclivegui.processors")
        self._processor_keys = list(self._scanned_processors.keys())

        for key in self._processor_keys:
            info = self._scanned_processors[key]
            display_name = f"{info['name']} ({info['file']})"
            self.processor_combo.addItem(display_name, key)

        self.statusBar().showMessage(
            f"Found {len(self._processor_keys)} processor(s) in package dlclivegui.processors", 3000
        )

    # ------------------------------------------------------------------
    # Recording path preview and session name persistence
    def _on_session_name_editing_finished(self) -> None:
        name = self.session_name_edit.text().strip()
        self._settings_store.set_session_name(name)
        self._update_recording_path_preview()

    # def _update_recording_path_preview(self) -> None:
    #     """Update the label showing where files will go (best-effort)."""
    #     if not hasattr(self, "recording_path_preview"):
    #         return
    #     out_dir = self.output_directory_edit.text().strip()
    #     sess = self.session_name_edit.text().strip() if hasattr(self, "session_name_edit") else ""
    #     base = self.filename_edit.text().strip()
    #     container = self.container_combo.currentText().strip() if hasattr(self, "container_combo") else "mp4"
    #     use_ts = self.use_timestamp_checkbox.isChecked() if hasattr(self, "use_timestamp_checkbox") else True

    #     # Preview is approximate (since run index/time is decided at start).
    #     sess_safe = sess.strip() or "session"
    #     run_hint = "run_<timestamp>" if use_ts else "run_<next>"
    #     stem_hint = Path(base).stem if base.strip() else "recording"  # shows user-provided stem or default
    #     full_hint = str(Path(out_dir).expanduser() / sess_safe / run_hint / f"{stem_hint}_<camera>.{container}")
    #     self.recording_path_preview.setText(f"<span style='color: gray;'>{full_hint}</span>")
    #     self.recording_path_preview.setToolTip(
    #         f"<b>Click to copy to clipboard :</b><br>{full_hint.replace('<camera>', '*')}"
    #     )

    def _update_recording_path_preview(self) -> None:
        """Update the label showing where files will go (best-effort)."""
        if not hasattr(self, "recording_path_preview"):
            return

        out_dir = self.output_directory_edit.text().strip()
        sess = self.session_name_edit.text().strip() if hasattr(self, "session_name_edit") else ""
        base = self.filename_edit.text().strip()
        container = self.container_combo.currentText().strip() if hasattr(self, "container_combo") else "mp4"
        use_ts = self.use_timestamp_checkbox.isChecked() if hasattr(self, "use_timestamp_checkbox") else True

        # Preview is approximate (since run index/time is decided at start).
        sess_safe = sess or "session"
        run_hint = "run_<timestamp>" if use_ts else "run_<next>"
        stem_hint = Path(base).stem if base else "recording"
        full_hint = str(Path(out_dir).expanduser() / sess_safe / run_hint / f"{stem_hint}_<camera>.{container}")

        self.recording_path_preview.set_full_text(full_hint)

    # def _copy_path_on_click(self, event):
    #     if event.button() == Qt.LeftButton:
    #         # Clear all HTML tags to get the raw path before copying
    #         path = self.recording_path_preview.text()
    #         path = path.replace("<span style='color: gray;'>", "").replace("</span>", "")
    #         clean_path = path.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    #         QGuiApplication.clipboard().setText(clean_path)
    #         QToolTip.showText(QCursor.pos(), "Copied path", self.recording_path_preview)

    def _on_use_timestamp_changed(self, _state: int) -> None:
        self._settings_store.set_use_timestamp(self.use_timestamp_checkbox.isChecked())
        self._update_recording_path_preview()

    def _on_bbox_color_changed(self, _index: int) -> None:
        enum_item = self.bbox_color_combo.currentData()
        if enum_item is None:
            return
        self._bbox_color = enum_item.value
        if self._current_frame is not None:
            self._display_frame(self._current_frame, force=True)

    # ------------------------------------------------------------------
    # Multi-camera
    def _open_camera_config_dialog(self) -> None:
        """Open the camera configuration dialog (non-modal, async inside)."""
        if self.multi_camera_controller.is_running():
            self._show_warning("Stop the main preview before configuring cameras.")
            return

        if self._cam_dialog is None:
            self._cam_dialog = CameraConfigDialog(self, self._config.multi_camera)
            self._cam_dialog.settings_changed.connect(self._on_multi_camera_settings_changed)
        else:
            # Refresh its UI from current settings when reopened
            self._cam_dialog._populate_from_settings()
            self._cam_dialog.dlc_camera_id = self._inference_camera_id

        self._cam_dialog.show()
        self._cam_dialog.raise_()
        self._cam_dialog.activateWindow()

    def _on_multi_camera_settings_changed(self, settings: MultiCameraSettings) -> None:
        """Handle changes to multi-camera settings."""
        self._config.multi_camera = settings
        self._update_active_cameras_label()
        self._refresh_dlc_camera_list()
        active_count = len(settings.get_active_cameras())
        self.statusBar().showMessage(f"Camera configuration updated: {active_count} active camera(s)", 3000)

    def _update_active_cameras_label(self) -> None:
        """Update the label showing active cameras."""
        active_cams = self._config.multi_camera.get_active_cameras()
        if not active_cams:
            self.active_cameras_label.setText("No cameras configured")
        elif len(active_cams) == 1:
            cam = active_cams[0]
            self.active_cameras_label.setText(f"{cam.name} [{cam.backend}:{cam.index}] @ {cam.fps:.1f} fps")
        else:
            cam_names = [f"{c.name}" for c in active_cams]
            self.active_cameras_label.setText(f"{len(active_cams)} cameras: {', '.join(cam_names)}")

    def _validate_configured_cameras(self) -> None:
        """Validate that configured cameras are available.

        Disables unavailable cameras and shows a warning dialog.
        """
        if getattr(self._cam_dialog, "_dialog_active", False):
            # Skip validation if camera config dialog is open
            return

        active_cams = self._config.multi_camera.get_active_cameras()
        if not active_cams:
            return

        unavailable: list[tuple[str, str, CameraSettings]] = []
        for cam in active_cams:
            cam_id = f"{cam.backend}:{cam.index}"
            available, error = CameraFactory.check_camera_available(cam)
            if not available:
                unavailable.append((cam.name or cam_id, error, cam))

        if unavailable:
            # Disable unavailable cameras
            for _, _, cam in unavailable:
                cam.enabled = False

            # Update the active cameras label
            self._update_active_cameras_label()

            # Build warning message
            error_lines = ["The following camera(s) are not available and have been disabled:"]
            for cam_name, error_msg, _ in unavailable:
                error_lines.append(f"  • {cam_name}: {error_msg}")
            error_lines.append("")
            error_lines.append("Please check camera connections or re-enable in camera settings.")
            self._show_warning("\n".join(error_lines))
            logger.warning("\n".join(error_lines))

    def _label_for_cam_id(self, cam_id: str) -> str:
        for cam in self._config.multi_camera.get_active_cameras():
            if get_camera_id(cam) == cam_id:
                return f"{cam.name} [{cam.backend}:{cam.index}]"
        return cam_id

    def _refresh_dlc_camera_list_running(self) -> None:
        """Populate the inference camera dropdown from currently running cameras."""
        self.dlc_camera_combo.blockSignals(True)
        self.dlc_camera_combo.clear()
        for cam_id in sorted(self._running_cams_ids):
            self.dlc_camera_combo.addItem(self._label_for_cam_id(cam_id), cam_id)

        # Keep current selection if still present, else select first running
        if self._inference_camera_id in self._running_cams_ids:
            idx = self.dlc_camera_combo.findData(self._inference_camera_id)
            if idx >= 0:
                self.dlc_camera_combo.setCurrentIndex(idx)
        elif self.dlc_camera_combo.count() > 0:
            self.dlc_camera_combo.setCurrentIndex(0)
            self._inference_camera_id = self.dlc_camera_combo.currentData()
        self.dlc_camera_combo.blockSignals(False)

    def _set_dlc_combo_to_id(self, cam_id: str) -> None:
        """Update combo selection to a given ID without firing signals."""
        self.dlc_camera_combo.blockSignals(True)
        idx = self.dlc_camera_combo.findData(cam_id)
        if idx >= 0:
            self.dlc_camera_combo.setCurrentIndex(idx)
        self.dlc_camera_combo.blockSignals(False)

    def _refresh_dlc_camera_list(self) -> None:
        """Populate the inference camera dropdown from active cameras."""
        self.dlc_camera_combo.blockSignals(True)
        self.dlc_camera_combo.clear()

        active_cams = self._config.multi_camera.get_active_cameras()
        for cam in active_cams:
            cam_id = get_camera_id(cam)  # e.g., "opencv:0" or "pylon:1"
            label = f"{cam.name} [{cam.backend}:{cam.index}]"
            self.dlc_camera_combo.addItem(label, cam_id)

        # Keep previous selection if still present, else default to first
        if self._inference_camera_id is not None:
            idx = self.dlc_camera_combo.findData(self._inference_camera_id)
            if idx >= 0:
                self.dlc_camera_combo.setCurrentIndex(idx)
            elif self.dlc_camera_combo.count() > 0:
                self.dlc_camera_combo.setCurrentIndex(0)
                self._inference_camera_id = self.dlc_camera_combo.currentData()
        else:
            if self.dlc_camera_combo.count() > 0:
                self.dlc_camera_combo.setCurrentIndex(0)
                self._inference_camera_id = self.dlc_camera_combo.currentData()

        self.dlc_camera_combo.blockSignals(False)

    def _on_dlc_camera_changed(self, _index: int) -> None:
        """Track user selection of the inference camera."""
        self._inference_camera_id = self.dlc_camera_combo.currentData()
        # Force redraw so bbox/pose overlays switch to the new tile immediately
        if self._current_frame is not None:
            self._display_frame(self._current_frame, force=True)

    # ------------------------------------------------------------------
    # Multi-camera event handlers
    def _render_overlays_for_recording(self, cam_id, frame):
        # Copy so we don't affect GUI preview pipeline
        output = frame.copy()
        offset, scale = (0, 0), (1.0, 1.0)

        # If this is the inference camera, apply pose overlays
        if cam_id == self._inference_camera_id and self._last_pose and self._last_pose.pose is not None:
            output = draw_pose(
                output,
                self._last_pose.pose,
                p_cutoff=self._p_cutoff,
                colormap=self._colormap,
                offset=offset,
                scale=scale,
            )
        if self._bbox_enabled:
            output = draw_bbox(
                frame=output,
                bbox_xyxy=(self._bbox_x0, self._bbox_y0, self._bbox_x1, self._bbox_y1),
                color_bgr=self._bbox_color,
                offset=offset,
                scale=scale,
            )
        return output

    def _on_multi_frame_ready(self, frame_data: MultiFrameData) -> None:
        """Handle frames from multiple cameras.

        Priority order for performance:
        1. DLC processing (highest priority - enqueue immediately, only for DLC camera)
        2. Recording (queued writes, non-blocking)
        3. Display (lowest priority - tiled and updated on separate timer)
        """
        self._multi_camera_frames = frame_data.frames
        src_id = frame_data.source_camera_id
        if src_id:
            self._fps_tracker.note_frame(src_id)  # Track FPS

        new_running = set(frame_data.frames.keys())
        if new_running != self._running_cams_ids:
            self._running_cams_ids = new_running
            self._refresh_dlc_camera_list_running()

        # Determine DLC camera (first active camera)
        selected_id = self._inference_camera_id
        available_ids = sorted(frame_data.frames.keys())
        if selected_id in frame_data.frames:
            dlc_cam_id = selected_id
        else:
            dlc_cam_id = available_ids[0] if available_ids else ""
            if dlc_cam_id is not None:
                self._inference_camera_id = dlc_cam_id
                self._set_dlc_combo_to_id(dlc_cam_id)
                self.statusBar().showMessage(
                    f"DLC inference camera changed to {self._label_for_cam_id(dlc_cam_id)}", 3000
                )
            else:  # No more cameras available
                if self._dlc_active:
                    self._stop_inference(show_message=True)
                self._display_dirty = True
                return

        # Check if this frame is from the DLC camera
        is_dlc_camera_frame = frame_data.source_camera_id == dlc_cam_id

        # Update tile info and raw frame only when DLC camera frame arrives
        if is_dlc_camera_frame and dlc_cam_id in frame_data.frames:
            frame = frame_data.frames[dlc_cam_id]
            self._raw_frame = frame
            self._dlc_tile_offset, self._dlc_tile_scale = compute_tile_info(dlc_cam_id, frame, frame_data.frames)

        # PRIORITY 1: DLC processing - only enqueue when DLC camera frame arrives!
        if self._dlc_active and is_dlc_camera_frame and dlc_cam_id in frame_data.frames:
            frame = frame_data.frames[dlc_cam_id]
            timestamp = frame_data.timestamps.get(dlc_cam_id, time.time())
            self._dlc.enqueue_frame(frame, timestamp)

        # PRIORITY 2: Recording (queued, non-blocking)
        if self._rec_manager.is_active and src_id in frame_data.frames:
            frame = frame_data.frames[src_id]

            if self.record_with_overlays_checkbox.isChecked():
                # Draw overlays for recording
                frame = self._render_overlays_for_recording(src_id, frame)

            ts = frame_data.timestamps.get(src_id, time.time())
            self._rec_manager.write_frame(src_id, frame, ts)

        # PRIORITY 3: Mark display dirty (tiling done in display timer)
        self._display_dirty = True

    def _on_multi_camera_started(self) -> None:
        """Handle all cameras started event."""
        self.preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(True)
        active_count = self.multi_camera_controller.get_active_count()
        self.statusBar().showMessage(f"Multi-camera preview started: {active_count} camera(s)", 5000)
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
        self._show_warning(f"Camera {camera_id} error: {message}\nRecording stopped.")
        self._refresh_dlc_camera_list_running()
        if self.dlc_camera_combo.count() <= 1:
            self._stop_inference()  # We now gracefully switch DLC camera if needed, but if none left, stop inference
        self._stop_recording()

    def _on_multi_camera_initialization_failed(self, failures: list) -> None:
        """Handle complete failure to initialize cameras."""
        # Build error message with details for each failed camera
        error_lines = ["Failed to initialize camera(s):"]
        for camera_id, error_msg in failures:
            error_lines.append(f"  • {camera_id}: {error_msg}")
        error_lines.append("")
        error_lines.append("Please check that the required camera backend is installed.")

        error_message = "\n".join(error_lines)
        self._show_error(error_message)
        logger.error(error_message)

    def _start_multi_camera_recording(self) -> None:
        """Start recording from all active cameras."""
        recording = self._recording_settings_from_ui()
        active_cams = self._config.multi_camera.get_active_cameras()
        if not active_cams:
            self._show_error("No active cameras to record from.")
            return

        session_name = self.session_name_edit.text().strip() if hasattr(self, "session_name_edit") else ""
        use_ts = self.use_timestamp_checkbox.isChecked() if hasattr(self, "use_timestamp_checkbox") else True

        run_dir = self._rec_manager.start_all(
            recording,
            active_cams,
            self._multi_camera_frames,
            session_name=session_name,
            use_timestamp=use_ts,
            all_or_nothing=False,
        )
        if run_dir is None:
            self._show_error("Failed to start recording.")
            return

        self._settings_store.set_session_name(session_name)
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.statusBar().showMessage(f"Recording {len(active_cams)} camera(s) to {run_dir}", 5000)
        self._update_camera_controls_enabled()

    def _stop_multi_camera_recording(self) -> None:
        if not self._rec_manager.is_active:
            return
        self._rec_manager.stop_all()
        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.statusBar().showMessage("Multi-camera recording stopped", 3000)
        self._update_camera_controls_enabled()

    # ------------------------------------------------------------------
    # Camera control
    def _show_logo_and_text(self):
        """Show the transparent logo with text below it in the preview area when not running."""

        size = self.video_label.size()

        if size.width() <= 0 or size.height() <= 0:
            return

        # Prepare blank canvas (transparent)
        composed = QPixmap(size)
        composed.fill(Qt.transparent)

        painter = QPainter(composed)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- Scale logo to at most 50% height (nice proportion) ---
        max_logo_height = int(size.height() * 0.45)
        logo = self._preview_pixmap.scaledToHeight(max_logo_height, Qt.SmoothTransformation)

        # Center the logo horizontally
        logo_x = (size.width() - logo.width()) // 2
        logo_y = int(size.height() * 0.15)  # small top margin

        painter.drawPixmap(logo_x, logo_y, logo)

        # --- Draw text BELOW the logo ---
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 22, QFont.Bold))

        text = "DeepLabCut-Live! "
        try:
            version = importlib.metadata.version("dlclivegui")
        except Exception:
            version = ""
        if version:
            text += f"\n(v{version})"

        # Position text under the logo with a small gap
        text_rect = QRect(
            0,
            logo_y + logo.height() + 15,  # 15px gap under logo
            size.width(),
            size.height() - (logo_y + logo.height() + 15),
        )

        painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, text)

        painter.end()
        self.video_label.setPixmap(composed)

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
        self._fps_tracker.clear()
        self._last_display_time = 0.0

        if hasattr(self, "camera_stats_label"):
            self.camera_stats_label.setText(f"Starting {len(active_cams)} camera(s)…")
        self.statusBar().showMessage(f"Starting preview ({len(active_cams)} camera(s))…", 3000)

        # Store active settings for single camera mode (for DLC, recording frame rate, etc.)
        self._active_camera_settings = active_cams[0] if active_cams else None
        for cam in active_cams:
            cam.properties.setdefault("fast_start", True)

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
        self._fps_tracker.clear()
        self._last_display_time = 0.0
        if hasattr(self, "camera_stats_label"):
            self.camera_stats_label.setText("Camera idle")
        # self._show_logo_and_text()

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
                logger.error(error_msg)
                return False

        self._dlc.configure(settings, processor=processor)
        self._model_path_store.save_if_valid(settings.model_path)
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
            # self.additional_options_edit,
            self.dlc_camera_combo,
        ]
        for widget in widgets:
            widget.setEnabled(allow_changes)

    def _update_camera_controls_enabled(self) -> None:
        multi_cam_recording = self._rec_manager.is_active

        # Check if preview is running
        preview_running = self.multi_camera_controller.is_running()

        allow_changes = not preview_running and not self._dlc_active and not multi_cam_recording

        # Recording settings (codec, crf) should be editable when not recording
        recording_editable = not multi_cam_recording
        self.codec_combo.setEnabled(recording_editable)
        self.crf_spin.setEnabled(recording_editable)

        # Config cameras button should be available when not in preview/recording
        self.config_cameras_button.setEnabled(allow_changes)

        # Disable loading configurations when preview/recording is active
        if hasattr(self, "load_config_action"):
            self.load_config_action.setEnabled(allow_changes)

    def _display_frame(self, frame: np.ndarray, *, force: bool = False) -> None:
        if frame is None:
            return
        now = time.perf_counter()
        if not force and (now - self._last_display_time) < self._display_interval:
            return
        self._last_display_time = now
        self._update_video_display(frame)

    def _update_display_from_pending(self) -> None:
        """Update display from pending frames (called by display timer)."""
        if not self._display_dirty:
            return
        if not self._multi_camera_frames:
            return

        self._display_dirty = False

        # Create tiled frame on demand (moved from camera thread for performance)
        tiled = create_tiled_frame(self._multi_camera_frames)
        if tiled is not None:
            self._current_frame = tiled
            self._update_video_display(tiled)

    def _update_metrics(self) -> None:
        # --- Camera stats ---
        if hasattr(self, "camera_stats_label"):
            running = self.multi_camera_controller.is_running()
            if running:
                active_count = self.multi_camera_controller.get_active_count()

                # Build per-camera FPS list for active cameras only
                active_cams = self._config.multi_camera.get_active_cameras()
                lines = []
                for cam in active_cams:
                    cam_id = get_camera_id(cam)  # e.g., "opencv:0" or "pylon:1"
                    fps = self._fps_tracker.fps(cam_id)
                    # Make a compact label: name [backend:index] @ fps
                    label = f"{cam.name or cam_id} [{cam.backend}:{cam.index}]"
                    if fps > 0:
                        lines.append(f"{label} @ {fps:.1f} fps")
                    else:
                        lines.append(f"{label} @ Measuring…")

                if active_count == 1:
                    # Single camera: show just the line
                    summary = lines[0] if lines else "Measuring…"
                else:
                    # Multi camera: join lines with separator
                    summary = " | ".join(lines)

                self.camera_stats_label.setText(summary)
            else:
                self.camera_stats_label.setText("Camera idle")

        # --- DLC processor stats ---
        if hasattr(self, "dlc_stats_label"):
            if self._dlc_active and self._dlc_initialized:
                stats = self._dlc.get_stats()
                summary = format_dlc_stats(stats)
                self.dlc_stats_label.setText(summary)
            else:
                self.dlc_stats_label.setText("DLC processor idle")

        # Update processor status (connection and recording state)
        if hasattr(self, "processor_status_label"):
            self._update_processor_status()

        # --- Recorder stats ---
        if hasattr(self, "recording_stats_label"):
            if self._rec_manager.is_active:
                summary = self._rec_manager.get_stats_summary()
                self._last_recorder_summary = summary
                self.recording_stats_label.setText(summary)
            else:
                self.recording_stats_label.setText(self._last_recorder_summary)

    def _update_processor_status(self) -> None:
        """Update processor connection and recording status, handle auto-recording."""
        if not self._dlc_active or not self._dlc_initialized:
            self.processor_status_label.setText("Processor: Not active")
            return

        # Get processor instance from _dlc
        processor = self._dlc._processor

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
                    if not self._rec_manager.is_active:
                        # Get session name from processor
                        session_name = getattr(processor, "session_name", "auto_session")
                        self._auto_record_session_name = session_name

                        # Processor overrides session name field + persist it
                        self.session_name_edit.setText(session_name)
                        self._settings_store.set_session_name(session_name)

                        # Optional: set base filename to session name (readable stable filenames)
                        self.filename_edit.setText(session_name)
                        self._update_recording_path_preview()

                        self._start_recording()
                        self.statusBar().showMessage(f"Auto-started recording: {session_name}", 3000)
                        logger.info(f"Auto-recording started for session: {session_name}")
                else:
                    # Stop video recording
                    if self._rec_manager.is_active:
                        self._stop_recording()
                        self.statusBar().showMessage("Auto-stopped recording", 3000)
                        logger.info("Auto-recording stopped")

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
        self._dlc.reset()
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
        self._dlc.reset()
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
        # logger.debug(f"Pose result: {result.pose}, Timestamp: {result.timestamp}")
        if self._current_frame is not None:
            self._display_frame(self._current_frame, force=True)

    def _on_dlc_error(self, message: str) -> None:
        self._stop_inference(show_message=False)
        self._show_error(message)

    def _update_video_display(self, frame: np.ndarray) -> None:
        display_frame = frame

        if self.show_predictions_checkbox.isChecked() and self._last_pose and self._last_pose.pose is not None:
            display_frame = draw_pose(
                frame,
                self._last_pose.pose,
                p_cutoff=self._p_cutoff,
                colormap=self._colormap,
                offset=self._dlc_tile_offset,
                scale=self._dlc_tile_scale,
            )

        if self._bbox_enabled:
            display_frame = draw_bbox(
                display_frame,
                (self._bbox_x0, self._bbox_y0, self._bbox_x1, self._bbox_y1),
                color_bgr=self._bbox_color,
                offset=self._dlc_tile_offset,
                scale=self._dlc_tile_scale,
            )

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Scale pixmap to fit label while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

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

    # ------------------------------------------------------------------
    # Helpers
    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        QMessageBox.critical(self, "Error", message)

    def _show_warning(self, message: str) -> None:
        """Display a warning message dialog."""
        self.statusBar().showMessage(f"⚠ {message}", 5000)
        QMessageBox.warning(self, "Warning", message)

    def _show_info(self, message: str) -> None:
        """Display an informational message dialog."""
        self.statusBar().showMessage(message, 5000)
        QMessageBox.information(self, "Information", message)

    def _populate_bbox_color_combo_with_swatches(self):
        self.bbox_color_combo.clear()
        for enum_item in BBoxColors:
            bgr = enum_item.value
            name = enum_item.name.title()
            pix = QPixmap(40, 16)
            pix.fill(Qt.transparent)
            p = QPainter(pix)
            p.fillRect(0, 0, 40, 16, Qt.black)  # border/background
            p.fillRect(1, 1, 38, 14, Qt.white)  # inner bg
            # Convert BGR to RGB for QPainter/QColor
            rgb = (bgr[2], bgr[1], bgr[0])
            p.fillRect(2, 2, 36, 12, QColor(*rgb))
            p.end()
            self.bbox_color_combo.addItem(QIcon(pix), name, enum_item)

    def _set_combo_from_color(self, bgr: tuple[int, int, int]) -> None:
        # Find combo entry whose enum value matches bgr
        for i in range(self.bbox_color_combo.count()):
            enum_item = self.bbox_color_combo.itemData(i)
            if enum_item is not None and getattr(enum_item, "value", None) == bgr:
                self.bbox_color_combo.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------
    # Qt overrides
    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - GUI behaviour
        if self.multi_camera_controller.is_running():
            self.multi_camera_controller.stop(wait=True)

        # Stop all multi-camera recorders
        self._rec_manager.stop_all()

        # Close the camera dialog if open (ensures its worker thread is canceled)
        if getattr(self, "_cam_dialog", None) is not None and self._cam_dialog.isVisible():
            try:
                self._cam_dialog.close()
            except Exception:
                pass
            self._cam_dialog = None

        self._dlc.shutdown()
        if hasattr(self, "_metrics_timer"):
            self._metrics_timer.stop()

        # Remember model path on exit
        self._model_path_store.save_if_valid(self.model_path_edit.text().strip())

        # Close the window
        super().closeEvent(event)
