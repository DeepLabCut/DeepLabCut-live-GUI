"""Configuration helpers for the DLC Live GUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from dlclivegui.utils.utils import is_model_file


@dataclass
class CameraSettings:
    """Configuration for a single camera device."""

    name: str = "Camera 0"
    index: int = 0
    fps: float = 25.0
    backend: str = "gentl"
    exposure: int = 500  # 0 = auto, otherwise microseconds
    gain: float = 10  # 0.0 = auto, otherwise gain value
    crop_x0: int = 0  # Left edge of crop region (0 = no crop)
    crop_y0: int = 0  # Top edge of crop region (0 = no crop)
    crop_x1: int = 0  # Right edge of crop region (0 = no crop)
    crop_y1: int = 0  # Bottom edge of crop region (0 = no crop)
    max_devices: int = 3  # Maximum number of devices to probe during detection
    rotation: int = 0  # Rotation degrees (0, 90, 180, 270)
    enabled: bool = True  # Whether this camera is active in multi-camera mode
    properties: dict[str, Any] = field(default_factory=dict)

    def apply_defaults(self) -> CameraSettings:
        """Ensure fps is a positive number and validate crop settings."""

        self.fps = float(self.fps) if self.fps else 30.0
        self.exposure = int(self.exposure) if self.exposure else 0
        self.gain = float(self.gain) if self.gain else 0.0
        self.crop_x0 = max(0, int(self.crop_x0)) if hasattr(self, "crop_x0") else 0
        self.crop_y0 = max(0, int(self.crop_y0)) if hasattr(self, "crop_y0") else 0
        self.crop_x1 = max(0, int(self.crop_x1)) if hasattr(self, "crop_x1") else 0
        self.crop_y1 = max(0, int(self.crop_y1)) if hasattr(self, "crop_y1") else 0
        return self

    def get_crop_region(self) -> tuple[int, int, int, int] | None:
        """Get crop region as (x0, y0, x1, y1) or None if no cropping."""
        if self.crop_x0 == 0 and self.crop_y0 == 0 and self.crop_x1 == 0 and self.crop_y1 == 0:
            return None
        return (self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1)

    def copy(self) -> CameraSettings:
        """Create a copy of this settings object."""
        return CameraSettings(
            name=self.name,
            index=self.index,
            fps=self.fps,
            backend=self.backend,
            exposure=self.exposure,
            gain=self.gain,
            crop_x0=self.crop_x0,
            crop_y0=self.crop_y0,
            crop_x1=self.crop_x1,
            crop_y1=self.crop_y1,
            max_devices=self.max_devices,
            rotation=self.rotation,
            enabled=self.enabled,
            properties=dict(self.properties),
        )


@dataclass
class MultiCameraSettings:
    """Configuration for multiple cameras."""

    cameras: list = field(default_factory=list)  # List of CameraSettings
    max_cameras: int = 4  # Maximum number of cameras that can be active
    tile_layout: str = "auto"  # "auto", "2x2", "1x4", "4x1"

    def get_active_cameras(self) -> list:
        """Get list of enabled cameras."""
        return [cam for cam in self.cameras if cam.enabled]

    def add_camera(self, settings: CameraSettings) -> bool:
        """Add a camera to the configuration. Returns True if successful."""
        if len(self.get_active_cameras()) >= self.max_cameras and settings.enabled:
            return False
        self.cameras.append(settings)
        return True

    def remove_camera(self, index: int) -> bool:
        """Remove camera at the given list index."""
        if 0 <= index < len(self.cameras):
            del self.cameras[index]
            return True
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiCameraSettings:
        """Create MultiCameraSettings from a dictionary."""
        cameras = []
        for cam_data in data.get("cameras", []):
            cam = CameraSettings(**cam_data)
            cam.apply_defaults()
            cameras.append(cam)
        return cls(
            cameras=cameras,
            max_cameras=data.get("max_cameras", 4),
            tile_layout=data.get("tile_layout", "auto"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cameras": [asdict(cam) for cam in self.cameras],
            "max_cameras": self.max_cameras,
            "tile_layout": self.tile_layout,
        }


@dataclass
class DLCProcessorSettings:
    """Configuration for DLCLive processing."""

    model_path: str = ""
    model_directory: str = "."  # Default directory for model browser (current dir if not set)
    device: str | None = (
        "auto"  # Device for inference (e.g., "cuda:0", "cpu"). None should be auto, but might default to cpu
    )
    dynamic: tuple = (False, 0.5, 10)  # Dynamic cropping: (enabled, margin, max_missing_frames)
    resize: float = 1.0  # Resize factor for input frames
    precision: str = "FP32"  # Inference precision ("FP32", "FP16")
    additional_options: dict[str, Any] = field(default_factory=dict)
    model_type: str = "pytorch"  # Only PyTorch models are supported
    single_animal: bool = True  # Only single-animal models are supported


@dataclass
class BoundingBoxSettings:
    """Configuration for bounding box visualization."""

    enabled: bool = False
    x0: int = 0
    y0: int = 0
    x1: int = 200
    y1: int = 100


@dataclass
class VisualizationSettings:
    """Configuration for pose visualization."""

    p_cutoff: float = 0.6  # Confidence threshold for displaying keypoints
    colormap: str = "hot"  # Matplotlib colormap for keypoints
    bbox_color: tuple[int, int, int] = (0, 0, 255)  # BGR color for bounding box (default: red)

    def get_bbox_color_bgr(self) -> tuple[int, int, int]:
        """Get bounding box color in BGR format."""
        if isinstance(self.bbox_color, (list, tuple)) and len(self.bbox_color) == 3:
            return tuple(int(c) for c in self.bbox_color)
        return (0, 0, 255)  # Default to red


@dataclass
class RecordingSettings:
    """Configuration for video recording."""

    enabled: bool = False
    directory: str = str(Path.home() / "Videos" / "deeplabcut-live")
    filename: str = "session.mp4"
    container: str = "mp4"
    codec: str = "libx264"
    crf: int = 23

    def output_path(self) -> Path:
        """Return the absolute output path for recordings."""

        directory = Path(self.directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        name = Path(self.filename)
        if name.suffix:
            filename = name
        else:
            filename = name.with_suffix(f".{self.container}")
        return directory / filename

    def writegear_options(self, fps: float) -> dict[str, Any]:
        """Return compression parameters for WriteGear."""

        fps_value = float(fps) if fps else 30.0
        codec_value = (self.codec or "libx264").strip() or "libx264"
        crf_value = int(self.crf) if self.crf is not None else 23
        return {
            "-input_framerate": f"{fps_value:.6f}",
            "-vcodec": codec_value,
            "-crf": str(crf_value),
        }


@dataclass
class ApplicationSettings:
    """Top level application configuration."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    multi_camera: MultiCameraSettings = field(default_factory=MultiCameraSettings)
    dlc: DLCProcessorSettings = field(default_factory=DLCProcessorSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    bbox: BoundingBoxSettings = field(default_factory=BoundingBoxSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApplicationSettings:
        """Create an :class:`ApplicationSettings` from a dictionary."""

        camera = CameraSettings(**data.get("camera", {})).apply_defaults()

        # Parse multi-camera settings
        multi_camera_data = data.get("multi_camera", {})
        if multi_camera_data:
            multi_camera = MultiCameraSettings.from_dict(multi_camera_data)
        else:
            multi_camera = MultiCameraSettings()

        dlc_data = dict(data.get("dlc", {}))
        # Parse dynamic parameter - can be list or tuple in JSON
        dynamic_raw = dlc_data.get("dynamic", [False, 0.5, 10])
        if isinstance(dynamic_raw, (list, tuple)) and len(dynamic_raw) == 3:
            dynamic = tuple(dynamic_raw)
        else:
            dynamic = (False, 0.5, 10)
        dlc = DLCProcessorSettings(
            model_path=str(dlc_data.get("model_path", "")),
            model_directory=str(dlc_data.get("model_directory", ".")),
            device=dlc_data.get("device"),  # None if not specified
            dynamic=dynamic,
            resize=float(dlc_data.get("resize", 1.0)),
            precision=str(dlc_data.get("precision", "FP32")),
            additional_options=dict(dlc_data.get("additional_options", {})),
        )
        recording_data = dict(data.get("recording", {}))
        recording_data.pop("options", None)
        recording = RecordingSettings(**recording_data)
        bbox = BoundingBoxSettings(**data.get("bbox", {}))
        visualization = VisualizationSettings(**data.get("visualization", {}))
        return cls(
            camera=camera,
            multi_camera=multi_camera,
            dlc=dlc,
            recording=recording,
            bbox=bbox,
            visualization=visualization,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration to a dictionary."""

        return {
            "camera": asdict(self.camera),
            "multi_camera": self.multi_camera.to_dict(),
            "dlc": asdict(self.dlc),
            "recording": asdict(self.recording),
            "bbox": asdict(self.bbox),
            "visualization": asdict(self.visualization),
        }

    @classmethod
    def load(cls, path: Path | str) -> ApplicationSettings:
        """Load configuration from ``path``."""

        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def save(self, path: Path | str) -> None:
        """Persist configuration to ``path``."""

        file_path = Path(path).expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


DEFAULT_CONFIG = ApplicationSettings()


class ModelPathStore:
    """Persist and resolve the last model path via QSettings."""

    def __init__(self, settings: QSettings | None = None):
        self._settings = settings or QSettings("DeepLabCut", "DLCLiveGUI")

    def load_last(self) -> str | None:
        val = self._settings.value("dlc/last_model_path")
        if not val:
            return None
        path = str(val)
        try:
            return path if is_model_file(path) else None
        except Exception:
            return None

    def save_if_valid(self, path: str) -> None:
        try:
            if path and is_model_file(path):
                self._settings.setValue("dlc/last_model_path", str(Path(path)))
        except Exception:
            pass

    def resolve(self, config_path: str | None) -> str:
        if config_path and is_model_file(config_path):
            return config_path
        persisted = self.load_last()
        if persisted and is_model_file(persisted):
            return persisted
        return ""
