"""Configuration helpers for the DLC Live GUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


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
    properties: Dict[str, Any] = field(default_factory=dict)

    def apply_defaults(self) -> "CameraSettings":
        """Ensure fps is a positive number and validate crop settings."""

        self.fps = float(self.fps) if self.fps else 30.0
        self.exposure = int(self.exposure) if self.exposure else 0
        self.gain = float(self.gain) if self.gain else 0.0
        self.crop_x0 = max(0, int(self.crop_x0)) if hasattr(self, "crop_x0") else 0
        self.crop_y0 = max(0, int(self.crop_y0)) if hasattr(self, "crop_y0") else 0
        self.crop_x1 = max(0, int(self.crop_x1)) if hasattr(self, "crop_x1") else 0
        self.crop_y1 = max(0, int(self.crop_y1)) if hasattr(self, "crop_y1") else 0
        return self

    def get_crop_region(self) -> Optional[tuple[int, int, int, int]]:
        """Get crop region as (x0, y0, x1, y1) or None if no cropping."""
        if self.crop_x0 == 0 and self.crop_y0 == 0 and self.crop_x1 == 0 and self.crop_y1 == 0:
            return None
        return (self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1)


@dataclass
class DLCProcessorSettings:
    """Configuration for DLCLive processing."""

    model_path: str = ""
    model_directory: str = "."  # Default directory for model browser (current dir if not set)
    device: Optional[str] = None  # Device for inference (e.g., "cuda:0", "cpu"). None = auto
    dynamic: tuple = (False, 0.5, 10)  # Dynamic cropping: (enabled, margin, max_missing_frames)
    resize: float = 1.0  # Resize factor for input frames
    precision: str = "FP32"  # Inference precision ("FP32", "FP16")
    additional_options: Dict[str, Any] = field(default_factory=dict)
    model_type: str = "pytorch"  # Only PyTorch models are supported


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

    def writegear_options(self, fps: float) -> Dict[str, Any]:
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
    dlc: DLCProcessorSettings = field(default_factory=DLCProcessorSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    bbox: BoundingBoxSettings = field(default_factory=BoundingBoxSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicationSettings":
        """Create an :class:`ApplicationSettings` from a dictionary."""

        camera = CameraSettings(**data.get("camera", {})).apply_defaults()
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
            camera=camera, dlc=dlc, recording=recording, bbox=bbox, visualization=visualization
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a dictionary."""

        return {
            "camera": asdict(self.camera),
            "dlc": asdict(self.dlc),
            "recording": asdict(self.recording),
            "bbox": asdict(self.bbox),
            "visualization": asdict(self.visualization),
        }

    @classmethod
    def load(cls, path: Path | str) -> "ApplicationSettings":
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
