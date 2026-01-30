# config_models.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Rotation = Literal[0, 90, 180, 270]
TileLayout = Literal["auto", "2x2", "1x4", "4x1"]
Precision = Literal["FP32", "FP16"]


class CameraSettingsModel(BaseModel):
    name: str = "Camera 0"
    index: int = 0
    fps: float = 25.0
    backend: str = "opencv"
    exposure: int = 500  # 0=auto else µs
    gain: float = 10.0  # 0.0=auto else value
    crop_x0: int = 0
    crop_y0: int = 0
    crop_x1: int = 0
    crop_y1: int = 0
    max_devices: int = 3
    rotation: Rotation = 0
    enabled: bool = True
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fps")
    @classmethod
    def _fps_positive(cls, v):
        return float(v) if v and v > 0 else 30.0

    @field_validator("exposure")
    @classmethod
    def _coerce_exposure(cls, v):  # allow None->0 and int
        return int(v) if v is not None else 0

    @field_validator("gain")
    @classmethod
    def _coerce_gain(cls, v):
        return float(v) if v is not None else 0.0

    @model_validator(mode="after")
    def _validate_crop(self):
        for f in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"):
            setattr(self, f, max(0, int(getattr(self, f))))
        # Optional: if any crop is set, enforce x1>x0 and y1>y0
        if any([self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1]):
            if not (self.crop_x1 > self.crop_x0 and self.crop_y1 > self.crop_y0):
                raise ValueError("Invalid crop rectangle: require x1>x0 and y1>y0 when cropping is enabled.")
        return self

    def get_crop_region(self) -> tuple[int, int, int, int] | None:
        if self.crop_x0 == self.crop_y0 == self.crop_x1 == self.crop_y1 == 0:
            return None
        return (self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraSettingsModel:
        return cls(**data)

    @classmethod
    def from_defaults(cls) -> CameraSettingsModel:
        return cls()

    def apply_defaults(self) -> CameraSettingsModel:
        default = self.from_defaults()
        for field in CameraSettingsModel.model_fields:
            if getattr(self, field) in (None, 0, 0.0):
                setattr(self, field, getattr(default, field))
        return self


class MultiCameraSettingsModel(BaseModel):
    cameras: list[CameraSettingsModel] = Field(default_factory=list)
    max_cameras: int = 4
    tile_layout: TileLayout = "auto"

    def get_active_cameras(self) -> list[CameraSettingsModel]:
        return [c for c in self.cameras if c.enabled]

    @model_validator(mode="after")
    def _enforce_max_active(self):
        if len(self.get_active_cameras()) > self.max_cameras:
            raise ValueError("Number of enabled cameras exceeds max_cameras.")
        return self

    def add_camera(self, camera: CameraSettingsModel) -> bool:
        """Add a new camera if under max_cameras limit."""
        if len(self.cameras) >= self.max_cameras:
            return False
        self.cameras.append(camera)
        return True

    def remove_camera(self, index: int) -> bool:
        """Remove camera at given index."""
        if 0 <= index < len(self.cameras):
            del self.cameras[index]
            return True
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiCameraSettingsModel:
        cameras_data = data.get("cameras", [])
        cameras = [CameraSettingsModel(**cam) for cam in cameras_data]
        max_cameras = data.get("max_cameras", 4)
        tile_layout = data.get("tile_layout", "auto")
        return cls(cameras=cameras, max_cameras=max_cameras, tile_layout=tile_layout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameras": [cam.model_dump() for cam in self.cameras],
            "max_cameras": self.max_cameras,
            "tile_layout": self.tile_layout,
        }


class DynamicCropModel(BaseModel):
    enabled: bool = False
    margin: float = Field(default=0.5, ge=0.0, le=1.0)
    max_missing_frames: int = Field(default=10, ge=0)

    @classmethod
    def from_tupleish(cls, v):
        # Accept (enabled, margin, max_missing_frames)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return cls(enabled=bool(v[0]), margin=float(v[1]), max_missing_frames=int(v[2]))
        if isinstance(v, dict):
            return cls(**v)
        if isinstance(v, cls):
            return v
        return cls()


class DLCProcessorSettingsModel(BaseModel):
    model_path: str = ""
    model_directory: str = "."
    device: str | None = "auto"  # "cuda:0", "cpu", or None
    dynamic: DynamicCropModel = Field(default_factory=DynamicCropModel)
    resize: float = Field(default=1.0, gt=0)
    precision: Precision = "FP32"
    additional_options: dict[str, Any] = Field(default_factory=dict)
    model_type: Literal["pytorch"] = "pytorch"
    single_animal: bool = True

    @field_validator("dynamic", mode="before")
    @classmethod
    def _coerce_dynamic(cls, v):
        return DynamicCropModel.from_tupleish(v)


class BoundingBoxSettingsModel(BaseModel):
    enabled: bool = False
    x0: int = 0
    y0: int = 0
    x1: int = 200
    y1: int = 100

    @model_validator(mode="after")
    def _bbox_logic(self):
        if self.enabled and not (self.x1 > self.x0 and self.y1 > self.y0):
            raise ValueError("Bounding box enabled but coordinates are invalid (x1>x0 and y1>y0 required).")
        return self


class VisualizationSettingsModel(BaseModel):
    p_cutoff: float = Field(default=0.6, ge=0.0, le=1.0)
    colormap: str = "hot"
    bbox_color: tuple[int, int, int] = (0, 0, 255)

    def get_bbox_color_bgr(self) -> tuple[int, int, int]:
        """Get bounding box color in BGR format"""
        if isinstance(self.bbox_color, (list, tuple)) and len(self.bbox_color) == 3:
            return tuple(int(c) for c in self.bbox_color)
        return (0, 0, 255)  # default red


class RecordingSettingsModel(BaseModel):
    enabled: bool = False
    directory: str = Field(default_factory=lambda: str(Path.home() / "Videos" / "deeplabcut-live"))
    filename: str = "session.mp4"
    container: Literal["mp4", "avi", "mov"] = "mp4"
    codec: str = "libx264"
    crf: int = Field(default=23, ge=0, le=51)

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


class ApplicationSettingsModel(BaseModel):
    # optional: add a semantic version for migrations
    version: int = 1
    camera: CameraSettingsModel = Field(default_factory=CameraSettingsModel)  # kept for backward compat
    multi_camera: MultiCameraSettingsModel = Field(default_factory=MultiCameraSettingsModel)
    dlc: DLCProcessorSettingsModel = Field(default_factory=DLCProcessorSettingsModel)
    recording: RecordingSettingsModel = Field(default_factory=RecordingSettingsModel)
    bbox: BoundingBoxSettingsModel = Field(default_factory=BoundingBoxSettingsModel)
    visualization: VisualizationSettingsModel = Field(default_factory=VisualizationSettingsModel)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApplicationSettingsModel:
        camera_data = data.get("camera", {})
        multi_camera_data = data.get("multi_camera", {})
        dlc_data = data.get("dlc", {})
        recording_data = data.get("recording", {})
        bbox_data = data.get("bbox", {})
        visualization_data = data.get("visualization", {})

        camera = CameraSettingsModel(**camera_data)
        multi_camera = MultiCameraSettingsModel.from_dict(multi_camera_data)
        dlc = DLCProcessorSettingsModel(**dlc_data)
        recording = RecordingSettingsModel(**recording_data)
        bbox = BoundingBoxSettingsModel(**bbox_data)
        visualization = VisualizationSettingsModel(**visualization_data)

        return cls(
            camera=camera,
            multi_camera=multi_camera,
            dlc=dlc,
            recording=recording,
            bbox=bbox,
            visualization=visualization,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "camera": self.camera.model_dump(),
            "multi_camera": self.multi_camera.to_dict(),
            "dlc": self.dlc.model_dump(),
            "recording": self.recording.model_dump(),
            "bbox": self.bbox.model_dump(),
            "visualization": self.visualization.model_dump(),
        }

    @classmethod
    def load(cls, path: Path | str) -> ApplicationSettingsModel:
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


DEFAULT_CONFIG = ApplicationSettingsModel()
