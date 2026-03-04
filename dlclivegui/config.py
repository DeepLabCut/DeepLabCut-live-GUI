# dlclivegui/config.py
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Rotation = Literal[0, 90, 180, 270]
TileLayout = Literal["auto", "2x2", "1x4", "4x1"]
Precision = Literal["FP32", "FP16"]
ModelType = Literal["pytorch", "tensorflow"]
ModelBackendType = Literal["dlc", "poet"]
BGR = tuple[int, int, int]  # (B, G, R) color format


class SkeletonColorMode(str, Enum):
    SOLID = "solid"
    GRADIENT_KEYPOINTS = "gradient_keypoints"  # use endpoint keypoint colors


class SkeletonStyle(BaseModel):
    mode: SkeletonColorMode = SkeletonColorMode.SOLID
    color: BGR = (0, 255, 255)  # default if SOLID
    thickness: int = 2  # base thickness in pixels
    gradient_steps: int = 16  # segments per edge when gradient
    scale_with_zoom: bool = True  # scale thickness with (sx, sy)

    def effective_thickness(self, sx: float, sy: float) -> int:
        if not self.scale_with_zoom:
            return max(1, int(self.thickness))
        return max(1, int(round(self.thickness * min(sx, sy))))


class CameraSettings(BaseModel):
    name: str = "Camera 0"
    index: int = 0
    backend: str = "opencv"

    # 0.0 = Auto (device default / don't request)
    fps: float = 0.0
    # 0 = Auto (device default / don't request)
    width: int = 0
    height: int = 0

    exposure: int = 0  # 0=auto else µs
    gain: float = 0.0  # 0.0=auto else value

    crop_x0: int = 0
    crop_y0: int = 0
    crop_x1: int = 0
    crop_y1: int = 0

    max_devices: int = 3
    rotation: Rotation = 0
    enabled: bool = True
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fps", mode="before")
    @classmethod
    def _coerce_fps(cls, v):
        """
        Accept:
          - None -> 0.0 (Auto)
          - 0 / 0.0 -> Auto
          - >0 -> requested fps
        """
        if v is None:
            return 0.0
        try:
            fv = float(v)
        except Exception:
            return 0.0
        # clamp negatives to Auto
        return fv if fv >= 0.0 else 0.0

    @field_validator("width", "height", mode="before")
    @classmethod
    def _coerce_resolution(cls, v):
        """
        Accept:
          - None -> 0 (Auto)
          - 0 -> Auto
          - >0 -> requested dimension
        """
        if v is None:
            return 0
        try:
            iv = int(v)
        except Exception:
            return 0
        return iv if iv >= 0 else 0

    @field_validator("exposure", mode="before")
    @classmethod
    def _coerce_exposure(cls, v):  # allow None->0 and int
        return int(v) if v is not None else 0

    @field_validator("gain", mode="before")
    @classmethod
    def _coerce_gain(cls, v):
        return float(v) if v is not None else 0.0

    @model_validator(mode="after")
    def _validate_crop(self):
        for f in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"):
            setattr(self, f, max(0, int(getattr(self, f))))

        # No crop
        if self.crop_x0 == self.crop_y0 == self.crop_x1 == self.crop_y1 == 0:
            return self

        # Allow x1/y1 == 0 to mean "to edge"
        # If x1 is explicitly set (>0), it must be > x0
        if self.crop_x1 > 0 and self.crop_x1 <= self.crop_x0:
            raise ValueError("Invalid crop rectangle: require x1 > x0 (or x1=0 for 'to edge').")

        # If y1 is explicitly set (>0), it must be > y0
        if self.crop_y1 > 0 and self.crop_y1 <= self.crop_y0:
            raise ValueError("Invalid crop rectangle: require y1 > y0 (or y1=0 for 'to edge').")

        return self

    def get_crop_region(self) -> tuple[int, int, int, int] | None:
        if self.crop_x0 == self.crop_y0 == self.crop_x1 == self.crop_y1 == 0:
            return None
        return (self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraSettings:
        return cls(**data)

    @classmethod
    def from_defaults(cls) -> CameraSettings:
        return cls()

    def apply_defaults(self) -> CameraSettings:
        """
        IMPORTANT:
        0 means "Auto" for fps/width/height/exposure/gain.
        So do NOT treat <=0 as "missing" for those fields.
        Only fill in defaults when the value is None.
        """
        default = self.from_defaults()

        # Fields where 0 is meaningful ("Auto"), so we must not replace 0 with defaults.
        auto_zero_fields = {"fps", "width", "height", "exposure", "gain"}

        for field in CameraSettings.model_fields:
            value = getattr(self, field)

            # Only replace None with defaults universally
            if value is None:
                setattr(self, field, getattr(default, field))
                continue

            # Careful: crop uses 0 legitimately too, though default is also 0
            if field not in auto_zero_fields and isinstance(value, (int, float)) and value < 0:
                setattr(self, field, getattr(default, field))

        return self

    @staticmethod
    def check_diff(old: CameraSettings, new: CameraSettings) -> dict:
        keys = (
            "width",
            "height",
            "fps",
            "exposure",
            "gain",
            "rotation",
            "crop_x0",
            "crop_y0",
            "crop_x1",
            "crop_y1",
            "enabled",
        )
        out = {}
        for k in keys:
            try:
                ov = getattr(old, k, None)
                nv = getattr(new, k, None)
                if ov != nv:
                    out[k] = (ov, nv)
            except Exception:
                pass
        return out


class MultiCameraSettings(BaseModel):
    cameras: list[CameraSettings] = Field(default_factory=list)
    max_cameras: int = 4
    tile_layout: TileLayout = "auto"

    def get_active_cameras(self) -> list[CameraSettings]:
        return [c for c in self.cameras if c.enabled]

    @model_validator(mode="after")
    def _enforce_max_active(self):
        if len(self.get_active_cameras()) > self.max_cameras:
            raise ValueError("Number of enabled cameras exceeds max_cameras.")
        return self

    def add_camera(self, camera: CameraSettings) -> bool:
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
    def from_dict(cls, data: dict[str, Any]) -> MultiCameraSettings:
        cameras_data = data.get("cameras", [])
        cameras = [CameraSettings(**cam) for cam in cameras_data]
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

    def to_tuple(self) -> tuple[bool, float, int]:
        return (self.enabled, self.margin, self.max_missing_frames)


class DLCProcessorSettings(BaseModel):
    model_path: str = ""
    model_directory: str = "."
    device: str | None = "auto"  # "cuda:0", "cpu", or None
    dynamic: DynamicCropModel = Field(default_factory=DynamicCropModel)
    resize: float = Field(default=1.0, gt=0)
    precision: Precision = "FP32"
    additional_options: dict[str, Any] = Field(default_factory=dict)
    model_type: ModelType = "pytorch"
    single_animal: bool = True

    @field_validator("dynamic", mode="before")
    @classmethod
    def _coerce_dynamic(cls, v):
        return DynamicCropModel.from_tupleish(v)

    @field_validator("model_type", mode="before")
    @classmethod
    def _coerce_model_type(cls, v):
        """
        Accept:
          - "pytorch"/"tensorflow"/etc as strings
          - Enum instances (e.g. Engine.PYTORCH) and store their .value
        Always return a lowercase string.
        """
        if v is None or v == "":
            return "pytorch"

        # If caller passed Engine enum or any Enum, use its value
        if isinstance(v, Enum):
            v = v.value

        # If caller passed something with a `.value` attribute (defensive)
        if not isinstance(v, str) and hasattr(v, "value"):
            v = v.value

        if not isinstance(v, str):
            raise TypeError(f"model_type must be a string or Enum, got {type(v)!r}")

        v = v.strip().lower()

        # Optional: enforce allowed values
        allowed = {"pytorch", "tensorflow"}
        if v not in allowed:
            raise ValueError(f"Unknown model type: {v!r}. Allowed: {sorted(allowed)}")

        return v


class BoundingBoxSettings(BaseModel):
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


class VisualizationSettings(BaseModel):
    p_cutoff: float = Field(default=0.6, ge=0.0, le=1.0)
    colormap: str = "hot"
    bbox_color: tuple[int, int, int] = (0, 0, 255)

    show_pose: bool = True
    show_skeleton: bool = False
    skeleton_style: SkeletonStyle = Field(default_factory=SkeletonStyle)

    def get_bbox_color_bgr(self) -> tuple[int, int, int]:
        """Get bounding box color in BGR format"""
        if isinstance(self.bbox_color, (list, tuple)) and len(self.bbox_color) == 3:
            return tuple(int(c) for c in self.bbox_color)
        return (0, 0, 255)  # default red

    def get_skeleton_color_bgr(self) -> tuple[int, int, int]:
        c = self.skeleton_style.color
        if isinstance(c, (list, tuple)) and len(c) == 3:
            return tuple(int(v) for v in c)
        return (0, 255, 255)  # default yellow


class RecordingSettings(BaseModel):
    enabled: bool = False
    directory: str = Field(default_factory=lambda: str(Path.home() / "Videos" / "deeplabcut-live"))
    filename: str = "session.mp4"
    container: Literal["mp4", "avi", "mov"] = "mp4"
    codec: str = "libx264"
    crf: int = Field(default=23, ge=0, le=51)
    with_overlays: bool = False

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


class ApplicationSettings(BaseModel):
    # optional: add a semantic version for migrations
    version: int = 1
    camera: CameraSettings = Field(default_factory=CameraSettings)  # kept for backward compat
    multi_camera: MultiCameraSettings = Field(default_factory=MultiCameraSettings)
    dlc: DLCProcessorSettings = Field(default_factory=DLCProcessorSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    bbox: BoundingBoxSettings = Field(default_factory=BoundingBoxSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    model_backend: ModelBackendType = "dlc"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApplicationSettings:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

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
