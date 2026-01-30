# config_models.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from dlclivegui.config import (
    ApplicationSettings,
    BoundingBoxSettings,
    CameraSettings,
    DLCProcessorSettings,
    MultiCameraSettings,
    RecordingSettings,
    VisualizationSettings,
)

Backend = Literal["gentl", "opencv", "basler", "aravis"]  # extend as needed
Rotation = Literal[0, 90, 180, 270]
TileLayout = Literal["auto", "2x2", "1x4", "4x1"]
Precision = Literal["FP32", "FP16"]


class CameraSettingsModel(BaseModel):
    name: str = "Camera 0"
    index: int = 0
    fps: float = 25.0
    backend: Backend = "gentl"
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


class RecordingSettingsModel(BaseModel):
    enabled: bool = False
    directory: str = Field(default_factory=lambda: str(Path.home() / "Videos" / "deeplabcut-live"))
    filename: str = "session.mp4"
    container: Literal["mp4", "avi", "mov"] = "mp4"
    codec: str = "libx264"
    crf: int = Field(default=23, ge=0, le=51)


class ApplicationSettingsModel(BaseModel):
    # optional: add a semantic version for migrations
    version: int = 1
    camera: CameraSettingsModel = Field(default_factory=CameraSettingsModel)  # kept for backward compat
    multi_camera: MultiCameraSettingsModel = Field(default_factory=MultiCameraSettingsModel)
    dlc: DLCProcessorSettingsModel = Field(default_factory=DLCProcessorSettingsModel)
    recording: RecordingSettingsModel = Field(default_factory=RecordingSettingsModel)
    bbox: BoundingBoxSettingsModel = Field(default_factory=BoundingBoxSettingsModel)
    visualization: VisualizationSettingsModel = Field(default_factory=VisualizationSettingsModel)


def dc_to_model(dc_cfg: ApplicationSettings) -> ApplicationSettingsModel:
    # Use your current dc.to_dict() then validate; preserves defaults + coercion
    return ApplicationSettingsModel.model_validate(dc_cfg.to_dict())


def model_to_dc(model: ApplicationSettingsModel) -> ApplicationSettings:
    # Build dataclasses from validated data
    cam_dc = CameraSettings(**model.camera.model_dump())
    mc_dc = MultiCameraSettings.from_dict(model.multi_camera.model_dump())
    dlc_dc = DLCProcessorSettings(**model.dlc.model_dump())
    rec_dc = RecordingSettings(**model.recording.model_dump())
    bbox_dc = BoundingBoxSettings(**model.bbox.model_dump())
    viz_dc = VisualizationSettings(**model.visualization.model_dump())

    return ApplicationSettings(
        camera=cam_dc,
        multi_camera=mc_dc,
        dlc=dlc_dc,
        recording=rec_dc,
        bbox=bbox_dc,
        visualization=viz_dc,
    )
