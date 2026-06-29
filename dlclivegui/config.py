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
TriggerRole = Literal["off", "external", "master", "follower"]
TriggerActivation = Literal["RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"]
TriggerStrobePolarity = Literal["ActiveHigh", "ActiveLow"]
TriggerStrobeOperation = Literal["Exposure", "FixedDuration"]

# Global settings
## GUI
GUI_MAX_DISPLAY_FPS: float = 30.0


## Debug
### Timing logs
SINGLE_CAMERA_WORKER_DO_LOG_TIMING: bool = False
MULTI_CAMERA_WORKER_DO_LOG_TIMING: bool = False
REC_DO_LOG_TIMING: bool = False
# MAIN_WINDOW_DO_LOG_TIMING: bool = False
#### Backends
BASLER_DO_LOG_TIMING: bool = False


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
    preserve_mono: bool = False  # if True, preserve mono images as mono (not BGR) when reading

    crop_x0: int = 0
    crop_y0: int = 0
    crop_x1: int = 0
    crop_y1: int = 0

    max_devices: int = 3
    rotation: Rotation = 0
    enabled: bool = True
    properties: dict[str, Any] = Field(default_factory=dict)

    def pretty(self) -> str:
        crop = (
            "none"
            if self.get_crop_region() is None
            else f"({self.crop_x0}, {self.crop_y0}) -> ({self.crop_x1 or 'edge'}, {self.crop_y1 or 'edge'})"
        )
        return (
            f"CameraSettings[\n"
            f"  name={self.name!r}, index={self.index}, backend={self.backend!r}, enabled={self.enabled}\n"
            f"  fps={self.fps}, size={self.width or 'auto'}x{self.height or 'auto'}, "
            f"exposure={self.exposure or 'auto'}, gain={self.gain or 'auto'}\n"
            f"  rotation={self.rotation}, crop={crop}\n"
            f"  preserve_mono={self.preserve_mono}, max_devices={self.max_devices}\n"
            f"]"
        )

    def __str__(self) -> str:
        return self.pretty()

    def __repr__(self) -> str:
        return self.pretty()

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

    def backend_options(self, backend: str | None = None) -> dict[str, Any]:
        key = backend or self.backend
        props = self.properties if isinstance(self.properties, dict) else {}
        ns = props.get(str(key).lower(), {})
        return ns if isinstance(ns, dict) else {}

    def get_trigger_settings(self, backend: str | None = None) -> CameraTriggerSettings:
        ns = self.backend_options(backend)
        return CameraTriggerSettings.from_any(ns.get("trigger"))

    def set_trigger_settings(self, trigger: CameraTriggerSettings, backend: str | None = None) -> None:
        key = backend or self.backend
        if not isinstance(self.properties, dict):
            self.properties = {}
        ns = self.properties.setdefault(str(key).lower(), {})
        if not isinstance(ns, dict):
            ns = {}
            self.properties[str(key).lower()] = ns
        ns["trigger"] = trigger.to_properties()

    def with_save_defaults(self) -> CameraSettings:
        out = self.model_copy(deep=True)

        backend = (out.backend or "").lower()
        if backend != "gentl":
            return out

        if not isinstance(out.properties, dict):
            out.properties = {}

        ns = out.properties.setdefault("gentl", {})
        if not isinstance(ns, dict):
            ns = {}
            out.properties["gentl"] = ns

        ns.setdefault("trigger", CameraTriggerSettings().to_properties())

        return out


class CameraTriggerSettings(BaseModel):
    """
    Generic hardware-trigger settings.

    Backend-specific code may ignore fields that are unsupported by a given
    camera/SDK.

    For GenTL/TIS DMK 37BUX287:
      - follower/external maps mainly to TriggerMode, TriggerSelector,
        TriggerActivation. TriggerSource may be read-only and is best-effort.
      - master output maps primarily to StrobeEnable, StrobePolarity,
        StrobeOperation, StrobeDuration, and StrobeDelay.
    """

    role: TriggerRole = "off"

    # Input trigger config: external/follower
    selector: str = "FrameStart"
    source: str = "auto"
    activation: TriggerActivation | str = "RisingEdge"

    # Generic/SFNC output config: master fallback for cameras exposing Line* nodes.
    output_line: str = "Line2"
    output_source: str = "ExposureActive"

    # Strobe output config: master path for TIS/DMK 37U cameras.
    strobe_polarity: TriggerStrobePolarity | str = "ActiveHigh"
    strobe_operation: TriggerStrobeOperation | str = "Exposure"
    strobe_duration: int | None = None  # µs, used when strobe_operation=FixedDuration
    strobe_delay: int | None = None  # µs

    # Runtime behavior
    timeout: float | None = None
    strict: bool = False

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, v):
        if v is None:
            return "off"

        s = str(v).strip().lower()
        aliases = {
            "": "off",
            "none": "off",
            "false": "off",
            "disabled": "off",
            "disable": "off",
            "off": "off",
            "true": "external",
            "on": "external",
            "trigger": "external",
            "triggered": "external",
            "external": "external",
            "follower": "follower",
            "slave": "follower",
            "master": "master",
            "main": "master",
        }
        return aliases.get(s, s)

    @field_validator("timeout", mode="before")
    @classmethod
    def _coerce_timeout(cls, v):
        if v in (None, ""):
            return None
        try:
            fv = float(v)
        except Exception:
            return None
        return fv if fv > 0 else None

    @field_validator("strobe_duration", "strobe_delay", mode="before")
    @classmethod
    def _coerce_optional_nonnegative_int(cls, v):
        if v in (None, ""):
            return None
        try:
            iv = int(float(v))
        except Exception:
            return None
        return iv if iv >= 0 else None

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, v):
        if v is None:
            return "auto"

        s = str(v).strip()
        if not s:
            return "auto"

        aliases = {
            "default": "auto",
            "automatic": "auto",
            "device": "auto",
            "camera": "auto",
        }
        return aliases.get(s.lower(), s)

    @classmethod
    def from_any(cls, value) -> CameraTriggerSettings:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return cls()

    def to_properties(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


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
        out = self.with_save_defaults()
        return {
            "cameras": [cam.model_dump() for cam in out.cameras],
            "max_cameras": out.max_cameras,
            "tile_layout": out.tile_layout,
        }

    def with_save_defaults(self) -> MultiCameraSettings:
        """Return a copy with save defaults applied to all cameras."""
        out = self.model_copy(deep=True)
        out.cameras = [cam.with_save_defaults() for cam in out.cameras]
        return out


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

    def get_bbox_color_bgr(self) -> tuple[int, int, int]:
        """Get bounding box color in BGR format"""
        if isinstance(self.bbox_color, (list, tuple)) and len(self.bbox_color) == 3:
            return tuple(int(c) for c in self.bbox_color)
        return (0, 0, 255)  # default red


class RecordingSettings(BaseModel):
    enabled: bool = False
    directory: str = Field(default_factory=lambda: str(Path.home() / "Videos" / "deeplabcut-live"))
    filename: str = "session.mp4"
    container: Literal["mp4", "avi", "mov"] = "mp4"
    codec: str = "libx264"
    crf: int = Field(default=23, ge=0, le=51)
    fast_encoding: bool = False

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

    def writegear_options(self, fps: float | None) -> dict[str, Any]:
        """Return FFmpeg/WriteGear compression parameters.

        The default settings prioritize compatibility and compression quality. If
        ``fast_encoding`` is enabled, additional low-latency encoder options are
        added for codecs that are known to support them.

        Args:
            fps: Desired input frame rate. If missing or non-positive, falls back
                to 30 FPS.

        Returns:
            Dictionary of WriteGear/FFmpeg options.
        """
        try:
            fps_value = float(fps or 0.0)
        except Exception:
            fps_value = 0.0
        if fps_value <= 0.0:
            fps_value = 30.0

        codec_value = (self.codec or "libx264").strip() or "libx264"
        crf_value = int(self.crf) if self.crf is not None else 23

        opts: dict[str, Any] = {
            "-input_framerate": f"{fps_value:.6f}",
            "-vcodec": codec_value,
            "-crf": str(crf_value),
        }

        if self.fast_encoding:
            if codec_value in {"libx264", "libx265"}:
                opts.update(
                    {
                        "-preset": "ultrafast",
                        "-tune": "zerolatency",
                    }
                )

        return opts


class ApplicationSettings(BaseModel):
    # optional: add a semantic version for migrations
    version: int = 1
    camera: CameraSettings = Field(default_factory=CameraSettings)  # kept for backward compat
    multi_camera: MultiCameraSettings = Field(default_factory=MultiCameraSettings)
    dlc: DLCProcessorSettings = Field(default_factory=DLCProcessorSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    bbox: BoundingBoxSettings = Field(default_factory=BoundingBoxSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApplicationSettings:
        camera_data = data.get("camera", {})
        multi_camera_data = data.get("multi_camera", {})
        dlc_data = data.get("dlc", {})
        recording_data = data.get("recording", {})
        bbox_data = data.get("bbox", {})
        visualization_data = data.get("visualization", {})

        camera = CameraSettings(**camera_data)
        multi_camera = MultiCameraSettings.from_dict(multi_camera_data)
        dlc = DLCProcessorSettings(**dlc_data)
        recording = RecordingSettings(**recording_data)
        bbox = BoundingBoxSettings(**bbox_data)
        visualization = VisualizationSettings(**visualization_data)

        return cls(
            camera=camera,
            multi_camera=multi_camera,
            dlc=dlc,
            recording=recording,
            bbox=bbox,
            visualization=visualization,
        )

    def to_dict(self) -> dict[str, Any]:
        camera = self.camera.with_save_defaults()
        multi_camera = self.multi_camera.with_save_defaults()

        return {
            "version": self.version,
            "camera": camera.model_dump(),
            "multi_camera": multi_camera.to_dict(),
            "dlc": self.dlc.model_dump(),
            "recording": self.recording.model_dump(),
            "bbox": self.bbox.model_dump(),
            "visualization": self.visualization.model_dump(),
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
