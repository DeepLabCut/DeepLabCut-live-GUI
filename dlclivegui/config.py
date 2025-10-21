"""Configuration helpers for the DLC Live GUI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class CameraSettings:
    """Configuration for a single camera device."""

    name: str = "Camera 0"
    index: int = 0
    width: int = 640
    height: int = 480
    fps: float = 30.0
    backend: str = "opencv"
    properties: Dict[str, Any] = field(default_factory=dict)

    def apply_defaults(self) -> "CameraSettings":
        """Ensure width, height and fps are positive numbers."""

        self.width = int(self.width) if self.width else 640
        self.height = int(self.height) if self.height else 480
        self.fps = float(self.fps) if self.fps else 30.0
        return self


@dataclass
class DLCProcessorSettings:
    """Configuration for DLCLive processing."""

    model_path: str = ""
    shuffle: Optional[int] = None
    trainingsetindex: Optional[int] = None
    processor: str = "cpu"
    processor_args: Dict[str, Any] = field(default_factory=dict)
    additional_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordingSettings:
    """Configuration for video recording."""

    enabled: bool = False
    directory: str = str(Path.home() / "Videos" / "deeplabcut-live")
    filename: str = "session.mp4"
    container: str = "mp4"
    options: Dict[str, Any] = field(default_factory=dict)

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


@dataclass
class ApplicationSettings:
    """Top level application configuration."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    dlc: DLCProcessorSettings = field(default_factory=DLCProcessorSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicationSettings":
        """Create an :class:`ApplicationSettings` from a dictionary."""

        camera = CameraSettings(**data.get("camera", {})).apply_defaults()
        dlc = DLCProcessorSettings(**data.get("dlc", {}))
        recording = RecordingSettings(**data.get("recording", {}))
        return cls(camera=camera, dlc=dlc, recording=recording)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a dictionary."""

        return {
            "camera": asdict(self.camera),
            "dlc": asdict(self.dlc),
            "recording": asdict(self.recording),
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
