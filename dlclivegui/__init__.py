"""DeepLabCut Live GUI package."""

from .config import (
    ApplicationSettings,
    CameraSettings,
    DLCProcessorSettings,
    MultiCameraSettings,
    RecordingSettings,
)

__all__ = [
    "ApplicationSettings",
    "CameraSettings",
    "DLCProcessorSettings",
    "MultiCameraSettings",
    "RecordingSettings",
    "main",
]
__version__ = "2.0.0rc0"  # PLACEHOLDER
