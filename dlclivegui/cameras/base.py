# dlclivegui/cameras/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..utils.config_models import CameraSettingsModel

_BACKEND_REGISTRY: dict[str, type[CameraBackend]] = {}


def register_backend(name: str):
    """
    Decorator to register a camera backend class.

    Usage:
        @register_backend("opencv")
        class OpenCVCameraBackend(CameraBackend):
            ...
    """

    def decorator(cls: type[CameraBackend]):
        if not issubclass(cls, CameraBackend):
            raise TypeError(f"Backend '{name}' must subclass CameraBackend")
        _BACKEND_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def register_backend_direct(name: str, cls: type[CameraBackend]):
    """Allow tests or dynamic plugins to register backends programmatically."""
    if not issubclass(cls, CameraBackend):
        raise TypeError(f"Backend '{name}' must subclass CameraBackend")
    _BACKEND_REGISTRY[name.lower()] = cls


def unregister_backend(name: str):
    """Remove a backend from the registry. Useful for tests."""
    _BACKEND_REGISTRY.pop(name.lower(), None)


def reset_backends():
    """Clear registry (useful for isolated unit tests)."""
    _BACKEND_REGISTRY.clear()


class CameraBackend(ABC):
    """Abstract base class for camera backends."""

    def __init__(self, settings: CameraSettingsModel):
        # Normalize to dataclass so all backends stay unchanged
        self.settings: CameraSettingsModel = settings

    @classmethod
    def name(cls) -> str:
        """Return the backend identifier."""
        return cls.__name__.lower()

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the backend can be used on this system."""
        return True

    @abstractmethod
    def stop(self) -> None:
        """Request a graceful stop."""
        # Subclasses may override when they need to interrupt blocking reads.
        raise NotImplementedError

    def device_name(self) -> str:
        """Return a human readable name for the device currently in use."""
        return self.settings.name

    @abstractmethod
    def open(self) -> None:
        """Open the capture device."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> tuple[np.ndarray, float]:
        """Read a frame and return the image with a timestamp."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release the capture device."""
        raise NotImplementedError
