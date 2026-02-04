# dlclivegui/cameras/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import CameraSettings

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

    def __init__(self, settings: CameraSettings):
        # Normalize to dataclass so all backends stay unchanged
        self.settings: CameraSettings = settings

    @classmethod
    def name(cls) -> str:
        """Return the backend identifier."""
        return cls.__name__.lower()

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the backend can be used on this system."""
        return True

    def stop(self) -> None:  # noqa B027
        """Optional: Request a graceful stop. No-op by default."""
        # Subclasses may override when they need to interrupt blocking reads.
        pass

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
