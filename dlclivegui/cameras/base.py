# dlclivegui/cameras/base.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from ..config import CameraSettings

_BACKEND_REGISTRY: dict[str, type[CameraBackend]] = {}

logger = logging.getLogger(__name__)


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
        logger.debug(f"Registered camera backend '{name}' -> {cls}")
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

    OPTIONS_KEY: ClassVar[str] = ""  # override in subclasses if they want to support options

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

    @classmethod
    def options_key(cls) -> str:
        """Return the key used to store this backend's options in CameraSettings."""
        return cls.OPTIONS_KEY

    @classmethod
    def parse_options(cls, settings: CameraSettings) -> Any:
        """Return a typed options object for this backend (or None)."""
        return None

    @classmethod
    def options_schema(cls) -> dict[str, Any] | None:
        """Optional: for UI/docs."""
        return None

    @classmethod
    def sanitize_for_probe(cls, settings: CameraSettings) -> CameraSettings:
        """
        Default: keep only the backend namespace and minimal safe toggles.
        Backends may override.
        """
        # shallow copy is fine if you deep-copy in factory already
        dc = settings.model_copy(deep=True)
        ns = (dc.properties or {}).get(cls.options_key(), {})
        dc.properties = {cls.options_key(): dict(ns)}
        return dc

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
