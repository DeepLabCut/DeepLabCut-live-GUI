"""Abstract camera backend definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..config import CameraSettings


class CameraBackend(ABC):
    """Abstract base class for camera backends."""

    def __init__(self, settings: CameraSettings):
        self.settings = settings

    @classmethod
    def name(cls) -> str:
        """Return the backend identifier."""

        return cls.__name__.lower()

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the backend can be used on this system."""

        return True

    def stop(self) -> None:
        """Request a graceful stop."""

        # Most backends do not require additional handling, but subclasses may
        # override when they need to interrupt blocking reads.

    @abstractmethod
    def open(self) -> None:
        """Open the capture device."""

    @abstractmethod
    def read(self) -> Tuple[np.ndarray, float]:
        """Read a frame and return the image with a timestamp."""

    @abstractmethod
    def close(self) -> None:
        """Release the capture device."""
