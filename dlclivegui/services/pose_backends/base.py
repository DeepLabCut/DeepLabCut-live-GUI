from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PoseBackend(ABC):
    """Common interface for pose estimation backends."""

    @abstractmethod
    def init_inference(self, init_frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_pose(self, frame: np.ndarray, frame_time: float | None = None) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        """Optional cleanup hook."""
        pass
