"""OpenCV based camera backend."""
from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

from .base import CameraBackend


class OpenCVCameraBackend(CameraBackend):
    """Fallback backend using :mod:`cv2.VideoCapture`."""

    def __init__(self, settings):
        super().__init__(settings)
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        backend_flag = self._resolve_backend(self.settings.properties.get("api"))
        self._capture = cv2.VideoCapture(int(self.settings.index), backend_flag)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera index {self.settings.index} with OpenCV"
            )
        self._configure_capture()

    def read(self) -> Tuple[np.ndarray, float]:
        if self._capture is None:
            raise RuntimeError("Camera has not been opened")
        success, frame = self._capture.read()
        if not success:
            raise RuntimeError("Failed to read frame from OpenCV camera")
        return frame, time.time()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def device_name(self) -> str:
        base_name = "OpenCV"
        if self._capture is not None and hasattr(self._capture, "getBackendName"):
            try:
                backend_name = self._capture.getBackendName()
            except Exception:  # pragma: no cover - backend specific
                backend_name = ""
            if backend_name:
                base_name = backend_name
        return f"{base_name} camera #{self.settings.index}"

    def _configure_capture(self) -> None:
        if self._capture is None:
            return
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.height))
        self._capture.set(cv2.CAP_PROP_FPS, float(self.settings.fps))
        for prop, value in self.settings.properties.items():
            if prop == "api":
                continue
            try:
                prop_id = int(prop)
            except (TypeError, ValueError):
                continue
            self._capture.set(prop_id, float(value))
        actual_width = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        if actual_width:
            self.settings.width = int(actual_width)
        if actual_height:
            self.settings.height = int(actual_height)
        if actual_fps:
            self.settings.fps = float(actual_fps)

    def _resolve_backend(self, backend: str | None) -> int:
        if backend is None:
            return cv2.CAP_ANY
        key = backend.upper()
        return getattr(cv2, f"CAP_{key}", cv2.CAP_ANY)
