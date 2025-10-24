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
            raise RuntimeError(f"Unable to open camera index {self.settings.index} with OpenCV")
        self._configure_capture()

    def read(self) -> Tuple[np.ndarray, float]:
        if self._capture is None:
            raise RuntimeError("Camera has not been opened")

        # Try grab first - this is non-blocking and helps detect connection issues faster
        grabbed = self._capture.grab()
        if not grabbed:
            # Check if camera is still opened - if not, it's a serious error
            if not self._capture.isOpened():
                raise RuntimeError("OpenCV camera connection lost")
            # Otherwise treat as temporary frame read failure (timeout-like)
            raise TimeoutError("Failed to grab frame from OpenCV camera (temporary)")

        # Now retrieve the frame
        success, frame = self._capture.retrieve()
        if not success or frame is None:
            raise TimeoutError("Failed to retrieve frame from OpenCV camera (temporary)")

        return frame, time.time()

    def close(self) -> None:
        if self._capture is not None:
            try:
                # Try to release properly
                self._capture.release()
            except Exception:
                pass
            finally:
                self._capture = None
            # Give the system a moment to fully release the device
            time.sleep(0.1)

    def stop(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            finally:
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
        # Don't set width/height - capture at camera's native resolution
        # Only set FPS if specified
        if self.settings.fps:
            self._capture.set(cv2.CAP_PROP_FPS, float(self.settings.fps))
        # Set any additional properties from the properties dict
        for prop, value in self.settings.properties.items():
            if prop == "api":
                continue
            try:
                prop_id = int(prop)
            except (TypeError, ValueError):
                continue
            self._capture.set(prop_id, float(value))
        # Update actual FPS from camera
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        if actual_fps:
            self.settings.fps = float(actual_fps)

    def _resolve_backend(self, backend: str | None) -> int:
        if backend is None:
            return cv2.CAP_ANY
        key = backend.upper()
        return getattr(cv2, f"CAP_{key}", cv2.CAP_ANY)
