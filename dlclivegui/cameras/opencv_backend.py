"""OpenCV based camera backend."""

from __future__ import annotations

import logging
import time
from typing import Tuple

import cv2
import numpy as np

from .base import CameraBackend

LOG = logging.getLogger(__name__)


class OpenCVCameraBackend(CameraBackend):
    """Fallback backend using :mod:`cv2.VideoCapture`."""

    def __init__(self, settings):
        super().__init__(settings)
        self._capture: cv2.VideoCapture | None = None
        # Parse resolution with defaults (720x540)
        self._resolution: Tuple[int, int] = self._parse_resolution(
            settings.properties.get("resolution")
        )

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

    def _parse_resolution(self, resolution) -> Tuple[int, int]:
        """Parse resolution setting.

        Args:
            resolution: Can be a tuple/list [width, height], or None

        Returns:
            Tuple of (width, height), defaults to (720, 540)
        """
        if resolution is None:
            return (720, 540)  # Default resolution

        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            try:
                return (int(resolution[0]), int(resolution[1]))
            except (ValueError, TypeError):
                return (720, 540)

        return (720, 540)

    def _configure_capture(self) -> None:
        if self._capture is None:
            return

        # Set resolution (width x height)
        width, height = self._resolution
        if not self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width)):
            LOG.warning(f"Failed to set frame width to {width}")
        if not self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height)):
            LOG.warning(f"Failed to set frame height to {height}")

        # Verify resolution was set correctly
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != width or actual_height != height:
            LOG.warning(
                f"Resolution mismatch: requested {width}x{height}, "
                f"got {actual_width}x{actual_height}"
            )

        # Set FPS if specified
        requested_fps = self.settings.fps
        if requested_fps:
            if not self._capture.set(cv2.CAP_PROP_FPS, float(requested_fps)):
                LOG.warning(f"Failed to set FPS to {requested_fps}")

        # Set any additional properties from the properties dict
        for prop, value in self.settings.properties.items():
            if prop in ("api", "resolution"):
                continue
            try:
                prop_id = int(prop)
            except (TypeError, ValueError) as e:
                LOG.warning(f"Could not parse property ID: {prop} ({e})")
                continue
            if not self._capture.set(prop_id, float(value)):
                LOG.warning(f"Failed to set property {prop_id} to {value}")

        # Update actual FPS from camera and warn if different from requested
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        if actual_fps:
            if requested_fps and abs(actual_fps - requested_fps) > 0.1:
                LOG.warning(f"FPS mismatch: requested {requested_fps:.2f}, got {actual_fps:.2f}")
            self.settings.fps = float(actual_fps)
            LOG.info(f"Camera configured with FPS: {actual_fps:.2f}")

    def _resolve_backend(self, backend: str | None) -> int:
        if backend is None:
            return cv2.CAP_ANY
        key = backend.upper()
        return getattr(cv2, f"CAP_{key}", cv2.CAP_ANY)
