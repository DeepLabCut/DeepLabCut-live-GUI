"""Basler camera backend implemented with :mod:`pypylon`."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

from .base import CameraBackend

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from pypylon import pylon
except Exception:  # pragma: no cover - optional dependency
    pylon = None  # type: ignore


class BaslerCameraBackend(CameraBackend):
    """Capture frames from Basler cameras using the Pylon SDK."""

    def __init__(self, settings):
        super().__init__(settings)
        self._camera: Optional["pylon.InstantCamera"] = None
        self._converter: Optional["pylon.ImageFormatConverter"] = None

    @classmethod
    def is_available(cls) -> bool:
        return pylon is not None

    def open(self) -> None:
        if pylon is None:  # pragma: no cover - optional dependency
            raise RuntimeError("pypylon is required for the Basler backend but is not installed")
        devices = self._enumerate_devices()
        if not devices:
            raise RuntimeError("No Basler cameras detected")
        device = self._select_device(devices)
        self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
        self._camera.Open()

        # Configure exposure
        exposure = self._settings_value("exposure", self.settings.properties)
        if exposure is not None:
            try:
                self._camera.ExposureTime.SetValue(float(exposure))
                actual = self._camera.ExposureTime.GetValue()
                if abs(actual - float(exposure)) > 1.0:  # Allow 1μs tolerance
                    LOG.warning(f"Exposure mismatch: requested {exposure}μs, got {actual}μs")
                else:
                    LOG.info(f"Exposure set to {actual}μs")
            except Exception as e:
                LOG.warning(f"Failed to set exposure to {exposure}μs: {e}")

        # Configure gain
        gain = self._settings_value("gain", self.settings.properties)
        if gain is not None:
            try:
                self._camera.Gain.SetValue(float(gain))
                actual = self._camera.Gain.GetValue()
                if abs(actual - float(gain)) > 0.1:  # Allow 0.1 tolerance
                    LOG.warning(f"Gain mismatch: requested {gain}, got {actual}")
                else:
                    LOG.info(f"Gain set to {actual}")
            except Exception as e:
                LOG.warning(f"Failed to set gain to {gain}: {e}")

        # Configure resolution
        requested_width = int(self.settings.properties.get("width", self.settings.width))
        requested_height = int(self.settings.properties.get("height", self.settings.height))
        try:
            self._camera.Width.SetValue(requested_width)
            self._camera.Height.SetValue(requested_height)
            actual_width = self._camera.Width.GetValue()
            actual_height = self._camera.Height.GetValue()
            if actual_width != requested_width or actual_height != requested_height:
                LOG.warning(
                    f"Resolution mismatch: requested {requested_width}x{requested_height}, "
                    f"got {actual_width}x{actual_height}"
                )
            else:
                LOG.info(f"Resolution set to {actual_width}x{actual_height}")
        except Exception as e:
            LOG.warning(f"Failed to set resolution to {requested_width}x{requested_height}: {e}")

        # Configure frame rate
        fps = self._settings_value("fps", self.settings.properties, fallback=self.settings.fps)
        if fps is not None:
            try:
                self._camera.AcquisitionFrameRateEnable.SetValue(True)
                self._camera.AcquisitionFrameRate.SetValue(float(fps))
                actual_fps = self._camera.AcquisitionFrameRate.GetValue()
                if abs(actual_fps - float(fps)) > 0.1:
                    LOG.warning(f"FPS mismatch: requested {fps:.2f}, got {actual_fps:.2f}")
                else:
                    LOG.info(f"Frame rate set to {actual_fps:.2f} FPS")
            except Exception as e:
                LOG.warning(f"Failed to set frame rate to {fps}: {e}")

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Read back final settings
        try:
            self.settings.width = int(self._camera.Width.GetValue())
            self.settings.height = int(self._camera.Height.GetValue())
        except Exception:
            pass
        try:
            self.settings.fps = float(self._camera.ResultingFrameRateAbs.GetValue())
            LOG.info(f"Camera configured with resulting FPS: {self.settings.fps:.2f}")
        except Exception:
            pass

    def read(self) -> Tuple[np.ndarray, float]:
        if self._camera is None or self._converter is None:
            raise RuntimeError("Basler camera not opened")
        try:
            grab_result = self._camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
        except Exception as exc:
            raise RuntimeError(f"Failed to retrieve image from Basler camera: {exc}")
        if not grab_result.GrabSucceeded():
            grab_result.Release()
            raise RuntimeError("Basler camera did not return an image")
        image = self._converter.Convert(grab_result)
        frame = image.GetArray()
        grab_result.Release()
        rotate = self._settings_value("rotate", self.settings.properties)
        if rotate:
            frame = self._rotate(frame, float(rotate))
        crop = self.settings.properties.get("crop")
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            left, right, top, bottom = map(int, crop)
            frame = frame[top:bottom, left:right]
        return frame, time.time()

    def close(self) -> None:
        if self._camera is not None:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
            if self._camera.IsOpen():
                self._camera.Close()
            self._camera = None
        self._converter = None

    def stop(self) -> None:
        if self._camera is not None and self._camera.IsGrabbing():
            try:
                self._camera.StopGrabbing()
            except Exception:
                pass

    def _enumerate_devices(self):
        factory = pylon.TlFactory.GetInstance()
        return factory.EnumerateDevices()

    def _select_device(self, devices):
        serial = self.settings.properties.get("serial") or self.settings.properties.get(
            "serial_number"
        )
        if serial:
            for device in devices:
                if getattr(device, "GetSerialNumber", None) and device.GetSerialNumber() == serial:
                    return device
        index = int(self.settings.index)
        if index < 0 or index >= len(devices):
            raise RuntimeError(
                f"Camera index {index} out of range for {len(devices)} Basler device(s)"
            )
        return devices[index]

    def _rotate(self, frame: np.ndarray, angle: float) -> np.ndarray:
        try:
            from imutils import rotate_bound  # pragma: no cover - optional
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Rotation requested for Basler camera but imutils is not installed"
            ) from exc
        return rotate_bound(frame, angle)

    @staticmethod
    def _settings_value(key: str, source: dict, fallback: Optional[float] = None):
        value = source.get(key, fallback)
        return None if value is None else value
