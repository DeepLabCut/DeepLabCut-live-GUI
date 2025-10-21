"""Basler camera backend implemented with :mod:`pypylon`."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from .base import CameraBackend

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
            raise RuntimeError(
                "pypylon is required for the Basler backend but is not installed"
            )
        devices = self._enumerate_devices()
        if not devices:
            raise RuntimeError("No Basler cameras detected")
        device = self._select_device(devices)
        self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
        self._camera.Open()

        exposure = self._settings_value("exposure", self.settings.properties)
        if exposure is not None:
            self._camera.ExposureTime.SetValue(float(exposure))
        gain = self._settings_value("gain", self.settings.properties)
        if gain is not None:
            self._camera.Gain.SetValue(float(gain))
        width = int(self.settings.properties.get("width", self.settings.width))
        height = int(self.settings.properties.get("height", self.settings.height))
        self._camera.Width.SetValue(width)
        self._camera.Height.SetValue(height)
        fps = self._settings_value("fps", self.settings.properties, fallback=self.settings.fps)
        if fps is not None:
            try:
                self._camera.AcquisitionFrameRateEnable.SetValue(True)
                self._camera.AcquisitionFrameRate.SetValue(float(fps))
            except Exception:
                # Some cameras expose different frame-rate features; ignore errors.
                pass

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

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
        serial = self.settings.properties.get("serial") or self.settings.properties.get("serial_number")
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
