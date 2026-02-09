"""Basler camera backend implemented with :mod:`pypylon`."""

# dlclivegui/cameras/backends/basler_backend.py
from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np

from ..base import CameraBackend, SupportLevel, register_backend

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from pypylon import pylon
except Exception:  # pragma: no cover - optional dependency
    pylon = None  # type: ignore


@register_backend("basler")
class BaslerCameraBackend(CameraBackend):
    """Capture frames from Basler cameras using the Pylon SDK."""

    OPTIONS_KEY: ClassVar[str] = "basler"

    def __init__(self, settings):
        super().__init__(settings)

        props = settings.properties if isinstance(settings.properties, dict) else {}
        ns = props.get(self.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}

        self._camera: pylon.InstantCamera | None = None
        self._converter: pylon.ImageFormatConverter | None = None

        # Resolution request (None = device default)
        self._requested_resolution: tuple[int, int] | None = self._get_requested_resolution_or_none()

        # Actuals for GUI
        self._actual_width: int | None = None
        self._actual_height: int | None = None
        self._actual_fps: float | None = None

    @property
    def actual_resolution(self) -> tuple[int, int] | None:
        if self._actual_width and self._actual_height:
            return (self._actual_width, self._actual_height)
        return None

    @property
    def actual_fps(self) -> float | None:
        return self._actual_fps

    @classmethod
    def is_available(cls) -> bool:
        return pylon is not None

    @classmethod
    def static_capabilities(cls) -> dict[str, SupportLevel]:
        caps = super().static_capabilities()
        caps.update(
            {
                "set_resolution": SupportLevel.SUPPORTED,
                "set_fps": SupportLevel.SUPPORTED,
                "set_exposure": SupportLevel.SUPPORTED,
                "set_gain": SupportLevel.SUPPORTED,
                "device_discovery": SupportLevel.BEST_EFFORT,
                "stable_identity": SupportLevel.SUPPORTED,
            }
        )
        return caps

    def open(self) -> None:
        if pylon is None:
            raise RuntimeError("pypylon is required for the Basler backend but is not installed")

        devices = self._enumerate_devices()
        if not devices:
            raise RuntimeError("No Basler cameras detected")

        device = self._select_device(devices)
        self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
        self._camera.Open()

        # Exposure
        exposure = self._settings_value("exposure", self.settings.properties)
        if exposure is not None:
            try:
                self._camera.ExposureTime.SetValue(float(exposure))
            except Exception:
                pass

        # Gain
        gain = self._settings_value("gain", self.settings.properties)
        if gain is not None:
            try:
                self._camera.Gain.SetValue(float(gain))
            except Exception:
                pass

        # Resolution (device default if None)
        self._configure_resolution()

        # Frame rate
        fps = self._settings_value("fps", self.settings.properties, fallback=self.settings.fps)
        if fps is not None:
            try:
                self._camera.AcquisitionFrameRateEnable.SetValue(True)
                self._camera.AcquisitionFrameRate.SetValue(float(fps))
                self._actual_fps = float(self._camera.AcquisitionFrameRate.GetValue())
            except Exception:
                self._actual_fps = None

        # Capture actual resolution even when using defaults
        try:
            self._actual_width = int(self._camera.Width.GetValue())
            self._actual_height = int(self._camera.Height.GetValue())
        except Exception:
            pass

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def read(self) -> tuple[np.ndarray, float]:
        if self._camera is None or self._converter is None:
            raise RuntimeError("Basler camera not opened")
        try:
            grab_result = self._camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
        except Exception as exc:
            raise RuntimeError("Failed to retrieve image from Basler camera.") from exc
        if not grab_result.GrabSucceeded():
            grab_result.Release()
            raise RuntimeError("Basler camera did not return an image")
        image = self._converter.Convert(grab_result)
        frame = image.GetArray()
        grab_result.Release()

        if self._actual_width is None or self._actual_height is None:
            h, w = frame.shape[:2]
            self._actual_width = int(w)
            self._actual_height = int(h)

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
            raise RuntimeError(f"Camera index {index} out of range for {len(devices)} Basler device(s)")
        return devices[index]

    def _rotate(self, frame: np.ndarray, angle: float) -> np.ndarray:
        try:
            from imutils import rotate_bound  # pragma: no cover - optional
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Rotation requested for Basler camera but imutils is not installed") from exc
        return rotate_bound(frame, angle)

    def _get_requested_resolution_or_none(self) -> tuple[int, int] | None:
        """
        Return (w, h) if user explicitly requested a resolution.
        Return None to keep device defaults.
        """
        props = self.settings.properties if isinstance(self.settings.properties, dict) else {}

        legacy = props.get("resolution")
        if isinstance(legacy, (list, tuple)) and len(legacy) == 2:
            try:
                w, h = int(legacy[0]), int(legacy[1])
                if w > 0 and h > 0:
                    return (w, h)
            except Exception:
                pass

        try:
            w = int(getattr(self.settings, "width", 0) or 0)
            h = int(getattr(self.settings, "height", 0) or 0)
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            pass

        return None

    def _configure_resolution(self) -> None:
        """
        Apply width/height only if explicitly requested.
        If None, keep device defaults.
        """
        if self._camera is None:
            return

        req = self._requested_resolution
        if req is None:
            LOG.info("Resolution: using device default.")
            return

        req_w, req_h = req
        try:
            self._camera.Width.SetValue(int(req_w))
            self._camera.Height.SetValue(int(req_h))

            aw = int(self._camera.Width.GetValue())
            ah = int(self._camera.Height.GetValue())
            self._actual_width = aw
            self._actual_height = ah

            if (aw, ah) != (req_w, req_h):
                LOG.warning(f"Resolution mismatch: requested {req_w}x{req_h}, got {aw}x{ah}")
            else:
                LOG.info(f"Resolution set to {aw}x{ah}")
        except Exception as exc:
            LOG.warning(f"Failed to set resolution to {req_w}x{req_h}: {exc}")

    @staticmethod
    def _settings_value(key: str, source: dict, fallback: float | None = None):
        value = source.get(key, fallback)
        return None if value is None else value
