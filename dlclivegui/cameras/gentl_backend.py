"""Generic GenTL backend implemented with Aravis."""
from __future__ import annotations

import ctypes
import time
from typing import Optional, Tuple

import numpy as np

from .base import CameraBackend

try:  # pragma: no cover - optional dependency
    import gi

    gi.require_version("Aravis", "0.6")
    from gi.repository import Aravis
except Exception:  # pragma: no cover - optional dependency
    gi = None  # type: ignore
    Aravis = None  # type: ignore


class GenTLCameraBackend(CameraBackend):
    """Capture frames from cameras that expose a GenTL interface."""

    def __init__(self, settings):
        super().__init__(settings)
        self._camera = None
        self._stream = None
        self._payload: Optional[int] = None

    @classmethod
    def is_available(cls) -> bool:
        return Aravis is not None

    def open(self) -> None:
        if Aravis is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Aravis (python-gi bindings) are required for the GenTL backend"
            )
        Aravis.update_device_list()
        num_devices = Aravis.get_n_devices()
        if num_devices == 0:
            raise RuntimeError("No GenTL cameras detected")
        device_id = self._select_device_id(num_devices)
        self._camera = Aravis.Camera.new(device_id)
        self._camera.set_exposure_time_auto(0)
        self._camera.set_gain_auto(0)
        exposure = self.settings.properties.get("exposure")
        if exposure is not None:
            self._set_exposure(float(exposure))
        crop = self.settings.properties.get("crop")
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            self._set_crop(crop)
        if self.settings.fps:
            try:
                self._camera.set_frame_rate(float(self.settings.fps))
            except Exception:
                pass
        self._stream = self._camera.create_stream()
        self._payload = self._camera.get_payload()
        self._stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))
        self._camera.start_acquisition()

    def read(self) -> Tuple[np.ndarray, float]:
        if self._stream is None:
            raise RuntimeError("GenTL stream not initialised")
        buffer = None
        while buffer is None:
            buffer = self._stream.try_pop_buffer()
            if buffer is None:
                time.sleep(0.01)
        frame = self._buffer_to_numpy(buffer)
        self._stream.push_buffer(buffer)
        return frame, time.time()

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass
        self._camera = None
        self._stream = None
        self._payload = None

    def stop(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass

    def _select_device_id(self, num_devices: int) -> str:
        index = int(self.settings.index)
        if index < 0 or index >= num_devices:
            raise RuntimeError(
                f"Camera index {index} out of range for {num_devices} GenTL device(s)"
            )
        return Aravis.get_device_id(index)

    def _set_exposure(self, exposure: float) -> None:
        if self._camera is None:
            return
        exposure = max(0.0, min(exposure, 1.0))
        self._camera.set_exposure_time(exposure * 1e6)

    def _set_crop(self, crop) -> None:
        if self._camera is None:
            return
        left, right, top, bottom = map(int, crop)
        width = right - left
        height = bottom - top
        self._camera.set_region(left, top, width, height)

    def _buffer_to_numpy(self, buffer) -> np.ndarray:
        pixel_format = buffer.get_image_pixel_format()
        bits_per_pixel = (pixel_format >> 16) & 0xFF
        if bits_per_pixel == 8:
            int_pointer = ctypes.POINTER(ctypes.c_uint8)
        else:
            int_pointer = ctypes.POINTER(ctypes.c_uint16)
        addr = buffer.get_data()
        ptr = ctypes.cast(addr, int_pointer)
        frame = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
        frame = frame.copy()
        if frame.ndim < 3:
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame
