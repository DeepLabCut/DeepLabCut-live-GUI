"""GenTL backend implemented using the Harvesters library."""
from __future__ import annotations

import glob
import os
import time
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .base import CameraBackend

try:  # pragma: no cover - optional dependency
    from harvesters.core import Harvester
except Exception:  # pragma: no cover - optional dependency
    Harvester = None  # type: ignore


class GenTLCameraBackend(CameraBackend):
    """Capture frames from GenTL-compatible devices via Harvesters."""

    _DEFAULT_CTI_PATTERNS: Tuple[str, ...] = (
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\IC4 GenTL Driver for USB3Vision Devices *\\bin\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Camera SDK\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files (x86)\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
    )

    def __init__(self, settings):
        super().__init__(settings)
        props = settings.properties
        self._cti_file: Optional[str] = props.get("cti_file")
        self._serial_number: Optional[str] = props.get("serial_number") or props.get("serial")
        self._pixel_format: str = props.get("pixel_format", "Mono8")
        self._rotate: int = int(props.get("rotate", 0)) % 360
        self._crop: Optional[Tuple[int, int, int, int]] = self._parse_crop(props.get("crop"))
        self._exposure: Optional[float] = props.get("exposure")
        self._gain: Optional[float] = props.get("gain")
        self._timeout: float = float(props.get("timeout", 2.0))
        self._cti_search_paths: Tuple[str, ...] = self._parse_cti_paths(props.get("cti_search_paths"))

        self._harvester: Optional[Harvester] = None
        self._acquirer = None

    @classmethod
    def is_available(cls) -> bool:
        return Harvester is not None

    def open(self) -> None:
        if Harvester is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'harvesters' package is required for the GenTL backend. "
                "Install it via 'pip install harvesters'."
            )

        self._harvester = Harvester()
        cti_file = self._cti_file or self._find_cti_file()
        self._harvester.add_file(cti_file)
        self._harvester.update()

        if not self._harvester.device_info_list:
            raise RuntimeError("No GenTL cameras detected via Harvesters")

        serial = self._serial_number
        index = int(self.settings.index or 0)
        if serial:
            available = self._available_serials()
            matches = [s for s in available if serial in s]
            if not matches:
                raise RuntimeError(
                    f"Camera with serial '{serial}' not found. Available cameras: {available}"
                )
            serial = matches[0]
        else:
            device_count = len(self._harvester.device_info_list)
            if index < 0 or index >= device_count:
                raise RuntimeError(
                    f"Camera index {index} out of range for {device_count} GenTL device(s)"
                )

        self._acquirer = self._create_acquirer(serial, index)

        remote = self._acquirer.remote_device
        node_map = remote.node_map

        self._configure_pixel_format(node_map)
        self._configure_resolution(node_map)
        self._configure_exposure(node_map)
        self._configure_gain(node_map)
        self._configure_frame_rate(node_map)

        self._acquirer.start()

    def read(self) -> Tuple[np.ndarray, float]:
        if self._acquirer is None:
            raise RuntimeError("GenTL image acquirer not initialised")

        with self._acquirer.fetch(timeout=self._timeout) as buffer:
            component = buffer.payload.components[0]
            channels = 3 if self._pixel_format in {"RGB8", "BGR8"} else 1
            if channels > 1:
                frame = component.data.reshape(
                    component.height, component.width, channels
                ).copy()
            else:
                frame = component.data.reshape(component.height, component.width).copy()

        frame = self._convert_frame(frame)
        timestamp = time.time()
        return frame, timestamp

    def stop(self) -> None:
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                pass

    def close(self) -> None:
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                pass
            try:
                destroy = getattr(self._acquirer, "destroy", None)
                if destroy is not None:
                    destroy()
            finally:
                self._acquirer = None

        if self._harvester is not None:
            try:
                self._harvester.reset()
            finally:
                self._harvester = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_cti_paths(self, value) -> Tuple[str, ...]:
        if value is None:
            return self._DEFAULT_CTI_PATTERNS
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Iterable):
            return tuple(str(item) for item in value)
        return self._DEFAULT_CTI_PATTERNS

    def _parse_crop(self, crop) -> Optional[Tuple[int, int, int, int]]:
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            return tuple(int(v) for v in crop)
        return None

    def _find_cti_file(self) -> str:
        patterns: List[str] = list(self._cti_search_paths)
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    return file_path
        raise RuntimeError(
            "Could not locate a GenTL producer (.cti) file. Set 'cti_file' in "
            "camera.properties or provide search paths via 'cti_search_paths'."
        )

    def _available_serials(self) -> List[str]:
        assert self._harvester is not None
        serials: List[str] = []
        for info in self._harvester.device_info_list:
            serial = getattr(info, "serial_number", "")
            if serial:
                serials.append(serial)
        return serials

    def _create_acquirer(self, serial: Optional[str], index: int):
        assert self._harvester is not None
        methods = [
            getattr(self._harvester, "create_image_acquirer", None),
            getattr(self._harvester, "create", None),
        ]
        methods = [m for m in methods if m is not None]
        errors: List[str] = []
        device_info = None
        if not serial:
            device_list = self._harvester.device_info_list
            if 0 <= index < len(device_list):
                device_info = device_list[index]
        for create in methods:
            try:
                if serial:
                    return create({"serial_number": serial})
            except Exception as exc:
                errors.append(f"{create.__name__} serial: {exc}")
        for create in methods:
            try:
                return create(index=index)
            except TypeError:
                try:
                    return create(index)
                except Exception as exc:
                    errors.append(f"{create.__name__} index positional: {exc}")
            except Exception as exc:
                errors.append(f"{create.__name__} index: {exc}")
        if device_info is not None:
            for create in methods:
                try:
                    return create(device_info)
                except Exception as exc:
                    errors.append(f"{create.__name__} device_info: {exc}")
        if not serial and index == 0:
            for create in methods:
                try:
                    return create()
                except Exception as exc:
                    errors.append(f"{create.__name__} default: {exc}")
        joined = "; ".join(errors) or "no creation methods available"
        raise RuntimeError(f"Failed to initialise GenTL image acquirer ({joined})")

    def _configure_pixel_format(self, node_map) -> None:
        try:
            if self._pixel_format in node_map.PixelFormat.symbolics:
                node_map.PixelFormat.value = self._pixel_format
        except Exception:
            pass

    def _configure_resolution(self, node_map) -> None:
        width = int(self.settings.width)
        height = int(self.settings.height)
        if self._rotate in (90, 270):
            width, height = height, width
        try:
            node_map.Width.value = self._adjust_to_increment(
                width, node_map.Width.min, node_map.Width.max, node_map.Width.inc
            )
        except Exception:
            pass
        try:
            node_map.Height.value = self._adjust_to_increment(
                height, node_map.Height.min, node_map.Height.max, node_map.Height.inc
            )
        except Exception:
            pass

    def _configure_exposure(self, node_map) -> None:
        if self._exposure is None:
            return
        for attr in ("ExposureAuto", "ExposureTime", "Exposure"):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                if attr == "ExposureAuto":
                    node.value = "Off"
                else:
                    node.value = float(self._exposure)
                    return
            except Exception:
                continue

    def _configure_gain(self, node_map) -> None:
        if self._gain is None:
            return
        for attr in ("GainAuto", "Gain"):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                if attr == "GainAuto":
                    node.value = "Off"
                else:
                    node.value = float(self._gain)
                    return
            except Exception:
                continue

    def _configure_frame_rate(self, node_map) -> None:
        if not self.settings.fps:
            return
        try:
            node_map.AcquisitionFrameRateEnable.value = True
        except Exception:
            pass
        try:
            node_map.AcquisitionFrameRate.value = float(self.settings.fps)
        except Exception:
            pass

    @staticmethod
    def _adjust_to_increment(value: int, minimum: int, maximum: int, increment: int) -> int:
        value = max(minimum, min(maximum, value))
        if increment <= 0:
            return value
        return minimum + ((value - minimum) // increment) * increment

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.astype(np.float32 if frame.dtype == np.float64 else frame.dtype)
        if result.dtype != np.uint8:
            max_val = np.max(result)
            if max_val > 0:
                result = (result / max_val * 255.0).astype(np.uint8)
            else:
                result = np.zeros_like(result, dtype=np.uint8)
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif result.ndim == 3 and result.shape[2] == 3 and self._pixel_format == "RGB8":
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        if self._rotate == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif self._rotate == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)
        elif self._rotate == 270:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self._crop is not None:
            top, bottom, left, right = self._crop
            height, width = result.shape[:2]
            top = max(0, min(height, top))
            bottom = max(top, min(height, bottom))
            left = max(0, min(width, left))
            right = max(left, min(width, right))
            result = result[top:bottom, left:right]

        return result.copy()
