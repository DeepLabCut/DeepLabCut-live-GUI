"""GenTL backend implemented using the Harvesters library."""

from __future__ import annotations

import glob
import logging
import os
import time
from collections.abc import Iterable

import cv2
import numpy as np

from ..base import CameraBackend, register_backend

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from harvesters.core import Harvester  # type: ignore

    try:
        from harvesters.core import HarvesterTimeoutError  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        HarvesterTimeoutError = TimeoutError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Harvester = None  # type: ignore
    HarvesterTimeoutError = TimeoutError  # type: ignore


@register_backend("gentl")
class GenTLCameraBackend(CameraBackend):
    """Capture frames from GenTL-compatible devices via Harvesters."""

    _DEFAULT_CTI_PATTERNS: tuple[str, ...] = (
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\IC4 GenTL Driver for USB3Vision Devices *\\bin\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Camera SDK\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files (x86)\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
    )

    def __init__(self, settings):
        super().__init__(settings)
        props = settings.properties
        self._cti_file: str | None = props.get("cti_file")
        self._serial_number: str | None = props.get("serial_number") or props.get("serial")
        self._pixel_format: str = props.get("pixel_format", "Mono8")
        self._rotate: int = int(props.get("rotate", 0)) % 360
        self._crop: tuple[int, int, int, int] | None = self._parse_crop(props.get("crop"))
        # Check settings first (from config), then properties (for backward compatibility)
        self._exposure: float | None = settings.exposure if settings.exposure else props.get("exposure")
        self._gain: float | None = settings.gain if settings.gain else props.get("gain")
        self._timeout: float = float(props.get("timeout", 2.0))
        self._cti_search_paths: tuple[str, ...] = self._parse_cti_paths(props.get("cti_search_paths"))
        # Parse resolution (width, height) with defaults
        self._resolution: tuple[int, int] | None = self._parse_resolution(props.get("resolution"))

        self._harvester = None
        self._acquirer = None
        self._device_label: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        return Harvester is not None

    @classmethod
    def get_device_count(cls) -> int:
        """Get the actual number of GenTL devices detected by Harvester.

        Returns the number of devices found, or -1 if detection fails.
        """
        if Harvester is None:
            return -1

        harvester = None
        try:
            harvester = Harvester()
            # Use the static helper to find CTI file with default patterns
            cti_file = cls._search_cti_file(cls._DEFAULT_CTI_PATTERNS)

            if not cti_file:
                return -1

            harvester.add_file(cti_file)
            harvester.update()
            return len(harvester.device_info_list)
        except Exception:
            return -1
        finally:
            if harvester is not None:
                try:
                    harvester.reset()
                except Exception:
                    pass

    def open(self) -> None:
        if Harvester is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'harvesters' package is required for the GenTL backend. Install it via 'pip install harvesters'."
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
                raise RuntimeError(f"Camera with serial '{serial}' not found. Available cameras: {available}")
            serial = matches[0]
        else:
            device_count = len(self._harvester.device_info_list)
            if index < 0 or index >= device_count:
                raise RuntimeError(f"Camera index {index} out of range for {device_count} GenTL device(s)")

        self._acquirer = self._create_acquirer(serial, index)

        remote = self._acquirer.remote_device
        node_map = remote.node_map

        # print(dir(node_map))
        """
        ['AcquisitionBurstFrameCount', 'AcquisitionControl', 'AcquisitionFrameRate', 'AcquisitionMode',
        'AcquisitionStart', 'AcquisitionStop', 'AnalogControl', 'AutoFunctionsROI', 'AutoFunctionsROIEnable',
        'AutoFunctionsROIHeight', 'AutoFunctionsROILeft', 'AutoFunctionsROIPreset', 'AutoFunctionsROITop',
        'AutoFunctionsROIWidth', 'BinningHorizontal', 'BinningVertical', 'BlackLevel', 'CameraRegisterAddress',
        'CameraRegisterAddressSpace', 'CameraRegisterControl', 'CameraRegisterRead', 'CameraRegisterValue',
        'CameraRegisterWrite', 'Contrast', 'DecimationHorizontal', 'DecimationVertical', 'Denoise',
        'DeviceControl', 'DeviceFirmwareVersion', 'DeviceModelName', 'DeviceReset', 'DeviceSFNCVersionMajor',
        'DeviceSFNCVersionMinor', 'DeviceSFNCVersionSubMinor', 'DeviceScanType', 'DeviceSerialNumber',
        'DeviceTLType', 'DeviceTLVersionMajor', 'DeviceTLVersionMinor', 'DeviceTLVersionSubMinor',
        'DeviceTemperature', 'DeviceTemperatureSelector', 'DeviceType', 'DeviceUserID', 'DeviceVendorName',
        'DigitalIO', 'ExposureAuto', 'ExposureAutoHighlightReduction', 'ExposureAutoLowerLimit',
        'ExposureAutoReference', 'ExposureAutoUpperLimit', 'ExposureAutoUpperLimitAuto', 'ExposureTime',
        'GPIn', 'GPOut', 'Gain', 'GainAuto', 'GainAutoLowerLimit', 'GainAutoUpperLimit', 'Gamma', 'Height',
        'HeightMax', 'IMXLowLatencyTriggerMode', 'ImageFormatControl', 'OffsetAutoCenter', 'OffsetX', 'OffsetY',
        'PayloadSize', 'PixelFormat', 'ReverseX', 'ReverseY', 'Root', 'SensorHeight', 'SensorWidth', 'Sharpness',
        'ShowOverlay', 'SoftwareAnalogControl', 'SoftwareTransformControl', 'SoftwareTransformEnable',
        'StrobeDelay', 'StrobeDuration', 'StrobeEnable', 'StrobeOperation', 'StrobePolarity', 'TLParamsLocked',
        'TestControl', 'TestPendingAck', 'TimestampLatch', 'TimestampLatchValue', 'TimestampReset', 'ToneMappingAuto',
        'ToneMappingControl', 'ToneMappingEnable', 'ToneMappingGlobalBrightness', 'ToneMappingIntensity',
        'TransportLayerControl', 'TriggerActivation', 'TriggerDebouncer', 'TriggerDelay', 'TriggerDenoise',
        'TriggerMask', 'TriggerMode', 'TriggerOverlap', 'TriggerSelector', 'TriggerSoftware', 'TriggerSource',
        'UserSetControl', 'UserSetDefault', 'UserSetLoad', 'UserSetSave', 'UserSetSelector', 'Width', 'WidthMax']
        """

        self._device_label = self._resolve_device_label(node_map)

        self._configure_pixel_format(node_map)
        self._configure_resolution(node_map)
        self._configure_exposure(node_map)
        self._configure_gain(node_map)
        self._configure_frame_rate(node_map)

        self._acquirer.start()

    def read(self) -> tuple[np.ndarray, float]:
        if self._acquirer is None:
            raise RuntimeError("GenTL image acquirer not initialised")

        try:
            with self._acquirer.fetch(timeout=self._timeout) as buffer:
                component = buffer.payload.components[0]
                channels = 3 if self._pixel_format in {"RGB8", "BGR8"} else 1
                array = np.asarray(component.data)
                expected = component.height * component.width * channels
                if array.size != expected:
                    array = np.frombuffer(bytes(component.data), dtype=array.dtype)
                try:
                    if channels > 1:
                        frame = array.reshape(component.height, component.width, channels).copy()
                    else:
                        frame = array.reshape(component.height, component.width).copy()
                except ValueError:
                    frame = array.copy()
        except HarvesterTimeoutError as exc:
            raise TimeoutError(str(exc) + " (GenTL timeout)") from exc

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

        self._device_label = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_cti_paths(self, value) -> tuple[str, ...]:
        if value is None:
            return self._DEFAULT_CTI_PATTERNS
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Iterable):
            return tuple(str(item) for item in value)
        return self._DEFAULT_CTI_PATTERNS

    def _parse_crop(self, crop) -> tuple[int, int, int, int] | None:
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            return tuple(int(v) for v in crop)
        return None

    def _parse_resolution(self, resolution) -> tuple[int, int] | None:
        """Parse resolution setting.

        Args:
            resolution: Can be a tuple/list [width, height], or None

        Returns:
            Tuple of (width, height) or None if not specified
            Default is (720, 540) if parsing fails but value is provided
        """
        if resolution is None:
            return (720, 540)  # Default resolution

        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            try:
                return (int(resolution[0]), int(resolution[1]))
            except (ValueError, TypeError):
                return (720, 540)

        return (720, 540)

    @staticmethod
    def _search_cti_file(patterns: tuple[str, ...]) -> str | None:
        """Search for a CTI file using the given patterns.

        Returns the first CTI file found, or None if none found.
        """
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    return file_path
        return None

    def _find_cti_file(self) -> str:
        """Find a CTI file using configured or default search paths.

        Raises RuntimeError if no CTI file is found.
        """
        cti_file = self._search_cti_file(self._cti_search_paths)
        if cti_file is None:
            raise RuntimeError(
                "Could not locate a GenTL producer (.cti) file. Set 'cti_file' in "
                "camera.properties or provide search paths via 'cti_search_paths'."
            )
        return cti_file

    def _available_serials(self) -> list[str]:
        assert self._harvester is not None
        serials: list[str] = []
        for info in self._harvester.device_info_list:
            serial = getattr(info, "serial_number", "")
            if serial:
                serials.append(serial)
        return serials

    def _create_acquirer(self, serial: str | None, index: int):
        assert self._harvester is not None
        methods = [
            getattr(self._harvester, "create", None),
            getattr(self._harvester, "create_image_acquirer", None),
        ]
        methods = [m for m in methods if m is not None]
        errors: list[str] = []
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
                actual = node_map.PixelFormat.value
                if actual != self._pixel_format:
                    LOG.warning(f"Pixel format mismatch: requested '{self._pixel_format}', got '{actual}'")
                else:
                    LOG.info(f"Pixel format set to '{actual}'")
            else:
                LOG.warning(
                    f"Pixel format '{self._pixel_format}' not in available formats: {node_map.PixelFormat.symbolics}"
                )
        except Exception as e:
            LOG.warning(f"Failed to set pixel format '{self._pixel_format}': {e}")

    def _configure_resolution(self, node_map) -> None:
        """Configure camera resolution (width and height)."""
        if self._resolution is None:
            return

        requested_width, requested_height = self._resolution
        actual_width, actual_height = None, None

        # Try to set width
        for width_attr in ("Width", "WidthMax"):
            try:
                node = getattr(node_map, width_attr)
                if width_attr == "Width":
                    # Get constraints
                    try:
                        min_w = node.min
                        max_w = node.max
                        inc_w = getattr(node, "inc", 1)
                        # Adjust to valid value
                        width = self._adjust_to_increment(requested_width, min_w, max_w, inc_w)
                        if width != requested_width:
                            LOG.info(
                                f"Width adjusted from {requested_width} to {width} "
                                f"(min={min_w}, max={max_w}, inc={inc_w})"
                            )
                        node.value = int(width)
                        actual_width = node.value
                        break
                    except Exception as e:
                        # Try setting without adjustment
                        try:
                            node.value = int(requested_width)
                            actual_width = node.value
                            break
                        except Exception:
                            LOG.warning(f"Failed to set width via {width_attr}: {e}")
                            continue
            except AttributeError:
                continue

        # Try to set height
        for height_attr in ("Height", "HeightMax"):
            try:
                node = getattr(node_map, height_attr)
                if height_attr == "Height":
                    # Get constraints
                    try:
                        min_h = node.min
                        max_h = node.max
                        inc_h = getattr(node, "inc", 1)
                        # Adjust to valid value
                        height = self._adjust_to_increment(requested_height, min_h, max_h, inc_h)
                        if height != requested_height:
                            LOG.info(
                                f"Height adjusted from {requested_height} to {height} "
                                f"(min={min_h}, max={max_h}, inc={inc_h})"
                            )
                        node.value = int(height)
                        actual_height = node.value
                        break
                    except Exception as e:
                        # Try setting without adjustment
                        try:
                            node.value = int(requested_height)
                            actual_height = node.value
                            break
                        except Exception:
                            LOG.warning(f"Failed to set height via {height_attr}: {e}")
                            continue
            except AttributeError:
                continue

        # Log final resolution
        if actual_width is not None and actual_height is not None:
            if actual_width != requested_width or actual_height != requested_height:
                LOG.warning(
                    f"Resolution mismatch: requested {requested_width}x{requested_height}, "
                    f"got {actual_width}x{actual_height}"
                )
            else:
                LOG.info(f"Resolution set to {actual_width}x{actual_height}")
        else:
            LOG.warning(f"Could not verify resolution setting (width={actual_width}, height={actual_height})")

    def _configure_exposure(self, node_map) -> None:
        if self._exposure is None:
            return

        # Try to disable auto exposure first
        for attr in ("ExposureAuto",):
            try:
                node = getattr(node_map, attr)
                node.value = "Off"
                LOG.info("Auto exposure disabled")
                break
            except AttributeError:
                continue
            except Exception as e:
                LOG.warning(f"Failed to disable auto exposure: {e}")

        # Set exposure value
        for attr in ("ExposureTime", "Exposure"):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                node.value = float(self._exposure)
                actual = node.value
                if abs(actual - self._exposure) > 1.0:  # Allow 1μs tolerance
                    LOG.warning(f"Exposure mismatch: requested {self._exposure}μs, got {actual}μs")
                else:
                    LOG.info(f"Exposure set to {actual}μs")
                return
            except Exception as e:
                LOG.warning(f"Failed to set exposure via {attr}: {e}")
                continue

        LOG.warning(f"Could not set exposure to {self._exposure}μs (no compatible attribute found)")

    def _configure_gain(self, node_map) -> None:
        if self._gain is None:
            return

        # Try to disable auto gain first
        for attr in ("GainAuto",):
            try:
                node = getattr(node_map, attr)
                node.value = "Off"
                LOG.info("Auto gain disabled")
                break
            except AttributeError:
                continue
            except Exception as e:
                LOG.warning(f"Failed to disable auto gain: {e}")

        # Set gain value
        for attr in ("Gain",):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                node.value = float(self._gain)
                actual = node.value
                if abs(actual - self._gain) > 0.1:  # Allow 0.1 tolerance
                    LOG.warning(f"Gain mismatch: requested {self._gain}, got {actual}")
                else:
                    LOG.info(f"Gain set to {actual}")
                return
            except Exception as e:
                LOG.warning(f"Failed to set gain via {attr}: {e}")
                continue

        LOG.warning(f"Could not set gain to {self._gain} (no compatible attribute found)")

    def _configure_frame_rate(self, node_map) -> None:
        if not self.settings.fps:
            return

        target = float(self.settings.fps)

        # Try to enable frame rate control
        for attr in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateControlEnable"):
            try:
                getattr(node_map, attr).value = True
                LOG.info(f"Frame rate control enabled via {attr}")
                break
            except Exception:
                continue

        # Set frame rate value
        for attr in ("AcquisitionFrameRate", "ResultingFrameRate", "AcquisitionFrameRateAbs"):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                node.value = target
                actual = node.value
                if abs(actual - target) > 0.1:
                    LOG.warning(f"FPS mismatch: requested {target:.2f}, got {actual:.2f}")
                else:
                    LOG.info(f"Frame rate set to {actual:.2f} FPS")
                return
            except Exception as e:
                LOG.warning(f"Failed to set frame rate via {attr}: {e}")
                continue

        LOG.warning(f"Could not set frame rate to {target} FPS (no compatible attribute found)")

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            max_val = float(frame.max()) if frame.size else 0.0
            scale = 255.0 / max_val if max_val > 0.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3 and self._pixel_format == "RGB8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._crop is not None:
            top, bottom, left, right = (int(v) for v in self._crop)
            top = max(0, top)
            left = max(0, left)
            bottom = bottom if bottom > 0 else frame.shape[0]
            right = right if right > 0 else frame.shape[1]
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)
            frame = frame[top:bottom, left:right]

        if self._rotate in (90, 180, 270):
            rotations = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            frame = cv2.rotate(frame, rotations[self._rotate])

        return frame.copy()

    def _resolve_device_label(self, node_map) -> str | None:
        candidates = [
            ("DeviceModelName", "DeviceSerialNumber"),
            ("DeviceDisplayName", "DeviceSerialNumber"),
        ]
        for name_attr, serial_attr in candidates:
            try:
                model = getattr(node_map, name_attr).value
            except AttributeError:
                continue
            serial = None
            try:
                serial = getattr(node_map, serial_attr).value
            except AttributeError:
                pass
            if model:
                model_str = str(model)
                serial_str = str(serial) if serial else None
                return f"{model_str} ({serial_str})" if serial_str else model_str
        return None

    def _adjust_to_increment(self, value: int, minimum: int, maximum: int, increment: int) -> int:
        value = max(minimum, min(maximum, int(value)))
        if increment <= 0:
            return value
        offset = value - minimum
        steps = offset // increment
        return minimum + steps * increment

    def device_name(self) -> str:
        if self._device_label:
            return self._device_label
        return super().device_name()
