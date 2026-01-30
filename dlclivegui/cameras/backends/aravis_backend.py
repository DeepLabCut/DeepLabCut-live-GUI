"""Aravis backend for GenICam cameras."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ..base import CameraBackend, register_backend

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis

    ARAVIS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Aravis = None  # type: ignore
    ARAVIS_AVAILABLE = False


@register_backend("aravis")
class AravisCameraBackend(CameraBackend):
    """Capture frames from GenICam-compatible devices via Aravis."""

    def __init__(self, settings):
        super().__init__(settings)
        props = settings.properties
        self._camera_id: str | None = props.get("camera_id")
        self._pixel_format: str = props.get("pixel_format", "Mono8")
        self._timeout: int = int(props.get("timeout", 2000000))  # microseconds
        self._n_buffers: int = int(props.get("n_buffers", 10))

        self._camera = None
        self._stream = None
        self._device_label: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Aravis is available on this system."""
        return ARAVIS_AVAILABLE

    @classmethod
    def get_device_count(cls) -> int:
        """Get the actual number of Aravis devices detected.

        Returns the number of devices found, or -1 if detection fails.
        """
        if not ARAVIS_AVAILABLE:
            return -1

        try:
            Aravis.update_device_list()
            return Aravis.get_n_devices()
        except Exception:
            return -1

    def open(self) -> None:
        """Open the Aravis camera device."""
        if not ARAVIS_AVAILABLE:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'aravis' library is required for the Aravis backend. "
                "Install it via your system package manager (e.g., 'sudo apt install gir1.2-aravis-0.8' on Ubuntu)."
            )

        # Update device list
        Aravis.update_device_list()
        n_devices = Aravis.get_n_devices()

        if n_devices == 0:
            raise RuntimeError("No Aravis cameras detected")

        # Open camera by ID or index
        if self._camera_id:
            self._camera = Aravis.Camera.new(self._camera_id)
            if self._camera is None:
                raise RuntimeError(f"Failed to open camera with ID '{self._camera_id}'")
        else:
            index = int(self.settings.index or 0)
            if index < 0 or index >= n_devices:
                raise RuntimeError(f"Camera index {index} out of range for {n_devices} Aravis device(s)")
            camera_id = Aravis.get_device_id(index)
            self._camera = Aravis.Camera.new(camera_id)
            if self._camera is None:
                raise RuntimeError(f"Failed to open camera at index {index}")

        # Get device information for label
        self._device_label = self._resolve_device_label()

        # Configure camera
        self._configure_pixel_format()
        self._configure_exposure()
        self._configure_gain()
        self._configure_frame_rate()

        # Create stream
        self._stream = self._camera.create_stream(None, None)
        if self._stream is None:
            raise RuntimeError("Failed to create Aravis stream")

        # Push buffers to stream
        payload_size = self._camera.get_payload()
        for _ in range(self._n_buffers):
            self._stream.push_buffer(Aravis.Buffer.new_allocate(payload_size))

        # Start acquisition
        self._camera.start_acquisition()

    def read(self) -> tuple[np.ndarray, float]:
        """Read a frame from the camera."""
        if self._camera is None or self._stream is None:
            raise RuntimeError("Aravis camera not initialized")

        # Pop buffer from stream
        buffer = self._stream.timeout_pop_buffer(self._timeout)

        if buffer is None:
            raise TimeoutError("Failed to grab frame from Aravis camera (timeout)")

        # Check buffer status
        status = buffer.get_status()
        if status != Aravis.BufferStatus.SUCCESS:
            self._stream.push_buffer(buffer)
            raise TimeoutError(f"Aravis buffer status error: {status}")

        # Get image data
        try:
            # Get buffer data as numpy array
            data = buffer.get_data()
            width = buffer.get_image_width()
            height = buffer.get_image_height()
            pixel_format = buffer.get_image_pixel_format()

            # Convert to numpy array
            if pixel_format == Aravis.PIXEL_FORMAT_MONO_8:
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif pixel_format == Aravis.PIXEL_FORMAT_RGB_8_PACKED:
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif pixel_format == Aravis.PIXEL_FORMAT_BGR_8_PACKED:
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            elif pixel_format in (Aravis.PIXEL_FORMAT_MONO_12, Aravis.PIXEL_FORMAT_MONO_16):
                # Handle 12-bit and 16-bit mono
                frame = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
                # Scale to 8-bit
                max_val = float(frame.max()) if frame.size else 0.0
                scale = 255.0 / max_val if max_val > 0.0 else 1.0
                frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                # Fallback for unknown formats - try to interpret as mono8
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame = frame.copy()
            timestamp = time.time()

        finally:
            # Always push buffer back to stream
            self._stream.push_buffer(buffer)

        return frame, timestamp

    def stop(self) -> None:
        """Stop camera acquisition."""
        if self._camera is not None:
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass

    def close(self) -> None:
        """Release the camera and stream."""
        if self._camera is not None:
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass

            # Clear stream buffers
            if self._stream is not None:
                try:
                    # Flush remaining buffers
                    while True:
                        buffer = self._stream.try_pop_buffer()
                        if buffer is None:
                            break
                except Exception:
                    pass
                self._stream = None

            # Release camera
            try:
                del self._camera
            except Exception:
                pass
            finally:
                self._camera = None

        self._device_label = None

    def device_name(self) -> str:
        """Return a human-readable device name."""
        if self._device_label:
            return self._device_label
        return super().device_name()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _configure_pixel_format(self) -> None:
        """Configure the camera pixel format."""
        if self._camera is None:
            return

        try:
            # Map common format names to Aravis pixel formats
            format_map = {
                "Mono8": Aravis.PIXEL_FORMAT_MONO_8,
                "Mono12": Aravis.PIXEL_FORMAT_MONO_12,
                "Mono16": Aravis.PIXEL_FORMAT_MONO_16,
                "RGB8": Aravis.PIXEL_FORMAT_RGB_8_PACKED,
                "BGR8": Aravis.PIXEL_FORMAT_BGR_8_PACKED,
            }

            if self._pixel_format in format_map:
                self._camera.set_pixel_format(format_map[self._pixel_format])
                LOG.info(f"Pixel format set to '{self._pixel_format}'")
            else:
                # Try setting as string
                self._camera.set_pixel_format_from_string(self._pixel_format)
                LOG.info(f"Pixel format set to '{self._pixel_format}' (from string)")
        except Exception as e:
            LOG.warning(f"Failed to set pixel format '{self._pixel_format}': {e}")

    def _configure_exposure(self) -> None:
        """Configure camera exposure time."""
        if self._camera is None:
            return

        # Get exposure from settings
        exposure = None
        if hasattr(self.settings, "exposure") and self.settings.exposure > 0:
            exposure = float(self.settings.exposure)

        if exposure is None:
            return

        try:
            # Disable auto exposure
            try:
                self._camera.set_exposure_time_auto(Aravis.Auto.OFF)
                LOG.info("Auto exposure disabled")
            except Exception as e:
                LOG.warning(f"Failed to disable auto exposure: {e}")

            # Set exposure time (in microseconds)
            self._camera.set_exposure_time(exposure)
            actual = self._camera.get_exposure_time()
            if abs(actual - exposure) > 1.0:  # Allow 1μs tolerance
                LOG.warning(f"Exposure mismatch: requested {exposure}μs, got {actual}μs")
            else:
                LOG.info(f"Exposure set to {actual}μs")
        except Exception as e:
            LOG.warning(f"Failed to set exposure to {exposure}μs: {e}")

    def _configure_gain(self) -> None:
        """Configure camera gain."""
        if self._camera is None:
            return

        # Get gain from settings
        gain = None
        if hasattr(self.settings, "gain") and self.settings.gain > 0.0:
            gain = float(self.settings.gain)

        if gain is None:
            return

        try:
            # Disable auto gain
            try:
                self._camera.set_gain_auto(Aravis.Auto.OFF)
                LOG.info("Auto gain disabled")
            except Exception as e:
                LOG.warning(f"Failed to disable auto gain: {e}")

            # Set gain value
            self._camera.set_gain(gain)
            actual = self._camera.get_gain()
            if abs(actual - gain) > 0.1:  # Allow 0.1 tolerance
                LOG.warning(f"Gain mismatch: requested {gain}, got {actual}")
            else:
                LOG.info(f"Gain set to {actual}")
        except Exception as e:
            LOG.warning(f"Failed to set gain to {gain}: {e}")

    def _configure_frame_rate(self) -> None:
        """Configure camera frame rate."""
        if self._camera is None or not self.settings.fps:
            return

        try:
            target_fps = float(self.settings.fps)
            self._camera.set_frame_rate(target_fps)
            actual_fps = self._camera.get_frame_rate()
            if abs(actual_fps - target_fps) > 0.1:
                LOG.warning(f"FPS mismatch: requested {target_fps:.2f}, got {actual_fps:.2f}")
            else:
                LOG.info(f"Frame rate set to {actual_fps:.2f} FPS")
        except Exception as e:
            LOG.warning(f"Failed to set frame rate to {self.settings.fps}: {e}")

    def _resolve_device_label(self) -> str | None:
        """Get a human-readable device label."""
        if self._camera is None:
            return None

        try:
            model = self._camera.get_model_name()
            vendor = self._camera.get_vendor_name()
            serial = self._camera.get_device_serial_number()

            if model and serial:
                if vendor:
                    return f"{vendor} {model} ({serial})"
                return f"{model} ({serial})"
            elif model:
                return model
            elif serial:
                return f"Camera {serial}"
        except Exception:
            pass

        return None
