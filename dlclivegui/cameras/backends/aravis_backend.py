"""Aravis backend for GenICam cameras."""

# dlclivegui/cameras/backends/aravis_backend.py
from __future__ import annotations

import logging
import time
from typing import ClassVar

import cv2
import numpy as np

from ...config import CameraSettings
from ..base import CameraBackend, SupportLevel, register_backend
from ..factory import DetectedCamera

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

    OPTIONS_KEY: ClassVar[str] = "aravis"

    def __init__(self, settings):
        super().__init__(settings)

        props = settings.properties if isinstance(settings.properties, dict) else {}
        ns = props.get(self.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}

        self._camera_id: str | None = ns.get("device_id") or props.get("device_id")
        self._pixel_format: str = ns.get("pixel_format") or props.get("pixel_format", "Mono8")
        self._timeout: int = int(ns.get("timeout", props.get("timeout", 2_000_000)))
        self._n_buffers: int = int(ns.get("n_buffers", props.get("n_buffers", 10)))

        # Resolution handling
        self._requested_resolution: tuple[int, int] | None = self._get_requested_resolution_or_none()
        self._actual_width: int | None = None
        self._actual_height: int | None = None
        self._actual_fps: float | None = None

        self._camera = None
        self._stream = None
        self._device_label: str | None = None

    @property
    def actual_resolution(self) -> tuple[int, int] | None:
        """Return the actual resolution of the camera after opening."""
        if self._actual_width is not None and self._actual_height is not None:
            return (self._actual_width, self._actual_height)
        return None

    @property
    def actual_fps(self) -> float | None:
        """Return the actual frame rate of the camera after opening."""
        return self._actual_fps

    @classmethod
    def is_available(cls) -> bool:
        """Check if Aravis is available on this system."""
        return ARAVIS_AVAILABLE

    @classmethod
    def static_capabilities(cls) -> dict[str, SupportLevel]:
        """Return a dict describing supported features for UI purposes."""
        caps = super().static_capabilities()
        caps.update(
            {
                "set_resolution": SupportLevel.SUPPORTED,
                "set_fps": SupportLevel.SUPPORTED,
                "set_exposure": SupportLevel.SUPPORTED,
                "set_gain": SupportLevel.SUPPORTED,
                "device_discovery": SupportLevel.SUPPORTED,
                "stable_identity": SupportLevel.SUPPORTED,
            }
        )
        return caps

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

    @classmethod
    def quick_ping(cls, index: int, *_args, **_kwargs) -> bool:
        """
        Cheap presence test for CameraFactory probing.
        Uses update_device_list() then bounds-check.
        """
        if not ARAVIS_AVAILABLE:
            return False
        try:
            Aravis.update_device_list()
            n = int(Aravis.get_n_devices() or 0)
            return 0 <= int(index) < n
        except Exception:
            return False

    @classmethod
    def discover_devices(cls, max_devices: int = 10, should_cancel=None, progress_cb=None):
        if not ARAVIS_AVAILABLE:
            return []

        # Refresh list once; indices may change after update_device_list()
        Aravis.update_device_list()

        snap = cls._arv_snapshot_devices(limit=max_devices)

        cams: list[DetectedCamera] = []
        for d in snap:
            if should_cancel and should_cancel():
                break
            if progress_cb:
                progress_cb(f"Found {d['label']}")

            path = d.get("physical_id") or d.get("address")

            cams.append(
                DetectedCamera(
                    index=int(d["index"]),
                    label=str(d["label"]),
                    device_id=d.get("device_id"),
                    path=path,
                )
            )
        return cams

    @classmethod
    def rebind_settings(cls, settings: CameraSettings) -> CameraSettings:
        """
        Best-effort quick rebind using only Aravis enumeration APIs (no camera open).
        Indices may change after Aravis.update_device_list().
        """
        if not ARAVIS_AVAILABLE:
            return settings

        props = settings.properties if isinstance(settings.properties, dict) else {}
        ns = props.get(cls.OPTIONS_KEY, {}) if isinstance(props.get(cls.OPTIONS_KEY), dict) else {}

        # Stored identifiers (some may be missing)
        stored_device_id = cls._safe_str(
            ns.get("device_id") or props.get("device_id") or ns.get("camera_id") or props.get("camera_id")
        )
        stored_physical = cls._safe_str(
            ns.get("device_physical_id") or ns.get("device_path") or props.get("device_path")
        )
        stored_vendor = cls._safe_str(ns.get("device_vendor"))
        stored_model = cls._safe_str(ns.get("device_model"))
        stored_serial = cls._safe_str(ns.get("device_serial_nbr") or ns.get("device_serial"))
        stored_name = cls._safe_str(ns.get("device_name"))

        # Nothing to rebind with
        if not any(
            [stored_device_id, stored_physical, (stored_vendor and stored_model and stored_serial), stored_name]
        ):
            return settings

        try:
            Aravis.update_device_list()  # must be called before get_device_*
            snap = cls._arv_snapshot_devices(limit=None)

            # 1) device_id exact match (fast)
            chosen = None
            if stored_device_id:
                for d in snap:
                    if d.get("device_id") == stored_device_id:
                        chosen = d
                        break

            # 2) physical_id exact match
            if chosen is None and stored_physical:
                for d in snap:
                    if d.get("physical_id") == stored_physical or d.get("address") == stored_physical:
                        chosen = d
                        break

            # 3) vendor/model/serial exact triple match
            if chosen is None and stored_vendor and stored_model and stored_serial:
                for d in snap:
                    if (d.get("vendor"), d.get("model"), d.get("serial")) == (
                        stored_vendor,
                        stored_model,
                        stored_serial,
                    ):
                        chosen = d
                        break

            # 4) name substring match against computed label
            if chosen is None and stored_name:
                needle = stored_name.lower()
                for d in snap:
                    label = (d.get("label") or "").lower()
                    if needle and needle in label:
                        chosen = d
                        break

            # 5) fallback to current index if still plausible
            if chosen is None:
                idx = int(getattr(settings, "index", 0) or 0)
                if 0 <= idx < len(snap):
                    chosen = snap[idx]
                else:
                    return settings

            # Apply new index
            settings.index = int(chosen["index"])

            # Refresh namespace fields (keeps GUI stable identity fresh)
            if isinstance(settings.properties, dict):
                out = settings.properties.setdefault(cls.OPTIONS_KEY, {})
                if isinstance(out, dict):
                    out["device_id"] = chosen.get("device_id")
                    out["device_physical_id"] = chosen.get("physical_id")
                    out["device_vendor"] = chosen.get("vendor")
                    out["device_model"] = chosen.get("model")
                    out["device_serial_nbr"] = chosen.get("serial")
                    out["device_protocol"] = chosen.get("protocol")
                    out["device_address"] = chosen.get("address")
                    out["device_name"] = chosen.get("label")  # computed label (no open)

                    # also keep 'device_path' aligned with physical id for GUI fallback
                    if chosen.get("physical_id"):
                        out["device_path"] = chosen.get("physical_id")

            return settings

        except Exception:
            # Never hard-fail creation just because rebinding couldn't happen
            return settings

    def open(self) -> None:
        if not ARAVIS_AVAILABLE:
            raise RuntimeError("Aravis library not available")

        Aravis.update_device_list()
        n_devices = Aravis.get_n_devices()
        if n_devices == 0:
            raise RuntimeError("No Aravis cameras detected")

        if self._camera_id:
            self._camera = Aravis.Camera.new(self._camera_id)
        else:
            index = int(self.settings.index or 0)
            if index < 0 or index >= n_devices:
                raise RuntimeError(f"Camera index {index} out of range for {n_devices} Aravis device(s)")
            camera_id = Aravis.get_device_id(index)
            self._camera = Aravis.Camera.new(camera_id)
            self._camera_id = self._safe_str(camera_id)

        if self._camera is None:
            raise RuntimeError("Failed to open Aravis camera")

        # --- Refresh identity and align index (best-effort, no heavy open needed) ---
        try:
            snap = self._arv_snapshot_devices(limit=None)

            opened_id = self._camera_id
            if opened_id is None:
                # Opened by index
                try:
                    opened_id = self._safe_str(Aravis.get_device_id(int(self.settings.index)))
                except Exception:
                    opened_id = None

            chosen = None
            if opened_id:
                for d in snap:
                    if d.get("device_id") == opened_id:
                        chosen = d
                        break

            # If we found it, align settings.index and refresh identity cache
            if chosen:
                self.settings.index = int(chosen["index"])
                if isinstance(self.settings.properties, dict):
                    ns = self.settings.properties.setdefault(self.OPTIONS_KEY, {})
                    if isinstance(ns, dict):
                        ns["device_id"] = chosen.get("device_id")
                        ns["device_physical_id"] = chosen.get("physical_id")
                        ns["device_vendor"] = chosen.get("vendor")
                        ns["device_model"] = chosen.get("model")
                        ns["device_serial_nbr"] = chosen.get("serial")
                        ns["device_protocol"] = chosen.get("protocol")
                        ns["device_address"] = chosen.get("address")
                        ns["device_path"] = chosen.get("physical_id") or chosen.get("address")
                else:
                    if isinstance(self.settings.properties, dict):
                        ns = self.settings.properties.setdefault(self.OPTIONS_KEY, {})
                        if isinstance(ns, dict):
                            ns["device_id"] = opened_id
        except Exception:
            pass

        # Compute higher-quality label from the opened camera object
        self._device_label = self._resolve_device_label()
        # Always populate minimal identity into backend namespace for GUI
        if isinstance(self.settings.properties, dict):
            ns = self.settings.properties.setdefault(self.OPTIONS_KEY, {})
            if isinstance(ns, dict):
                # Always write a device_id after a successful open
                try:
                    if self._camera_id:
                        ns["device_id"] = self._camera_id
                    else:
                        ns["device_id"] = self._safe_str(Aravis.get_device_id(int(self.settings.index)))
                except Exception:
                    pass
                if self._device_label:
                    ns["device_name"] = self._device_label

        self._configure_pixel_format()
        self._configure_resolution()
        self._configure_exposure()
        self._configure_gain()
        self._configure_frame_rate()

        # Capture actual resolution even when using defaults
        try:
            self._actual_width = int(self._camera.get_integer("Width"))
            self._actual_height = int(self._camera.get_integer("Height"))
        except Exception:
            pass

        try:
            self._actual_fps = float(self._camera.get_float("AcquisitionFrameRate"))
        except Exception:
            self._actual_fps = None

        self._stream = self._camera.create_stream(None, None)
        if self._stream is None:
            raise RuntimeError("Failed to create Aravis stream")

        payload_size = self._camera.get_payload()
        for _ in range(self._n_buffers):
            self._stream.push_buffer(Aravis.Buffer.new_allocate(payload_size))

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

            if self._actual_width is None or self._actual_height is None:
                self._actual_width = int(width)
                self._actual_height = int(height)

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
    @staticmethod
    def _safe_str(x) -> str | None:
        try:
            if x is None:
                return None
            s = str(x).strip()
            return s if s else None
        except Exception:
            return None

    @classmethod
    def _arv_snapshot_devices(cls, limit: int | None = None) -> list[dict]:
        """
        Fast snapshot of the current Aravis device list without opening cameras.
        Requires Aravis.update_device_list() before calling.
        """
        n = int(Aravis.get_n_devices() or 0)  # valid until next update_device_list()
        if limit is not None:
            n = min(n, int(limit))

        devices: list[dict] = []
        for i in range(n):
            try:
                dev_id = cls._safe_str(Aravis.get_device_id(i))
            except Exception:
                dev_id = None

            try:
                physical = cls._safe_str(Aravis.get_device_physical_id(i))
            except Exception:
                physical = None
            try:
                vendor = cls._safe_str(Aravis.get_device_vendor(i))
            except Exception:
                vendor = None
            try:
                model = cls._safe_str(Aravis.get_device_model(i))
            except Exception:
                model = None
            try:
                serial = cls._safe_str(Aravis.get_device_serial_nbr(i))
            except Exception:
                serial = None
            try:
                protocol = cls._safe_str(Aravis.get_device_protocol(i))
            except Exception:
                protocol = None
            try:
                address = cls._safe_str(Aravis.get_device_address(i))
            except Exception:
                address = None

            # Construct a stable-ish human label without opening the camera
            label_parts = [p for p in (vendor, model) if p]
            label = " ".join(label_parts) if label_parts else None
            if serial:
                label = f"{label} ({serial})" if label else f"({serial})"
            if not label:
                label = dev_id or f"Aravis #{i}"

            devices.append(
                {
                    "index": int(i),
                    "device_id": dev_id,
                    "physical_id": physical,
                    "vendor": vendor,
                    "model": model,
                    "serial": serial,
                    "protocol": protocol,
                    "address": address,
                    "label": label,
                }
            )
        return devices

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
            self._camera.set_integer("Width", int(req_w))
            self._camera.set_integer("Height", int(req_h))

            aw = int(self._camera.get_integer("Width"))
            ah = int(self._camera.get_integer("Height"))
            self._actual_width = aw
            self._actual_height = ah

            if (aw, ah) != (req_w, req_h):
                LOG.warning(f"Resolution mismatch: requested {req_w}x{req_h}, got {aw}x{ah}")
            else:
                LOG.info(f"Resolution set to {aw}x{ah}")
        except Exception as exc:
            LOG.warning(f"Failed to set resolution to {req_w}x{req_h}: {exc}")

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
