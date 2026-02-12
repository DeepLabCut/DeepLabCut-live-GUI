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

        self._props: dict = settings.properties if isinstance(settings.properties, dict) else {}

        # Optional fast-start hint for probe workers (best-effort; doesn't change behavior yet)
        self._fast_start: bool = bool(self.ns.get("fast_start", False))
        self._apply_transforms: bool = bool(self.ns.get("apply_transforms", False))

        # Stable identity (serial-based). Prefer new namespace; fall back to legacy keys read-only.
        self._device_id: str | None = None
        dev_id = self.ns.get("device_id")
        if dev_id:
            self._device_id = str(dev_id)
        else:
            # legacy fallback (read-only)
            legacy_serial = None
            try:
                legacy_serial = self._props.get("serial") or self._props.get("serial_number")
            except Exception:
                legacy_serial = None
            if legacy_serial:
                self._device_id = str(legacy_serial)

        self._requested_resolution: tuple[int, int] | None = self._get_requested_resolution_or_none()

        # ---- Runtime handles (set during open) ----
        self._camera: pylon.InstantCamera | None = None
        self._converter: pylon.ImageFormatConverter | None = None

        # ---- Actuals for GUI telemetry ----
        self._actual_width: int | None = None
        self._actual_height: int | None = None
        self._actual_fps: float | None = None
        self._actual_exposure: float | None = None
        self._actual_gain: float | None = None

    @property
    def actual_resolution(self) -> tuple[int, int] | None:
        if self._actual_width and self._actual_height:
            return (self._actual_width, self._actual_height)
        return None

    @property
    def actual_fps(self) -> float | None:
        return self._actual_fps

    @property
    def actual_exposure(self) -> float | None:
        return self._actual_exposure

    @property
    def actual_gain(self) -> float | None:
        return self._actual_gain

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

    @property
    def ns(self) -> dict:
        """Basler namespace view (read-only). Always derived from current settings.properties."""
        return self.__class__._ns_from_settings(self.settings)

    @classmethod
    def _ns_from_settings(cls, settings) -> dict:
        """Return basler namespace dict from a settings object (read-only, safe)."""
        props = settings.properties if isinstance(settings.properties, dict) else {}
        ns = props.get(cls.OPTIONS_KEY, {})
        return ns if isinstance(ns, dict) else {}

    def _ensure_mutable_ns(self) -> dict:
        """Ensure settings.properties and its basler namespace dict exist; return the namespace."""
        if not isinstance(self.settings.properties, dict):
            self.settings.properties = {}
        ns = self.settings.properties.get(self.OPTIONS_KEY)
        if not isinstance(ns, dict):
            ns = {}
            self.settings.properties[self.OPTIONS_KEY] = ns
        return ns

    @classmethod
    def _enumerate_devices_cls(cls):
        """Enumerate DeviceInfo entries (unit-testable via monkeypatch)."""
        if pylon is None:
            return []
        factory = pylon.TlFactory.GetInstance()
        return factory.EnumerateDevices()

    @classmethod
    def get_device_count(cls) -> int:
        """Return the number of Basler devices visible to Pylon."""
        try:
            return len(cls._enumerate_devices_cls())
        except Exception:
            return 0

    @classmethod
    def quick_ping(cls, index: int, *args, **kwargs) -> bool:
        """Best-effort presence check; avoids opening the device."""
        try:
            devices = cls._enumerate_devices_cls()
            idx = int(index)
            return 0 <= idx < len(devices)
        except Exception:
            return False

    @classmethod
    def discover_devices(
        cls,
        *,
        max_devices: int = 10,
        should_cancel=None,
        progress_cb=None,
    ):
        """
        Return a rich list of DetectedCamera with stable identity (serial).
        Best-effort: works for USB3/GigE; fields depend on SDK/device.
        """
        if pylon is None:
            return []

        from ..factory import DetectedCamera  # local import to keep module load light

        devices = cls._enumerate_devices_cls()
        out = []

        # Bound by max_devices to match factory expectations
        n = min(len(devices), int(max_devices) if max_devices is not None else len(devices))

        for i in range(n):
            if should_cancel and should_cancel():
                break
            if progress_cb:
                progress_cb(f"Reading Basler device info ({i + 1}/{n})…")

            di = devices[i]

            # Best-effort getters; not all are present on all transports
            serial = None
            try:
                serial = di.GetSerialNumber()
            except Exception:
                serial = None

            # Friendly label: Vendor Model (Serial)
            vendor = model = friendly = full_name = None
            try:
                vendor = di.GetVendorName()
            except Exception:
                pass
            try:
                model = di.GetModelName()
            except Exception:
                pass
            try:
                friendly = di.GetFriendlyName()
            except Exception:
                pass
            try:
                full_name = di.GetFullName()
            except Exception:
                pass

            label_parts = []
            if vendor:
                label_parts.append(str(vendor))
            if model:
                label_parts.append(str(model))
            if not label_parts and friendly:
                label_parts.append(str(friendly))
            label = " ".join(label_parts) if label_parts else f"Basler #{i}"
            if serial:
                label = f"{label} ({serial})"

            out.append(
                DetectedCamera(
                    index=i,
                    label=label,
                    device_id=str(serial) if serial else None,  # <-- stable identity
                    path=str(full_name) if full_name else None,
                )
            )

        return out

    @classmethod
    def rebind_settings(cls, settings):
        """
        If settings.properties['basler']['device_id'] (serial) exists,
        update settings.index to match the current device list order.
        """
        if pylon is None:
            return settings

        dc = settings.model_copy(deep=True)

        ns = cls._ns_from_settings(dc)
        serial = ns.get("device_id") or ns.get("serial")  # allow legacy-in-namespace

        # Legacy top-level fallback (read-only compatibility)
        if not serial:
            props = dc.properties if isinstance(dc.properties, dict) else {}
            serial = props.get("serial") or props.get("serial_number")

        if not serial:
            return dc

        try:
            devices = cls._enumerate_devices_cls()
            for i, di in enumerate(devices):
                try:
                    if di.GetSerialNumber() == serial:
                        dc.index = int(i)

                        # Ensure we persist stable ID in the basler namespace
                        if not isinstance(dc.properties, dict):
                            dc.properties = {}
                        bns = dc.properties.get(cls.OPTIONS_KEY)
                        if not isinstance(bns, dict):
                            bns = {}
                            dc.properties[cls.OPTIONS_KEY] = bns

                        bns["device_id"] = str(serial)  # canonical
                        # optional friendly name cache (nice for UI)
                        try:
                            bns["device_name"] = str(di.GetFriendlyName())
                        except Exception:
                            pass
                        try:
                            bns["device_path"] = str(di.GetFullName())
                        except Exception:
                            pass

                        return dc
                except Exception:
                    continue
        except Exception:
            pass

        return dc

    @classmethod
    def sanitize_for_probe(cls, settings):
        """
        Keep only basler namespace + set all requested controls to Auto
        so probing is fast and doesn't force modes.
        """
        dc = settings.model_copy(deep=True)

        # Keep only backend namespace dict
        ns = cls._ns_from_settings(dc)
        dc.properties = {cls.OPTIONS_KEY: dict(ns)}  # shallow copy ok

        # Force Auto for probe; do NOT set heavy parameters
        dc.width = 0
        dc.height = 0
        dc.fps = 0.0
        dc.exposure = 0
        dc.gain = 0.0
        dc.rotation = 0
        dc.crop_x0 = dc.crop_y0 = dc.crop_x1 = dc.crop_y1 = 0

        return dc

    @staticmethod
    def _positive_float(value) -> float | None:
        """Return float(value) if > 0 else None."""
        try:
            v = float(value)
            return v if v > 0 else None
        except Exception:
            return None

    @staticmethod
    def _apply_crop(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if x1 <= 0:
            x1 = w
        if y1 <= 0:
            y1 = h
        x0 = max(0, min(int(x0), w))
        y0 = max(0, min(int(y0), h))
        x1 = max(x0, min(int(x1), w))
        y1 = max(y0, min(int(y1), h))
        return frame[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else frame

    def open(self) -> None:
        if pylon is None:
            raise RuntimeError("pypylon is required for the Basler backend but is not installed")

        # ----------------------------
        # Device enumeration & selection
        # ----------------------------
        devices = self._enumerate_devices()
        if not devices:
            raise RuntimeError("No Basler cameras detected")

        device = self._select_device(devices)

        self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
        self._camera.Open()

        # ----------------------------
        # Exposure (0 = Auto → do not set)
        # ----------------------------
        exposure = self._positive_float(getattr(self.settings, "exposure", 0))

        if exposure is not None:
            try:
                self._camera.ExposureTime.SetValue(exposure)
            except Exception:
                LOG.debug("ExposureTime not writable or not supported", exc_info=True)

        # ----------------------------
        # Gain (0 = Auto → do not set)
        # ----------------------------
        gain = self._positive_float(getattr(self.settings, "gain", 0))

        if gain is not None:
            try:
                self._camera.Gain.SetValue(gain)
            except Exception:
                LOG.debug("Gain not writable or not supported", exc_info=True)

        # ----------------------------
        # Resolution (None → device default)
        # ----------------------------
        # Re-evaluate in case settings were rebound before open()
        self._requested_resolution = self._get_requested_resolution_or_none()
        self._configure_resolution()

        # ----------------------------
        # Frame rate (0.0 = Auto → do not set)
        # ----------------------------
        fps = self._positive_float(getattr(self.settings, "fps", 0.0))

        if fps is not None:
            try:
                # Some models require enable flag to be writable
                if hasattr(self._camera, "AcquisitionFrameRateEnable"):
                    try:
                        self._camera.AcquisitionFrameRateEnable.SetValue(True)
                    except Exception:
                        pass
                self._camera.AcquisitionFrameRate.SetValue(fps)
            except Exception:
                LOG.debug("Frame rate not writable or not supported", exc_info=True)

        # ----------------------------
        # Read back actual values (telemetry for GUI / probe)
        # ----------------------------
        try:
            self._actual_fps = float(self._camera.AcquisitionFrameRate.GetValue())
        except Exception:
            self._actual_fps = None

        try:
            self._actual_width = int(self._camera.Width.GetValue())
            self._actual_height = int(self._camera.Height.GetValue())
        except Exception:
            pass

        try:
            self._actual_exposure = float(self._camera.ExposureTime.GetValue())
        except Exception:
            self._actual_exposure = None

        try:
            self._actual_gain = float(self._camera.Gain.GetValue())
        except Exception:
            self._actual_gain = None

        # ----------------------------
        # Start acquisition (skip for fast probe)
        # ----------------------------
        if not self._fast_start:
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        else:
            LOG.debug("Fast-start probe: skipping StartGrabbing and converter")

        LOG.info(
            "[Basler] open device_id=%s index=%s fast_start=%s requested=(%sx%s @ %s fps exp=%s gain=%s)",
            getattr(self, "_device_id", None),
            getattr(self.settings, "index", None),
            getattr(self, "_fast_start", None),
            getattr(self.settings, "width", None),
            getattr(self.settings, "height", None),
            getattr(self.settings, "fps", None),
            getattr(self.settings, "exposure", None),
            getattr(self.settings, "gain", None),
        )
        # ----------------------------
        # Persist stable identity into namespace (migration-safe)
        # ----------------------------
        try:
            serial = device.GetSerialNumber()
            if serial:
                ns = self._ensure_mutable_ns()
                ns["device_id"] = str(serial)
                try:
                    ns["device_name"] = str(device.GetFriendlyName())
                except Exception:
                    pass
        except Exception:
            pass

    def read(self) -> tuple[np.ndarray, float]:
        if self._camera is None:
            raise RuntimeError("Basler camera not opened")
        if self._converter is None:
            raise RuntimeError("Basler camera opened in fast-start probe mode; cannot read frames")
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

        # --- Optional transforms ---
        if self._apply_transforms:
            # Rotation from CameraSettings
            rotation = int(getattr(self.settings, "rotation", 0) or 0)
            if rotation:
                frame = self._rotate(frame, rotation)

            # Crop from CameraSettings
            x0 = int(getattr(self.settings, "crop_x0", 0) or 0)
            y0 = int(getattr(self.settings, "crop_y0", 0) or 0)
            x1 = int(getattr(self.settings, "crop_x1", 0) or 0)
            y1 = int(getattr(self.settings, "crop_y1", 0) or 0)

            if x0 or y0 or x1 or y1:
                frame = self._apply_crop(frame, x0, y0, x1, y1)

        return frame, time.time()

    def close(self) -> None:
        LOG.info(
            "[Basler] close called camera_exists=%s grabbing=%s open=%s",
            self._camera is not None,
            bool(self._camera and self._camera.IsGrabbing()),
            bool(self._camera and self._camera.IsOpen()),
        )
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
        return self.__class__._enumerate_devices_cls()

    def _select_device(self, devices):
        # 1) Namespaced / cached stable identity (preferred)
        serial = self._device_id

        if serial:
            for device in devices:
                try:
                    if device.GetSerialNumber() == serial:
                        return device
                except Exception:
                    continue

        # 2) Legacy top-level fallback (read-only compatibility)
        legacy = None
        try:
            legacy = self._props.get("serial") or self._props.get("serial_number")
        except Exception:
            legacy = None

        if legacy:
            for device in devices:
                try:
                    if device.GetSerialNumber() == legacy:
                        return device
                except Exception:
                    continue

        # 3) Index fallback
        index = int(self.settings.index)
        if index < 0 or index >= len(devices):
            raise RuntimeError(f"Camera index {index} out of range for {len(devices)} Basler device(s)")

        return devices[index]

    @staticmethod
    def _snap_to_node(value: int, node) -> int:
        """
        Best-effort clamp/snap for Basler integer nodes (Width/Height).
        Works with real Pylon nodes and is unit-testable with fakes.

        If node lacks GetMin/GetMax/GetInc, returns value unchanged.
        """
        v = int(value)

        try:
            vmin = int(node.GetMin())
            vmax = int(node.GetMax())
            v = max(vmin, min(v, vmax))
        except Exception:
            # Node doesn't support min/max querying; keep as-is
            return v

        try:
            inc = int(node.GetInc())
            if inc > 1:
                # snap down to nearest valid increment
                v = vmin + ((v - vmin) // inc) * inc
        except Exception:
            pass

        return int(v)

    def _configure_resolution(self) -> None:
        """
        Apply width/height only if explicitly requested (GUI or override).
        If None, keep device defaults.

        Best-effort: if camera enforces increments/ranges, snap to valid values.
        """
        if self._camera is None:
            return

        req = self._requested_resolution
        if req is None:
            LOG.info("Resolution: using device default.")
            return

        req_w, req_h = int(req[0]), int(req[1])

        try:
            # Best-effort clamp/snap (helps with Basler increment constraints)
            try:
                req_w = self._snap_to_node(req_w, self._camera.Width)
                req_h = self._snap_to_node(req_h, self._camera.Height)
            except Exception:
                pass

            # Apply requested values
            self._camera.Width.SetValue(int(req_w))
            self._camera.Height.SetValue(int(req_h))

            # Read back actual applied values
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

    def _get_requested_resolution_or_none(self) -> tuple[int, int] | None:
        """
        Return (w, h) if a resolution was explicitly requested.

        Priority:
        1) CameraSettings.width/height (GUI fields)
        2) properties['basler']['resolution'] (namespaced optional override)
        3) properties['resolution'] (legacy fallback)
        Return None to keep device defaults.

        Note: 'Auto' in GUI is represented by 0 for width/height.
        """

        def _coerce_pair(val) -> tuple[int, int] | None:
            if isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    w = int(val[0])
                    h = int(val[1])
                    if w > 0 and h > 0:
                        return (w, h)
                except Exception:
                    return None
            return None

        # 1) GUI fields first
        try:
            w = int(getattr(self.settings, "width", 0) or 0)
            h = int(getattr(self.settings, "height", 0) or 0)
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            pass

        # 2) Namespaced optional override
        props = self.settings.properties if isinstance(self.settings.properties, dict) else {}
        ns = props.get(self.OPTIONS_KEY, {})
        if isinstance(ns, dict):
            pair = _coerce_pair(ns.get("resolution"))
            if pair:
                return pair

        # 3) Legacy fallback (read-only compatibility)
        pair = _coerce_pair(props.get("resolution"))
        if pair:
            return pair

        return None
