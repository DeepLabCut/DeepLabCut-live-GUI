"""Basler camera backend implemented with :mod:`pypylon`."""

# dlclivegui/cameras/backends/basler_backend.py
from __future__ import annotations

import logging
import time
from typing import ClassVar

from ...config import BASLER_DO_LOG_TIMING, CameraTriggerSettings
from ...utils.stats import WorkerTimingStats
from ...utils.timestamps import FrameTimestampMetadata
from ..base import CameraBackend, CapturedFrame, SupportLevel, register_backend

LOG = logging.getLogger(__name__)


# NOTE @C-Achard: This could be added in settings eventually
# Forces pypylon to create N emulation virtual cameras,
# mostly for testing. This should not be enabled for release.
ENABLE_PYLON_EMU = True
if ENABLE_PYLON_EMU:
    import os

    os.environ["PYLON_CAMEMU"] = "4"

try:  # pragma: no cover - optional dependency
    from pypylon import pylon
except Exception:  # pragma: no cover - optional dependency
    pylon = None  # type: ignore

DEBUG_TRIGGER_LOGS = False


@register_backend("basler")
class BaslerCameraBackend(CameraBackend):
    """Capture frames from Basler cameras using the Pylon SDK."""

    OPTIONS_KEY: ClassVar[str] = "basler"

    # Keep RetrieveResult calls short enough that controller shutdown can stop
    # worker threads promptly while waiting for external hardware triggers.
    _MAX_HARDWARE_TRIGGER_RETRIEVE_TIMEOUT_MS: ClassVar[int] = 1000

    def __init__(self, settings):
        super().__init__(settings)

        self._props: dict = settings.properties if isinstance(settings.properties, dict) else {}
        self._preserve_mono: bool = bool(
            getattr(settings, "preserve_mono", False) or self.ns.get("preserve_mono", False)
        )
        self._camera_pixel_format: str | None = None
        self._logged_first_frame: bool = False

        # Optional fast-start hint for probe workers
        # (may skip StartGrabbing and converter setup for faster capability probing; not suitable for normal capture)
        self._fast_start: bool = bool(self.ns.get("fast_start", False))
        self._retrieve_timeout_ms: int = 100  # default; may be overridden by trigger settings
        self._timestamp_tick_frequency_hz: float | None = None
        self._timestamp_tick_frequency_source: str | None = None

        # ---- Trigger settings ----
        raw_trigger = self.ns.get("trigger", self._props.get("trigger"))
        raw_trigger_strict = isinstance(raw_trigger, dict) and bool(raw_trigger.get("strict", False))

        try:
            self._trigger = CameraTriggerSettings.from_any(raw_trigger)
        except Exception as exc:
            if raw_trigger_strict:
                raise ValueError(f"Strict mode failure - Invalid Basler trigger configuration: {exc}") from exc

            LOG.warning(
                "Invalid Basler trigger config; falling back to trigger role=off: %s. "
                "Enable strict mode to force this to raise.",
                exc,
            )
            self._trigger = CameraTriggerSettings()

        trigger_timeout = self._positive_float(self._trigger_attr(self._trigger, "timeout", None))
        if trigger_timeout is not None:
            # pypylon RetrieveResult timeout is milliseconds.
            self._retrieve_timeout_ms = max(1, int(float(trigger_timeout) * 1000.0))
        else:
            self._retrieve_timeout_ms = 100

        if self.waits_for_hardware_trigger:
            self._retrieve_timeout_ms = min(
                self._retrieve_timeout_ms,
                self._MAX_HARDWARE_TRIGGER_RETRIEVE_TIMEOUT_MS,
            )

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

        # ---- Timing stats for logging (optional) ----
        msg = self._device_id or f"index:{getattr(settings, 'index', '?')}"
        timing_id = f"Basler {msg}"
        self._timing = WorkerTimingStats(
            timing_id,
            logger=LOG,
            log_interval=1.0,
            enabled=BASLER_DO_LOG_TIMING,
        )

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

    @property
    def actual_pixel_format(self) -> str | None:
        """Camera/native pixel format reported by Basler, e.g. 'Mono8'."""
        return self._camera_pixel_format

    @property
    def actual_output_format(self) -> str | None:
        """Backend output frame format emitted to the app, e.g. 'Mono8' or 'BGR8'."""
        if not self._camera_pixel_format:
            return None
        return "Mono8" if self._should_output_mono() else "BGR8"

    @property
    def recommended_preserve_mono(self) -> bool | None:
        if not self._camera_pixel_format:
            return None
        return self._is_camera_mono()

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
                "hardware_trigger": SupportLevel.BEST_EFFORT,
                "preserve_mono": SupportLevel.SUPPORTED,
                "hardware_frame_timestamps": SupportLevel.BEST_EFFORT,
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

    def _read_camera_pixel_format(self) -> str:
        pixel_format = self._feature_value(self._feature("PixelFormat"), "")
        self._camera_pixel_format = str(pixel_format or "")
        return self._camera_pixel_format

    def _is_camera_mono(self) -> bool:
        return bool(self._camera_pixel_format and self._camera_pixel_format.startswith("Mono"))

    def _should_output_mono(self) -> bool:
        return bool(self._preserve_mono and self._is_camera_mono())

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

    def trigger_once(self) -> None:
        if self._camera is None:
            raise RuntimeError("Basler camera not opened")

        # pypylon commonly exposes ExecuteSoftwareTrigger on InstantCamera.
        method = getattr(self._camera, "ExecuteSoftwareTrigger", None)
        if method is not None:
            method()
            return

        command = self._feature("TriggerSoftware")
        if command is not None:
            try:
                command.Execute()
                return
            except Exception as exc:
                raise RuntimeError(f"Failed to execute Basler software trigger: {exc}") from exc

        raise RuntimeError("Basler software trigger command is not available")

    def _configure_frame_rate(self) -> None:
        if self._camera is None:
            return

        fps = self._positive_float(getattr(self.settings, "fps", 0.0))
        if fps is None:
            LOG.info("[Basler] FPS: auto/free-run, not forcing AcquisitionFrameRate")
            return

        enable = self._feature("AcquisitionFrameRateEnable")
        rate = self._feature("AcquisitionFrameRate")

        try:
            if enable is not None:
                enable.SetValue(True)

            if rate is None:
                LOG.warning("[Basler] AcquisitionFrameRate node not available; cannot set FPS=%s", fps)
                return

            try:
                min_v = rate.GetMin()
                max_v = rate.GetMax()
                LOG.info("[Basler] AcquisitionFrameRate range: min=%s max=%s requested=%s", min_v, max_v, fps)
            except Exception:
                pass

            rate.SetValue(float(fps))

        except Exception as exc:
            LOG.warning("[Basler] Failed to set AcquisitionFrameRate=%s: %s", fps, exc, exc_info=True)

        # Readbacks
        readbacks = {}
        for name in (
            "AcquisitionFrameRateEnable",
            "AcquisitionFrameRate",
            "ResultingFrameRate",
            "ResultingAcquisitionFrameRate",
            "AcquisitionResultingFrameRate",
            "BslResultingAcquisitionFrameRate",
            "ExposureAuto",
            "ExposureTime",
            "ExposureTimeAbs",
            "Width",
            "Height",
            "PixelFormat",
            "TestImageSelector",
            "ImageFileMode",
        ):
            feature = self._feature(name)
            if feature is not None:
                readbacks[name] = self._feature_value(feature, None)

        LOG.info("[Basler] FPS readback requested=%s values=%s", fps, readbacks)

        try:
            self._actual_fps = float(readbacks.get("AcquisitionFrameRate"))
        except Exception:
            self._actual_fps = None

    def _configure_converter(self) -> None:
        """Configure pypylon image converter.

        Default behavior remains BGR8 for compatibility.

        If preserve_mono=True and the camera PixelFormat is Mono*,
        return Mono8 frames as 2D arrays to avoid 3x BGR expansion.
        """
        if self._camera is None:
            return

        camera_pixel_format = self._camera_pixel_format or self._read_camera_pixel_format()

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        if self._should_output_mono():
            self._converter.OutputPixelFormat = pylon.PixelType_Mono8
            LOG.info(
                "[Basler] Converter configured for Mono8 output (camera PixelFormat=%s preserve_mono=%s)",
                camera_pixel_format,
                self._preserve_mono,
            )
        else:
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            LOG.info(
                "[Basler] Converter configured for BGR8 output (camera PixelFormat=%s preserve_mono=%s)",
                camera_pixel_format,
                self._preserve_mono,
            )

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

        # Exposure
        if getattr(self.settings, "exposure", 0) > 0:
            try:
                if hasattr(self._camera, "ExposureAuto"):
                    self._camera.ExposureAuto.SetValue("Off")
                if hasattr(self._camera, "ExposureTime"):
                    self._camera.ExposureTime.SetValue(float(self.settings.exposure))
                if hasattr(self._camera, "ExposureTimeAbs"):
                    self._camera.ExposureTimeAbs.SetValue(float(self.settings.exposure))
                LOG.info("[Basler] Exposure set to %s us (auto off)", self.settings.exposure)
            except Exception as exc:
                LOG.warning("[Basler] Failed to set exposure: %s", exc)

        # Gain
        if getattr(self.settings, "gain", 0.0) > 0.0:
            try:
                if hasattr(self._camera, "GainAuto"):
                    self._camera.GainAuto.SetValue("Off")
                self._camera.Gain.SetValue(float(self.settings.gain))
                LOG.info("[Basler] Gain set to %s dB (auto off)", self.settings.gain)
            except Exception as exc:
                LOG.warning("[Basler] Failed to set gain: %s", exc)

        # ----------------------------
        # Resolution (None → device default)
        # ----------------------------
        # Re-evaluate in case settings were rebound before open()
        self._requested_resolution = self._get_requested_resolution_or_none()
        self._configure_resolution()

        # ----------------------------
        # Frame rate (0.0 = Auto → do not set)
        # ----------------------------
        self._configure_frame_rate()

        # ----------------------------
        # Trigger configuration
        # ----------------------------
        self._debug_trigger_nodes(context="before configuration")
        self._configure_trigger()
        self._debug_trigger_nodes(context="after configuration")

        try:
            ns = self._ensure_mutable_ns()
            ns["trigger_actual"] = self._trigger_to_dict(self._trigger)
        except Exception:
            pass

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

        self._read_camera_pixel_format()

        # ----------------------------
        # Start acquisition (skip for fast probe)
        # ----------------------------
        if not self._fast_start:
            # --- HARD RESET of stream state (critical after fast-start probe) ---
            try:
                if hasattr(self._camera, "StopGrabbing") and self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
            except Exception:
                pass

            # Converter BEFORE StartGrabbing
            self._configure_converter()

            # Force stream configuration reset
            try:
                if hasattr(self._camera, "MaxNumBuffer"):
                    self._camera.MaxNumBuffer.SetValue(10)
            except Exception:
                pass

            self._camera.StartGrabbing(
                # pylon.GrabStrategy_LatestImageOnly,
                pylon.GrabStrategy_OneByOne,
            )
            LOG.info(
                "[Basler] grabbing=%s max_buffers=%s",
                self._camera.IsGrabbing(),
                self._camera.MaxNumBuffer.GetValue() if hasattr(self._camera, "MaxNumBuffer") else "N/A",
            )
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

        # Get hardware tick frequency for timestamp conversion
        try:
            node = getattr(self._camera, "GevTimestampTickFrequency", None)
            if node is not None and node.IsReadable():
                self._timestamp_tick_frequency_hz = float(node.GetValue())
                self._timestamp_tick_frequency_source = "GevTimestampTickFrequency"
                LOG.info(
                    "[Basler] timestamp tick frequency: %.3f Hz from GevTimestampTickFrequency",
                    self._timestamp_tick_frequency_hz,
                )
        except Exception:
            LOG.debug("[Basler] Could not read GevTimestampTickFrequency", exc_info=True)

        if not self._timestamp_tick_frequency_hz or self._timestamp_tick_frequency_hz <= 0:
            self._timestamp_tick_frequency_hz = 1_000_000_000.0
            self._timestamp_tick_frequency_source = "assumed_default_1ghz"
            LOG.info(
                "[Basler] timestamp tick frequency unavailable; assuming %.3f Hz",
                self._timestamp_tick_frequency_hz,
            )

        # Persist stable identity into namespace
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

    def _make_timestamp_metadata(self, grab_result) -> FrameTimestampMetadata | None:
        try:
            ticks = int(grab_result.GetTimeStamp())
        except Exception:
            return None

        if ticks == 0:
            # Basler returns 0 if the timestamp is not available (e.g. for some GigE cameras)
            return None

        freq = getattr(self, "_timestamp_tick_frequency_hz", None)
        seconds = ticks / freq if freq and freq > 0 else None

        return FrameTimestampMetadata(
            source="grab_result.GetTimeStamp",
            backend="basler",
            default_reported="seconds" if seconds is not None else "raw_value",
            seconds=seconds,
            wall_clock_time=None,
            raw_value=ticks,
            raw_unit="ticks",
            tick_frequency_hz=freq,
            timebase="Basler camera timestamp counter",
            kind="camera_clock",
            extra={
                "tick_frequency_source": self._timestamp_tick_frequency_source,
            },
        )

    def read(self) -> CapturedFrame:
        if self._camera is None:
            raise RuntimeError("Basler camera not opened")
        if self._converter is None:
            raise RuntimeError("Basler camera opened in fast-start probe mode; cannot read frames")

        grab_result = None

        try:
            with self._timing.measure("Basler.retrieve"):
                grab_result = self._camera.RetrieveResult(
                    int(getattr(self, "_retrieve_timeout_ms", 100)),
                    pylon.TimeoutHandling_ThrowException,
                )

            with self._timing.measure("Basler.check_result"):
                if not grab_result.GrabSucceeded():
                    grab_result.Release()
                    grab_result = None
                    self._timing.note_error()
                    self._timing.maybe_log()
                    raise RuntimeError("Basler camera did not return an image")

            with self._timing.measure("Basler.convert"):
                image = self._converter.Convert(grab_result)

            with self._timing.measure("Basler.get_array"):
                frame = image.GetArray()

            with self._timing.measure("Basler.timestamp"):
                software_timestamp = time.time()
                timestamp_metadata = self._make_timestamp_metadata(grab_result)

            if not self._logged_first_frame:
                self._logged_first_frame = True
                LOG.info(
                    "[Basler] first frame device_id=%s shape=%s dtype=%s nbytes=%.2f MB "
                    "camera_pixel_format=%s output_format=%s preserve_mono=%s",
                    self._device_id,
                    frame.shape,
                    frame.dtype,
                    frame.nbytes / (1024 * 1024),
                    self._camera_pixel_format,
                    self.actual_output_format,
                    self._preserve_mono,
                )

            with self._timing.measure("Basler.release"):
                grab_result.Release()
                grab_result = None

            if self._actual_width is None or self._actual_height is None:
                h, w = frame.shape[:2]
                self._actual_width = int(w)
                self._actual_height = int(h)

            self._timing.note_frame()
            self._timing.maybe_log()

            return CapturedFrame(
                frame=frame,
                software_timestamp=software_timestamp,
                timestamp_metadata=timestamp_metadata,
            )

        except Exception as exc:
            if grab_result is not None:
                try:
                    grab_result.Release()
                except Exception:
                    pass

            if self.waits_for_hardware_trigger:
                self._timing.note_timeout()
                self._timing.maybe_log()
                raise TimeoutError(f"Basler timeout while waiting for hardware trigger: {exc}") from exc

            self._timing.note_error()
            self._timing.maybe_log()
            raise RuntimeError("Failed to retrieve image from Basler camera.") from exc

    def close(self) -> None:
        LOG.info(
            "[Basler] close called camera_exists=%s grabbing=%s open=%s",
            self._camera is not None,
            bool(self._camera and self._camera.IsGrabbing()),
            bool(self._camera and self._camera.IsOpen()),
        )
        if self._camera is not None:
            if self._camera.IsGrabbing():
                try:
                    self._camera.StopGrabbing()
                except Exception:
                    pass

            if self._camera.IsOpen():
                try:
                    self._restore_trigger_idle()
                except Exception:
                    pass

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

    @property
    def waits_for_hardware_trigger(self) -> bool:
        role = str(self._trigger_attr(getattr(self, "_trigger", None), "role", "off") or "off").lower()
        return role in {"external", "follower"}

    @staticmethod
    def _trigger_attr(trigger, name: str, default=None):
        if isinstance(trigger, dict):
            return trigger.get(name, default)
        return getattr(trigger, name, default)

    @staticmethod
    def _trigger_to_dict(trigger) -> dict:
        if trigger is None:
            return {}
        if isinstance(trigger, dict):
            return dict(trigger)
        if hasattr(trigger, "model_dump"):
            try:
                return trigger.model_dump(exclude_none=True)
            except Exception:
                pass
        return {}

    def _feature(self, name: str):
        if self._camera is None:
            return None
        try:
            return getattr(self._camera, name)
        except Exception:
            return None

    @staticmethod
    def _feature_value(feature, default=None):
        if feature is None:
            return default
        try:
            return feature.GetValue()
        except Exception:
            return default

    @staticmethod
    def _feature_symbolics(feature) -> list[str]:
        if feature is None:
            return []

        for method_name in ("GetSymbolics", "GetEntries"):
            try:
                method = getattr(feature, method_name, None)
                if method is None:
                    continue

                values = method()
                out = []

                for value in values:
                    try:
                        if hasattr(value, "GetSymbolic"):
                            out.append(str(value.GetSymbolic()))
                        else:
                            out.append(str(value))
                    except Exception:
                        continue

                return [v for v in out if v]
            except Exception:
                continue

        return []

    def _set_enum_feature(self, name: str, value: str, *, strict: bool = False) -> bool:
        feature = self._feature(name)

        if feature is None:
            if strict:
                raise RuntimeError(f"Basler feature '{name}' is not available")
            LOG.debug("Basler feature '%s' is not available; skipping", name)
            return False

        symbolics = self._feature_symbolics(feature)
        if symbolics and value not in symbolics:
            if strict:
                raise RuntimeError(f"Basler feature '{name}' does not support '{value}'. Available: {symbolics}")
            LOG.warning("Basler feature '%s' does not support '%s'. Available: %s", name, value, symbolics)
            return False

        try:
            feature.SetValue(value)
            return True
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to set Basler feature '{name}' to '{value}': {exc}") from exc
            LOG.warning("Failed to set Basler feature '%s' to '%s': %s", name, value, exc)
            return False

    def _set_numeric_feature(self, name: str, value, *, strict: bool = False) -> bool:
        feature = self._feature(name)

        if feature is None:
            if strict:
                raise RuntimeError(f"Basler feature '{name}' is not available")
            LOG.debug("Basler feature '%s' is not available; skipping", name)
            return False

        try:
            feature.SetValue(value)
            return True
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to set Basler feature '{name}' to '{value}': {exc}") from exc
            LOG.warning("Failed to set Basler feature '%s' to '%s': %s", name, value, exc)
            return False

    def _debug_trigger_nodes(self, *, context: str = "") -> None:
        if not LOG.isEnabledFor(logging.DEBUG) or not DEBUG_TRIGGER_LOGS:
            return

        names = (
            "TriggerSelector",
            "TriggerMode",
            "TriggerSource",
            "TriggerActivation",
            "TriggerDelay",
            "TriggerDelayAbs",
            "AcquisitionMode",
            "LineSelector",
            "LineMode",
            "LineSource",
            "LineInverter",
        )

        label = f"Basler trigger debug {context}".strip()

        for name in names:
            feature = self._feature(name)
            if feature is None:
                continue

            value = self._feature_value(feature, None)
            symbolics = self._feature_symbolics(feature)

            extras = []
            if symbolics:
                extras.append(f"symbolics={symbolics}")

            for method_name in ("IsReadable", "IsWritable"):
                try:
                    method = getattr(feature, method_name, None)
                    if method is not None:
                        extras.append(f"{method_name}={method()}")
                except Exception:
                    pass

            LOG.debug("%s: %s=%r %s", label, name, value, " ".join(extras))

    def _resolve_trigger_source(self, requested: str, *, strict: bool) -> tuple[str, bool]:
        requested = str(requested or "auto").strip()
        feature = self._feature("TriggerSource")
        available = self._feature_symbolics(feature)

        if not available:
            if strict:
                raise RuntimeError("Basler feature 'TriggerSource' is not available or has no symbolics")
            LOG.warning("Basler feature 'TriggerSource' is not available; disabling trigger input.")
            return requested, False

        if requested in available:
            return requested, True

        if requested.lower() == "auto":
            for candidate in ("Line1", "Line2", "Line3", "Line4", "Line0", "Software", "Action1"):
                if candidate in available:
                    LOG.info("Basler TriggerSource auto-selected '%s'. Available: %s", candidate, available)
                    return candidate, True

            LOG.warning("Could not auto-select a Basler TriggerSource. Available: %s", available)
            return requested, False

        if strict:
            raise RuntimeError(f"Basler feature 'TriggerSource' does not support '{requested}'. Available: {available}")

        LOG.warning("Basler TriggerSource '%s' is not available. Available: %s", requested, available)
        return requested, False

    def _configure_trigger(self) -> None:
        cfg = getattr(self, "_trigger", CameraTriggerSettings())
        self._trigger = cfg
        role = str(self._trigger_attr(cfg, "role", "off") or "off").strip().lower()
        strict = bool(self._trigger_attr(cfg, "strict", False))

        if role in {"off", "disabled"}:
            self._configure_trigger_off(strict=strict)
            return

        if role in {"external", "follower"}:
            self._configure_trigger_input(cfg, strict=strict)
            return

        if role == "software":
            self._configure_trigger_software(cfg, strict=strict)
            return

        if role == "master":
            self._configure_trigger_master(cfg, strict=strict)
            return

        if strict:
            raise RuntimeError(f"Unsupported Basler trigger role: {role!r}")

        LOG.warning("Unsupported Basler trigger role '%s'; disabling trigger.", role)
        self._configure_trigger_off(strict=False)

    def _configure_trigger_off(self, *, strict: bool = False) -> None:
        # Select FrameStart first when possible so TriggerMode=Off applies to
        # the frame-start trigger path.
        self._set_enum_feature("TriggerSelector", "FrameStart", strict=False)
        self._set_enum_feature("TriggerMode", "Off", strict=strict)

    def _configure_trigger_input(self, cfg, *, strict: bool = False) -> None:
        role = str(self._trigger_attr(cfg, "role", "external") or "external").strip().lower()
        selector = str(self._trigger_attr(cfg, "selector", "FrameStart") or "FrameStart")
        activation = str(self._trigger_attr(cfg, "activation", "RisingEdge") or "RisingEdge")
        source = str(self._trigger_attr(cfg, "source", "auto") or "auto").strip()
        delay = self._trigger_attr(cfg, "delay", None)

        # Disable trigger while changing trigger-related parameters.
        self._set_enum_feature("TriggerMode", "Off", strict=False)

        selector_ok = self._set_enum_feature("TriggerSelector", selector, strict=strict)

        resolved_source, source_supported = self._resolve_trigger_source(source, strict=strict)
        source_ok = False
        if source_supported:
            source_ok = self._set_enum_feature("TriggerSource", resolved_source, strict=strict)

        activation_ok = self._set_enum_feature("TriggerActivation", activation, strict=False)

        if delay is not None:
            delay_value = float(delay)
            if not self._set_numeric_feature("TriggerDelay", delay_value, strict=False):
                self._set_numeric_feature("TriggerDelayAbs", delay_value, strict=False)

        self._set_enum_feature("AcquisitionMode", "Continuous", strict=False)

        if not selector_ok:
            LOG.warning("Could not apply Basler TriggerSelector=%s; disabling trigger.", selector)
            self._configure_trigger_off(strict=False)
            self._trigger = CameraTriggerSettings()
            return

        if not source_ok:
            LOG.warning(
                "Could not apply Basler TriggerSource=%s resolved=%s; disabling trigger.",
                source,
                resolved_source,
            )
            self._configure_trigger_off(strict=False)
            self._trigger = CameraTriggerSettings()
            return

        if not self._set_enum_feature("TriggerMode", "On", strict=strict):
            LOG.warning("Could not enable Basler TriggerMode=On; disabling trigger.")
            self._configure_trigger_off(strict=False)
            self._trigger = CameraTriggerSettings()
            return

        LOG.info(
            "Basler trigger input configured: role=%s selector=%s source=%s activation=%s "
            "selector_ok=%s source_ok=%s activation_ok=%s",
            role,
            selector,
            resolved_source,
            activation,
            selector_ok,
            source_ok,
            activation_ok,
        )

    def _configure_trigger_software(self, cfg, *, strict: bool = False) -> None:
        selector = str(self._trigger_attr(cfg, "selector", "FrameStart") or "FrameStart")
        delay = self._trigger_attr(cfg, "delay", None)

        self._set_enum_feature("TriggerMode", "Off", strict=False)

        selector_ok = self._set_enum_feature("TriggerSelector", selector, strict=strict)
        source_ok = self._set_enum_feature("TriggerSource", "Software", strict=strict)

        if delay is not None:
            delay_value = float(delay)
            if not self._set_numeric_feature("TriggerDelay", delay_value, strict=False):
                self._set_numeric_feature("TriggerDelayAbs", delay_value, strict=False)

        self._set_enum_feature("AcquisitionMode", "Continuous", strict=False)

        if not selector_ok or not source_ok:
            LOG.warning(
                "Could not configure Basler software trigger selector_ok=%s source_ok=%s; disabling trigger.",
                selector_ok,
                source_ok,
            )
            self._configure_trigger_off(strict=False)
            self._trigger = CameraTriggerSettings()
            return

        if not self._set_enum_feature("TriggerMode", "On", strict=strict):
            LOG.warning("Could not enable Basler software TriggerMode=On; disabling trigger.")
            self._configure_trigger_off(strict=False)
            self._trigger = CameraTriggerSettings()
            return

        LOG.info("Basler software trigger configured: selector=%s source=Software", selector)

    def _configure_trigger_master(self, cfg, *, strict: bool = False) -> None:
        output_line = str(self._trigger_attr(cfg, "output_line", "Line2") or "Line2")
        output_source = str(self._trigger_attr(cfg, "output_source", "ExposureActive") or "ExposureActive")

        # Master camera should acquire freely.
        self._configure_trigger_off(strict=False)

        selected = self._set_enum_feature("LineSelector", output_line, strict=strict)
        if not selected:
            msg = f"Could not select Basler output line '{output_line}'"
            if strict:
                raise RuntimeError(msg)
            LOG.warning("%s; skipping master output configuration.", msg)
            return

        mode_ok = self._set_enum_feature("LineMode", "Output", strict=strict)
        source_ok = self._set_enum_feature("LineSource", output_source, strict=strict)

        if mode_ok and source_ok:
            LOG.info(
                "Basler trigger master configured via Line*: output_line=%s output_source=%s",
                output_line,
                output_source,
            )
            return

        msg = (
            "Could not configure Basler trigger master output completely "
            f"(LineMode ok={mode_ok}, LineSource ok={source_ok})."
        )

        if strict:
            raise RuntimeError(msg)

        LOG.warning(msg)

    def _restore_trigger_idle(self) -> None:
        role = str(self._trigger_attr(getattr(self, "_trigger", None), "role", "off") or "off").lower()

        try:
            if role in {"external", "follower", "software"}:
                self._set_enum_feature("TriggerMode", "Off", strict=False)

            elif role == "master":
                self._set_enum_feature("LineSource", "Off", strict=False)
                self._set_enum_feature("LineMode", "Input", strict=False)

        except Exception:
            LOG.debug("Best-effort Basler trigger restore failed", exc_info=True)

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
