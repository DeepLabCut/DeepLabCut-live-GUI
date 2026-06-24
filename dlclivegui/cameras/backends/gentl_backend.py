"""GenTL backend implemented using the Harvesters library."""

# dlclivegui/cameras/backends/gentl_backend.py
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from ...config import CameraTriggerSettings
from ..base import CameraBackend, SupportLevel, register_backend
from ..factory import DetectedCamera
from .utils import gentl_discovery as cti_finder

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
    """Capture frames from GenTL-compatible devices via Harvesters.

    Notes
    -----
    Multi-camera operation uses a shared Harvester per CTI set. Some GenTL
    producers, including the Imaging Source USB3 Vision producer, can report no
    devices if a second independent Harvester enumerates while another camera is
    already open/streaming. Therefore open() acquires a shared Harvester and
    never calls Harvester.update() during runtime open; initial enumeration is
    handled by SharedHarvesterPool when the shared Harvester is created.
    """

    OPTIONS_KEY: ClassVar[str] = "gentl"
    _OPEN_LOCK: ClassVar[threading.RLock] = threading.RLock()

    _DEFAULT_CTI_PATTERNS: ClassVar[tuple[str, ...]] = (
        # Windows-only defaults; harmless/no-op on other platforms.
        r"C:\Program Files\The Imaging Source Europe GmbH\IC4 GenTL Driver for USB3Vision Devices *\bin\*.cti",
        r"C:\Program Files\The Imaging Source Europe GmbH\TIS Grabber\bin\win64_x64\*.cti",
        r"C:\Program Files\The Imaging Source Europe GmbH\TIS Camera SDK\bin\win64_x64\*.cti",
        r"C:\Program Files (x86)\The Imaging Source Europe GmbH\TIS Grabber\bin\win64_x64\*.cti",
    )
    _COLOR_PIXEL_FORMATS: ClassVar[tuple[str, ...]] = (
        "BGR8",
        "RGB8",
        "BayerRG8",
        "BayerGB8",
        "BayerGR8",
        "BayerBG8",
    )
    _MONO_PIXEL_FORMATS: ClassVar[tuple[str, ...]] = (
        "Mono8",
        "Mono10",
        "Mono12",
        "Mono16",
    )

    # Source marker stored in properties["gentl"]["cti_files_source"].
    # auto: persisted by auto-discovery; may be stale and can fall back.
    # user: explicitly set by user; strict if stale/missing.
    _CTI_FILES_SOURCE_AUTO: ClassVar[str] = "auto"
    _CTI_FILES_SOURCE_USER: ClassVar[str] = "user"

    # Keep individual Harvester.fetch() calls short enough that controller
    # shutdown can stop worker threads promptly. Hardware-trigger waits are
    # handled by repeated polling in SingleCameraWorker.
    _MAX_HARDWARE_TRIGGER_FETCH_TIMEOUT: ClassVar[float] = 1.0

    def __init__(self, settings):
        super().__init__(settings)

        props = settings.properties if isinstance(settings.properties, dict) else {}
        ns = props.get(self.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}

        self._fast_start: bool = bool(ns.get("fast_start", False))

        raw_device_id = ns.get("device_id") or props.get("device_id")
        legacy_serial = ns.get("serial_number") or ns.get("serial") or props.get("serial_number") or props.get("serial")

        self._device_id: str | None = str(raw_device_id).strip() if raw_device_id else None
        self._serial_number: str | None = self._serial_from_identity(self._device_id, legacy_serial)

        self._pixel_format: str = ns.get("pixel_format") or props.get("pixel_format", "auto")
        self._pixel_format = str(self._pixel_format).strip()
        self._camera_pixel_format: str | None = None
        self._actual_output_format: str | None = None

        self._rotate: int = int(ns.get("rotate", props.get("rotate", 0))) % 360
        self._crop: tuple[int, int, int, int] | None = self._parse_crop(ns.get("crop", props.get("crop")))

        self._exposure: float | None = self._positive_float(getattr(settings, "exposure", 0))
        if self._exposure is None:
            self._exposure = self._positive_float(ns.get("exposure", props.get("exposure")))

        self._gain: float | None = self._positive_float(getattr(settings, "gain", 0.0))
        if self._gain is None:
            self._gain = self._positive_float(ns.get("gain", props.get("gain")))

        self._timeout: float = float(ns.get("timeout", props.get("timeout", 2.0)))
        raw_trigger = ns.get("trigger", props.get("trigger"))
        raw_trigger_strict = isinstance(raw_trigger, dict) and bool(raw_trigger.get("strict", False))

        try:
            self._trigger = CameraTriggerSettings.from_any(raw_trigger)
        except Exception as exc:
            if raw_trigger_strict:
                raise ValueError(f"Strict mode failure - Invalid GenTL trigger configuration: {exc}") from exc

            LOG.warning(
                "Invalid GenTL trigger config; falling back to trigger role=off: %s. "
                "Enable strict mode to force this to raise.",
                exc,
            )
            self._trigger = CameraTriggerSettings()

        trigger_timeout = self._positive_float(self._trigger_attr(self._trigger, "timeout", None))
        if trigger_timeout is not None:
            role = str(self._trigger_attr(self._trigger, "role", "off") or "off").strip().lower()

            if role in {"external", "follower"}:
                # Do not let a long hardware-trigger wait block shutdown.
                # SingleCameraWorker treats these fetch timeouts as expected
                # polling misses while waits_for_hardware_trigger is true.
                self._timeout = min(float(trigger_timeout), self._MAX_HARDWARE_TRIGGER_FETCH_TIMEOUT)
            else:
                # For non-trigger-waiting modes, preserve legacy behavior.
                self._timeout = float(trigger_timeout)

        self._requested_resolution: tuple[int, int] | None = self._get_requested_resolution_or_none()

        self._actual_width: int | None = None
        self._actual_height: int | None = None
        self._actual_fps: float | None = None
        self._actual_gain: float | None = None
        self._actual_exposure: float | None = None

        self._harvester = None
        self._acquirer = None
        self._shared_entry = None
        self._device_label: str | None = None
        self._cti_files_source_used: str | None = None

    # ------------------------------------------------------------------
    # Public telemetry / capabilities
    # ------------------------------------------------------------------

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
        """Camera/native pixel format selected on the GenICam PixelFormat node."""
        return self._camera_pixel_format or (self._pixel_format if self._pixel_format != "auto" else None)

    @property
    def actual_output_format(self) -> str | None:
        """Current GenTL backend emits OpenCV-native BGR uint8 frames."""
        return self._actual_output_format or "BGR8"

    @classmethod
    def is_available(cls) -> bool:
        return Harvester is not None

    @classmethod
    def static_capabilities(cls) -> dict[str, SupportLevel]:
        return {
            "set_resolution": SupportLevel.SUPPORTED,
            "set_fps": SupportLevel.SUPPORTED,
            "set_exposure": SupportLevel.SUPPORTED,
            "set_gain": SupportLevel.SUPPORTED,
            "device_discovery": SupportLevel.SUPPORTED,
            "stable_identity": SupportLevel.SUPPORTED,
            "hardware_trigger": SupportLevel.BEST_EFFORT,
        }

    def _debug_trigger_nodes(self, node_map, *, context: str = "") -> None:
        names = (
            "TriggerMode",
            "TriggerSelector",
            "TriggerSource",
            "TriggerActivation",
            "AcquisitionMode",
            # Generic line nodes, if available.
            "LineSelector",
            "LineMode",
            "LineSource",
            # TIS 37U / DMK 37BUX287 strobe/output nodes.
            "GPIn",
            "GPOut",
            "StrobeEnable",
            "StrobePolarity",
            "StrobeOperation",
            "StrobeDuration",
            "StrobeDelay",
        )

        label = f"GenTL trigger debug {context}".strip()

        for name in names:
            node = self._node(node_map, name)
            if node is None:
                continue

            value = self._node_value(node_map, name, None)

            extras = []

            symbolics = self._node_symbolics(node)
            if symbolics:
                extras.append(f"symbolics={symbolics}")

            for attr in ("access_mode", "is_writable", "is_readable"):
                try:
                    extras.append(f"{attr}={getattr(node, attr)}")
                except Exception:
                    pass

            LOG.debug("%s: %s=%r %s", label, name, value, " ".join(extras))

    def _debug_frame_rate_nodes(self, node_map, *, context: str = "") -> None:
        names = (
            "AcquisitionFrameRateEnable",
            "AcquisitionFrameRateControlEnable",
            "AcquisitionFrameRate",
            "AcquisitionFrameRateAbs",
            "AcquisitionResultingFrameRate",
            "ResultingFrameRate",
            "AcquisitionFrameRateResulting",
            "DeviceFrameRate",
            "ExposureAuto",
            "ExposureTime",
            "ExposureTimeAbs",
            "DeviceLinkThroughputLimit",
            "DeviceLinkThroughputLimitMode",
            "PayloadSize",
            "Width",
            "Height",
            "PixelFormat",
        )

        label = f"GenTL FPS debug {context}".strip()

        for name in names:
            node = self._node(node_map, name)
            if node is None:
                continue

            value = self._node_value(node_map, name, None)

            extras = []
            for attr in ("min", "max", "inc"):
                try:
                    extras.append(f"{attr}={getattr(node, attr)}")
                except Exception:
                    pass

            LOG.debug("%s: %s=%r %s", label, name, value, " ".join(extras))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def get_device_count(cls) -> int:
        """Return the number of GenTL devices, or -1 if detection fails."""
        if Harvester is None:
            return -1

        harvester = None
        try:
            harvester, _, _ = cls._build_harvester_for_discovery(strict_single=False)
            if harvester is None:
                return -1
            return len(harvester.device_info_list or [])
        except Exception:
            return -1
        finally:
            cls._safe_reset_harvester(harvester)

    @classmethod
    def discover_devices(
        cls,
        *,
        max_devices: int = 10,
        should_cancel: callable[[], bool] | None = None,
        progress_cb: callable[[str], None] | None = None,
    ):
        """Rich discovery path for CameraFactory.detect_cameras()."""
        if Harvester is None:
            return []

        def _canceled() -> bool:
            return bool(should_cancel and should_cancel())

        harvester = None
        try:
            if progress_cb:
                progress_cb("Initializing GenTL discovery…")

            harvester, loaded, _ = cls._build_harvester_for_discovery(strict_single=False)
            if harvester is None or not loaded:
                if progress_cb:
                    progress_cb("No GenTL producers could be loaded.")
                return []

            if progress_cb:
                progress_cb(f"Loaded {len(loaded)} GenTL producer(s). Scanning devices…")

            infos = list(harvester.device_info_list or [])
            limit = min(len(infos), max_devices if max_devices > 0 else len(infos))
            out: list[DetectedCamera] = []

            for idx in range(limit):
                if _canceled():
                    break

                info = infos[idx]
                label = cls._label_from_info(info, idx)
                device_id = cls._device_id_from_info(info)

                out.append(
                    DetectedCamera(
                        index=idx,
                        label=label,
                        device_id=device_id,
                        vid=None,
                        pid=None,
                        path=None,
                        backend_hint=None,
                    )
                )

                if progress_cb:
                    progress_cb(f"Found: {label}")

            out.sort(key=lambda c: c.index)
            return out
        except Exception:
            LOG.debug("GenTL rich discovery failed", exc_info=True)
            return []
        finally:
            cls._safe_reset_harvester(harvester)

    @classmethod
    def quick_ping(cls, index: int, _unused=None) -> bool:
        """Fast presence check by index using a temporary discovery Harvester."""
        if Harvester is None:
            return False

        harvester = None
        try:
            harvester, _, _ = cls._build_harvester_for_discovery(strict_single=False)
            if harvester is None:
                return False
            infos = harvester.device_info_list or []
            return 0 <= int(index) < len(infos)
        except Exception:
            return False
        finally:
            cls._safe_reset_harvester(harvester)

    @classmethod
    def _build_harvester_for_discovery(cls, *, strict_single: bool = False):
        """Build a temporary Harvester for discovery-only operations."""
        if Harvester is None:
            return None, [], None

        candidates, diag = cti_finder.discover_cti_files(
            include_env=True,
            cti_search_paths=list(cls._DEFAULT_CTI_PATTERNS),
            must_exist=True,
        )
        if not candidates:
            return None, [], diag

        cti_files = list(candidates)
        if strict_single:
            cti_files = cti_finder.choose_cti_files(
                cti_files,
                policy=cti_finder.GenTLDiscoveryPolicy.RAISE_IF_MULTIPLE,
                max_files=1,
            )

        harvester = Harvester()
        loaded: list[str] = []

        for cti in cti_files:
            ok, reason = cls._cti_preflight(cti)
            if not ok:
                LOG.warning("Skipping CTI '%s' during discovery preflight: %s", cti, reason)
                continue
            try:
                harvester.add_file(cti)
                loaded.append(cti)
            except Exception as exc:
                LOG.warning("Failed to load CTI '%s' during discovery: %s", cti, exc)

        if not loaded:
            cls._safe_reset_harvester(harvester)
            return None, [], diag

        try:
            harvester.update()
        except Exception as exc:
            LOG.error("Harvester.update() failed during discovery: %s. CTIs loaded: %s", exc, loaded)
            cls._safe_reset_harvester(harvester)
            return None, [], diag

        return harvester, loaded, diag

    # ------------------------------------------------------------------
    # Settings rebinding
    # ------------------------------------------------------------------

    @classmethod
    def rebind_settings(cls, settings):
        """Map stable identity to current index when necessary.

        Serial identities are stable enough for open() to select directly, so
        they intentionally avoid extra Harvester enumeration during multi-camera
        startup.
        """
        if Harvester is None:
            return settings

        props = settings.properties if isinstance(settings.properties, dict) else {}
        ns = props.get(cls.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}

        target_id = ns.get("device_id") or ns.get("serial_number") or ns.get("serial")
        if not target_id:
            return settings

        target_id_str = str(target_id).strip()
        if target_id_str.startswith("serial:"):
            cls._persist_serial_identity(settings, target_id_str)
            return settings

        if target_id_str.startswith("fp:"):
            return settings  # open() will match by fingerprint via _select_device → _match_device

        # Non-serial fallback retained for older configs / fingerprint IDs.
        harvester = None
        try:
            explicit_files = ns.get("cti_files") or props.get("cti_files")
            explicit_file = ns.get("cti_file") or props.get("cti_file")
            source = str(ns.get("cti_files_source", "")).strip().lower()
            is_auto_cache = source == cls._CTI_FILES_SOURCE_AUTO

            if explicit_files or explicit_file:
                candidates, _ = cti_finder.discover_cti_files(
                    cti_file=explicit_file,
                    cti_files=cti_finder.cti_files_as_list(explicit_files),
                    include_env=False,
                    must_exist=True,
                )
                if not candidates and is_auto_cache:
                    harvester, _, _ = cls._build_harvester_for_discovery(strict_single=False)
                elif candidates:
                    harvester = Harvester()
                    loaded = []
                    for cti in candidates:
                        try:
                            harvester.add_file(cti)
                            loaded.append(cti)
                        except Exception:
                            continue
                    if not loaded:
                        return settings
                    harvester.update()
            else:
                harvester, _, _ = cls._build_harvester_for_discovery(strict_single=False)

            if harvester is None:
                return settings

            infos = list(harvester.device_info_list or [])
            match_index, match_serial = cls._match_device(infos, target_id_str)
            if match_index is None:
                return settings

            settings.index = int(match_index)
            ns2 = cls._ensure_ns_for_settings(settings)
            ns2["device_id"] = target_id_str
            if match_serial:
                ns2["serial_number"] = str(match_serial)
            return settings
        except Exception:
            return settings
        finally:
            cls._safe_reset_harvester(harvester)

    # ------------------------------------------------------------------
    # Open / read / close
    # ------------------------------------------------------------------

    def open(self) -> None:
        if Harvester is None:  # pragma: no cover
            raise RuntimeError(
                "The 'harvesters' package is required for the GenTL backend. Install it via 'pip install harvesters'."
            )

        with type(self)._OPEN_LOCK:
            loaded, failed = self._resolve_and_persist_ctis()
            try:
                infos = self._acquire_shared_harvester(loaded)
                if not infos:
                    self._reset_harvester()
                    raise RuntimeError(
                        "No GenTL cameras detected via Harvesters after loading producers.\n\n"
                        f"Loaded CTIs: {loaded}\n"
                        f"Failed CTIs: {failed}\n"
                        "Fix: ensure your camera vendor's GenTL producer is installed and working."
                    )

                selected_index, selected_serial, selected_info = self._select_device(infos)
                self.settings.index = int(selected_index)

                with self._shared_entry.lock:
                    self._acquirer = self._create_image_acquirer(selected_serial, int(selected_index))
                    node_map = self._acquirer.remote_device.node_map
                    self._device_label = self._resolve_device_label(node_map)

                    self._configure_pixel_format(node_map)
                    self._configure_resolution(node_map)
                    self._configure_exposure(node_map)
                    self._configure_gain(node_map)
                    self._configure_frame_rate(node_map)
                    self._configure_trigger(node_map)  # keep low in the list
                    self._debug_trigger_nodes(node_map, context="after configuration before acquisition")
                    self._ensure_settings_ns()["trigger_actual"] = self._trigger_to_dict(self._trigger)
                    self._read_telemetry(node_map)
                    self._persist_device_metadata(selected_info, selected_serial)

                    if self._fast_start:
                        LOG.info("GenTL open() in fast_start probe mode: acquisition not started.")
                        return

                    self._acquirer.start()

                try:
                    self._read_telemetry(node_map)
                    self._debug_frame_rate_nodes(node_map, context="after starting acquisition")
                except Exception:
                    LOG.warning(
                        "Failed to read telemetry after starting acquisition; some 'actual' values may be missing.",
                        exc_info=True,
                    )

                LOG.debug(
                    "Opened GenTL camera index=%s serial=%s label=%s",
                    selected_index,
                    selected_serial,
                    self._device_label,
                )
            except Exception as exc:
                try:
                    self.close()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Failed to open GenTL camera.\n\nLoaded CTIs: {loaded}\nFailed CTIs: {failed}\nReason: {exc}"
                ) from exc

    @property
    def waits_for_hardware_trigger(self) -> bool:
        role = str(self._trigger_attr(getattr(self, "_trigger", None), "role", "off") or "off").lower()
        return role in {"external", "follower"}

    @staticmethod
    def _output_format_for_frame(frame: np.ndarray) -> str:
        if frame.ndim == 2:
            if frame.dtype == np.uint8:
                return "Mono8"
            return f"Mono{frame.dtype}"
        if frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 3 and frame.dtype == np.uint8:
                return "BGR8"
            if channels == 4 and frame.dtype == np.uint8:
                return "BGRA8"
            return f"{channels}ch-{frame.dtype}"
        return str(frame.dtype)

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
            if self.waits_for_hardware_trigger:
                raise TimeoutError(str(exc) + " (GenTL timeout; waiting for hardware trigger?)") from exc
            raise TimeoutError(str(exc) + " (GenTL timeout)") from exc

        frame = self._convert_frame(frame)
        timestamp = time.time()

        if self._actual_width is None or self._actual_height is None:
            h, w = frame.shape[:2]
            self._actual_width = int(w)
            self._actual_height = int(h)

        if self._actual_exposure is None or self._actual_gain is None:
            try:
                self._read_telemetry(self._acquirer.remote_device.node_map)
            except Exception:
                pass
        self._actual_output_format = self._output_format_for_frame(frame)

        return frame, timestamp

    def stop(self) -> None:
        if self._acquirer is not None:
            try:
                self._call_with_optional_lock(self._acquirer.stop)
            except Exception:
                pass

    def close(self) -> None:
        if self._acquirer is not None:
            try:
                self._call_with_optional_lock(self._acquirer.stop)
            except Exception:
                pass

            try:
                node_map = self._acquirer.remote_device.node_map
                self._call_with_optional_lock(self._restore_trigger_idle, node_map)
            except Exception:
                pass

            try:
                destroy = getattr(self._acquirer, "destroy", None)
                if destroy is not None:
                    self._call_with_optional_lock(destroy)
            finally:
                self._acquirer = None

        if self._harvester is not None or self._shared_entry is not None:
            self._reset_harvester()

        self._device_label = None

    # ------------------------------------------------------------------
    # CTI / shared Harvester helpers
    # ------------------------------------------------------------------

    def _resolve_and_persist_ctis(self) -> tuple[list[str], list[tuple[str, str]]]:
        ns = self._ensure_settings_ns()
        ns.setdefault("cti_search_paths", list(self._DEFAULT_CTI_PATTERNS))
        ns.setdefault("cti_files_source", self._CTI_FILES_SOURCE_AUTO)

        cti_files = self._resolve_cti_files_for_settings()
        ns["cti_files_source"] = (
            self._cti_files_source_used or ns.get("cti_files_source") or self._CTI_FILES_SOURCE_AUTO
        )

        loaded: list[str] = []
        failed: list[tuple[str, str]] = []
        for cti in cti_files:
            ok, reason = self._cti_preflight(cti)
            if ok:
                loaded.append(str(cti))
            else:
                failed.append((str(cti), reason or "preflight failed"))
                LOG.warning("Skipping CTI '%s': %s", cti, reason)

        ns["cti_files"] = [str(p) for p in cti_files]
        ns["cti_files_loaded"] = loaded[:]
        ns["cti_files_failed"] = [{"cti": c, "error": e} for c, e in failed]
        if loaded:
            ns["cti_file"] = loaded[0]
        elif cti_files:
            ns["cti_file"] = str(cti_files[0])

        if not loaded:
            self._reset_harvester()
            raise RuntimeError(
                "No GenTL producer (.cti) could be loaded.\n\n"
                f"Resolved CTIs: {cti_files}\n"
                f"Failures: {failed}\n"
                "Fix: remove/repair incompatible producers "
                "or set properties.gentl.cti_file to a known working producer."
            )

        return loaded, failed

    def _acquire_shared_harvester(self, loaded: list[str]) -> list:
        ns = self._ensure_settings_ns()
        try:
            self._shared_entry = cti_finder.SharedHarvesterPool.acquire(loaded)
            self._harvester = self._shared_entry.harvester

            actual_loaded = list(getattr(self._shared_entry, "loaded_files", loaded))
            actual_failed = dict(getattr(self._shared_entry, "failed_files", {}))

            ns["cti_files_loaded"] = actual_loaded
            if actual_failed:
                existing_failed = ns.get("cti_files_failed")
                merged_failed = list(existing_failed) if isinstance(existing_failed, list) else []
                merged_failed.extend({"cti": str(cti), "error": str(error)} for cti, error in actual_failed.items())
                ns["cti_files_failed"] = merged_failed

            with self._shared_entry.lock:
                infos = list(self._harvester.device_info_list or [])

            LOG.debug(
                "Using shared GenTL Harvester for %d device(s), refcount=%s",
                len(infos),
                cti_finder.SharedHarvesterPool.get_refcount(self._shared_entry),
            )
            return infos

        except Exception as exc:
            exc_loaded = list(getattr(exc, "loaded_files", []))
            exc_failed = dict(getattr(exc, "failed_files", {}))

            if exc_loaded or exc_failed:
                ns["cti_files_loaded"] = [str(p) for p in exc_loaded]
                existing_failed = ns.get("cti_files_failed")
                merged_failed = list(existing_failed) if isinstance(existing_failed, list) else []
                merged_failed.extend({"cti": str(cti), "error": str(error)} for cti, error in exc_failed.items())
                ns["cti_files_failed"] = merged_failed

            if self._shared_entry is not None:
                try:
                    cti_finder.SharedHarvesterPool.release(self._shared_entry)
                except Exception:
                    pass

            self._shared_entry = None
            self._harvester = None

            raise RuntimeError(
                f"Failed to initialize shared GenTL producer state.\n\nCTIs: {loaded}\nReason: {exc}"
            ) from exc

    def _reset_harvester(self) -> None:
        try:
            if self._shared_entry is not None:
                cti_finder.SharedHarvesterPool.release(self._shared_entry)
                self._shared_entry = None
            else:
                self._reset_select_harvester(self._harvester)
        finally:
            self._harvester = None

    @staticmethod
    def _reset_select_harvester(harvester) -> None:
        GenTLCameraBackend._safe_reset_harvester(harvester)

    @staticmethod
    def _safe_reset_harvester(harvester) -> None:
        if harvester is not None:
            try:
                harvester.reset()
            except Exception:
                pass

    @staticmethod
    def _cti_preflight(path: str) -> tuple[bool, str | None]:
        p = Path(str(path))
        try:
            if not p.exists():
                return False, "missing at load time"
            if not p.is_file():
                return False, "not a file at load time"
            with p.open("rb"):
                pass
            return True, None
        except PermissionError:
            return False, "permission denied at load time"
        except OSError as e:
            return False, f"os error at load time: {e}"

    def _resolve_cti_files_for_settings(self) -> list[str]:
        """Resolve CTI files using explicit user overrides, auto cache, then discovery."""
        props = self.settings.properties if isinstance(self.settings.properties, dict) else {}
        ns = props.get(self.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}

        source = ns.get("cti_files_source")
        source = str(source).strip().lower() if source is not None else None

        ns_cti_files = ns.get("cti_files")
        ns_cti_file = ns.get("cti_file")
        legacy_cti_files = props.get("cti_files")
        legacy_cti_file = props.get("cti_file")

        if legacy_cti_files or legacy_cti_file:
            self._cti_files_source_used = self._CTI_FILES_SOURCE_USER
            candidates, diag = cti_finder.discover_cti_files(
                cti_file=str(legacy_cti_file) if legacy_cti_file else None,
                cti_files=cti_finder.cti_files_as_list(legacy_cti_files) if legacy_cti_files else None,
                include_env=False,
                must_exist=True,
            )
            if not candidates:
                raise RuntimeError(
                    "No valid GenTL producer (.cti) found from properties.cti_file/cti_files.\n\n"
                    f"Discovery details:\n{diag.summarize()}"
                )
            return list(candidates)

        if ns_cti_files or ns_cti_file:
            is_auto_cache = source == self._CTI_FILES_SOURCE_AUTO
            self._cti_files_source_used = self._CTI_FILES_SOURCE_AUTO if is_auto_cache else self._CTI_FILES_SOURCE_USER
            candidates, diag = cti_finder.discover_cti_files(
                cti_file=str(ns_cti_file) if ns_cti_file else None,
                cti_files=cti_finder.cti_files_as_list(ns_cti_files) if ns_cti_files else None,
                include_env=False,
                must_exist=True,
            )
            if candidates:
                return list(candidates)
            if not is_auto_cache:
                raise RuntimeError(
                    "No valid GenTL producer (.cti) found from properties.gentl.cti_file/cti_files.\n\n"
                    f"Discovery details:\n{diag.summarize()}"
                )
            LOG.info("Auto-persisted GenTL CTIs stale/missing; falling back to discovery.")

        self._cti_files_source_used = self._CTI_FILES_SOURCE_AUTO
        search_paths = ns.get("cti_search_paths", props.get("cti_search_paths"))
        extra_dirs = ns.get("cti_dirs", props.get("cti_dirs"))
        search_patterns = (
            cti_finder.cti_files_as_list(search_paths) if search_paths is not None else list(self._DEFAULT_CTI_PATTERNS)
        )

        candidates, diag = cti_finder.discover_cti_files(
            cti_search_paths=search_patterns,
            include_env=True,
            extra_dirs=cti_finder.cti_files_as_list(extra_dirs) if extra_dirs is not None else None,
            recursive_env_search=False,
            recursive_extra_search=False,
            must_exist=True,
        )
        if not candidates:
            raise RuntimeError(
                "Could not locate any GenTL producer (.cti) file.\n\n"
                "Fix options:\n"
                "  - Set camera.properties.gentl.cti_file to the full path of a .cti file\n"
                "  - Or set GENICAM_GENTL64_PATH / GENICAM_GENTL32_PATH to include the producer directory\n"
                "  - Or provide camera.properties.gentl.cti_search_paths with glob patterns\n\n"
                f"Discovery details:\n{diag.summarize(redact_env=False)}"
            )
        return list(candidates)

    # ------------------------------------------------------------------
    # Device selection / identity helpers
    # ------------------------------------------------------------------

    def _select_device(self, infos: list) -> tuple[int, str | None, object]:
        requested_index = int(self.settings.index or 0)
        target_device_id = self._device_id or self._ensure_settings_ns().get("device_id")

        selected_index: int | None = None
        selected_serial: str | None = None

        if target_device_id:
            target = str(target_device_id).strip()
            selected_index, selected_serial = self._match_device(infos, target)
            if selected_index is None:
                available = [str(self._info_get(i, "serial_number", "") or "").strip() for i in infos]
                raise RuntimeError(f"GenTL device '{target}' not found. Available serials: {available}")

        elif self._serial_number:
            serial = str(self._serial_number).strip()
            selected_index, selected_serial = self._match_device(infos, serial)
            if selected_index is None:
                available = [str(self._info_get(i, "serial_number", "") or "").strip() for i in infos]
                raise RuntimeError(f"GenTL camera with serial '{serial}' not found. Available serials: {available}")

        else:
            if requested_index < 0 or requested_index >= len(infos):
                raise RuntimeError(f"Camera index {requested_index} out of range for {len(infos)} GenTL device(s)")
            selected_index = requested_index
            serial = self._info_get(infos[selected_index], "serial_number", "")
            selected_serial = str(serial).strip() if serial else None

        return int(selected_index), selected_serial, infos[int(selected_index)]

    @classmethod
    def _match_device(cls, infos: list, target: str) -> tuple[int | None, str | None]:
        if not target:
            return None, None

        serial_target = target.split("serial:", 1)[1].strip() if target.startswith("serial:") else target

        for idx, info in enumerate(infos):
            if cls._device_id_from_info(info) == target:
                serial = cls._info_get(info, "serial_number", None)
                return idx, str(serial).strip() if serial else None

        exact: list[tuple[int, str]] = []
        for idx, info in enumerate(infos):
            sn = str(cls._info_get(info, "serial_number", "") or "").strip()
            if sn == serial_target:
                exact.append((idx, sn))
        if exact:
            return exact[0]

        partial = []
        for idx, info in enumerate(infos):
            sn = str(cls._info_get(info, "serial_number", "") or "").strip()
            if serial_target and serial_target in sn:
                partial.append((idx, sn))
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise RuntimeError(
                f"Ambiguous GenTL serial match for '{serial_target}'. Candidates: {[sn for _, sn in partial]}"
            )

        return None, None

    @staticmethod
    def _device_id_from_info(info) -> str | None:
        serial = GenTLCameraBackend._first_info_value(
            info,
            "serial_number",
            "SerialNumber",
            "device_serial_number",
            "sn",
            "serial",
        )
        if serial:
            return f"serial:{serial}"

        parts = []
        for key, names in (
            ("vendor", ("vendor", "vendor_name", "manufacturer", "DeviceVendorName")),
            ("model", ("model", "model_name", "DeviceModelName")),
            ("user", ("user_defined_name", "user_id", "DeviceUserID", "DeviceUserId", "device_user_id")),
            ("tl", ("tl_type", "transport_layer_type", "DeviceTLType")),
            ("uid", ("id_", "id", "device_id", "uid", "guid", "mac_address", "interface_id", "display_name")),
        ):
            value = GenTLCameraBackend._first_info_value(info, *names)
            if value:
                parts.append(f"{key}={value}")
        return "fp:" + "|".join(parts) if parts else None

    @staticmethod
    def _first_info_value(info, *names: str) -> str | None:
        for name in names:
            value = GenTLCameraBackend._info_get(info, name, None)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _info_get(info, key: str, default=None):
        try:
            if hasattr(info, "get"):
                value = info.get(key)
                if value is not None:
                    return value
        except Exception:
            pass
        try:
            value = getattr(info, key, None)
            if value is not None:
                return value
        except Exception:
            pass
        return default

    @staticmethod
    def _label_from_info(info, index: int) -> str:
        display = GenTLCameraBackend._info_get(info, "display_name", None)
        if display:
            return str(display).strip()

        vendor = str(GenTLCameraBackend._info_get(info, "vendor", "") or "").strip()
        model = str(GenTLCameraBackend._info_get(info, "model", "") or "").strip()
        serial = str(GenTLCameraBackend._info_get(info, "serial_number", "") or "").strip()
        label = f"{vendor} {model}".strip() if (vendor or model) else f"GenTL device {index}"
        return f"{label} ({serial})" if serial else label

    @staticmethod
    def _serial_from_identity(device_id: str | None, legacy_serial) -> str | None:
        if device_id:
            did = str(device_id).strip()
            if did.startswith("serial:"):
                return did.split("serial:", 1)[1].strip() or None
            if not did.startswith("fp:"):
                return did
        return str(legacy_serial).strip() if legacy_serial else None

    @classmethod
    def _persist_serial_identity(cls, settings, device_id: str) -> None:
        serial = device_id.split("serial:", 1)[1].strip()
        if not serial:
            return
        ns = cls._ensure_ns_for_settings(settings)
        ns["device_id"] = device_id
        ns["serial_number"] = serial

    def _persist_device_metadata(self, selected_info, selected_serial: str | None) -> None:
        ns = self._ensure_settings_ns()
        computed_id = self._device_id_from_info(selected_info)

        if computed_id:
            ns["device_id"] = computed_id
        elif selected_serial:
            ns["device_id"] = f"serial:{selected_serial}"

        if selected_serial:
            ns["serial_number"] = str(selected_serial)
            ns["device_serial_number"] = str(selected_serial)

        if self._device_label:
            ns["device_name"] = str(self._device_label)

        for out_key, info_key in (
            ("device_display_name", "display_name"),
            ("device_info_id", "id_"),
            ("device_vendor", "vendor"),
            ("device_model", "model"),
            ("device_tl_type", "tl_type"),
            ("device_user_defined_name", "user_defined_name"),
            ("device_version", "version"),
            ("device_access_status", "access_status"),
        ):
            value = self._info_get(selected_info, info_key, "")
            ns[out_key] = value if out_key == "device_access_status" else str(value or "")

    @classmethod
    def _ensure_ns_for_settings(cls, settings) -> dict:
        if not isinstance(settings.properties, dict):
            settings.properties = {}
        ns = settings.properties.get(cls.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}
        settings.properties[cls.OPTIONS_KEY] = ns
        return ns

    def _ensure_settings_ns(self) -> dict:
        return self._ensure_ns_for_settings(self.settings)

    # ------------------------------------------------------------------
    # Existing compatibility helpers
    # ------------------------------------------------------------------

    def _call_with_optional_lock(self, func, *args, **kwargs):
        if self._shared_entry is not None:
            with self._shared_entry.lock:
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    def _create_image_acquirer(self, selected_serial: str | None, selected_index: int):
        if self._harvester is None:
            raise RuntimeError("Harvester is not initialized")
        try:
            if selected_serial:
                return self._harvester.create({"serial_number": str(selected_serial)})
            return self._harvester.create(int(selected_index))
        except TypeError:
            if selected_serial:
                return self._harvester.create({"serial_number": str(selected_serial)})
            return self._harvester.create(index=int(selected_index))

    def _available_serials(self) -> list[str]:
        assert self._harvester is not None
        return [
            str(s).strip()
            for s in (self._info_get(i, "serial_number", "") for i in self._harvester.device_info_list)
            if s
        ]

    def _create_acquirer(self, serial: str | None, index: int):
        """Compatibility wrapper for older code/tests."""
        return self._create_image_acquirer(serial, index)

    # ------------------------------------------------------------------
    # Camera configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _node(node_map, name: str):
        try:
            return getattr(node_map, name)
        except Exception:
            return None

    @staticmethod
    def _node_value(node_map, name: str, default=None):
        """Best-effort read of a GenICam node value.

        Debug helpers must not make open() fail just because a value cannot be read.
        Harvesters-style fake/test nodes usually expose `.value`; some SDK-style
        nodes may expose `GetValue()`.
        """
        node = GenTLCameraBackend._node(node_map, name)
        if node is None:
            return default

        try:
            return node.value
        except Exception:
            pass

        try:
            getter = getattr(node, "GetValue", None)
            if getter is not None:
                return getter()
        except Exception:
            pass

        return default

    @staticmethod
    def _node_symbolics(node) -> list[str]:
        try:
            return list(getattr(node, "symbolics", []) or [])
        except Exception:
            return []

    @staticmethod
    def _node_value(node_map, name: str, default=None):
        """Best-effort read of a GenICam node value."""
        try:
            node = getattr(node_map, name)
        except Exception:
            return default

        try:
            return node.value
        except Exception:
            return default

    @classmethod
    def _node_float(cls, node_map, *names: str, allow_zero: bool = False) -> float | None:
        """Return the first positive float value from a list of GenICam node names."""
        for name in names:
            value = cls._node_value(node_map, name, None)
            try:
                fvalue = float(value)
            except Exception:
                continue

            if fvalue > 0 or (allow_zero and fvalue == 0):
                return fvalue

        return None

    @classmethod
    def _node_str(cls, node_map, *names: str) -> str | None:
        """Return the first non-empty string value from a list of GenICam node names."""
        for name in names:
            value = cls._node_value(node_map, name, None)
            if value is None:
                continue

            text = str(value).strip()
            if text:
                return text

        return None

    def _set_enum_node(self, node_map, name: str, value: str, *, strict: bool = False) -> bool:
        node = self._node(node_map, name)
        if node is None:
            if strict:
                raise RuntimeError(f"GenICam node '{name}' is not available")
            LOG.debug("GenICam node '%s' is not available; skipping", name)
            return False

        symbolics = self._node_symbolics(node)
        if symbolics and value not in symbolics:
            if strict:
                raise RuntimeError(f"GenICam node '{name}' does not support '{value}'. Available: {symbolics}")
            LOG.warning("GenICam node '%s' does not support '%s'. Available: %s", name, value, symbolics)
            return False

        try:
            node.value = value
            return True
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to set GenICam node '{name}' to '{value}': {exc}") from exc
            LOG.warning("Failed to set GenICam node '%s' to '%s': %s", name, value, exc)
            return False

    @staticmethod
    def _trigger_attr(trigger, name: str, default=None):
        if isinstance(trigger, dict):
            return trigger.get(name, default)
        return getattr(trigger, name, default)

    @staticmethod
    def _trigger_to_dict(trigger) -> dict[str, Any]:
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

    def _resolve_trigger_source(self, node_map, requested: str, *, strict: bool) -> tuple[str, bool]:
        """Resolve TriggerSource against the camera-supported GenICam enum values.

        Model-level default is "auto"; this backend maps it to the first preferred
        source supported by the actual camera.
        """
        requested = str(requested or "auto").strip()
        node = self._node(node_map, "TriggerSource")
        available = self._node_symbolics(node)

        if not available:
            if strict:
                raise RuntimeError("GenICam node 'TriggerSource' is not available or has no symbolics")
            LOG.warning("GenICam node 'TriggerSource' is not available; disabling trigger input.")
            return requested, False

        if requested in available:
            return requested, True

        if requested.lower() == "auto":
            for candidate in ("Line0", "Line1", "Line2", "Any"):
                if candidate in available:
                    LOG.info(
                        "GenTL TriggerSource auto-selected '%s'. Available: %s",
                        candidate,
                        available,
                    )
                    return candidate, True

            LOG.warning(
                "Could not auto-select a GenTL TriggerSource. Available: %s",
                available,
            )
            return requested, False

        if strict:
            raise RuntimeError(f"GenICam node 'TriggerSource' does not support '{requested}'. Available: {available}")

        LOG.warning(
            "GenTL TriggerSource '%s' is not available. Available: %s",
            requested,
            available,
        )
        return requested, False

    def _configure_pixel_format(self, node_map) -> None:
        try:
            pixel_format_node = getattr(node_map, "PixelFormat", None)
            if pixel_format_node is None:
                return

            available = list(getattr(pixel_format_node, "symbolics", []) or [])
            if not available:
                return

            requested = str(self._pixel_format or "auto").strip()

            if requested.lower() == "auto":
                selected = None

                for fmt in self._COLOR_PIXEL_FORMATS:
                    if fmt in available:
                        selected = fmt
                        break

                if selected is None:
                    for fmt in self._MONO_PIXEL_FORMATS:
                        if fmt in available:
                            selected = fmt
                            break

                if selected is None:
                    selected = available[0]

            else:
                selected = requested
                if selected not in available:
                    LOG.warning(
                        "Pixel format '%s' not available. Available formats: %s. Falling back to auto.",
                        selected,
                        available,
                    )
                    selected = None
                    for fmt in self._COLOR_PIXEL_FORMATS + self._MONO_PIXEL_FORMATS:
                        if fmt in available:
                            selected = fmt
                            break
                    if selected is None:
                        selected = available[0]

            pixel_format_node.value = selected
            self._pixel_format = str(pixel_format_node.value)
            self._camera_pixel_format = self._pixel_format

            LOG.debug("GenTL pixel format selected: %s", self._pixel_format)

        except Exception as e:
            LOG.warning("Failed to configure pixel format '%s': %s", self._pixel_format, e)
            if self._pixel_format and self._pixel_format.lower() != "auto":
                self._camera_pixel_format = self._pixel_format

    def _configure_trigger(self, node_map) -> None:
        cfg = self._trigger
        role = str(self._trigger_attr(cfg, "role", "off") or "off").strip().lower()
        strict = bool(self._trigger_attr(cfg, "strict", False))

        if role in {"off", "disabled"}:
            self._configure_trigger_off(node_map, strict=strict)
            return

        if role in {"external", "follower"}:
            self._configure_trigger_input(node_map, cfg, strict=strict)
            return

        if role == "master":
            self._configure_trigger_master(node_map, cfg, strict=strict)
            return

        if strict:
            raise RuntimeError(f"Unsupported GenTL trigger role: {role!r}")

        LOG.warning("Unsupported GenTL trigger role '%s'; disabling trigger.", role)
        self._configure_trigger_off(node_map, strict=False)

    def _configure_trigger_off(self, node_map, *, strict: bool = False) -> None:
        self._set_enum_node(node_map, "TriggerMode", "Off", strict=strict)

    def _configure_trigger_input(self, node_map, cfg, *, strict: bool = False) -> None:
        role = str(self._trigger_attr(cfg, "role", "external") or "external").strip().lower()
        selector = str(self._trigger_attr(cfg, "selector", "FrameStart") or "FrameStart")
        activation = str(self._trigger_attr(cfg, "activation", "RisingEdge") or "RisingEdge")
        source = str(self._trigger_attr(cfg, "source", "auto") or "auto").strip()

        # Disable trigger while changing trigger-related nodes.
        self._set_enum_node(node_map, "TriggerMode", "Off", strict=False)

        selector_ok = self._set_enum_node(node_map, "TriggerSelector", selector, strict=strict)

        resolved_source, source_supported = self._resolve_trigger_source(
            node_map,
            source,
            strict=strict,
        )

        source_ok = False
        if source_supported:
            source_ok = self._set_enum_node(
                node_map,
                "TriggerSource",
                resolved_source,
                strict=strict,
            )

        activation_ok = self._set_enum_node(
            node_map,
            "TriggerActivation",
            activation,
            strict=False,
        )

        # TriggerSelector and TriggerSource are required routing nodes.
        # If either failed in non-strict mode, do not arm TriggerMode=On.
        # Otherwise the camera may wait on a previous/default input line.
        if not (selector_ok and source_ok):
            LOG.warning(
                "Could not apply GenTL trigger input routing "
                "(selector_ok=%s, source_ok=%s); disabling trigger. "
                "requested role=%s selector=%s source=%s resolved_source=%s activation=%s",
                selector_ok,
                source_ok,
                role,
                selector,
                source,
                resolved_source,
                activation,
            )
            self._configure_trigger_off(node_map, strict=False)
            self._trigger = CameraTriggerSettings()
            return

        if not activation_ok:
            LOG.warning(
                "Could not apply GenTL TriggerActivation=%s; using camera default/current activation.",
                activation,
            )

        self._set_enum_node(node_map, "AcquisitionMode", "Continuous", strict=False)

        if not self._set_enum_node(node_map, "TriggerMode", "On", strict=strict):
            LOG.warning("Could not enable GenTL TriggerMode=On; disabling trigger.")
            self._configure_trigger_off(node_map, strict=False)
            self._trigger = CameraTriggerSettings()
            return

        LOG.info(
            "GenTL trigger input configured: role=%s selector=%s source_requested=%s "
            "source=%s activation=%s selector_ok=%s source_ok=%s activation_ok=%s",
            role,
            selector,
            source,
            resolved_source,
            activation,
            selector_ok,
            source_ok,
            activation_ok,
        )

    def _configure_trigger_master(self, node_map, cfg, *, strict: bool = False) -> None:
        """Configure this camera as a free-running master that emits STROBE_OUT pulses.

        For DMK 37BUX287 / TIS 37U series, the physical output is controlled by
        StrobeEnable/StrobePolarity/StrobeOperation rather than SFNC LineSelector/
        LineMode/LineSource nodes.
        """
        output_line = str(self._trigger_attr(cfg, "output_line", "Line2") or "Line2")
        output_source = str(self._trigger_attr(cfg, "output_source", "ExposureActive") or "ExposureActive")

        # Optional extra fields if present in trigger dict/model.
        strobe_polarity = str(self._trigger_attr(cfg, "strobe_polarity", "ActiveHigh") or "ActiveHigh")
        strobe_operation = str(self._trigger_attr(cfg, "strobe_operation", "Exposure") or "Exposure")
        strobe_duration = self._trigger_attr(cfg, "strobe_duration", None)
        strobe_delay = self._trigger_attr(cfg, "strobe_delay", None)

        # Master camera should be free-running.
        self._configure_trigger_off(node_map, strict=False)

        # ------------------------------------------------------------------
        # Preferred path for The Imaging Source 37U / DMK 37BUX287:
        # StrobeEnable, StrobePolarity, StrobeOperation, StrobeDuration, StrobeDelay
        # ------------------------------------------------------------------
        strobe_enable_node = self._node(node_map, "StrobeEnable")

        if strobe_enable_node is not None:
            # Disable first while changing parameters.
            self._set_enum_node(node_map, "StrobeEnable", "Off", strict=False)

            polarity_ok = self._set_enum_node(
                node_map,
                "StrobePolarity",
                strobe_polarity,
                strict=False,
            )

            operation_ok = self._set_enum_node(
                node_map,
                "StrobeOperation",
                strobe_operation,
                strict=False,
            )

            if strobe_duration is not None:
                try:
                    node = self._node(node_map, "StrobeDuration")
                    if node is not None:
                        node.value = int(strobe_duration)
                        LOG.info("Configured GenTL StrobeDuration=%s", int(strobe_duration))
                except Exception as exc:
                    if strict:
                        raise RuntimeError(f"Failed to set StrobeDuration={strobe_duration}: {exc}") from exc
                    LOG.warning("Failed to set StrobeDuration=%s: %s", strobe_duration, exc)

            if strobe_delay is not None:
                try:
                    node = self._node(node_map, "StrobeDelay")
                    if node is not None:
                        node.value = int(strobe_delay)
                        LOG.info("Configured GenTL StrobeDelay=%s", int(strobe_delay))
                except Exception as exc:
                    if strict:
                        raise RuntimeError(f"Failed to set StrobeDelay={strobe_delay}: {exc}") from exc
                    LOG.warning("Failed to set StrobeDelay=%s: %s", strobe_delay, exc)

            enable_ok = self._set_enum_node(
                node_map,
                "StrobeEnable",
                "On",
                strict=strict,
            )

            if enable_ok:
                LOG.info(
                    "GenTL trigger master configured via Strobe*: "
                    "StrobeEnable=On StrobePolarity=%s polarity_ok=%s "
                    "StrobeOperation=%s operation_ok=%s",
                    strobe_polarity,
                    polarity_ok,
                    strobe_operation,
                    operation_ok,
                )
                return

            if strict:
                raise RuntimeError("Could not enable GenTL StrobeEnable=On")

            LOG.warning(
                "StrobeEnable node exists but could not be enabled; falling back to generic Line* output configuration."
            )

        # ------------------------------------------------------------------
        # Generic SFNC fallback for cameras that expose LineSelector/LineMode/LineSource.
        # ------------------------------------------------------------------
        line_selector = self._node(node_map, "LineSelector")
        if line_selector is not None:
            line_selected = self._set_enum_node(
                node_map,
                "LineSelector",
                output_line,
                strict=strict,
            )

            if not line_selected:
                LOG.warning(
                    "Could not select GenTL output line '%s'; skipping Line* output configuration.",
                    output_line,
                )
            else:
                mode_ok = self._set_enum_node(node_map, "LineMode", "Output", strict=strict)
                source_ok = self._set_enum_node(node_map, "LineSource", output_source, strict=strict)

                if mode_ok and source_ok:
                    LOG.info(
                        "GenTL trigger master configured via Line*: output_line=%s output_source=%s",
                        output_line,
                        output_source,
                    )
                    return

                LOG.warning(
                    "GenTL Line* trigger output configuration incomplete (LineMode ok=%s, LineSource ok=%s).",
                    mode_ok,
                    source_ok,
                )

        msg = (
            "Could not configure GenTL trigger master output. "
            "No supported Strobe* or Line* output path was successfully configured."
        )

        if strict:
            raise RuntimeError(msg)

        LOG.warning(msg)

    def _restore_trigger_idle(self, node_map) -> None:
        """Best-effort restore to a safe non-triggering state after acquisition stops.

        Important:
        - This should be called after acquirer.stop(), not while acquisition is active.
        - It is intentionally non-strict because shutdown should not fail if a node
        is missing or read-only.
        """
        role = str(self._trigger_attr(getattr(self, "_trigger", None), "role", "off") or "off").lower()

        try:
            if role in {"external", "follower"}:
                self._set_enum_node(node_map, "TriggerMode", "Off", strict=False)

            elif role == "master":
                # Stop driving output if the camera exposes these nodes.
                self._set_enum_node(node_map, "LineSource", "Off", strict=False)
                self._set_enum_node(node_map, "LineMode", "Input", strict=False)

        except Exception:
            LOG.debug("Best-effort GenTL trigger restore failed", exc_info=True)

    def _configure_resolution(self, node_map) -> None:
        if self._requested_resolution is None:
            return

        requested_width, requested_height = self._requested_resolution
        actual_width, actual_height = None, None

        try:
            node = node_map.Width
            width = self._adjust_to_increment(requested_width, node.min, node.max, getattr(node, "inc", 1))
            node.value = int(width)
            actual_width = int(node.value)
        except Exception as e:
            LOG.warning("Failed to set width: %s", e)

        try:
            node = node_map.Height
            height = self._adjust_to_increment(requested_height, node.min, node.max, getattr(node, "inc", 1))
            node.value = int(height)
            actual_height = int(node.value)
        except Exception as e:
            LOG.warning("Failed to set height: %s", e)

        if actual_width is not None and actual_height is not None:
            self._actual_width = actual_width
            self._actual_height = actual_height
            if (actual_width, actual_height) != (requested_width, requested_height):
                LOG.warning(
                    "Resolution mismatch: requested %sx%s, got %sx%s",
                    requested_width,
                    requested_height,
                    actual_width,
                    actual_height,
                )

    def _configure_exposure(self, node_map) -> None:
        if self._exposure is None:
            return

        try:
            node_map.ExposureAuto.value = "Off"
        except Exception:
            pass

        for attr in ("ExposureTime", "Exposure"):
            try:
                node = getattr(node_map, attr)
                node.value = float(self._exposure)
                return
            except AttributeError:
                continue
            except Exception as e:
                LOG.warning("Failed to set exposure via %s: %s", attr, e)
        LOG.warning("Could not set exposure to %s µs", self._exposure)

    def _configure_gain(self, node_map) -> None:
        if self._gain is None:
            return

        try:
            node_map.GainAuto.value = "Off"
        except Exception:
            pass

        try:
            node_map.Gain.value = float(self._gain)
        except Exception as e:
            LOG.warning("Could not set gain to %s: %s", self._gain, e)

    def _configure_frame_rate(self, node_map) -> None:
        if not self.settings.fps:
            return

        target = float(self.settings.fps)
        LOG.info("Configuring GenTL frame rate: requested %.3f FPS", target)

        for attr in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateControlEnable"):
            try:
                node = getattr(node_map, attr)
                before = getattr(node, "value", None)
                node.value = True
                after = getattr(node, "value", None)
                LOG.info("Enabled GenTL %s: before=%r after=%r", attr, before, after)
                break
            except Exception:
                pass

        for attr in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
            try:
                node = getattr(node_map, attr)
                before = getattr(node, "value", None)
                node.value = target
                after = getattr(node, "value", None)

                LOG.info(
                    "Set GenTL %s: before=%r requested=%.3f after=%r",
                    attr,
                    before,
                    target,
                    after,
                )

                try:
                    accepted = float(after)
                    if accepted > 0:
                        self._actual_fps = accepted
                except Exception:
                    pass

                return

            except AttributeError:
                continue
            except Exception as e:
                LOG.warning("Failed to set frame rate via %s: %s", attr, e)

        LOG.warning("Could not set frame rate to %s FPS", target)

    def _read_telemetry(self, node_map) -> None:
        try:
            self._actual_width = int(node_map.Width.value)
            self._actual_height = int(node_map.Height.value)
        except Exception:
            pass

        # Prefer true/resulting frame-rate readback nodes.
        resulting_fps = self._node_float(
            node_map,
            "AcquisitionResultingFrameRate",
            "ResultingFrameRate",
            "AcquisitionFrameRateResulting",
            "DeviceFrameRate",
        )

        # Fallback to requested/accepted frame-rate nodes only if no resulting node exists.
        requested_fps = self._node_float(
            node_map,
            "AcquisitionFrameRate",
            "AcquisitionFrameRateAbs",
        )

        if resulting_fps is not None:
            self._actual_fps = resulting_fps
        elif requested_fps is not None:
            self._actual_fps = requested_fps

        exposure = self._node_float(
            node_map,
            "ExposureTime",
            "ExposureTimeAbs",
            "Exposure",
            allow_zero=True,
        )
        if exposure is not None:
            self._actual_exposure = exposure

        gain = self._node_float(
            node_map,
            "Gain",
            "GainRaw",
            allow_zero=True,
        )
        if gain is not None:
            self._actual_gain = gain

        # Persist useful telemetry into properties["gentl"] for GUI/debugging.
        try:
            ns = self._ensure_settings_ns()

            if self._actual_width and self._actual_height:
                ns["actual_resolution"] = [int(self._actual_width), int(self._actual_height)]

            if self._actual_fps is not None:
                ns["actual_fps"] = float(self._actual_fps)

            if resulting_fps is not None:
                ns["actual_resulting_frame_rate"] = float(resulting_fps)

            if requested_fps is not None:
                ns["actual_acquisition_frame_rate"] = float(requested_fps)

            if self._actual_exposure is not None:
                ns["actual_exposure"] = float(self._actual_exposure)

            if self._actual_gain is not None:
                ns["actual_gain"] = float(self._actual_gain)

            exposure_auto = self._node_str(node_map, "ExposureAuto")
            if exposure_auto is not None:
                ns["actual_exposure_auto"] = exposure_auto

            throughput = self._node_float(node_map, "DeviceLinkThroughputLimit", allow_zero=True)
            if throughput is not None:
                ns["actual_device_link_throughput_limit"] = float(throughput)

            throughput_mode = self._node_str(node_map, "DeviceLinkThroughputLimitMode")
            if throughput_mode is not None:
                ns["actual_device_link_throughput_limit_mode"] = throughput_mode

            pixel_format = self._node_str(node_map, "PixelFormat")
            if pixel_format is not None:
                self._camera_pixel_format = pixel_format
                ns["actual_pixel_format"] = pixel_format
                ns["detected_pixel_format"] = pixel_format

            output_format = self.actual_output_format
            if output_format is not None:
                ns["actual_output_format"] = output_format

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Frame conversion / local helpers
    # ------------------------------------------------------------------

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            max_val = float(frame.max()) if frame.size else 0.0
            scale = 255.0 / max_val if max_val > 0.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)

        fmt = str(self._pixel_format or "").strip()

        if frame.ndim == 2:
            if fmt == "BayerRG8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            elif fmt == "BayerGB8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
            elif fmt == "BayerGR8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
            elif fmt == "BayerBG8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        elif frame.ndim == 3 and frame.shape[2] == 3:
            if fmt == "RGB8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # BGR8 is already OpenCV-native.

        if self._crop is not None:
            top, bottom, left, right = (int(v) for v in self._crop)
            top = max(0, top)
            left = max(0, left)
            bottom = bottom if bottom > 0 else frame.shape[0]
            right = right if right > 0 else frame.shape[1]
            frame = frame[top : min(frame.shape[0], bottom), left : min(frame.shape[1], right)]

        if self._rotate in (90, 180, 270):
            rotations = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            frame = cv2.rotate(frame, rotations[self._rotate])

        return frame.copy()

    def _resolve_device_label(self, node_map) -> str | None:
        for name_attr, serial_attr in (
            ("DeviceModelName", "DeviceSerialNumber"),
            ("DeviceDisplayName", "DeviceSerialNumber"),
        ):
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
                return f"{model} ({serial})" if serial else str(model)
        return None

    def _parse_crop(self, crop) -> tuple[int, int, int, int] | None:
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            return tuple(int(v) for v in crop)
        return None

    def _get_requested_resolution_or_none(self) -> tuple[int, int] | None:
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

    @staticmethod
    def _adjust_to_increment(value: int, minimum: int, maximum: int, increment: int) -> int:
        value = max(minimum, min(maximum, int(value)))
        if increment <= 0:
            return value
        return minimum + ((value - minimum) // increment) * increment

    @staticmethod
    def _positive_float(value) -> float | None:
        try:
            number = float(value)
            return number if number > 0 else None
        except Exception:
            return None

    def device_name(self) -> str:
        if self._device_label:
            return self._device_label
        return super().device_name()
