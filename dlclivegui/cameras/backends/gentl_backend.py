"""GenTL backend implemented using the Harvesters library."""

# dlclivegui/cameras/backends/gentl_backend.py
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

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
        self._rotate: int = int(ns.get("rotate", props.get("rotate", 0))) % 360
        self._crop: tuple[int, int, int, int] | None = self._parse_crop(ns.get("crop", props.get("crop")))

        self._exposure: float | None = self._positive_float(getattr(settings, "exposure", 0))
        if self._exposure is None:
            self._exposure = self._positive_float(ns.get("exposure", props.get("exposure")))

        self._gain: float | None = self._positive_float(getattr(settings, "gain", 0.0))
        if self._gain is None:
            self._gain = self._positive_float(ns.get("gain", props.get("gain")))

        self._timeout: float = float(ns.get("timeout", props.get("timeout", 2.0)))
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
        }

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
                    self._configure_trigger(node_map)
                    self._configure_resolution(node_map)
                    self._configure_exposure(node_map)
                    self._configure_gain(node_map)
                    self._configure_frame_rate(node_map)
                    self._read_telemetry(node_map)
                    self._persist_device_metadata(selected_info, selected_serial)

                    if self._fast_start:
                        LOG.info("GenTL open() in fast_start probe mode: acquisition not started.")
                        return

                    self._acquirer.start()

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

        if self._actual_width is None or self._actual_height is None:
            h, w = frame.shape[:2]
            self._actual_width = int(w)
            self._actual_height = int(h)

        if self._actual_exposure is None or self._actual_gain is None:
            try:
                self._read_telemetry(self._acquirer.remote_device.node_map)
            except Exception:
                pass

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
            with self._shared_entry.lock:
                infos = list(self._harvester.device_info_list or [])
            ns["cti_files_loaded"] = list(getattr(self._shared_entry, "loaded_files", loaded))
            LOG.debug(
                "Using shared GenTL Harvester for %d device(s), refcount=%s",
                len(infos),
                cti_finder.SharedHarvesterPool.get_refcount(self._shared_entry),
            )
            return infos
        except Exception as exc:
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

            LOG.debug("GenTL pixel format selected: %s", self._pixel_format)

        except Exception as e:
            LOG.warning("Failed to configure pixel format '%s': %s", self._pixel_format, e)

    def _configure_trigger(self, node_map) -> None:
        try:
            trigger_mode = getattr(node_map, "TriggerMode", None)
            if trigger_mode is not None and "Off" in getattr(trigger_mode, "symbolics", []):
                trigger_mode.value = "Off"
        except Exception as e:
            LOG.warning("Failed to disable trigger mode: %s", e)

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
        for attr in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateControlEnable"):
            try:
                getattr(node_map, attr).value = True
                break
            except Exception:
                pass

        for attr in ("AcquisitionFrameRate", "ResultingFrameRate", "AcquisitionFrameRateAbs"):
            try:
                getattr(node_map, attr).value = target
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

        try:
            self._actual_fps = float(node_map.ResultingFrameRate.value)
        except Exception:
            self._actual_fps = None

        try:
            self._actual_exposure = float(node_map.ExposureTime.value)
        except Exception:
            self._actual_exposure = None

        try:
            self._actual_gain = float(node_map.Gain.value)
        except Exception:
            self._actual_gain = None

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
