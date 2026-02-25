"""GenTL backend implemented using the Harvesters library."""

#  dlclivegui/cameras/backends/gentl_backend.py
from __future__ import annotations

import logging
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
    """Capture frames from GenTL-compatible devices via Harvesters."""

    OPTIONS_KEY: ClassVar[str] = "gentl"
    _LEGACY_DEFAULT_CTI_PATTERNS: tuple[str, ...] = (  # Windows-only, ignored on other platforms
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\IC4 GenTL Driver for USB3Vision Devices *\\bin\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Camera SDK\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files (x86)\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
    )
    # Source marker stored in properties["gentl"]["cti_files_source"]
    # auto : persisted by auto-discovery (env vars, patterns, etc.). Cache, may be stale, re-discover if missing.
    # user : explicitly set by user via properties.gentl.cti_file(s). Cache, strict raise if missing.
    _CTI_FILES_SOURCE_AUTO: ClassVar[str] = "auto"
    _CTI_FILES_SOURCE_USER: ClassVar[str] = "user"

    def __init__(self, settings):
        super().__init__(settings)

        # --- Properties namespace handling (new UI stores backend options under properties["gentl"]) ---
        props = settings.properties if isinstance(settings.properties, dict) else {}
        ns = props.get(self.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}

        # --- Fast probe mode (CameraProbeWorker sets this) ---
        # When fast_start=True, open() should avoid starting acquisition if possible.
        self._fast_start: bool = bool(ns.get("fast_start", False))

        # --- Stable identity / serial selection ---
        # New UI stores stable identity as ns["device_id"], with recommended formats:
        #   - "serial:<SERIAL>" for true serials
        #   - "fp:<fingerprint...>" when serial is missing/ambiguous
        #
        # We keep legacy "serial_number"/"serial" behavior as fallback.
        raw_device_id = ns.get("device_id") or props.get("device_id")
        legacy_serial = ns.get("serial_number") or ns.get("serial") or props.get("serial_number") or props.get("serial")

        self._device_id: str | None = str(raw_device_id).strip() if raw_device_id else None

        # Decide what to use for actual device selection in open():
        # - If device_id is "serial:XXXX" -> use XXXX as serial_number
        # - Otherwise, keep legacy serial if present; open() may still use index if serial is None
        self._serial_number: str | None = None
        if self._device_id:
            did = self._device_id
            if did.startswith("serial:"):
                self._serial_number = did.split("serial:", 1)[1].strip() or None
            elif did.startswith("fp:"):
                # fingerprint: not directly usable as serial; rebind_settings should map fp -> index
                self._serial_number = legacy_serial  # keep legacy if any, otherwise None
            else:
                # If device_id is provided without prefix, treat it as a "serial-like" value for backward compatibility
                self._serial_number = did
        else:
            self._serial_number = str(legacy_serial).strip() if legacy_serial else None

        # --- Pixel format / image transforms (legacy + backend options) ---
        self._pixel_format: str = ns.get("pixel_format") or props.get("pixel_format", "Mono8")
        self._rotate: int = int(ns.get("rotate", props.get("rotate", 0))) % 360
        self._crop: tuple[int, int, int, int] | None = self._parse_crop(ns.get("crop", props.get("crop")))

        # --- Exposure / Gain: 0 means Auto (do not set) ---
        exp_val = getattr(settings, "exposure", 0)
        gain_val = getattr(settings, "gain", 0.0)

        self._exposure: float | None = (
            float(exp_val) if isinstance(exp_val, (int, float)) and float(exp_val) > 0 else None
        )
        if self._exposure is None:
            v = ns.get("exposure", props.get("exposure"))
            try:
                self._exposure = float(v) if v is not None and float(v) > 0 else None
            except Exception:
                self._exposure = None

        self._gain: float | None = (
            float(gain_val) if isinstance(gain_val, (int, float)) and float(gain_val) > 0 else None
        )
        if self._gain is None:
            v = ns.get("gain", props.get("gain"))
            try:
                self._gain = float(v) if v is not None and float(v) > 0 else None
            except Exception:
                self._gain = None

        # --- Acquisition timeout ---
        self._timeout: float = float(ns.get("timeout", props.get("timeout", 2.0)))

        # --- Resolution request (None = device default / Auto) ---
        # Uses settings.width/settings.height if set; falls back to legacy props["resolution"] if present.
        self._requested_resolution: tuple[int, int] | None = self._get_requested_resolution_or_none()

        # --- Actuals for GUI ---
        self._actual_width: int | None = None
        self._actual_height: int | None = None
        self._actual_fps: float | None = None
        self._actual_gain: float | None = None
        self._actual_exposure: float | None = None

        # --- Harvesters resources ---
        self._harvester = None
        self._acquirer = None
        self._device_label: str | None = None

        self._cti_files_source_used: str | None = None

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

    @classmethod
    def get_device_count(cls) -> int:
        """Get the number of GenTL devices detected by Harvester.

        Returns the number of devices found, or -1 if detection fails.
        """
        if Harvester is None:
            return -1

        harvester = None
        try:
            harvester, _, _ = cls._build_harvester_for_discovery(strict_single=False)
            if harvester is None:
                return -1
            infos = harvester.device_info_list or []
            return len(infos)
        except Exception:
            return -1
        finally:
            if harvester is not None:
                try:
                    harvester.reset()
                except Exception:
                    pass

    @staticmethod
    def _cti_preflight(path: str) -> tuple[bool, str | None]:
        """
        Best-effort check right before calling Harvester.add_file().
        Still subject to race conditions (e.g. file deleted after this check),
        but helps diagnose common issues like missing files or permission errors more gracefully and early.
        Returns (ok, reason_if_not_ok).
        """
        p = Path(str(path))
        try:
            if not p.exists():
                return False, "missing at load time"
            if not p.is_file():
                return False, "not a file at load time"
            # Optional: try opening for read to detect permission/locking issues early
            with p.open("rb"):
                pass
            return True, None
        except PermissionError:
            return False, "permission denied at load time"
        except OSError as e:
            return False, f"os error at load time: {e}"

    def _resolve_cti_files_for_settings(self) -> list[str]:
        """
        Resolve CTI files to load.

        - User override (properties.gentl.cti_file/cti_files OR legacy properties.cti_file/cti_files):
            * strict: must exist, otherwise raise
            * source = "user"
        - Auto-persisted cache (properties.gentl.cti_files_source == "auto"):
            * try persisted ctis first
            * if stale/missing, fall back to discovery
            * source = "auto"
        - Default: discovery (env + configured patterns/dirs) => source = "auto"

        NOTE : legacy properties.cti_file(s) always take strict precedence as user override if present,
        even if source marker says "auto".
        Never raise just because multiple CTIs exist.
        Raise only when none are found (after allowed fallback).
        """
        props = self.settings.properties if isinstance(self.settings.properties, dict) else {}
        ns = props.get(self.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}

        # Read source marker
        source = ns.get("cti_files_source")
        source = str(source).strip().lower() if source is not None else None

        # Explicit CTIs (namespace first, then legacy top-level)
        ns_cti_files = ns.get("cti_files")
        ns_cti_file = ns.get("cti_file")
        legacy_cti_files = props.get("cti_files")
        legacy_cti_file = props.get("cti_file")

        # ------------------------------------------------------------
        # 1) Legacy explicit CTIs: always treat as user override (strict)
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------------------
        # 2) Namespace explicit CTIs: behavior depends on cti_files_source marker
        #    - source=="auto": treat as cache, stale => fallback to discovery
        #    - otherwise: strict user override
        # ------------------------------------------------------------------------
        if ns_cti_files or ns_cti_file:
            is_auto_cache = source == self._CTI_FILES_SOURCE_AUTO

            # Default to "user" if the marker is missing/unknown.
            self._cti_files_source_used = self._CTI_FILES_SOURCE_AUTO if is_auto_cache else self._CTI_FILES_SOURCE_USER

            candidates, diag = cti_finder.discover_cti_files(
                cti_file=str(ns_cti_file) if ns_cti_file else None,
                cti_files=cti_finder.cti_files_as_list(ns_cti_files) if ns_cti_files else None,
                include_env=False,
                must_exist=True,
            )

            if candidates:
                return list(candidates)

            # If auto cache is stale, fall back to discovery
            if is_auto_cache:
                LOG.info(
                    "Auto-persisted GenTL CTIs appear stale/missing; falling back to discovery. "
                    "Persisted cti_file=%s cti_files=%s",
                    ns_cti_file,
                    ns_cti_files,
                )
                # Fall through to discovery (below)
            else:
                # User override: strict failure
                raise RuntimeError(
                    "No valid GenTL producer (.cti) found from properties.gentl.cti_file/cti_files.\n\n"
                    f"Discovery details:\n{diag.summarize()}"
                )

        # ------------------------------------------------------------
        # 3) Discovery path: env vars + patterns/dirs (source = "auto")
        # ------------------------------------------------------------
        self._cti_files_source_used = self._CTI_FILES_SOURCE_AUTO

        search_paths = ns.get("cti_search_paths", props.get("cti_search_paths"))
        extra_dirs = ns.get("cti_dirs", props.get("cti_dirs"))

        candidates, diag = cti_finder.discover_cti_files(
            cti_search_paths=cti_finder.cti_files_as_list(search_paths) if search_paths is not None else None,
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
                f"Discovery details:\n{diag.summarize()}"
            )

        return list(candidates)

    @classmethod
    def _build_harvester_for_discovery(
        cls,
        *,
        strict_single: bool = False,  # retained for optional future use
    ):
        """
        Build a Harvester instance and load CTI producers for class-level operations
        (discover_devices, quick_ping, get_device_count, rebind_settings).

        Default policy: try to load ALL discovered producers.
        """
        if Harvester is None:
            return None, [], None

        candidates, diag = cti_finder.discover_cti_files(
            include_env=True,
            cti_search_paths=list(cls._LEGACY_DEFAULT_CTI_PATTERNS),
            must_exist=True,
        )

        if not candidates:
            return None, [], diag

        # Default: load all candidates
        cti_files = list(candidates)

        # Optional strict mode (off by default)
        if strict_single:
            # If you ever want strict, use choose_cti_files here; otherwise ignore.
            cti_files = cti_finder.choose_cti_files(
                cti_files, policy=cti_finder.GenTLDiscoveryPolicy.RAISE_IF_MULTIPLE, max_files=1
            )

        harvester = Harvester()
        loaded: list[str] = []
        failures: list[tuple[str, str]] = []

        for cti in cti_files:
            ok, reason = cls._cti_preflight(cti)
            if not ok:
                failures.append((str(cti), reason or "Check failed"))
                LOG.warning("Skipping CTI '%s' during discovery preflight: %s", cti, reason)
                continue

            try:
                harvester.add_file(cti)
                loaded.append(cti)
            except Exception as exc:
                failures.append((str(cti), str(exc)))
                LOG.warning("Failed to load CTI '%s' during discovery: %s", cti, exc)

        if not loaded:
            try:
                harvester.reset()
            except Exception:
                pass
            return None, [], diag

        try:
            harvester.update()
        except Exception as exc:
            LOG.error(
                "Harvester.update() failed during discovery: %s "
                "Device list not usable, treating as discovery failure."
                "CTIs loaded before failure : %s",
                exc,
                loaded,
            )
            try:
                harvester.reset()
            except Exception:
                pass
            # Update failure
            return None, [], diag

        return harvester, loaded, diag

    def open(self) -> None:
        if Harvester is None:  # pragma: no cover
            raise RuntimeError(
                "The 'harvesters' package is required for the GenTL backend. Install it via 'pip install harvesters'."
            )

        # Ensure properties namespace exists for persistence back to UI
        if not isinstance(self.settings.properties, dict):
            self.settings.properties = {}
        props = self.settings.properties
        ns = props.get(self.OPTIONS_KEY, {})
        if not isinstance(ns, dict):
            ns = {}
            props[self.OPTIONS_KEY] = ns

        # Resolve CTIs (may return many). This no longer raises just because there are multiple.
        cti_files = self._resolve_cti_files_for_settings()
        ns["cti_files_source"] = (
            self._cti_files_source_used or ns.get("cti_files_source") or self._CTI_FILES_SOURCE_AUTO
        )

        self._harvester = Harvester()

        loaded: list[str] = []
        failed: list[tuple[str, str]] = []

        for cti in cti_files:
            ok, reason = self._cti_preflight(cti)
            if not ok:
                failed.append((str(cti), reason or "preflight failed"))
                LOG.warning("Skipping CTI '%s': %s", cti, reason)
                continue

            try:
                self._harvester.add_file(cti)
                loaded.append(cti)
            except Exception as exc:
                failed.append((str(cti), str(exc)))
                LOG.warning("Failed to load CTI '%s': %s", cti, exc)

        # Persist diagnostics for UI / debugging
        ns["cti_files"] = [str(p) for p in cti_files]  # all resolved candidates
        ns["cti_files_loaded"] = [str(p) for p in loaded]  # successfully added to harvester
        ns["cti_files_failed"] = [{"cti": c, "error": e} for c, e in failed]  # load failures

        # Keep single-cti convenience key for backward compatibility / display
        if loaded:
            ns["cti_file"] = str(loaded[0])
        elif cti_files:
            ns["cti_file"] = str(cti_files[0])  # best effort

        if not loaded:
            self._reset_harvester()
            raise RuntimeError(
                "No GenTL producer (.cti) could be loaded.\n\n"
                f"Resolved CTIs: {cti_files}\n"
                f"Failures: {failed}\n"
                "Fix: remove/repair incompatible producers or "
                "set properties.gentl.cti_file to a known working producer."
            )

        # Update device list after loading producers
        self._harvester.update()

        if not self._harvester.device_info_list:
            self._reset_harvester()
            raise RuntimeError(
                "No GenTL cameras detected via Harvesters after loading producers.\n\n"
                f"Loaded CTIs: {loaded}\n"
                f"Failed CTIs: {failed}\n"
                "Fix: ensure your camera vendor's GenTL producer is installed and working."
            )

        infos = list(self._harvester.device_info_list)

        # Helper: robustly read device_info fields (dict-like or attribute-like)
        def _info_get(info, key: str, default=None):
            try:
                if hasattr(info, "get"):
                    v = info.get(key)
                    if v is not None:
                        return v
            except Exception:
                pass
            try:
                v = getattr(info, key, None)
                if v is not None:
                    return v
            except Exception:
                pass
            return default

        # ------------------------------------------------------------------
        # Device selection (stable device_id > serial > index)
        # ------------------------------------------------------------------
        requested_index = int(self.settings.index or 0)
        selected_index: int | None = None
        selected_serial: str | None = None

        target_device_id = self._device_id or ns.get("device_id") or props.get("device_id")
        if target_device_id:
            target_device_id = str(target_device_id).strip()

            # Exact match against computed device_id
            for idx, info in enumerate(infos):
                try:
                    did = self._device_id_from_info(info)
                except Exception:
                    did = None
                if did and did == target_device_id:
                    selected_index = idx
                    selected_serial = _info_get(info, "serial_number", None)
                    selected_serial = str(selected_serial).strip() if selected_serial else None
                    break

            # If device_id is "serial:XXXX", match serial directly
            if selected_index is None and target_device_id.startswith("serial:"):
                serial_target = target_device_id.split("serial:", 1)[1].strip()
                if serial_target:
                    exact = []
                    for idx, info in enumerate(infos):
                        sn = _info_get(info, "serial_number", "")
                        sn = str(sn).strip() if sn is not None else ""
                        if sn == serial_target:
                            exact.append((idx, sn))
                    if exact:
                        selected_index = exact[0][0]
                        selected_serial = exact[0][1]
                    else:
                        sub = []
                        for idx, info in enumerate(infos):
                            sn = _info_get(info, "serial_number", "")
                            sn = str(sn).strip() if sn is not None else ""
                            if serial_target and serial_target in sn:
                                sub.append((idx, sn))
                        if len(sub) == 1:
                            selected_index = sub[0][0]
                            selected_serial = sub[0][1] or None
                        elif len(sub) > 1:
                            candidates = [sn for _, sn in sub]
                            raise RuntimeError(
                                f"Ambiguous GenTL serial match for '{serial_target}'. Candidates: {candidates}"
                            )

        # Legacy serial selection fallback
        if selected_index is None:
            serial = self._serial_number
            if serial:
                serial = str(serial).strip()
                exact = []
                for idx, info in enumerate(infos):
                    sn = _info_get(info, "serial_number", "")
                    sn = str(sn).strip() if sn is not None else ""
                    if sn == serial:
                        exact.append((idx, sn))
                if exact:
                    selected_index = exact[0][0]
                    selected_serial = exact[0][1]
                else:
                    sub = []
                    for idx, info in enumerate(infos):
                        sn = _info_get(info, "serial_number", "")
                        sn = str(sn).strip() if sn is not None else ""
                        if serial and serial in sn:
                            sub.append((idx, sn))
                    if len(sub) == 1:
                        selected_index = sub[0][0]
                        selected_serial = sub[0][1] or None
                    elif len(sub) > 1:
                        candidates = [sn for _, sn in sub]
                        raise RuntimeError(f"Ambiguous GenTL serial match for '{serial}'. Candidates: {candidates}")
                    else:
                        available = [str(_info_get(i, "serial_number", "")).strip() for i in infos]
                        raise RuntimeError(f"Camera with serial '{serial}' not found. Available cameras: {available}")

        # Index fallback
        if selected_index is None:
            device_count = len(infos)
            if requested_index < 0 or requested_index >= device_count:
                raise RuntimeError(f"Camera index {requested_index} out of range for {device_count} GenTL device(s)")
            selected_index = requested_index
            sn = _info_get(infos[selected_index], "serial_number", "")
            selected_serial = str(sn).strip() if sn else None

        # Update settings.index to actual selected index (UI stability)
        self.settings.index = int(selected_index)
        selected_info = infos[int(selected_index)]

        # Create ImageAcquirer via Harvester.create(...)
        try:
            if selected_serial:
                self._acquirer = self._harvester.create({"serial_number": str(selected_serial)})
            else:
                self._acquirer = self._harvester.create(int(selected_index))
        except TypeError:
            if selected_serial:
                self._acquirer = self._harvester.create({"serial_number": str(selected_serial)})
            else:
                self._acquirer = self._harvester.create(index=int(selected_index))

        remote = self._acquirer.remote_device
        node_map = remote.node_map

        self._device_label = self._resolve_device_label(node_map)

        # Apply configuration
        self._configure_pixel_format(node_map)
        self._configure_resolution(node_map)
        self._configure_exposure(node_map)
        self._configure_gain(node_map)
        self._configure_frame_rate(node_map)

        # Read back telemetry
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

        # Persist identity + metadata
        computed_id = None
        try:
            computed_id = self._device_id_from_info(selected_info)
        except Exception:
            computed_id = None

        if computed_id:
            ns["device_id"] = computed_id
        elif selected_serial:
            ns["device_id"] = f"serial:{selected_serial}"

        if selected_serial:
            ns["serial_number"] = str(selected_serial)
            ns["device_serial_number"] = str(selected_serial)

        if self._device_label:
            ns["device_name"] = str(self._device_label)

        ns["device_display_name"] = str(_info_get(selected_info, "display_name", "") or "")
        ns["device_info_id"] = str(_info_get(selected_info, "id_", "") or "")
        ns["device_vendor"] = str(_info_get(selected_info, "vendor", "") or "")
        ns["device_model"] = str(_info_get(selected_info, "model", "") or "")
        ns["device_tl_type"] = str(_info_get(selected_info, "tl_type", "") or "")
        ns["device_user_defined_name"] = str(_info_get(selected_info, "user_defined_name", "") or "")
        ns["device_version"] = str(_info_get(selected_info, "version", "") or "")
        ns["device_access_status"] = _info_get(selected_info, "access_status", None)

        # Start acquisition unless fast_start
        if getattr(self, "_fast_start", False):
            LOG.info("GenTL open() in fast_start probe mode: acquisition not started.")
            return

        self._acquirer.start()

    @staticmethod
    def _device_id_from_info(info) -> str | None:
        """
        Build a stable-ish device identifier from Harvester device_info_list entries.
        This helper supports both dict-like and attribute-like representations.
        """

        def _read(name: str):
            # dict-like
            try:
                if hasattr(info, "get"):
                    v = info.get(name)  # type: ignore[attr-defined]
                    if v is not None:
                        return v
            except Exception:
                pass
            # attribute-like
            try:
                return getattr(info, name, None)
            except Exception:
                return None

        def _get(*names: str) -> str | None:
            for n in names:
                v = _read(n)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
            return None

        # Prefer serial if present (best stable key when available)
        serial = _get("serial_number", "SerialNumber", "device_serial_number", "sn", "serial")
        if serial:
            return f"serial:{serial}"

        # Fallback components (best-effort; names may vary per producer)
        vendor = _get("vendor", "vendor_name", "manufacturer", "DeviceVendorName")
        model = _get("model", "model_name", "DeviceModelName")
        user_id = _get("user_defined_name", "user_id", "DeviceUserID", "DeviceUserId", "device_user_id")
        tl_type = _get("tl_type", "transport_layer_type", "DeviceTLType")

        unique = _get("id_", "id", "device_id", "uid", "guid", "mac_address", "interface_id", "display_name")

        parts = []
        for k, v in (("vendor", vendor), ("model", model), ("user", user_id), ("tl", tl_type), ("uid", unique)):
            if v:
                parts.append(f"{k}={v}")

        if not parts:
            return None

        return "fp:" + "|".join(parts)

    @classmethod
    def discover_devices(
        cls,
        *,
        max_devices: int = 10,
        should_cancel: callable[[], bool] | None = None,
        progress_cb: callable[[str], None] | None = None,
    ):
        """
        Rich discovery path for CameraFactory.detect_cameras().
        Returns a list of DetectedCamera with device_id filled when possible.

        Cross-platform CTI discovery:
        - Uses GENICAM_GENTL64_PATH / GENICAM_GENTL32_PATH when available
        - Falls back to built-in Windows patterns
        - Best-effort loads multiple CTI producers
        """
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
            if not infos:
                return []

            out: list[DetectedCamera] = []
            limit = min(len(infos), max_devices if max_devices > 0 else len(infos))

            for idx in range(limit):
                if _canceled():
                    break

                info = infos[idx]

                display_name = None
                try:
                    display_name = (
                        info.get("display_name") if hasattr(info, "get") else getattr(info, "display_name", None)
                    )
                except Exception:
                    display_name = None

                if display_name:
                    label = str(display_name).strip()
                else:
                    vendor = (
                        getattr(info, "vendor", None) or (info.get("vendor") if hasattr(info, "get") else None) or ""
                    )
                    model = getattr(info, "model", None) or (info.get("model") if hasattr(info, "get") else None) or ""
                    serial = (
                        getattr(info, "serial_number", None)
                        or (info.get("serial_number") if hasattr(info, "get") else None)
                        or ""
                    )
                    vendor, model, serial = str(vendor).strip(), str(model).strip(), str(serial).strip()
                    label = f"{vendor} {model}".strip() if (vendor or model) else f"GenTL device {idx}"
                    if serial:
                        label = f"{label} ({serial})"

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
            return []
        finally:
            if harvester is not None:
                try:
                    harvester.reset()
                except Exception:
                    pass

    @classmethod
    def rebind_settings(cls, settings):
        """
        If a stable identity exists in settings.properties['gentl'], map it to the
        correct current index (and serial_number if available).

        Strategy:
        - If CTIs were persisted:
            * if source == "auto" and they are stale -> fall back to discovery
            * otherwise use them (best stability)
        - Otherwise, fall back to env-var + pattern discovery (best-effort).
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

        source = ns.get("cti_files_source")
        source = str(source).strip().lower() if source is not None else None
        is_auto_cache = source == cls._CTI_FILES_SOURCE_AUTO

        harvester = None
        try:
            explicit_files = ns.get("cti_files") or props.get("cti_files")
            explicit_file = ns.get("cti_file") or props.get("cti_file")

            if explicit_files or explicit_file:
                candidates, _diag = cti_finder.discover_cti_files(
                    cti_file=explicit_file,
                    cti_files=cti_finder.cti_files_as_list(explicit_files),
                    include_env=False,
                    must_exist=True,
                )

                if not candidates and is_auto_cache:
                    # Auto cache stale -> fallback to discovery
                    harvester, _loaded, _diag2 = cls._build_harvester_for_discovery(strict_single=False)
                    if harvester is None:
                        return settings
                elif not candidates:
                    # User override stale or unknown -> no rebind
                    return settings
                else:
                    harvester = Harvester()
                    loaded: list[str] = []
                    for cti in candidates:
                        try:
                            harvester.add_file(cti)
                            loaded.append(cti)
                        except Exception:
                            continue
                    if not loaded:
                        cls._reset_select_harvester(harvester)
                        if is_auto_cache:
                            harvester, _loaded, _diag2 = cls._build_harvester_for_discovery(strict_single=False)
                            if harvester is None:
                                return settings
                        else:
                            return settings
                    else:
                        harvester.update()
            else:
                harvester, _loaded, _diag = cls._build_harvester_for_discovery(strict_single=False)
                if harvester is None:
                    return settings

            infos = list(harvester.device_info_list or [])
            if not infos:
                return settings

            target_id_str = str(target_id).strip()
            match_index = None
            match_serial = None

            # 1) Exact match by computed device_id
            for idx, info in enumerate(infos):
                dev_id = cls._device_id_from_info(info)
                if dev_id and dev_id == target_id_str:
                    match_index = idx
                    match_serial = getattr(info, "serial_number", None)
                    break

            # 2) Fallback: treat target as serial-ish substring
            if match_index is None:
                for idx, info in enumerate(infos):
                    serial = getattr(info, "serial_number", None)
                    if serial and target_id_str in str(serial):
                        match_index = idx
                        match_serial = serial
                        break

            if match_index is None:
                return settings

            # Apply rebinding
            settings.index = int(match_index)

            # Ensure namespace exists
            if not isinstance(settings.properties, dict):
                settings.properties = {}
            ns2 = settings.properties.setdefault(cls.OPTIONS_KEY, {})
            if not isinstance(ns2, dict):
                ns2 = {}
                settings.properties[cls.OPTIONS_KEY] = ns2

            if match_serial:
                ns2["serial_number"] = str(match_serial)
            ns2["device_id"] = target_id_str

            return settings

        except Exception:
            return settings
        finally:
            if harvester is not None:
                try:
                    harvester.reset()
                except Exception:
                    pass

    @classmethod
    def quick_ping(cls, index: int, _unused=None) -> bool:
        """
        Fast check: is there a device at this index according to Harvester?
        Does not open/start acquisition.
        """
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
            if harvester is not None:
                try:
                    harvester.reset()
                except Exception:
                    pass

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

        if self._actual_exposure is None:
            try:
                self._actual_exposure = float(self._acquirer.node_map.ExposureTime.value)
            except Exception:
                self._actual_exposure = None

        if self._actual_gain is None:
            try:
                self._actual_gain = float(self._acquirer.node_map.Gain.value)
            except Exception:
                self._actual_gain = None

        return frame, timestamp

    def stop(self) -> None:
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                pass

    @staticmethod
    def _reset_select_harvester(harvester) -> None:
        if harvester is not None:
            try:
                harvester.reset()
            except Exception:
                pass

    def _reset_harvester(self) -> None:
        try:
            self._reset_select_harvester(self._harvester)
        finally:
            self._harvester = None

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

    def _parse_crop(self, crop) -> tuple[int, int, int, int] | None:
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            return tuple(int(v) for v in crop)
        return None

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

    def _configure_resolution(self, node_map) -> None:
        """
        Configure camera resolution only if explicitly requested.
        If None, keep device defaults.
        """
        req = self._requested_resolution
        if req is None:
            LOG.info("Resolution: using device default.")
            return

        requested_width, requested_height = req
        actual_width, actual_height = None, None

        # Width
        try:
            node = node_map.Width
            min_w, max_w = node.min, node.max
            inc_w = getattr(node, "inc", 1)
            width = self._adjust_to_increment(requested_width, min_w, max_w, inc_w)
            node.value = int(width)
            actual_width = node.value
        except Exception as e:
            LOG.warning(f"Failed to set width: {e}")

        # Height
        try:
            node = node_map.Height
            min_h, max_h = node.min, node.max
            inc_h = getattr(node, "inc", 1)
            height = self._adjust_to_increment(requested_height, min_h, max_h, inc_h)
            node.value = int(height)
            actual_height = node.value
        except Exception as e:
            LOG.warning(f"Failed to set height: {e}")

        if actual_width is not None and actual_height is not None:
            self._actual_width = int(actual_width)
            self._actual_height = int(actual_height)
            if (actual_width, actual_height) != (requested_width, requested_height):
                LOG.warning(
                    f"Resolution mismatch: requested {requested_width}x{requested_height}, "
                    f"got {actual_width}x{actual_height}"
                )
            else:
                LOG.info(f"Resolution set to {actual_width}x{actual_height}")

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
