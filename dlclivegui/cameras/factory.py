"""Backend discovery and construction utilities."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Iterable  # CHANGED
from contextlib import contextmanager
from dataclasses import dataclass

from ..config import CameraSettings
from .base import CameraBackend


def _opencv_get_log_level(cv2):
    """Return OpenCV log level using new utils.logging API when available, else legacy."""
    # Preferred (OpenCV ≥ 4.x): cv2.utils.logging.getLogLevel()
    try:
        return cv2.utils.logging.getLogLevel()
    except Exception:
        # Legacy (older OpenCV): cv2.getLogLevel()
        try:
            return cv2.getLogLevel()
        except Exception:
            return None  # unknown / not supported


def _opencv_set_log_level(cv2, level: int):
    """Set OpenCV log level using new utils.logging API when available, else legacy."""
    # Preferred (OpenCV ≥ 4.x): cv2.utils.logging.setLogLevel(level)
    try:
        cv2.utils.logging.setLogLevel(level)
        return
    except Exception:
        # Legacy (older OpenCV): cv2.setLogLevel(level)
        try:
            cv2.setLogLevel(level)
        except Exception:
            pass  # not supported on this build


@contextmanager
def _suppress_opencv_logging():
    """Temporarily suppress OpenCV logging during camera probing (backwards compatible)."""
    try:
        import cv2

        # Resolve a 'silent' level cross-version.
        # In newer OpenCV it's 0 (LOG_LEVEL_SILENT).
        SILENT = 0
        old_level = _opencv_get_log_level(cv2)

        _opencv_set_log_level(cv2, SILENT)
        try:
            yield
        finally:
            # Restore if we were able to read it
            if old_level is not None:
                _opencv_set_log_level(cv2, int(old_level))
    except ImportError:
        # OpenCV not installed; nothing to suppress
        yield


@dataclass
class DetectedCamera:
    """Information about a camera discovered during probing."""

    index: int
    label: str


_BACKENDS: dict[str, tuple[str, str]] = {
    "opencv": ("dlclivegui.cameras.opencv_backend", "OpenCVCameraBackend"),
    "basler": ("dlclivegui.cameras.basler_backend", "BaslerCameraBackend"),
    "gentl": ("dlclivegui.cameras.gentl_backend", "GenTLCameraBackend"),
    "aravis": ("dlclivegui.cameras.aravis_backend", "AravisCameraBackend"),
}


def _sanitize_for_probe(settings: CameraSettings) -> CameraSettings:
    """
    Return a light, side-effect-minimized copy of CameraSettings for availability probes.
    - Zero FPS (let driver pick default)
    - Keep only 'api' hint in properties, force fast_start=True
    - Do not change 'enabled'
    """
    probe = copy.deepcopy(settings)
    probe.fps = 0.0  # don't force FPS during probe
    props = probe.properties if isinstance(probe.properties, dict) else {}
    api = props.get("api")
    probe.properties = {}
    if api is not None:
        probe.properties["api"] = api
    probe.properties["fast_start"] = True
    return probe


class CameraFactory:
    """Create camera backend instances based on configuration."""

    @staticmethod
    def backend_names() -> Iterable[str]:
        """Return the identifiers of all known backends."""
        return tuple(_BACKENDS.keys())

    @staticmethod
    def available_backends() -> dict[str, bool]:
        """Return a mapping of backend names to availability flags."""
        availability: dict[str, bool] = {}
        for name in _BACKENDS:
            try:
                backend_cls = CameraFactory._resolve_backend(name)
            except RuntimeError:
                availability[name] = False
                continue
            availability[name] = backend_cls.is_available()
        return availability

    @staticmethod
    def detect_cameras(
        backend: str,
        max_devices: int = 10,
        *,
        should_cancel: Callable[[], bool] | None = None,  # NEW
        progress_cb: Callable[[str], None] | None = None,  # NEW
    ) -> list[DetectedCamera]:
        """Probe ``backend`` for available cameras.

        Parameters
        ----------
        backend:
            The backend identifier, e.g. ``"opencv"``.
        max_devices:
            Upper bound for the indices that should be probed.
            For backends with get_device_count (GenTL, Aravis), the actual device count is queried.
        should_cancel:
            Optional callable that returns True if discovery should be canceled.
            When cancellation is requested, the function returns the cameras found so far.
        progress_cb:
            Optional callable to receive human-readable progress messages.

        Returns
        -------
        list of :class:`DetectedCamera`
            Sorted list of detected cameras with human readable labels (partial if canceled).
        """

        def _canceled() -> bool:
            return bool(should_cancel and should_cancel())

        try:
            backend_cls = CameraFactory._resolve_backend(backend)
        except RuntimeError:
            return []
        if not backend_cls.is_available():
            return []

        # Resolve device count if possible
        num_devices = max_devices
        if hasattr(backend_cls, "get_device_count"):
            try:
                if _canceled():
                    return []
                actual_count = backend_cls.get_device_count()
                if actual_count >= 0:
                    num_devices = actual_count
            except Exception:
                pass

        detected: list[DetectedCamera] = []
        # Suppress OpenCV warnings/errors during probing (e.g., "can't open camera by index")
        with _suppress_opencv_logging():
            try:
                for index in range(num_devices):
                    if _canceled():
                        # return partial results immediately
                        break

                    if progress_cb:
                        progress_cb(f"Probing {backend}:{index}…")

                    # Prefer quick presence check first
                    quick_ok = None
                    if hasattr(backend_cls, "quick_ping"):
                        try:
                            quick_ok = bool(backend_cls.quick_ping(index))  # type: ignore[attr-defined]
                        except TypeError:
                            quick_ok = bool(backend_cls.quick_ping(index, None))  # type: ignore[attr-defined]
                        except Exception:
                            quick_ok = None
                    if quick_ok is False:
                        # Definitely not present, skip heavy open
                        continue

                    settings = CameraSettings(
                        name=f"Probe {index}",
                        index=index,
                        fps=30.0,
                        backend=backend,
                        properties={},
                    )
                    backend_instance = backend_cls(settings)

                    try:
                        # This open() may block for a short time depending on driver/backend.
                        backend_instance.open()
                    except Exception:
                        # Not available → continue probing next index
                        pass
                    else:
                        label = backend_instance.device_name() or f"{backend.title()} #{index}"
                        detected.append(DetectedCamera(index=index, label=label))
                        if progress_cb:
                            progress_cb(f"Found {label}")
                    finally:
                        try:
                            backend_instance.close()
                        except Exception:
                            pass

                    # Check cancel again between indices
                    if _canceled():
                        break

            except KeyboardInterrupt:
                # Graceful early exit with partial results
                if progress_cb:
                    progress_cb("Discovery interrupted.")
            # any other exception bubbles up to caller

        detected.sort(key=lambda camera: camera.index)
        return detected

    @staticmethod
    def create(settings: CameraSettings) -> CameraBackend:
        """Instantiate a backend for ``settings``."""
        backend_name = (settings.backend or "opencv").lower()
        try:
            backend_cls = CameraFactory._resolve_backend(backend_name)
        except RuntimeError as exc:  # pragma: no cover - runtime configuration
            raise RuntimeError(f"Unknown camera backend '{backend_name}': {exc}") from exc
        if not backend_cls.is_available():
            raise RuntimeError(
                f"Camera backend '{backend_name}' is not available. "
                "Ensure the required drivers and Python packages are installed."
            )
        return backend_cls(settings)

    @staticmethod
    def check_camera_available(settings: CameraSettings) -> tuple[bool, str]:
        """Check if a camera is present/accessible without pushing heavy settings like FPS."""
        backend_name = (settings.backend or "opencv").lower()

        try:
            backend_cls = CameraFactory._resolve_backend(backend_name)
        except RuntimeError as exc:
            return False, f"Backend '{backend_name}' not installed: {exc}"

        if not backend_cls.is_available():
            return False, f"Backend '{backend_name}' is not available (missing drivers/packages)"

        # Prefer quick presence test if the backend provides it (e.g., OpenCV.quick_ping)
        if hasattr(backend_cls, "quick_ping"):
            try:
                with _suppress_opencv_logging():
                    idx = int(settings.index)
                    # Most backends expose quick_ping(index [, backend_flag])
                    ok = False
                    try:
                        ok = backend_cls.quick_ping(idx)  # type: ignore[attr-defined]
                    except TypeError:
                        # Fallback signature with backend flag if required by the specific backend
                        ok = backend_cls.quick_ping(idx, None)  # type: ignore[attr-defined]
                    if ok:
                        return True, ""
                    return False, "Device not present"
            except Exception as exc:
                return False, f"Quick probe failed: {exc}"

        # 2) Fallback: try a very lightweight open/close with sanitized settings
        try:
            probe_settings = _sanitize_for_probe(settings)
            backend_instance = backend_cls(probe_settings)
            with _suppress_opencv_logging():
                backend_instance.open()
                backend_instance.close()
            return True, ""
        except Exception as exc:
            return False, f"Camera not accessible: {exc}"

    @staticmethod
    def _resolve_backend(name: str) -> type[CameraBackend]:
        try:
            module_name, class_name = _BACKENDS[name]
        except KeyError as exc:
            raise RuntimeError("backend not registered") from exc
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(str(exc)) from exc
        backend_cls = getattr(module, class_name)
        if not issubclass(backend_cls, CameraBackend):  # pragma: no cover - safety
            raise RuntimeError(f"Backend '{name}' does not implement CameraBackend")
        return backend_cls
