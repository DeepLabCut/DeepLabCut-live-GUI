"""Backend discovery and construction utilities."""

# dlclivegui/cameras/factory.py
from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from os import environ

from ..config import CameraSettings
from .base import _BACKEND_REGISTRY, CameraBackend

logger = logging.getLogger(__name__)
_BACKEND_IMPORT_ERRORS: dict[str, str] = {}


@dataclass
class DetectedCamera:
    """Information about a camera discovered during probing."""

    index: int
    label: str


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


# Lazy loader for backends (ensures @register_backend runs)
_BUILTIN_BACKEND_PACKAGES = (
    "dlclivegui.cameras.backends",  # import every submodule once
)
_BACKENDS_IMPORTED = False


def _ensure_backends_loaded() -> None:
    """Import all built-in backend modules once so their decorators run."""
    global _BACKENDS_IMPORTED
    if _BACKENDS_IMPORTED:
        return

    for pkg_name in _BUILTIN_BACKEND_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as exc:
            _BACKEND_IMPORT_ERRORS[pkg_name] = f"{type(exc).__name__}: {exc}"
            logger.exception("FAILED to import backend package '%s': %s", pkg_name, exc)
            if environ.get("DLC_CAMERA_BACKENDS_STRICT_IMPORT", "").strip().lower() in ("1", "true", "yes"):
                raise
            # Package might not exist (fine if all backends are third-party via tests/plugins)
            continue

        # Import every submodule of the package (triggers decorator side-effects)
        pkg_path = getattr(pkg, "__path__", None)
        if not pkg_path:
            continue

        for _finder, mod_name, _is_pkg in pkgutil.iter_modules(pkg_path, prefix=pkg_name + "."):
            try:
                importlib.import_module(mod_name)
                logger.debug("Loaded camera backend module: %s", mod_name)
            except Exception as exc:
                # Record and log loudly WITH traceback
                _BACKEND_IMPORT_ERRORS[mod_name] = f"{type(exc).__name__}: {exc}"
                logger.exception("FAILED to import backend module '%s': %s", mod_name, exc)

                # Optional fail-fast mode for CI/dev
                if environ.get("DLC_CAMERA_BACKENDS_STRICT_IMPORT", "").strip().lower() in ("1", "true", "yes"):
                    raise

    _BACKENDS_IMPORTED = True


class CameraFactory:
    """Create camera backend instances based on configuration."""

    @staticmethod
    def backend_names() -> Iterable[str]:
        """Return the identifiers of all known backends."""
        _ensure_backends_loaded()
        return tuple(_BACKEND_REGISTRY.keys())

    @staticmethod
    def available_backends() -> dict[str, bool]:
        """Return a mapping of backend names to availability flags."""
        _ensure_backends_loaded()
        availability: dict[str, bool] = {}
        for name in _BACKEND_REGISTRY:
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
        should_cancel: Callable[[], bool] | None = None,
        progress_cb: Callable[[str], None] | None = None,
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
        _ensure_backends_loaded()

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
                    settings = backend_cls.sanitize_for_probe(settings)
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
        # always ensure backends are loaded before creating,
        # to get accurate error reporting for unknown backends
        _ensure_backends_loaded()
        dc = settings
        backend_name = (dc.backend or "opencv").lower()
        try:
            backend_cls = CameraFactory._resolve_backend(backend_name)
            try:
                backend_cls.parse_options(settings)  # ensures bad config fails loudly here
            except Exception as exc:
                raise RuntimeError(f"Invalid {backend_name} options: {exc}") from exc

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
        # always ensure backends are loaded before checking,
        # to get accurate error reporting for unknown backends
        _ensure_backends_loaded()
        dc = settings
        backend_name = (dc.backend or "opencv").lower()

        try:
            backend_cls = CameraFactory._resolve_backend(backend_name)
        except RuntimeError as exc:
            return False, f"Backend '{backend_name}' not installed: {exc}"

        if not backend_cls.is_available():
            return False, f"Backend '{backend_name}' is not available (missing drivers/packages)"

        # Prefer quick presence test
        if hasattr(backend_cls, "quick_ping"):
            try:
                with _suppress_opencv_logging():
                    idx = int(dc.index)
                    ok = False
                    try:
                        ok = backend_cls.quick_ping(idx)  # type: ignore[attr-defined]
                    except TypeError:
                        ok = backend_cls.quick_ping(idx, None)  # type: ignore[attr-defined]
                    if ok:
                        return True, ""
                    return False, "Device not present"
            except Exception as exc:
                return False, f"Quick probe failed: {exc}"

        # Fallback: lightweight open/close with sanitized settings
        try:
            probe_settings = backend_cls.sanitize_for_probe(dc)
            backend_instance = backend_cls(probe_settings)
            with _suppress_opencv_logging():
                backend_instance.open()
                backend_instance.close()
            return True, ""
        except Exception as exc:
            return False, f"Camera not accessible: {exc}"

    @staticmethod
    def backend_import_errors() -> dict[str, str]:
        _ensure_backends_loaded()
        return dict(_BACKEND_IMPORT_ERRORS)

    @staticmethod
    def _resolve_backend(name: str) -> type[CameraBackend]:
        _ensure_backends_loaded()
        key = name.lower()
        try:
            return _BACKEND_REGISTRY[key]
        except KeyError as exc:
            available = ", ".join(sorted(_BACKEND_REGISTRY.keys())) or "(none)"

            # Show import failures that might explain missing registration
            # (filter to your backend packages to avoid noise)
            failing = (
                "\n".join(f"  - {mod}: {_BACKEND_IMPORT_ERRORS[mod]}" for mod in sorted(_BACKEND_IMPORT_ERRORS.keys()))
                or "  (no import errors recorded)"
            )

            msg = (
                f"Backend '{key}' not registered. Available: {available}\n"
                f"Backend module import errors (most likely cause):\n{failing}\n"
                "Tip: enable strict import failures with DLC_CAMERA_BACKENDS_STRICT_IMPORT=1"
            )
            raise RuntimeError(msg) from exc
