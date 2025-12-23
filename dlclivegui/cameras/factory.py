"""Backend discovery and construction utilities."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Tuple, Type

from ..config import CameraSettings
from .base import CameraBackend


@contextmanager
def _suppress_opencv_logging() -> Generator[None, None, None]:
    """Temporarily suppress OpenCV logging during camera probing."""
    try:
        import cv2

        old_level = cv2.getLogLevel()
        cv2.setLogLevel(0)  # LOG_LEVEL_SILENT
        try:
            yield
        finally:
            cv2.setLogLevel(old_level)
    except ImportError:
        yield


@dataclass
class DetectedCamera:
    """Information about a camera discovered during probing."""

    index: int
    label: str


_BACKENDS: Dict[str, Tuple[str, str]] = {
    "opencv": ("dlclivegui.cameras.opencv_backend", "OpenCVCameraBackend"),
    "basler": ("dlclivegui.cameras.basler_backend", "BaslerCameraBackend"),
    "gentl": ("dlclivegui.cameras.gentl_backend", "GenTLCameraBackend"),
    "aravis": ("dlclivegui.cameras.aravis_backend", "AravisCameraBackend"),
}


class CameraFactory:
    """Create camera backend instances based on configuration."""

    @staticmethod
    def backend_names() -> Iterable[str]:
        """Return the identifiers of all known backends."""

        return tuple(_BACKENDS.keys())

    @staticmethod
    def available_backends() -> Dict[str, bool]:
        """Return a mapping of backend names to availability flags."""

        availability: Dict[str, bool] = {}
        for name in _BACKENDS:
            try:
                backend_cls = CameraFactory._resolve_backend(name)
            except RuntimeError:
                availability[name] = False
                continue
            availability[name] = backend_cls.is_available()
        return availability

    @staticmethod
    def detect_cameras(backend: str, max_devices: int = 10) -> List[DetectedCamera]:
        """Probe ``backend`` for available cameras.

        Parameters
        ----------
        backend:
            The backend identifier, e.g. ``"opencv"``.
        max_devices:
            Upper bound for the indices that should be probed.
            For backends with get_device_count (GenTL, Aravis), the actual device count is queried.

        Returns
        -------
        list of :class:`DetectedCamera`
            Sorted list of detected cameras with human readable labels.
        """

        try:
            backend_cls = CameraFactory._resolve_backend(backend)
        except RuntimeError:
            return []
        if not backend_cls.is_available():
            return []

        # For GenTL backend, try to get actual device count
        num_devices = max_devices
        if hasattr(backend_cls, "get_device_count"):
            try:
                actual_count = backend_cls.get_device_count()
                if actual_count >= 0:
                    num_devices = actual_count
            except Exception:
                pass

        detected: List[DetectedCamera] = []
        # Suppress OpenCV warnings/errors during probing (e.g., "can't open camera by index")
        with _suppress_opencv_logging():
            for index in range(num_devices):
                settings = CameraSettings(
                    name=f"Probe {index}",
                    index=index,
                    fps=30.0,
                    backend=backend,
                    properties={},
                )
                backend_instance = backend_cls(settings)
                try:
                    backend_instance.open()
                except Exception:
                    continue
                else:
                    label = backend_instance.device_name()
                    if not label:
                        label = f"{backend.title()} #{index}"
                    detected.append(DetectedCamera(index=index, label=label))
                finally:
                    try:
                        backend_instance.close()
                    except Exception:
                        pass
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
        """Check if a camera is available without keeping it open.

        Parameters
        ----------
        settings : CameraSettings
            The camera settings to check.

        Returns
        -------
        tuple[bool, str]
            A tuple of (is_available, error_message).
            If available, error_message is empty.
        """
        backend_name = (settings.backend or "opencv").lower()

        # Check if backend module is available
        try:
            backend_cls = CameraFactory._resolve_backend(backend_name)
        except RuntimeError as exc:
            return False, f"Backend '{backend_name}' not installed: {exc}"

        # Check if backend reports as available (drivers installed)
        if not backend_cls.is_available():
            return False, f"Backend '{backend_name}' is not available (missing drivers/packages)"

        # Try to actually open the camera briefly
        try:
            backend_instance = backend_cls(settings)
            backend_instance.open()
            backend_instance.close()
            return True, ""
        except Exception as exc:
            return False, f"Camera not accessible: {exc}"

    @staticmethod
    def _resolve_backend(name: str) -> Type[CameraBackend]:
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
