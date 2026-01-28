"""Backend discovery and construction utilities."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Generator, Iterable  # CHANGED
from contextlib import contextmanager
from dataclasses import dataclass

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


_BACKENDS: dict[str, tuple[str, str]] = {
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
        """Check if a camera is available without keeping it open."""
        backend_name = (settings.backend or "opencv").lower()

        try:
            backend_cls = CameraFactory._resolve_backend(backend_name)
        except RuntimeError as exc:
            return False, f"Backend '{backend_name}' not installed: {exc}"

        if not backend_cls.is_available():
            return False, f"Backend '{backend_name}' is not available (missing drivers/packages)"

        try:
            backend_instance = backend_cls(settings)
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
