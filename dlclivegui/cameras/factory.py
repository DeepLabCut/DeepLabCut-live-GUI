"""Backend discovery and construction utilities."""
from __future__ import annotations

import importlib
from typing import Dict, Iterable, Tuple, Type

from ..config import CameraSettings
from .base import CameraBackend


_BACKENDS: Dict[str, Tuple[str, str]] = {
    "opencv": ("dlclivegui.cameras.opencv_backend", "OpenCVCameraBackend"),
    "basler": ("dlclivegui.cameras.basler_backend", "BaslerCameraBackend"),
    "gentl": ("dlclivegui.cameras.gentl_backend", "GenTLCameraBackend"),
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
