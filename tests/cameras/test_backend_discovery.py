# tests/cameras/test_backend_discovery.py
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from dlclivegui.cameras import factory as cam_factory
from dlclivegui.cameras.base import _BACKEND_REGISTRY, reset_backends
from dlclivegui.config import CameraSettings


def _write_temp_backend_package(tmp_path: Path, pkg_name: str = "test_backends_pkg") -> str:
    """
    Create a temporary backend package with a single backend module that registers
    itself using the @register_backend decorator.

    Returns the *package name* to be used in CameraFactory's discovery list.
    """
    pkg_root = tmp_path / pkg_name
    pkg_root.mkdir(parents=True, exist_ok=True)
    (pkg_root / "__init__.py").write_text("# test backends package\n", encoding="utf-8")

    # A backend module which registers itself as "lazyfake"
    backend_code = textwrap.dedent(
        """
        from dlclivegui.cameras.base import register_backend, CameraBackend
        from dlclivegui.config import CameraSettings
        import numpy as np
        import time

        @register_backend("lazyfake")
        class LazyFakeBackend(CameraBackend):
            @classmethod
            def is_available(cls) -> bool:
                return True

            def open(self) -> None:
                # No-op open for testing
                self._opened = True

            def read(self):
                # Small deterministic frame + timestamp
                frame = np.zeros((2, 3, 3), dtype=np.uint8)
                return frame, time.time()

            def close(self) -> None:
                self._opened = False

            # Optional: friendly name for detect_cameras label
            def device_name(self) -> str:
                return self.settings.name or f"LazyFake #{self.settings.index}"
        """
    )
    (pkg_root / "fake_backend.py").write_text(backend_code, encoding="utf-8")
    return pkg_name


@pytest.fixture
def temp_backends_pkg(tmp_path, monkeypatch):
    """
    Fixture that creates a temporary backend package and configures CameraFactory
    to import from it during lazy discovery. Resets the global registry/import flags.
    """
    # 1) Create on-disk package with a single backend
    pkg_name = _write_temp_backend_package(tmp_path)

    # 2) Ensure Python can import it
    sys.path.insert(0, str(tmp_path))
    try:
        # 3) Reset registry & lazy-import flags
        reset_backends()
        monkeypatch.setattr(cam_factory, "_BACKENDS_IMPORTED", False, raising=False)
        monkeypatch.setattr(cam_factory, "_BUILTIN_BACKEND_PACKAGES", (pkg_name,), raising=False)

        sys.modules.pop(pkg_name, None)
        sys.modules.pop(f"{pkg_name}.fake_backend", None)

        yield pkg_name
    finally:
        # Cleanup sys.path
        try:
            sys.path.remove(str(tmp_path))
        except ValueError:
            pass
        reset_backends()


def test_backend_lazy_discovery_from_package(temp_backends_pkg):
    """
    Verify that calling CameraFactory.backend_names() triggers lazy import and
    registers the backend found in the temporary package.
    """
    # Initially empty
    assert len(_BACKEND_REGISTRY) == 0

    names = set(cam_factory.CameraFactory.backend_names())
    assert "lazyfake" in names, f"Expected 'lazyfake' in discovered backends, got {names}"
    # Registry should now contain our backend
    assert "lazyfake" in _BACKEND_REGISTRY


def test_detect_and_create_with_discovered_backend(temp_backends_pkg):
    """
    Verify CameraFactory.detect_cameras() and CameraFactory.create() work
    with the lazily-discovered backend.
    """
    # Trigger discovery
    names = set(cam_factory.CameraFactory.backend_names())
    assert "lazyfake" in names

    # detect_cameras should instantiate/open/close without error and yield a label
    detected = cam_factory.CameraFactory.detect_cameras("lazyfake", max_devices=1)
    assert isinstance(detected, list)
    assert len(detected) >= 1
    # Our backend returns device_name() -> "Probe 0" (from factory) or our override in device_name
    assert detected[0].index == 0
    assert isinstance(detected[0].label, str)
    assert len(detected[0].label) > 0

    # create() should return an instance of our registered backend using a model-only settings
    s = CameraSettings(name="UnitCam", backend="lazyfake", index=0, fps=30.0)
    backend = cam_factory.CameraFactory.create(s)
    # A minimal behavior check: open/read/close work
    backend.open()
    frame, ts = backend.read()
    backend.close()

    assert frame is not None and getattr(frame, "shape", None) is not None
    assert frame.shape == (2, 3, 3)
    assert isinstance(ts, float)
