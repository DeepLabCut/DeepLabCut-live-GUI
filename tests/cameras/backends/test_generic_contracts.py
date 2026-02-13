"""
Backend-agnostic contract tests for camera backends.

Hard failures:
- backend registry / factory / discovery calls must not crash
- capabilities must be well-formed
- if backend is available, create() must return a usable backend object
- close() must be idempotent

Soft signals (warnings):
- missing quick_ping / discover_devices / rebind_settings (helps future dev work)
- capability claims that do not match provided methods
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from dlclivegui.cameras.factory import CameraFactory, DetectedCamera
from dlclivegui.config import CameraSettings


def _try_import_gui_apply_identity():
    try:
        from dlclivegui.gui.camera_config.camera_config_dialog import _apply_detected_identity  # type: ignore

        return _apply_detected_identity
    except Exception:
        return None


def _minimal_settings(backend: str, index: int = 0, *, properties: dict[str, Any] | None = None) -> CameraSettings:
    return CameraSettings(
        name=f"ContractTest-{backend}",
        backend=backend,
        index=index,
        properties=properties or {},
        enabled=True,
        width=0,
        height=0,
        fps=0.0,
        exposure=0,
        gain=0.0,
        rotation=0,
        crop_x0=0,
        crop_y0=0,
        crop_x1=0,
        crop_y1=0,
    )


@pytest.fixture(scope="module")
def all_registered_backends() -> list[str]:
    return list(CameraFactory.backend_names())


@pytest.mark.unit
def test_all_registered_backends_have_well_formed_capabilities(all_registered_backends):
    for name in all_registered_backends:
        caps = CameraFactory.backend_capabilities(name)
        assert isinstance(caps, dict), f"{name}: capabilities must be a dict"
        assert all(isinstance(k, str) for k in caps.keys()), f"{name}: capability keys must be str"
        assert all(hasattr(v, "value") for v in caps.values()), f"{name}: capability values must be enum-like"


@pytest.mark.unit
def test_available_backends_map_is_well_formed():
    availability = CameraFactory.available_backends()
    assert isinstance(availability, dict)
    assert all(isinstance(k, str) for k in availability.keys())
    assert all(isinstance(v, bool) for v in availability.values())


@pytest.mark.unit
def test_detect_cameras_is_safe_for_all_backends(all_registered_backends):
    """
    Must never crash; it should return [] for unavailable SDKs.
    """
    for name in all_registered_backends:
        cams = CameraFactory.detect_cameras(name, max_devices=2)
        assert isinstance(cams, list), f"{name}: detect_cameras must return a list"
        for c in cams:
            assert hasattr(c, "index") and hasattr(c, "label"), f"{name}: detected items must have index/label"
            assert isinstance(c.index, int)
            assert isinstance(c.label, str)


@pytest.mark.unit
def test_optional_accelerators_warn_if_missing(all_registered_backends):
    """
    Non-failing warnings to encourage implementers to add helpful fast paths.
    """
    for name in all_registered_backends:
        # Resolve backend class indirectly by trying to create minimal settings only if available.
        # We can still warn based on capabilities + expected methods.
        caps = CameraFactory.backend_capabilities(name)

        # Determine if stable identity / discovery are claimed
        stable_claim = getattr(caps.get("stable_identity", None), "value", None)
        disco_claim = getattr(caps.get("device_discovery", None), "value", None)

        # Check method presence on backend class (best-effort)
        try:
            # This is internal-ish, but it’s the most direct way to inspect class methods.
            backend_cls = CameraFactory._resolve_backend(name)  # type: ignore[attr-defined]
        except Exception:
            # If backend can't resolve, that's a real issue, but it will be caught elsewhere.
            continue

        missing = []
        if not hasattr(backend_cls, "quick_ping"):
            missing.append("quick_ping")
        if not hasattr(backend_cls, "discover_devices"):
            missing.append("discover_devices")
        if not hasattr(backend_cls, "rebind_settings"):
            missing.append("rebind_settings")

        # Soft warnings: missing accelerators
        if missing:
            warnings.warn(
                f"[backend-contract] {name}: missing optional accelerators: {', '.join(missing)}",
                UserWarning,
                stacklevel=2,
            )

        # Soft warnings: claimed capability but method missing
        if stable_claim in ("supported", "best_effort") and not hasattr(backend_cls, "rebind_settings"):
            warnings.warn(
                f"[backend-contract] {name}: capabilities claim stable_identity={stable_claim} "
                f"but rebind_settings() is missing",
                UserWarning,
                stacklevel=2,
            )
        if disco_claim in ("supported", "best_effort") and not hasattr(backend_cls, "discover_devices"):
            warnings.warn(
                f"[backend-contract] {name}: capabilities claim device_discovery={disco_claim} "
                f"but discover_devices() is missing",
                UserWarning,
                stacklevel=2,
            )


@pytest.mark.unit
def test_create_contract_for_available_backends(all_registered_backends, backend_sdk_patchers):
    """
    If a backend is available (possibly after applying a fake SDK patch), create() should work.
    If not available, create() should fail cleanly (factory raises).
    """
    CameraFactory.available_backends()

    for name in all_registered_backends:
        # Apply optional SDK stub/patch if provided (keeps this test file backend-agnostic)
        patcher = backend_sdk_patchers.get(name)
        if patcher:
            patcher()

        availability = CameraFactory.available_backends()
        is_avail = availability.get(name, False)

        settings = _minimal_settings(name, index=0)

        if not is_avail:
            # Must fail cleanly if unavailable
            with pytest.raises(RuntimeError):
                CameraFactory.create(settings)
            continue

        # Available -> must create successfully
        be = CameraFactory.create(settings)

        # Minimal base contract
        assert hasattr(be, "open")
        assert hasattr(be, "read")
        assert hasattr(be, "close")
        assert callable(be.device_name)

        # close() should be idempotent even if never opened
        be.close()
        be.close()


@pytest.mark.unit
def test_gui_identity_helper_is_backend_agnostic(all_registered_backends):
    """
    This checks the GUI helper itself is backend-agnostic, not that each backend populates identity.
    We only validate that applying a DetectedCamera results in namespaced properties.
    """
    apply_identity = _try_import_gui_apply_identity()
    if apply_identity is None:
        pytest.skip("GUI helpers not importable (PySide6 likely missing in test env).")

    for backend in all_registered_backends:
        cam = _minimal_settings(backend, index=0, properties={})
        detected = DetectedCamera(
            index=0,
            label=f"Label-{backend}",
            device_id=f"device-{backend}",
            vid=0x1234,
            pid=0x5678,
            path=f"path-{backend}",
            backend_hint=None,
        )

        apply_identity(cam, detected, backend)

        assert isinstance(cam.properties, dict)
        assert backend in cam.properties
        ns = cam.properties[backend]
        assert isinstance(ns, dict)
        assert ns.get("device_id") == detected.device_id
        assert ns.get("device_name") == detected.label
