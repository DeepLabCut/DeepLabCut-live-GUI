# tests/cameras/backends/conftest.py
import importlib
import os

import pytest


# -----------------------------
# Dependency detection helpers
# -----------------------------
def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


ARAVIS_AVAILABLE = _has_module("gi")  # Aravis via GObject introspection
PYPYLON_AVAILABLE = _has_module("pypylon")  # Basler pypylon SDK


# -----------------------------
# Pytest configuration
# -----------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require hardware/SDKs (aravis/pypylon/gentl). "
        "By default these are skipped. You can also set BACKENDS_RUN_HARDWARE=1.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Document custom markers
    config.addinivalue_line("markers", "hardware: tests that touch real devices or SDKs")
    config.addinivalue_line("markers", "aravis: tests for Aravis backend")
    config.addinivalue_line("markers", "pypylon: tests for Basler/pypylon backend")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Auto-skip tests if the corresponding dependency is not present,
    and only run hardware-marked tests when explicitly requested.
    """
    run_hardware_flag = bool(config.getoption("--run-hardware"))
    run_hardware_env = os.getenv("BACKENDS_RUN_HARDWARE", "").strip() in {"1", "true", "yes"}
    run_hardware = run_hardware_flag or run_hardware_env

    skip_no_aravis = pytest.mark.skip(reason="Aravis/gi is not available")
    skip_no_pypylon = pytest.mark.skip(reason="Basler pypylon is not available")
    skip_hardware = pytest.mark.skip(
        reason="Hardware/SDK tests disabled. Use --run-hardware or set BACKENDS_RUN_HARDWARE=1"
    )

    for item in items:
        # Per-backend availability skips
        if "aravis" in item.keywords and not ARAVIS_AVAILABLE:
            item.add_marker(skip_no_aravis)
        if "pypylon" in item.keywords and not PYPYLON_AVAILABLE:
            item.add_marker(skip_no_pypylon)

        # Global hardware gate (only applies to tests marked 'hardware')
        if "hardware" in item.keywords and not run_hardware:
            item.add_marker(skip_hardware)


# -----------------------------
# Useful fixtures for backends
# -----------------------------
@pytest.fixture
def reset_backend_registry():
    """
    Ensure backend registry is clean for tests that rely on registration behavior.
    Automatically imports the package module that registers backends.
    """
    from dlclivegui.cameras.base import reset_backends

    reset_backends()
    try:
        # Import once so decorators run and register built-ins where possible.
        import dlclivegui.cameras.backends  # noqa: F401
    except Exception:
        # If import fails (optional deps), tests can still register backends directly.
        pass
    yield
    reset_backends()  # cleanup


@pytest.fixture
def force_aravis_unavailable(monkeypatch):
    """
    Force the Aravis backend to behave as if Aravis is not installed.
    Useful for testing error paths without modifying the environment.
    """
    import dlclivegui.cameras.backends.aravis_backend as ar

    # Simulate missing optional dependency
    monkeypatch.setattr(ar, "ARAVIS_AVAILABLE", False, raising=False)
    # Make sure the module symbol itself is treated as absent
    monkeypatch.setattr(ar, "Aravis", None, raising=False)
    yield


@pytest.fixture
def force_pypylon_unavailable(monkeypatch):
    """
    Force Basler/pypylon to be unavailable for error-path testing.
    """
    try:
        import dlclivegui.cameras.backends.basler_backend as bas
    except Exception:
        # If the module doesn't exist in your tree, ignore.
        yield
        return
    monkeypatch.setattr(bas, "PYPYLON_AVAILABLE", False, raising=False)
    monkeypatch.setattr(bas, "pylon", None, raising=False)
    yield
