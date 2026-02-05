# tests/cameras/backends/utils/test_backend_loader.py
from __future__ import annotations

from types import SimpleNamespace

import pytest

import dlclivegui.cameras.backends.utils.backend_loader as backend_loader


@pytest.fixture(autouse=True)
def _reset_import_errors(monkeypatch):
    """
    Ensure each test starts clean:
    - no previous import errors
    - strict env var unset
    """
    backend_loader._BACKEND_IMPORT_ERRORS.clear()
    monkeypatch.delenv("DLC_CAMERA_BACKENDS_STRICT_IMPORT", raising=False)
    yield
    backend_loader._BACKEND_IMPORT_ERRORS.clear()
    monkeypatch.delenv("DLC_CAMERA_BACKENDS_STRICT_IMPORT", raising=False)


def test_load_backend_modules_autodiscovers_and_imports(monkeypatch, caplog):
    """
    When modules=None, we:
    - import the package
    - iterate pkgutil.iter_modules(pkg.__path__, prefix=...)
    - import each discovered module
    """
    package = "dlclivegui.cameras.backends"

    imported = []

    # Fake package module returned by importlib.import_module(package)
    fake_pkg = SimpleNamespace(__name__=package, __path__=["/fake/path"])

    def fake_import_module(name: str):
        imported.append(name)
        if name == package:
            return fake_pkg
        # importing backend modules succeeds
        return SimpleNamespace(__name__=name)

    # pkgutil.iter_modules yields ModuleInfo-like objects with .name
    def fake_iter_modules(path, prefix=""):
        assert path == fake_pkg.__path__
        assert prefix == package + "."
        return [
            SimpleNamespace(name=prefix + "opencv"),
            SimpleNamespace(name=prefix + "gige"),
        ]

    monkeypatch.setattr(backend_loader.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(backend_loader.pkgutil, "iter_modules", fake_iter_modules)

    caplog.set_level("DEBUG")
    backend_loader.load_backend_modules(package)

    # Ensure we imported the package and both submodules
    assert imported == [
        package,
        package + ".opencv",
        package + ".gige",
    ]

    # Optional: ensure debug logging happened for module imports
    assert any("Loaded camera backend module" in rec.message for rec in caplog.records)


def test_load_backend_modules_with_explicit_modules_skips_discovery(monkeypatch):
    """
    When modules is provided, we should NOT import the package or call pkgutil.iter_modules.
    """
    imported = []

    def fake_import_module(name: str):
        imported.append(name)
        return SimpleNamespace(__name__=name)

    def fail_iter_modules(*args, **kwargs):
        raise AssertionError("pkgutil.iter_modules should not be called when modules is provided")

    monkeypatch.setattr(backend_loader.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(backend_loader.pkgutil, "iter_modules", fail_iter_modules)

    backend_loader.load_backend_modules(
        package="dlclivegui.cameras.backends",
        modules=["a.b.c", "x.y.z"],
    )

    assert imported == ["a.b.c", "x.y.z"]


def test_import_failure_is_logged_and_recorded(monkeypatch, caplog):
    """
    If a backend module fails to import:
    - it should not raise (default)
    - it should log exception
    - it should record the error in _BACKEND_IMPORT_ERRORS
    """
    package = "dlclivegui.cameras.backends"
    fake_pkg = SimpleNamespace(__name__=package, __path__=["/fake/path"])

    def fake_import_module(name: str):
        if name == package:
            return fake_pkg
        if name.endswith(".broken"):
            raise ImportError("boom")
        return SimpleNamespace(__name__=name)

    def fake_iter_modules(path, prefix=""):
        return [
            SimpleNamespace(name=prefix + "ok"),
            SimpleNamespace(name=prefix + "broken"),
        ]

    monkeypatch.setattr(backend_loader.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(backend_loader.pkgutil, "iter_modules", fake_iter_modules)

    caplog.set_level("DEBUG")
    backend_loader.load_backend_modules(package)

    # Should record only the broken import
    errors = backend_loader.backend_import_errors()
    assert package + ".broken" in errors
    assert "ImportError" in errors[package + ".broken"] or "boom" in errors[package + ".broken"]

    # Should have logged loudly
    assert any("FAILED to import backend module" in rec.message for rec in caplog.records)


def test_strict_import_mode_raises(monkeypatch):
    """
    If DLC_CAMERA_BACKENDS_STRICT_IMPORT is set, import failures should be raised.
    """
    monkeypatch.setenv("DLC_CAMERA_BACKENDS_STRICT_IMPORT", "1")

    package = "dlclivegui.cameras.backends"
    fake_pkg = SimpleNamespace(__name__=package, __path__=["/fake/path"])

    def fake_import_module(name: str):
        if name == package:
            return fake_pkg
        raise ImportError("hard fail")

    def fake_iter_modules(path, prefix=""):
        return [SimpleNamespace(name=prefix + "broken")]

    monkeypatch.setattr(backend_loader.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(backend_loader.pkgutil, "iter_modules", fake_iter_modules)

    with pytest.raises(ImportError, match="hard fail"):
        backend_loader.load_backend_modules(package)


def test_backend_import_errors_returns_copy(monkeypatch):
    """
    backend_import_errors() should return a copy so callers can't mutate internal state.
    """
    backend_loader._BACKEND_IMPORT_ERRORS["x"] = "y"
    snapshot = backend_loader.backend_import_errors()

    assert snapshot == {"x": "y"}
    snapshot["x"] = "changed"

    # internal should be unchanged
    assert backend_loader._BACKEND_IMPORT_ERRORS["x"] == "y"
