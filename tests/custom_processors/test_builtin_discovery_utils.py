# tests/custom_processors/test_builtin_discovery_utils.py
from __future__ import annotations

import importlib
import uuid
from pathlib import Path

import pytest

from dlclivegui.processors.processor_utils import (
    default_processors_dir,
    display_processor_info,
    instantiate_from_scan,
    load_processors_from_file,
    scan_processor_folder,
    scan_processor_package,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_temp_processor_file(tmp_path: Path, stem: str | None = None) -> Path:
    """
    Create a temporary processor module that exposes get_available_processors()
    so we don't depend on dlclive.Processor being importable.

    The dummy processor has safe __init__ and no side-effects.
    """
    stem = stem or f"tmp_proc_{uuid.uuid4().hex}"
    py_file = tmp_path / f"{stem}.py"

    py_file.write_text(
        # Use get_available_processors to bypass dlclive import in loader.
        """
class DummyProc:
    PROCESSOR_NAME = "Dummy Processor"
    PROCESSOR_DESCRIPTION = "A safe, dummy processor for tests"
    PROCESSOR_PARAMS = {
        "foo": {"type": "int", "default": 1, "description": "dummy param"}
    }

    def __init__(self, **kwargs):
        self.kwargs = kwargs

def get_available_processors():
    # Return the normalized mapping the loader expects
    return {
        "DummyProc": {
            "class": DummyProc,
            "name": DummyProc.PROCESSOR_NAME,
            "description": DummyProc.PROCESSOR_DESCRIPTION,
            "params": DummyProc.PROCESSOR_PARAMS,
        }
    }
"""
    )
    return py_file


def _assert_processor_info_shape(info: dict):
    """Common assertions for the normalized processor info dict."""
    assert "class" in info
    assert "name" in info
    assert "description" in info
    assert "params" in info
    # The scan functions additionally attach these:
    assert "file" in info
    assert "class_name" in info
    assert "file_path" in info


# ---------------------------------------------------------------------------
# Tests: default_processors_dir
# ---------------------------------------------------------------------------


def test_default_processors_dir_exists():
    proc_dir = Path(default_processors_dir())
    assert proc_dir.exists(), f"Default processors dir does not exist: {proc_dir}"
    assert proc_dir.is_dir(), f"Default processors path is not a dir: {proc_dir}"


# ---------------------------------------------------------------------------
# Tests: scan_processor_package (built-in package)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("dlclivegui.processors") is None,
    reason="dlclivegui.processors package not importable in this test environment",
)
def test_scan_processor_package_populates_and_has_valid_shape():
    data = scan_processor_package("dlclivegui.processors")
    assert isinstance(data, dict)
    assert len(data) > 0, "Expected at least one processor from the package"

    # Validate key format and info shape on the first item
    key, info = next(iter(data.items()))
    assert "::" in key, f"Key should look like 'module.py::ClassName', got: {key}"
    assert key.split("::")[0].endswith(".py"), f"Key should start with a .py module, got: {key}"

    _assert_processor_info_shape(info)


# ---------------------------------------------------------------------------
# Tests: load_processors_from_file and scan_processor_folder (custom temp files)
# ---------------------------------------------------------------------------


def test_load_processors_from_file_prefers_registry(tmp_path: Path):
    py_file = _write_temp_processor_file(tmp_path)
    result = load_processors_from_file(py_file)
    assert isinstance(result, dict)
    assert "DummyProc" in result
    info = result["DummyProc"]
    # For load_processors_from_file (registry path), the minimal fields are present:
    assert "class" in info and info["class"].__name__ == "DummyProc"
    assert info["name"] == "Dummy Processor"
    assert "params" in info and "foo" in info["params"]


def test_scan_processor_folder_discovers_files_and_normalizes_shape(tmp_path: Path):
    # One valid file + one ignored file starting with underscore
    _write_temp_processor_file(tmp_path, stem="visible_proc")
    ignored = tmp_path / "_ignore_me.py"
    ignored.write_text("IGNORED = True\n")

    data = scan_processor_folder(tmp_path)
    assert isinstance(data, dict)
    assert len(data) == 1, f"Expected only the visible file to be discovered, got {list(data.keys())}"
    key, info = next(iter(data.items()))
    assert key.startswith("visible_proc.py::"), f"Unexpected key name: {key}"
    _assert_processor_info_shape(info)


def test_instantiate_from_scan_returns_instance(tmp_path: Path):
    _write_temp_processor_file(tmp_path, stem="instantiable_proc")

    # Discover via folder scan so we get the normalized shape
    scanned = scan_processor_folder(tmp_path)
    assert len(scanned) == 1
    key = next(iter(scanned.keys()))
    instance = instantiate_from_scan(scanned, key, foo=123, bar="baz")

    # The dummy class stores kwargs on self.kwargs
    assert instance.__class__.__name__ == "DummyProc"
    assert instance.kwargs == {"foo": 123, "bar": "baz"}


def test_display_processor_info_prints(capsys, tmp_path: Path):
    # Build a minimal processors dict to print
    _write_temp_processor_file(tmp_path, stem="printable_proc")
    scanned = scan_processor_folder(tmp_path)
    # Re-map to the shape expected by display_processor_info (which takes a dict keyed by class name)
    # Here we transform the normalized dict back to {class_name: info}
    simplified = {info["class_name"]: info for info in scanned.values()}

    display_processor_info(simplified)
    captured = capsys.readouterr().out

    assert "AVAILABLE PROCESSORS" in captured
    # Ensure processor entry shows up
    assert "Dummy Processor" in captured
    assert "Parameters:" in captured
    assert "- foo (int)" in captured or "foo" in captured  # depends on your formatter
