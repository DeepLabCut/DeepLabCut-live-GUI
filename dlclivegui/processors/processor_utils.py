from __future__ import annotations

import importlib.util
import inspect
import logging
import pkgutil
import sys
from importlib import import_module
from importlib.resources import as_file, files
from pathlib import Path

logger = logging.getLogger(__name__)


def default_processors_dir() -> str:
    with as_file(files("dlclivegui").joinpath("processors")) as path:
        return str(path)


def _processor_base_class():
    from dlclive.processor import Processor

    return Processor


def _is_processor_subclass(
    obj,
    *,
    include_base: bool = False,
) -> bool:
    """Return whether obj is a selectable Processor subclass."""
    if not inspect.isclass(obj):
        return False

    try:
        processor_base = _processor_base_class()
    except Exception:
        logger.exception("Could not import dlclive.Processor")
        return False

    try:
        if obj is processor_base:
            return bool(include_base)

        if not issubclass(obj, processor_base):
            return False

        # Check only the class itself, not inherited values. This lets concrete
        # subclasses of a non-discoverable base remain discoverable by default.
        # getattr would return the inherited value.
        if obj.__dict__.get("PROCESSOR_DISCOVERABLE", True) is False:
            return False

        return True
    except Exception:
        logger.exception(
            "Error checking whether %r is a Processor subclass",
            obj,
        )
        return False


def _add_processor_results(
    target: dict[str, dict],
    processors: dict[str, dict],
    *,
    file_name: str,
    file_path: str,
) -> None:
    """Normalize discovered processors and add them to a scan result."""
    for class_name, processor_info in processors.items():
        key = f"{file_name}::{class_name}"
        info = dict(processor_info)
        info.update(
            {
                "file": file_name,
                "class_name": class_name,
                "file_path": file_path,
            }
        )
        target[key] = info


def _processor_info_from_class(cls, fallback_name: str) -> dict:
    return {
        "class": cls,
        "name": getattr(cls, "PROCESSOR_NAME", fallback_name),
        "description": getattr(cls, "PROCESSOR_DESCRIPTION", ""),
        "params": getattr(cls, "PROCESSOR_PARAMS", {}),
    }


def discover_processor_classes(module, *, only_defined_in_module: bool = True) -> dict[str, dict]:
    """Discover dlclive.Processor subclasses in a module.

    Includes indirect subclasses of Processor.

    Args:
        module: Imported Python module.
        only_defined_in_module: If True, ignore Processor subclasses imported
            from other modules to avoid duplicate registry entries.
    """
    processors: dict[str, dict] = {}

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if only_defined_in_module and getattr(obj, "__module__", None) != module.__name__:
            continue

        if not _is_processor_subclass(obj):
            continue

        processors[name] = _processor_info_from_class(obj, name)

    return processors


def scan_processor_folder(folder_path: str | Path) -> dict[str, dict]:
    all_processors = {}
    folder = Path(folder_path)

    for py_file in folder.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            processors = load_processors_from_file(py_file)
            _add_processor_results(
                all_processors,
                processors,
                file_name=py_file.name,
                file_path=str(py_file),
            )
        except Exception:
            logger.exception(f"Error loading {py_file}")

    return all_processors


def scan_processor_package(package_name: str = "dlclivegui.processors") -> dict[str, dict]:
    """
    Discover and load processor classes from a package namespace.
    """
    all_processors: dict[str, dict] = {}

    try:
        pkg = import_module(package_name)
    except Exception:
        logger.exception(f"Could not import package '{package_name}'")
        return all_processors

    # Iterate submodules under dlclivegui.processors
    for _, mod_name, ispkg in pkgutil.iter_modules(pkg.__path__, prefix=package_name + "."):
        if ispkg:
            continue
        try:
            mod = import_module(mod_name)
            processors = discover_processor_classes(mod)
            _add_processor_results(
                all_processors,
                processors,
                file_name=mod_name.split(".")[-1] + ".py",
                file_path=getattr(mod, "__file__", ""),
            )

        except Exception:
            logger.exception(f"Error importing processor module '{mod_name}'")

    return all_processors


def load_processors_from_file(file_path: str | Path) -> dict[str, dict]:
    """
    Load all processor classes from a Python file.

    Returns:
        dict[str, dict]: { "ClassOrId": {...info...}, ... }
    """
    file_path = str(file_path)
    stem = Path(file_path).stem

    # Use a unique module name per file to avoid collisions
    module_name = f"dlclivegui_plugins.{stem}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {file_path}")

        # Ensure a clean slate for refreshes
        sys.modules.pop(module_name, None)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Make visible during import for intra-module imports
        spec.loader.exec_module(module)

        # Fallback path: discover subclasses of dlclive.Processor
        return discover_processor_classes(module, only_defined_in_module=False)

    except Exception:
        # Full traceback helps a ton when a plugin fails to import
        logger.exception(f"Error loading processors from {file_path}")
        return {}


def instantiate_from_scan(processors_dict: dict[str, dict], processor_key: str, **kwargs):
    """
    Instantiate a processor from scan_processor_folder results.

    Args:
        processors_dict: Dict returned by scan_processor_folder
        processor_key: Key like "file.py::ClassName"
        **kwargs: Parameters for processor constructor

    Returns:
        Processor instance

    Example:
        processors = scan_processor_folder("./dlc_processors")
        processor = instantiate_from_scan(
            processors,
            "dlc_processor_socket.py::MyProcessor_socket",
            use_filter=True
        )
    """
    if processor_key not in processors_dict:
        available = ", ".join(processors_dict.keys())
        raise ValueError(f"Unknown processor '{processor_key}'. Available: {available}")

    processor_info = processors_dict[processor_key]
    processor_class = processor_info["class"]
    return processor_class(**kwargs)


def display_processor_info(processors):
    """Display processor information in a user-friendly format."""
    print("\n" + "=" * 70)
    print("AVAILABLE PROCESSORS")
    print("=" * 70)

    for idx, (class_name, info) in enumerate(processors.items(), 1):
        print(f"\n[{idx}] {info['name']}")
        print(f"    Class: {class_name}")
        print(f"    Description: {info['description']}")
        print("    Parameters:")
        for param_name, param_info in info["params"].items():
            print(f"      - {param_name} ({param_info['type']})")
            print(f"        Default: {param_info['default']}")
            print(f"        {param_info['description']}")
