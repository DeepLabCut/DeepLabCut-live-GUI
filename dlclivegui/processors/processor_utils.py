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


def scan_processor_folder(folder_path):
    all_processors = {}
    folder = Path(folder_path)

    for py_file in folder.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            processors = load_processors_from_file(py_file)
            for class_or_id, processor_info in processors.items():
                key = f"{py_file.name}::{class_or_id}"
                processor_info["file"] = py_file.name
                processor_info["class_name"] = class_or_id
                processor_info["file_path"] = str(py_file)
                all_processors[key] = processor_info
        except Exception:
            logger.exception(f"Error loading {py_file}")

    return all_processors


def scan_processor_package(package_name: str = "dlclivegui.processors") -> dict[str | dict]:
    """
    Discover and load processor classes from a package namespace.
    Returns a dict keyed as 'module.py::ClassName' with the same
    structure you use today.
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

            # Prefer module-level registry function if present
            if hasattr(mod, "get_available_processors"):
                processors = mod.get_available_processors()
            else:
                # Fallback: scan for dlclive.Processor subclasses
                from dlclive import Processor

                processors = {}
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name)
                    try:
                        if isinstance(obj, type) and obj is not Processor and issubclass(obj, Processor):
                            processors[attr_name] = {
                                "class": obj,
                                "name": getattr(obj, "PROCESSOR_NAME", attr_name),
                                "description": getattr(obj, "PROCESSOR_DESCRIPTION", ""),
                                "params": getattr(obj, "PROCESSOR_PARAMS", {}),
                            }
                    except Exception:
                        # Non-class or weird metaclass; ignore
                        pass

            # Normalize into your “file::class” shape
            module_file = mod.__name__.split(".")[-1] + ".py"
            for class_name, info in processors.items():
                key = f"{module_file}::{class_name}"
                info = dict(info)  # copy
                info["file"] = module_file
                info["class_name"] = class_name
                info["file_path"] = mod.__file__ or ""
                all_processors[key] = info

        except Exception:
            logger.exception(f"Error importing processor module '{mod_name}'")

    return all_processors


def load_processors_from_file(file_path: str | Path):
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

        # Preferred path: the module exposes get_available_processors()
        if hasattr(module, "get_available_processors"):
            processors = module.get_available_processors()
            if not isinstance(processors, dict):
                raise TypeError(f"{file_path}: get_available_processors() must return a dict, got {type(processors)}")
            return processors

        # Fallback path: discover subclasses of dlclive.Processor
        from dlclive import Processor

        processors: dict[str, dict] = {}
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Processor:
                continue
            # Guard: module might define other classes; only include Processor subclasses
            try:
                if issubclass(obj, Processor):
                    processors[name] = {
                        "class": obj,
                        "name": getattr(obj, "PROCESSOR_NAME", name),
                        "description": getattr(obj, "PROCESSOR_DESCRIPTION", ""),
                        "params": getattr(obj, "PROCESSOR_PARAMS", {}),
                    }
            except Exception:
                # Some "classes" can fail issubclass checks; ignore safely
                continue

        return processors

    except Exception:
        # Full traceback helps a ton when a plugin fails to import
        logger.exception(f"Error loading processors from {file_path}")
        return {}


def instantiate_from_scan(processors_dict, processor_key, **kwargs):
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
