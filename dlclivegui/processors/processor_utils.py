import importlib.util
import inspect
from pathlib import Path


def load_processors_from_file(file_path):
    """
    Load all processor classes from a Python file.

    Args:
        file_path: Path to Python file containing processors

    Returns:
        dict: Dictionary of available processors
    """
    # Load module from file
    spec = importlib.util.spec_from_file_location("processors", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check if module has get_available_processors function
    if hasattr(module, "get_available_processors"):
        return module.get_available_processors()

    # Fallback: scan for Processor subclasses
    from dlclive import Processor

    processors = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Processor) and obj != Processor:
            processors[name] = {
                "class": obj,
                "name": getattr(obj, "PROCESSOR_NAME", name),
                "description": getattr(obj, "PROCESSOR_DESCRIPTION", ""),
                "params": getattr(obj, "PROCESSOR_PARAMS", {}),
            }
    return processors


def scan_processor_folder(folder_path):
    """
    Scan a folder for all Python files with processor definitions.

    Args:
        folder_path: Path to folder containing processor files

    Returns:
        dict: Dictionary mapping unique processor keys to processor info:
            {
                "file_name.py::ClassName": {
                    "class": ProcessorClass,
                    "name": "Display Name",
                    "description": "...",
                    "params": {...},
                    "file": "file_name.py",
                    "class_name": "ClassName"
                }
            }
    """
    all_processors = {}
    folder = Path(folder_path)

    for py_file in folder.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            processors = load_processors_from_file(py_file)
            for class_name, processor_info in processors.items():
                # Create unique key: file::class
                key = f"{py_file.name}::{class_name}"
                # Add file and class name to info
                processor_info["file"] = py_file.name
                processor_info["class_name"] = class_name
                processor_info["file_path"] = str(py_file)
                all_processors[key] = processor_info
        except Exception as e:
            print(f"Error loading {py_file}: {e}")

    return all_processors


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
        print(f"    Parameters:")
        for param_name, param_info in info["params"].items():
            print(f"      - {param_name} ({param_info['type']})")
            print(f"        Default: {param_info['default']}")
            print(f"        {param_info['description']}")
