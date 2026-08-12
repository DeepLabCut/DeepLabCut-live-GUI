from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

# Legacy compatibility registry.
# GUI discovery no longer depends on this registry.
PROCESSOR_REGISTRY: dict[str, type] = {}


def register_processor(cls):
    """Register a processor for backward compatibility.

    New processor modules do not need this decorator. Processor discovery now
    finds eligible dlclive.Processor subclasses directly.
    """
    warnings.warn(
        "@register_processor is deprecated and no longer required for GUI "
        "discovery. Define a discoverable Processor subclass instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    registry_key = str(getattr(cls, "PROCESSOR_ID", cls.__name__))

    existing = PROCESSOR_REGISTRY.get(registry_key)
    if existing is not None and existing is not cls:
        logger.warning(
            "Duplicate legacy processor registration key %r: %s vs %s",
            registry_key,
            existing.__name__,
            cls.__name__,
        )

    PROCESSOR_REGISTRY[registry_key] = cls
    return cls


def get_available_processors() -> dict[str, dict]:
    """Return processors registered through the legacy decorator.

    Deprecated:
        GUI discovery now inspects Processor subclasses directly.
    """
    warnings.warn(
        "get_available_processors() is deprecated. Use "
        "discover_processor_classes(), scan_processor_package(), or "
        "scan_processor_folder() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return {
        name: {
            "class": cls,
            "name": getattr(cls, "PROCESSOR_NAME", name),
            "description": getattr(
                cls,
                "PROCESSOR_DESCRIPTION",
                "",
            ),
            "params": getattr(cls, "PROCESSOR_PARAMS", {}),
        }
        for name, cls in PROCESSOR_REGISTRY.items()
    }


def instantiate_processor(
    class_name: str,
    **kwargs,
):
    """Instantiate a processor from the legacy registry.

    Deprecated:
        Use instantiate_from_scan() with scanner output instead.
    """
    warnings.warn(
        "instantiate_processor() is deprecated. Use instantiate_from_scan() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if class_name not in PROCESSOR_REGISTRY:
        available = ", ".join(sorted(PROCESSOR_REGISTRY))
        raise ValueError(f"Unknown processor {class_name!r}. Available legacy registrations: {available}")

    return PROCESSOR_REGISTRY[class_name](**kwargs)
