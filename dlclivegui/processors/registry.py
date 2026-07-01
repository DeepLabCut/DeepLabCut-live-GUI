import logging

logger = logging.getLogger(__name__)

# Registry for GUI discovery
PROCESSOR_REGISTRY = {}


def register_processor(cls):
    registry_key = getattr(cls, "PROCESSOR_ID", cls.__name__)
    if registry_key in PROCESSOR_REGISTRY:
        msg = (
            f"Duplicate processor registration key '{registry_key}': "
            f"{PROCESSOR_REGISTRY[registry_key].__name__} vs {cls.__name__}"
        )
        logger.warning(msg)
    PROCESSOR_REGISTRY[registry_key] = cls
    return cls


def get_available_processors():
    """
    Get list of available processor classes.

    Returns:
        dict: Dictionary mapping registry keys to processor info.
    """
    return {
        name: {
            "class": cls,
            "name": getattr(cls, "PROCESSOR_NAME", name),
            "description": getattr(cls, "PROCESSOR_DESCRIPTION", ""),
            "params": getattr(cls, "PROCESSOR_PARAMS", {}),
        }
        for name, cls in PROCESSOR_REGISTRY.items()
    }


def instantiate_processor(class_name, **kwargs):
    """
    Instantiate a processor by class name with given parameters.

    Args:
        class_name: Registry key (e.g., "MyProcessorSocket")
        **kwargs: Constructor kwargs

    Raises:
        ValueError: If class_name is not in registry
    """
    if class_name not in PROCESSOR_REGISTRY:
        available = ", ".join(PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown processor '{class_name}'. Available: {available}")
    return PROCESSOR_REGISTRY[class_name](**kwargs)
