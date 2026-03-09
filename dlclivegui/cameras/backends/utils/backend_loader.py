"""A small utility to load camera backend modules dynamically.
Allows to check for import errors."""

# dlclivegui/cameras/backend_loader.py
from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from collections.abc import Iterable

LOG = logging.getLogger(__name__)

# Track import errors so you can show them if a backend is missing later
_BACKEND_IMPORT_ERRORS: dict[str, str] = {}


def backend_import_errors() -> dict[str, str]:
    """Expose import errors for diagnostics."""
    return dict(_BACKEND_IMPORT_ERRORS)


def load_backend_modules(package: str, modules: Iterable[str] | None = None) -> None:
    """
    Import backend modules so their @register_backend decorators execute.

    - package: e.g. "dlclivegui.cameras.backends"
    - modules: optional explicit module list (useful in tests)
    """
    if modules is None:
        # auto-discover modules inside the package
        pkg = importlib.import_module(package)
        prefix = pkg.__name__ + "."
        names = [m.name for m in pkgutil.iter_modules(pkg.__path__, prefix=prefix)]
    else:
        names = list(modules)

    for mod_name in names:
        try:
            importlib.import_module(mod_name)
            LOG.debug("Loaded camera backend module: %s", mod_name)
        except Exception as exc:
            # Loud + full traceback
            LOG.exception("FAILED to import backend module '%s': %s", mod_name, exc)
            _BACKEND_IMPORT_ERRORS[mod_name] = repr(exc)

            # Optional "fail hard" mode for CI / dev / strict deployments
            if os.environ.get("DLC_CAMERA_BACKENDS_STRICT_IMPORT", "").strip() in ("1", "true", "yes"):
                raise


if __name__ == "__main__":
    # For manual testing
    load_backend_modules("dlclivegui.cameras.backends")
