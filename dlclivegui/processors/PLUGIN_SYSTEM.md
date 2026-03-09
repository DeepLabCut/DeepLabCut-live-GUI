# DeepLabCut Live GUI — Processor Plugin System

This repository includes a **plugin-style processor system** that lets the GUI discover and instantiate **DLCLive processors** dynamically.

Processors are Python classes (typically subclasses of `dlclive.Processor`) that can optionally:

- receive pose estimates during inference (via `process(pose, **kwargs)`),
- broadcast pose-derived data to external clients (e.g., for experiment control),
- expose metadata so the GUI can list them and (optionally) build simple parameter UIs.

> **Security / control note:** The GUI should treat processors as **optional, user-controlled extensions**. In our current design, the GUI exposes an opt-in toggle (recommended label: **“Allow processor control”**) that gates whether processor plugins are instantiated and whether the GUI reads/acts on processor state.

---

## Overview

### Useful files

- `dlclivegui/processors/dlc_processor_socket.py` — Example socket-based processor base class + examples
- `dlclivegui/processors/processor_utils.py` — Scanning + instantiation helpers used by the GUI

---

## Architecture

### 1) Processor registry (module-level)

A typical processor module defines a registry and a decorator. The decorator registers classes into `PROCESSOR_REGISTRY` using either `PROCESSOR_ID` (if present) or the class name.

```python
# Registry for GUI discovery
PROCESSOR_REGISTRY = {}

def register_processor(cls):
    registry_key = getattr(cls, "PROCESSOR_ID", cls.__name__)
    PROCESSOR_REGISTRY[registry_key] = cls
    return cls
```

Register processors by decorating the class:

```python
@register_processor
class ExampleProcessor(BaseProcessorSocket):
    PROCESSOR_NAME = "Example Processor"
    PROCESSOR_DESCRIPTION = "Example description"
    PROCESSOR_PARAMS = {}
```

### 2) Processor metadata

Each processor class should define metadata attributes to help GUI discovery:

```python
class MyProcessorSocket(BaseProcessorSocket):
    PROCESSOR_NAME = "Mouse Pose Processor"  # Human-readable
    PROCESSOR_DESCRIPTION = "Broadcasts processed pose values"
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("127.0.0.1", 6000),
            "description": "Server address (host, port)",
        },
        "authkey": {
            "type": "bytes",
            "default": b"secret password",
            "description": "Authentication key for clients",
        },
        "use_filter": {
            "type": "bool",
            "default": False,
            "description": "Apply One-Euro filter",
        },
    }
```

> **Recommendation:** For security, prefer binding to `127.0.0.1` unless you explicitly want LAN exposure.

### 3) Module-level discovery helpers (optional)

Processor modules can expose:

- `get_available_processors()` — returns a dictionary of available processors and metadata

Example:

```python
def get_available_processors():
    return {
        name: {
            "class": cls,
            "name": getattr(cls, "PROCESSOR_NAME", name),
            "description": getattr(cls, "PROCESSOR_DESCRIPTION", ""),
            "params": getattr(cls, "PROCESSOR_PARAMS", {}),
        }
        for name, cls in PROCESSOR_REGISTRY.items()
    }
```

---

## Discovery & instantiation (current utilities)

The GUI uses utilities from `dlclivegui/processors/processor_utils.py`:

- `scan_processor_folder(folder_path)` — discover processors from `*.py` files in a folder
- `scan_processor_package(package_name="dlclivegui.processors")` — discover processors from a package namespace
- `instantiate_from_scan(processors_dict, processor_key, **kwargs)` — instantiate a processor from scan output

### Key format

Scan results are dictionaries keyed like:

```
"some_file.py::SomeProcessorClassOrId"
```

Each entry contains (at least):

- `class`: the processor class object
- `name`: display name
- `description`: description text
- `params`: parameter schema
- `file`: module filename
- `class_name`: class/registry key
- `file_path`: full path to the module file

### Example: scanning and instantiating

```python
from dlclivegui.processors.processor_utils import (
    scan_processor_package,
    scan_processor_folder,
    instantiate_from_scan,
)

# Built-in processors
processors = scan_processor_package("dlclivegui.processors")

# Or user folder processors
# processors = scan_processor_folder("/path/to/custom_processors")

# List
for key, info in processors.items():
    print(f"{info['name']} ({key}) — {info['description']}")

# Instantiate
selected_key = next(iter(processors.keys()))
proc = instantiate_from_scan(processors, selected_key, bind=("127.0.0.1", 6000))
```

---

## GUI integration & the “Allow processor control” gate

### Recommended behavior

To keep processor behavior explicit and opt-in, the GUI provides a toggle (**Allow processor-based control**) with these semantics:

- **Disabled (default):**
  - the GUI does **not instantiate** any processor plugin;
  - the GUI does **not read or act** on processor state (connections, recording flags, remote commands);
  - inference runs with `processor=None`.
  - *processor code may be imported by the discovery process*

- **Enabled:**
  - the GUI may instantiate the selected processor and (optionally) reflect processor state in the UI.
  - the processor will be used by the `DLCLive` instance during inference.

This lets users decide whether they want to run processor plugins and whether those plugins may influence UI/recording behavior.

> We recommend users to follow this design patter when designing their own processors
> to help ensure predictable behavior and clear user control over processor-based features.<br>
> **We are not responsible for any unexpected behavior caused by custom processors,**
> **and the examples are provided as-is with no guarantees.**

---

## Socket-based processors (example base class)

The built-in `BaseProcessorSocket` (in `dlc_processor_socket.py`) demonstrates a simple approach for:

- accepting multiple clients,
- receiving control messages (e.g., start/stop recording),
- broadcasting payloads to connected clients,
- cleaning up reliably on shutdown.

### Key points

- Socket server is optional: `BaseProcessorSocket` supports `start_server(...)`.
- Connections are tracked in `self.conns`.
- `broadcast(payload)` sends to all clients; failing clients are dropped.
- `stop()` closes clients and listener, joins threads, and attempts to wake `accept()` during shutdown.

> **Tip:** If you publish processors for others to use, keep module import side-effect free (define classes/functions only).

---

## Adding a new processor

1) Create a new module file in a processor folder (or inside `dlclivegui/processors/`).

2) Define a processor class and metadata:

```python
from dlclive import Processor

PROCESSOR_REGISTRY = {}

def register_processor(cls):
    PROCESSOR_REGISTRY[getattr(cls, "PROCESSOR_ID", cls.__name__)] = cls
    return cls

@register_processor
class MyNewProcessor(Processor):
    PROCESSOR_NAME = "My New Processor"
    PROCESSOR_DESCRIPTION = "Does something cool"
    PROCESSOR_PARAMS = {
        "my_param": {"type": "bool", "default": True, "description": "Enable cool feature"}
    }

    def process(self, pose, **kwargs):
        # Do something with pose
        return pose


def get_available_processors():
    return {
        name: {
            "class": cls,
            "name": getattr(cls, "PROCESSOR_NAME", name),
            "description": getattr(cls, "PROCESSOR_DESCRIPTION", ""),
            "params": getattr(cls, "PROCESSOR_PARAMS", {}),
        }
        for name, cls in PROCESSOR_REGISTRY.items()
    }
```

3) Refresh processors in the GUI, select your processor, and start inference (with processor control enabled if required).

---

## Parameter schema types

Supported `PROCESSOR_PARAMS` types:

- `"bool"` — checkbox
- `"int"` — integer input
- `"float"` — float input
- `"str"` — string input
- `"bytes"` — string that gets encoded to bytes
- `"tuple"` — tuple (e.g., `(host, port)`)
- `"dict"` — dictionary
- `"list"` — list

---

## Notes on external processors

External processors are arbitrary Python code. Only load processors you trust.



## License

This project is distributed under its project license.
See `LICENSE` in the repository.
