# DeepLabCut Live GUI: Processor Plugin System

This repository includes a **plugin-style processor system** that lets the GUI discover and instantiate **DLCLive processors** dynamically.

Processors are Python classes that subclass `dlclive.processor.Processor`, directly or indirectly, and can optionally:

- Receive pose estimates during inference through `process(pose, **kwargs)`
- Broadcast pose-derived data, for example for experiment control
- Expose metadata so the GUI can list them and support processor configuration

> The GUI should treat processors as **optional, user-controlled extensions**.
> In our current design, the GUI exposes an opt-in toggle, **Allow processor-based control**, that controls whether processor plugins are instantiated and whether the GUI reads or acts on processor state.

## Overview

### Useful files

- `dlclivegui/processors/dlc_processor_socket.py`: Example socket-based processor base class
- `dlclivegui/processors/examples.py`: Example processor implementations, such as One-Euro filtering
- `dlclivegui/processors/processor_utils.py`: Scanning and instantiation helpers used by the GUI

## Architecture

### 1) Processor class discovery

A processor module defines one or more classes that subclass `dlclive.processor.Processor`, directly or indirectly.

The GUI discovers eligible processor classes by inspecting the imported module.

```python
from dlclive.processor import Processor


class ExampleProcessor(Processor):
    PROCESSOR_NAME = "Example Processor"
    PROCESSOR_DESCRIPTION = "Example description"
    PROCESSOR_PARAMS = {}

    def process(self, pose, **kwargs):
        return pose
```

Only processor classes defined in the scanned module are included. Processor classes imported from another module are ignored to avoid duplicate entries.

Reusable base classes that should not appear in the GUI can explicitly opt out:

```python
class BaseProcessorSocket(Processor):
    PROCESSOR_DISCOVERABLE = False
```

Concrete subclasses of a non-discoverable base class remain discoverable by default.

### 2) Processor metadata

Each processor class should define metadata attributes to help GUI discovery:

```python
class MyProcessorSocket(BaseProcessorSocket):
    PROCESSOR_NAME = "Use Pose Processor"  # Human-readable
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


## Discovery & instantiation

The GUI uses utilities from `dlclivegui/processors/processor_utils.py`:

- `discover_processor_classes(module)`: discover eligible processor classes in an imported module
- `scan_processor_folder(folder_path)`: discover processors from `*.py` files in a folder
- `scan_processor_package(package_name="dlclivegui.processors")`: discover processors from a package namespace
- `instantiate_from_scan(processors_dict, processor_key, **kwargs)`: instantiate a processor from scan output

Package and folder scanning use different module-loading mechanisms, but both use the same class-based processor discovery.

### Key format

Scan results are dictionaries keyed like:

```text
some_file.py::SomeProcessorClass
```

Each entry contains at least:

- `class`: the processor class object
- `name`: display name
- `description`: description text
- `params`: parameter schema
- `file`: module filename
- `class_name`: processor class name
- `file_path`: full path to the module file

### Example: scanning and instantiating

```python
from dlclivegui.processors.processor_utils import (
    instantiate_from_scan,
    scan_processor_folder,
    scan_processor_package,
)


# Built-in processors
processors = scan_processor_package("dlclivegui.processors")

# Or user folder processors
# processors = scan_processor_folder("/path/to/custom_processors")

# List
for key, info in processors.items():
    print(f"{info['name']} ({key}): {info['description']}")

# Instantiate
selected_key = next(iter(processors))
proc = instantiate_from_scan(
    processors,
    selected_key,
    bind=("127.0.0.1", 6000),
)
```

### Legacy registration compatibility

Earlier processor modules may still import and use:

```python
from dlclivegui.processors import PROCESSOR_REGISTRY, register_processor
```

The registry and decorator remain temporarily available for compatibility with existing processor modules. However:

- GUI discovery does not use `PROCESSOR_REGISTRY`
- GUI discovery does not call `get_available_processors()`
- Decorating a class is not required for discovery
- An existing decorated class remains discoverable because the decorator returns the original class

New processor modules should rely on subclass discovery instead of defining a registry or discovery function.

## GUI integration & enabling custom processors

### Recommended behavior

To keep processor behavior explicit and opt-in, the GUI provides an **Use custom processor** toggle with these effects:

- **Disabled by default:**
  - The GUI does **not instantiate** any processor plugin
  - The GUI does **not read or act** on processor state, such as connections, recording flags, or remote commands
  - Inference runs with `processor=None`
  - Processor code may still be imported by the discovery process

- **Enabled:**
  - The GUI may instantiate the selected processor and reflect processor state in the UI
  - The processor is used by the `DLCLive` instance during inference

This lets users decide whether they want to run processor plugins and whether those plugins may influence UI or recording behavior.

> We recommend that users follow this design pattern when creating processors to help ensure predictable behavior and clear user control over processor-based features.
> **We are not responsible for unexpected behavior caused by custom processors, and the examples are provided as-is with no guarantees.**

## Socket-based processors

The built-in `BaseProcessorSocket` in `dlc_processor_socket.py` demonstrates a simple approach for:

- Accepting multiple clients
- Receiving control messages, such as start and stop recording,
- Broadcasting payloads to connected clients,
- Cleaning up reliably on shutdown.

`BaseProcessorSocket` is a reusable base class and is not shown as a selectable processor in the GUI:

```python
PROCESSOR_DISCOVERABLE = False
```

Concrete subclasses defined in processor modules are discovered normally.

### Key points

- The socket server is optional: `BaseProcessorSocket` supports `start_server(...)`.
- Connections are tracked in `self.conns`.
- `broadcast(payload)` sends to all clients, and failing clients are dropped.
- `stop()` closes clients and the listener, joins threads, and attempts to wake `accept()` during shutdown.

> **Tip:** If you publish processors for others to use, keep module imports side-effect free where possible. Define classes and functions during import, and initialize sockets, hardware, or other resources when the processor is instantiated.

## Adding a new processor

1. Create a new module file in a processor folder or inside `dlclivegui/processors/`.

2. Define a processor class and metadata:
   ```python
    from dlclive.processor import Processor

    class MyNewProcessor(Processor):
        PROCESSOR_NAME = "My New Processor"
        PROCESSOR_DESCRIPTION = "Does something useful"
        PROCESSOR_PARAMS = {
            "my_param": {
                "type": "bool",
                "default": True,
                "description": "Enable optional behavior",
            }
        }

        def __init__(self, my_param: bool = True):
            super().__init__()
            self.my_param = my_param

        def process(self, pose, **kwargs):
            # Do something with pose
            return pose
    ```
    No registration decorator, module-level registry, or `get_available_processors()` function is required.

3. Refresh processors in the GUI, select your processor, and start inference with processor control enabled if required.

## Parameter schema types

Supported `PROCESSOR_PARAMS` types:

- `"bool"`: checkbox
- `"int"`: integer input
- `"float"`: float input
- `"str"`: string input
- `"bytes"`: string that gets encoded to bytes
- `"tuple"`: tuple, for example `(host, port)`
- `"dict"`: dictionary
- `"list"`: list

The processor constructor remains the base definition of accepted arguments and values.

## Notes on external processors

External processors are arbitrary Python code and are imported during discovery. Only load processors you trust.

Where possible, processor modules should avoid import-time side effects and initialize files, sockets, hardware, or other resources only when the processor is instantiated.
