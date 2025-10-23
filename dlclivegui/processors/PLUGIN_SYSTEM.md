# DLC Processor Plugin System

This folder contains a plugin-style architecture for DLC processors that allows GUI tools to discover and instantiate processors dynamically.

## Architecture

### 1. Processor Registry

Each processor file should define a `PROCESSOR_REGISTRY` dictionary and helper functions:

```python
# Registry for GUI discovery
PROCESSOR_REGISTRY = {}

# At end of file, register your processors
PROCESSOR_REGISTRY["MyProcessor_socket"] = MyProcessor_socket
```

### 2. Processor Metadata

Each processor class should define metadata attributes for GUI discovery:

```python
class MyProcessor_socket(BaseProcessor_socket):
    # Metadata for GUI discovery
    PROCESSOR_NAME = "Mouse Pose Processor"  # Human-readable name
    PROCESSOR_DESCRIPTION = "Calculates mouse center, heading, and head angle"
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("0.0.0.0", 6000),
            "description": "Server address (host, port)"
        },
        "use_filter": {
            "type": "bool",
            "default": False,
            "description": "Apply One-Euro filter"
        },
        # ... more parameters
    }
```

### 3. Discovery Functions

Two helper functions enable GUI discovery:

```python
def get_available_processors():
    """Returns dict of available processors with metadata."""
    
def instantiate_processor(class_name, **kwargs):
    """Instantiates a processor by name with given parameters."""
```

## GUI Integration

### Simple Usage

```python
from dlc_processor_socket import get_available_processors, instantiate_processor

# 1. Get available processors
processors = get_available_processors()

# 2. Display to user (e.g., in dropdown)
for class_name, info in processors.items():
    print(f"{info['name']} - {info['description']}")

# 3. User selects "MyProcessor_socket"
selected_class = "MyProcessor_socket"

# 4. Show parameter form based on info['params']
processor_info = processors[selected_class]
for param_name, param_info in processor_info['params'].items():
    # Create input widget for param_type and default value
    pass

# 5. Instantiate with user's values
processor = instantiate_processor(
    selected_class,
    bind=("127.0.0.1", 7000),
    use_filter=True
)
```

### Scanning Multiple Files

To scan a folder for processor files:

```python
import importlib.util
from pathlib import Path

def load_processors_from_file(file_path):
    """Load processors from a single file."""
    spec = importlib.util.spec_from_file_location("processors", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'get_available_processors'):
        return module.get_available_processors()
    return {}

# Scan folder
for py_file in Path("dlc_processors").glob("*.py"):
    processors = load_processors_from_file(py_file)
    # Display processors to user
```

## Examples

### 1. Command-line Example

```bash
python example_gui_usage.py
```

This demonstrates:
- Loading processors
- Displaying metadata
- Instantiating with default/custom parameters
- Simulated GUI workflow

### 2. tkinter GUI

```bash
python processor_gui_simple.py
```

This provides a full GUI with:
- Dropdown to select processor
- Auto-generated parameter form
- Create/Stop buttons
- Status display

## Adding New Processors

To make a new processor discoverable:

1. **Define metadata attributes:**
```python
class MyNewProcessor(BaseProcessor_socket):
    PROCESSOR_NAME = "My New Processor"
    PROCESSOR_DESCRIPTION = "Does something cool"
    PROCESSOR_PARAMS = {
        "my_param": {
            "type": "bool",
            "default": True,
            "description": "Enable cool feature"
        }
    }
```

2. **Register in PROCESSOR_REGISTRY:**
```python
PROCESSOR_REGISTRY["MyNewProcessor"] = MyNewProcessor
```

3. **Done!** GUI will automatically discover it.

## Parameter Types

Supported parameter types in `PROCESSOR_PARAMS`:

- `"bool"` - Boolean checkbox
- `"int"` - Integer input
- `"float"` - Float input
- `"str"` - String input
- `"bytes"` - String that gets encoded to bytes
- `"tuple"` - Tuple (e.g., `(host, port)`)
- `"dict"` - Dictionary (e.g., filter parameters)
- `"list"` - List

## Benefits

1. **No hardcoding** - GUI doesn't need to know about specific processors
2. **Easy extension** - Add new processors without modifying GUI code
3. **Self-documenting** - Parameters include descriptions
4. **Type-safe** - Parameter metadata includes type information
5. **Modular** - Each processor file can be independent

## File Structure

```
dlc_processors/
├── dlc_processor_socket.py      # Base + MyProcessor with registry
├── my_custom_processor.py       # Your custom processor (with registry)
├── example_gui_usage.py         # Command-line example
├── processor_gui_simple.py      # tkinter GUI example
└── PLUGIN_SYSTEM.md            # This file
```
