# GUI Integration Guide

## Quick Answer

Here's how to use `scan_processor_folder` in your GUI:

```python
from example_gui_usage import scan_processor_folder, instantiate_from_scan

# 1. Scan folder
all_processors = scan_processor_folder("./processors")

# 2. Populate dropdown with keys (for backend) and display names (for user)
for key, info in all_processors.items():
    # key = "file.py::ClassName" (use this for instantiation)
    # display_name = "Human Name (file.py)" (show this to user)
    display_name = f"{info['name']} ({info['file']})"
    dropdown.add_item(key, display_name)

# 3. When user selects, get the key from dropdown
selected_key = dropdown.get_selected_value()  # e.g., "dlc_processor_socket.py::MyProcessor_socket"

# 4. Get processor info
processor_info = all_processors[selected_key]

# 5. Build parameter form from processor_info['params']
for param_name, param_info in processor_info['params'].items():
    add_input_field(param_name, param_info['type'], param_info['default'])

# 6. When user clicks Create, instantiate using the key
user_params = get_form_values()
processor = instantiate_from_scan(all_processors, selected_key, **user_params)
```

## The Key Insight

**The key returned by `scan_processor_folder` is what you use to instantiate!**

```python
# OLD problem: "I have a name, how do I load it?"
# NEW solution: Use the key directly

all_processors = scan_processor_folder(folder)
# Returns: {"file.py::ClassName": {processor_info}, ...}

# The KEY "file.py::ClassName" uniquely identifies the processor
# Pass this key to instantiate_from_scan()

processor = instantiate_from_scan(all_processors, "file.py::ClassName", **params)
```

## What's in the returned dict?

```python
all_processors = {
    "dlc_processor_socket.py::MyProcessor_socket": {
        "class": <class 'MyProcessor_socket'>,  # The actual class
        "name": "Mouse Pose Processor",         # Human-readable name
        "description": "Calculates mouse...",   # Description
        "params": {                             # All parameters
            "bind": {
                "type": "tuple",
                "default": ("0.0.0.0", 6000),
                "description": "Server address"
            },
            # ... more parameters
        },
        "file": "dlc_processor_socket.py",     # Source file
        "class_name": "MyProcessor_socket",    # Class name
        "file_path": "/full/path/to/file.py"  # Full path
    }
}
```

## GUI Workflow

### Step 1: Scan Folder
```python
all_processors = scan_processor_folder("./processors")
```

### Step 2: Populate Dropdown
```python
# Store keys in order (for mapping dropdown index -> key)
self.processor_keys = list(all_processors.keys())

# Create display names for dropdown
display_names = [
    f"{info['name']} ({info['file']})"
    for info in all_processors.values()
]
dropdown.set_items(display_names)
```

### Step 3: User Selects Processor
```python
def on_processor_selected(dropdown_index):
    # Get the key
    key = self.processor_keys[dropdown_index]

    # Get processor info
    info = all_processors[key]

    # Show description
    description_label.text = info['description']

    # Build parameter form
    for param_name, param_info in info['params'].items():
        add_parameter_field(
            name=param_name,
            type=param_info['type'],
            default=param_info['default'],
            help_text=param_info['description']
        )
```

### Step 4: User Clicks Create
```python
def on_create_clicked():
    # Get selected key
    key = self.processor_keys[dropdown.current_index]

    # Get user's parameter values
    user_params = parameter_form.get_values()

    # Instantiate using the key!
    self.processor = instantiate_from_scan(
        all_processors,
        key,
        **user_params
    )

    print(f"Created: {self.processor.__class__.__name__}")
```

## Why This Works

1. **Unique Keys**: `"file.py::ClassName"` format ensures uniqueness even if multiple files have same class name

2. **All Info Included**: Each dict entry has everything needed (class, metadata, parameters)

3. **Simple Lookup**: Just use the key to get processor info or instantiate

4. **No Manual Imports**: `scan_processor_folder` handles all module loading

5. **Type Safety**: Parameter metadata includes types for validation

## Complete Example

See `processor_gui.py` for a full working tkinter GUI that demonstrates:
- Folder scanning
- Processor selection
- Parameter form generation
- Instantiation

Run it with:
```bash
python processor_gui.py
```

## Files

- `dlc_processor_socket.py` - Processors with metadata and registry
- `example_gui_usage.py` - Scanning and instantiation functions + examples
- `processor_gui.py` - Full tkinter GUI
- `GUI_USAGE_GUIDE.py` - Pseudocode and examples
- `README.md` - Documentation on the plugin system
