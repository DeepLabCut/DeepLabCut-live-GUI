# DeepLabCut Live GUI

A modernised PyQt6 GUI for running [DeepLabCut-live](https://github.com/DeepLabCut/DeepLabCut-live) experiments. The application
streams frames from a camera, optionally performs DLCLive inference, and records video using the
[vidgear](https://github.com/abhiTronix/vidgear) toolkit.

## Features

- Python 3.11+ compatible codebase with a PyQt6 interface.
- Modular architecture with dedicated modules for camera control, video recording, configuration
  management, and DLCLive processing.
- Single JSON configuration file that captures camera settings, DLCLive parameters, and recording
  options. All fields can be edited directly within the GUI.
- Optional DLCLive inference with pose visualisation over the live video feed.
- Recording support via vidgear's `WriteGear`, including custom encoder options.

## Installation

1. Install the package and its dependencies:

   ```bash
   pip install deeplabcut-live-gui
   ```

   The GUI requires additional runtime packages for optional features:

   - [DeepLabCut-live](https://github.com/DeepLabCut/DeepLabCut-live) for pose estimation.
   - [vidgear](https://github.com/abhiTronix/vidgear) for video recording.
   - [OpenCV](https://opencv.org/) for camera access.

   These libraries are listed in `setup.py` and will be installed automatically when the package is
   installed via `pip`.

2. Launch the GUI:

   ```bash
   dlclivegui
   ```

## Configuration

The GUI works with a single JSON configuration describing the experiment. The configuration contains
three main sections:

```json
{
  "camera": {
    "index": 0,
    "width": 1280,
    "height": 720,
    "fps": 60.0,
    "backend": "opencv",
    "properties": {}
  },
  "dlc": {
    "model_path": "/path/to/exported-model",
    "processor": "cpu",
    "shuffle": 1,
    "trainingsetindex": 0,
    "processor_args": {},
    "additional_options": {}
  },
  "recording": {
    "enabled": true,
    "directory": "~/Videos/deeplabcut",
    "filename": "session.mp4",
    "container": "mp4",
    "options": {
      "compression_mode": "mp4"
    }
  }
}
```

Use **File → Load configuration…** to open an existing configuration, or **File → Save configuration**
to persist the current settings. Every field in the GUI is editable, and values entered in the
interface will be written back to the JSON file.

### Camera backends

Set `camera.backend` to one of the supported drivers:

- `opencv` – standard `cv2.VideoCapture` fallback available on every platform.
- `basler` – uses the Basler Pylon SDK via `pypylon` (install separately).
- `gentl` – uses Aravis for GenTL-compatible cameras (requires `python-gi` bindings).

Backend specific parameters can be supplied through the `camera.properties` object. For example:

```json
{
  "camera": {
    "index": 0,
    "backend": "basler",
    "properties": {
      "serial": "40123456",
      "exposure": 15000,
      "gain": 6.0
    }
  }
}
```

If optional dependencies are missing, the GUI will show the backend as unavailable in the drop-down
but you can still configure it for a system where the drivers are present.

## Development

The core modules of the package are organised as follows:

- `dlclivegui.config` – dataclasses for loading, storing, and saving application settings.
- `dlclivegui.cameras` – modular camera backends (OpenCV, Basler, GenTL) and factory helpers.
- `dlclivegui.camera_controller` – camera capture worker running in a dedicated `QThread`.
- `dlclivegui.video_recorder` – wrapper around `WriteGear` for video output.
- `dlclivegui.dlc_processor` – asynchronous DLCLive inference with optional pose overlay.
- `dlclivegui.gui` – PyQt6 user interface and application entry point.

Run a quick syntax check with:

```bash
python -m compileall dlclivegui
```

## License

This project is licensed under the GNU Lesser General Public License v3.0. See the `LICENSE` file for
more information.
