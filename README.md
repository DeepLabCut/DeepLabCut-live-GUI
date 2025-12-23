# DeepLabCut Live GUI

A modern PySide6 GUI for running [DeepLabCut-live](https://github.com/DeepLabCut/DeepLabCut-live) experiments with real-time pose estimation. The application streams frames from industrial or consumer cameras, performs DLCLive inference, and records high-quality video with synchronized pose data.

## Features

### Core Functionality
- **Modern Python Stack**: Python 3.10+ compatible codebase with PySide6 interface
- **Multi-Backend Camera Support**: OpenCV, GenTL (Harvesters), Aravis, and Basler (pypylon)
- **Real-Time Pose Estimation**: Live DLCLive inference with configurable models (TensorFlow, PyTorch)
- **High-Performance Recording**: Hardware-accelerated video encoding via FFmpeg
- **Flexible Configuration**: Single JSON file for all settings with GUI editing

### Camera Features
- **Multiple Backends**:
  - OpenCV - Universal webcam support
  - GenTL - Industrial cameras via Harvesters (Windows/Linux)
  - Aravis - GenICam/GigE cameras (Linux/macOS)
  - Basler - Basler cameras via pypylon
- **Smart Device Detection**: Automatic camera enumeration without unnecessary probing
- **Camera Controls**: Exposure time, gain, frame rate, and ROI cropping
- **Live Preview**: Real-time camera feed with rotation support (0°, 90°, 180°, 270°)

### DLCLive Features
- **Model Support**: Only PyTorch models! (in theory also tensorflow models work)
- **Processor System**: Plugin architecture for custom pose processing
- **Auto-Recording**: Automatic video recording triggered by processor commands
- **Performance Metrics**: Real-time FPS, latency, and queue monitoring
- **Pose Visualization**: Optional overlay of detected keypoints on live feed

### Recording Features
- **Hardware Encoding**: NVENC (NVIDIA GPU) and software codecs (libx264, libx265)
- **Configurable Quality**: CRF-based quality control
- **Multiple Formats**: MP4, AVI, MOV containers
- **Timestamp Support**: Frame-accurate timestamps for synchronization
- **Performance Monitoring**: Write FPS, buffer status, and dropped frame tracking

### User Interface
- **Intuitive Layout**: Organized control panels with clear separation of concerns
- **Configuration Management**: Load/save settings, support for multiple configurations
- **Status Indicators**: Real-time feedback on camera, inference, and recording status
- **Bounding Box Tool**: Visual overlay for ROI definition

## Installation

### Basic Installation

```bash
pip install deeplabcut-live-gui
```

This installs the core package with OpenCV camera support.

### Full Installation with Optional Dependencies

```bash
# Install with gentl support
pip install deeplabcut-live-gui[gentl]
```

### Platform-Specific Camera Backend Setup

#### Windows (GenTL for Industrial Cameras)
1. Install camera vendor drivers and SDK
2. Ensure GenTL producer (.cti) files are accessible
3. Common locations:
   - `C:\Program Files\The Imaging Source Europe GmbH\IC4 GenTL Driver\bin\`
   - Check vendor documentation for CTI file location

#### Linux (Aravis for GenICam Cameras - Recommended)
NOT tested
```bash
# Ubuntu/Debian
sudo apt-get install gir1.2-aravis-0.8 python3-gi

# Fedora
sudo dnf install aravis python3-gobject
```

#### macOS (Aravis)
NOT tested
```bash
brew install aravis
pip install pygobject
```

#### Basler Cameras (All Platforms)
NOT tested
```bash
# Install Pylon SDK from Basler website
# Then install pypylon
pip install pypylon
```

### Hardware Acceleration (Optional)

For NVIDIA GPU encoding (highly recommended for high-resolution/high-FPS recording):
```bash
# Ensure NVIDIA drivers are installed
# FFmpeg with NVENC support will be used automatically
```

## Quick Start

1. **Launch the GUI**:
   ```bash
   dlclivegui
   ```

2. **Select Camera Backend**: Choose from the dropdown (opencv, gentl, aravis, basler)

3. **Configure Camera**: Set FPS, exposure, gain, and other parameters

4. **Start Preview**: Click "Start Preview" to begin camera streaming

5. **Optional - Load DLC Model**: Browse to your exported DLCLive model directory

6. **Optional - Start Inference**: Click "Start pose inference" for real-time tracking

7. **Optional - Record Video**: Configure output path and click "Start recording"

## Configuration

The GUI uses a single JSON configuration file containing all experiment settings:

```json
{
  "camera": {
    "name": "Camera 0",
    "index": 0,
    "fps": 60.0,
    "backend": "gentl",
    "exposure": 10000,
    "gain": 5.0,
    "crop_x0": 0,
    "crop_y0": 0,
    "crop_x1": 0,
    "crop_y1": 0,
    "max_devices": 3,
    "properties": {}
  },
  "dlc": {
    "model_path": "/path/to/exported-model",
    "model_type": "pytorch",
  },
  "recording": {
    "enabled": true,
    "directory": "~/Videos/deeplabcut-live",
    "filename": "session.mp4",
    "container": "mp4",
    "codec": "h264_nvenc",
    "crf": 23
  },
  "bbox": {
    "enabled": false,
    "x0": 0,
    "y0": 0,
    "x1": 200,
    "y1": 100
  }
}
```

### Configuration Management

- **Load**: File → Load configuration… (or Ctrl+O)
- **Save**: File → Save configuration (or Ctrl+S)
- **Save As**: File → Save configuration as… (or Ctrl+Shift+S)

All GUI fields are automatically synchronized with the configuration file.

## Camera Backends

### Backend Selection Guide

| Backend | Platform | Use Case | Auto-Detection |
|---------|----------|----------|----------------|
| **opencv** | All | Webcams, simple USB cameras | Basic |
| **gentl** | Windows, Linux | Industrial cameras via CTI files | Yes |
| **aravis** | Linux, macOS | GenICam/GigE cameras | Yes |
| **basler** | All | Basler cameras specifically | Yes |

### Backend-Specific Configuration

#### OpenCV
```json
{
  "camera": {
    "backend": "opencv",
    "index": 0,
    "fps": 30.0
  }
}
```
**Note**: Exposure and gain controls are disabled for OpenCV backend due to limited driver support.

#### GenTL (Harvesters)
```json
{
  "camera": {
    "backend": "gentl",
    "index": 0,
    "fps": 60.0,
    "exposure": 15000,
    "gain": 8.0,
  }
}
```


See [Camera Backend Documentation](docs/camera_support.md) for detailed setup instructions.

## DLCLive Integration

### Model Types

The GUI supports PyTorch DLCLive models:

1. **PyTorch**: PyTorch-based models (requires PyTorch installation)

Select the model type from the dropdown before starting inference.

### Processor System

The GUI includes a plugin system for custom pose processing:

```python
# Example processor
class MyProcessor:
    def process(self, pose, timestamp):
        # Custom processing logic
        x, y = pose[0, :2]  # First keypoint
        print(f"Position: ({x}, {y})")
    def save(self):
      pass
```

Place processors in `dlclivegui/processors/` and refresh to load them.

See [Processor Plugin Documentation](docs/PLUGIN_SYSTEM.md) for details.

### Auto-Recording Feature

Enable "Auto-record video on processor command" to automatically start/stop recording based on processor signals. Useful for event-triggered recording in behavioral experiments.

## Performance Optimization

### High-Speed Camera Tips

1. **Use Hardware Encoding**: Select `h264_nvenc` codec for NVIDIA GPUs
2. **Adjust Buffer Count**: Increase buffers for GenTL/Aravis backends
   ```json
   "properties": {"n_buffers": 20}
   ```
3. **Optimize CRF**: Lower CRF = higher quality but larger files (default: 23)
4. **Disable Visualization**: Uncheck "Display pose predictions" during recording
5. **Crop Region**: Use cropping to reduce frame size before inference

### Project Structure

```
dlclivegui/
├── __init__.py
├── gui.py                 # Main PySide6 application
├── config.py             # Configuration dataclasses
├── camera_controller.py  # Camera capture thread
├── dlc_processor.py      # DLCLive inference thread
├── video_recorder.py     # Video encoding thread
├── cameras/              # Camera backend modules
│   ├── base.py          # Abstract base class
│   ├── factory.py       # Backend registry and detection
│   ├── opencv_backend.py
│   ├── gentl_backend.py
│   ├── aravis_backend.py
│   └── basler_backend.py
└── processors/           # Pose processor plugins
    ├── processor_utils.py
    └── dlc_processor_socket.py
```


## Documentation

- [Camera Support](docs/camera_support.md) - All camera backends and setup
- [Aravis Backend](docs/aravis_backend.md) - GenICam camera setup (Linux/macOS)
- [Processor Plugins](docs/PLUGIN_SYSTEM.md) - Custom pose processing
- [Installation Guide](docs/install.md) - Detailed setup instructions
- [Timestamp Format](docs/timestamp_format.md) - Timestamp synchronization

## System Requirements


### Recommended
- Python 3.10+
- 8 GB RAM
- NVIDIA GPU with CUDA support (for DLCLive inference and video encoding)
- USB 3.0 or GigE network (for industrial cameras)
- SSD storage (for high-speed recording)

### Tested Platforms
- Windows 11

## License

This project is licensed under the GNU Lesser General Public License v3.0. See the [LICENSE](LICENSE) file for more information.

## Citation

Cite the original DeepLabCut-live paper:
```bibtex
@article{Kane2020,
  title={Real-time, low-latency closed-loop feedback using markerless posture tracking},
  author={Kane, Gary A and Lopes, Gonçalo and Saunders, Jonny L and Mathis, Alexander and Mathis, Mackenzie W},
  journal={eLife},
  year={2020},
  doi={10.7554/eLife.61909}
}
```
