# DeepLabCut-live-GUI Features

## Table of Contents

- [Camera Control](#camera-control)
- [Real-Time Pose Estimation](#real-time-pose-estimation)
- [Video Recording](#video-recording)
- [Configuration Management](#configuration-management)
- [Processor System](#processor-system)
- [User Interface](#user-interface)
- [Performance Monitoring](#performance-monitoring)
- [Advanced Features](#advanced-features)

---

## Camera Control

### Multi-Backend Support

The GUI supports four different camera backends, each optimized for different use cases:

#### OpenCV Backend
- **Platform**: Windows, Linux, macOS
- **Best For**: Webcams, simple USB cameras
- **Installation**: Built-in with OpenCV
- **Limitations**: Limited exposure/gain control

#### GenTL Backend (Harvesters)
- **Platform**: Windows, Linux
- **Best For**: Industrial cameras with GenTL producers
- **Installation**: Requires vendor CTI files
- **Features**: Full camera control, smart device detection

#### Aravis Backend
- **Platform**: Linux (best), macOS
- **Best For**: GenICam/GigE Vision cameras
- **Installation**: System packages (`gir1.2-aravis-0.8`)
- **Features**: Excellent Linux support, native GigE

#### Basler Backend (pypylon)
- **Platform**: Windows, Linux, macOS
- **Best For**: Basler cameras specifically
- **Installation**: Pylon SDK + pypylon
- **Features**: Vendor-specific optimizations

### Camera Settings

#### Frame Rate Control
- Range: 1-240 FPS (hardware dependent)
- Real-time FPS monitoring
- Automatic camera validation

#### Exposure Control
- Auto mode (value = 0)
- Manual mode (microseconds)
- Range: 0-1,000,000 μs
- Real-time adjustment (backend dependent)

#### Gain Control
- Auto mode (value = 0.0)
- Manual mode (gain value)
- Range: 0.0-100.0
- Useful for low-light conditions

#### Region of Interest (ROI) Cropping
- Define crop region: (x0, y0, x1, y1)
- Applied before recording and inference
- Reduces processing load
- Maintains aspect ratio

#### Image Rotation
- 0°, 90°, 180°, 270° rotation
- Applied to all outputs
- Useful for mounted cameras

### Smart Camera Detection

The GUI intelligently detects available cameras:

1. **Backend-Specific**: Each backend reports available cameras
2. **No Blind Probing**: GenTL and Aravis query actual device count
3. **Fast Refresh**: Only check connected devices
4. **Detailed Labels**: Shows vendor, model, serial number

Example detection output:
```
[CameraDetection] Available cameras for backend 'gentl':
  ['0:DMK 37BUX287 (26320523)', '1:Basler acA1920 (40123456)']
```

---

## Real-Time Pose Estimation

### DLCLive Integration

#### Model Support
- **TensorFlow (Base)**: Original DeepLabCut models
- **PyTorch**: PyTorch-exported models
- Model selection via dropdown
- Automatic model validation

#### Inference Pipeline
1. **Frame Acquisition**: Camera thread → Queue
2. **Preprocessing**: Crop, resize (optional)
3. **Inference**: DLCLive model processing
4. **Pose Output**: (x, y) coordinates per keypoint
5. **Visualization**: Optional overlay on video

#### Performance Metrics
- **Inference FPS**: Actual processing rate
- **Latency**: Time from capture to pose output
  - Last latency (ms)
  - Average latency (ms)
- **Queue Status**: Frame buffer depth
- **Dropped Frames**: Count of skipped frames

### Pose Visualization

#### Overlay Options
- **Toggle**: "Display pose predictions" checkbox
- **Keypoint Markers**: Green circles at (x, y) positions
- **Real-Time Update**: Synchronized with video feed
- **No Performance Impact**: Rendering optimized

#### Bounding Box Visualization
- **Purpose**: Visual ROI definition
- **Configuration**: (x0, y0, x1, y1) coordinates
- **Color**: Red rectangle overlay
- **Use Cases**:
  - Crop region preview
  - Analysis area marking
  - Multi-region tracking

### Initialization Feedback

Visual indicators during model loading:
1. **"Initializing DLCLive!"** - Blue button during load
2. **"DLCLive running!"** - Green button when ready
3. Status bar updates with progress

---

## Video Recording

### Recording Capabilities

#### Hardware-Accelerated Encoding
- **NVENC (NVIDIA)**: GPU-accelerated H.264/H.265
  - Codecs: `h264_nvenc`, `hevc_nvenc`
  - 10x faster than software encoding
  - Minimal CPU usage
- **Software Encoding**: CPU-based fallback
  - Codecs: `libx264`, `libx265`
  - Universal compatibility

#### Container Formats
- **MP4**: Most compatible, web-ready
- **AVI**: Legacy support
- **MOV**: Apple ecosystem

#### Quality Control
- **CRF (Constant Rate Factor)**: 0-51
  - 0 = Lossless (huge files)
  - 23 = Default (good quality)
  - 28 = High compression
  - 51 = Lowest quality
- **Presets**: ultrafast, fast, medium, slow

### Recording Features

#### Timestamp Synchronization
- Frame-accurate timestamps
- Microsecond precision
- Synchronized with pose data
- Stored in separate files

#### Performance Monitoring
- **Write FPS**: Actual encoding rate
- **Queue Size**: Buffer depth (~ms)
- **Latency**: Encoding delay
- **Frames Written/Enqueued**: Progress tracking
- **Dropped Frames**: Quality indicator

#### Buffer Management
- Configurable queue size
- Automatic overflow handling
- Warning on frame drops
- Backpressure indication

### Auto-Recording Feature

Processor-triggered recording:

1. **Enable**: Check "Auto-record video on processor command"
2. **Processor Control**: Custom processor sets recording flag
3. **Automatic Start**: GUI starts recording when flag set
4. **Session Naming**: Uses processor-defined session name
5. **Automatic Stop**: GUI stops when flag cleared

**Use Cases**:
- Event-triggered recording
- Trial-based experiments
- Conditional data capture
- Remote control via socket

---

## Configuration Management

### Configuration File Structure

Single JSON file contains all settings:

```json
{
  "camera": { ... },
  "dlc": { ... },
  "recording": { ... },
  "bbox": { ... }
}
```

### Features

#### Save/Load Operations
- **Load**: File → Load configuration (Ctrl+O)
- **Save**: File → Save configuration (Ctrl+S)
- **Save As**: File → Save configuration as (Ctrl+Shift+S)
- **Auto-sync**: GUI fields update from file

#### Multiple Configurations
- Switch between experiments quickly
- Per-animal configurations
- Environment-specific settings
- Backup and version control

#### Validation
- Type checking on load
- Default values for missing fields
- Error messages for invalid entries
- Safe fallback to defaults

### Configuration Sections

#### Camera Settings (`camera`)
```json
{
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
}
```

#### DLC Settings (`dlc`)
```json
{
  "model_path": "/path/to/model",
  "model_type": "base",
  "additional_options": {
    "resize": 0.5,
    "processor": "cpu",
    "pcutoff": 0.6
  }
}
```

#### Recording Settings (`recording`)
```json
{
  "enabled": true,
  "directory": "~/Videos/dlc",
  "filename": "session.mp4",
  "container": "mp4",
  "codec": "h264_nvenc",
  "crf": 23
}
```

#### Bounding Box Settings (`bbox`)
```json
{
  "enabled": false,
  "x0": 0,
  "y0": 0,
  "x1": 200,
  "y1": 100
}
```

---

## Processor System

### Plugin Architecture

Custom pose processors for real-time analysis and control.

#### Processor Interface

```python
class MyProcessor:
    """Custom processor example."""

    def process(self, pose, timestamp):
        """Process pose data in real-time.

        Args:
            pose: numpy array (n_keypoints, 3) - x, y, likelihood
            timestamp: float - frame timestamp
        """
        # Extract keypoint positions
        nose_x, nose_y = pose[0, :2]

        # Custom logic
        if nose_x > 320:
            self.trigger_event()

        # Return results (optional)
        return {"position": (nose_x, nose_y)}
```

#### Loading Processors

1. Place processor file in `dlclivegui/processors/`
2. Click "Refresh" in processor dropdown
3. Select processor from list
4. Start inference to activate

#### Built-in Processors

**Socket Processor** (`dlc_processor_socket.py`):
- TCP socket server for remote control
- Commands: `START_RECORDING`, `STOP_RECORDING`
- Session management
- Multi-client support

### Auto-Recording Integration

Processors can control recording:

```python
class RecordingProcessor:
    def __init__(self):
        self._vid_recording = False
        self.session_name = "default"

    @property
    def video_recording(self):
        return self._vid_recording

    def start_recording(self, session):
        self.session_name = session
        self._vid_recording = True

    def stop_recording(self):
        self._vid_recording = False
```

The GUI monitors `video_recording` property and automatically starts/stops recording.

---

## User Interface

### Layout

#### Control Panel (Left)
- **Camera Settings**: Backend, index, FPS, exposure, gain, crop
- **DLC Settings**: Model path, type, processor, options
- **Recording Settings**: Path, filename, codec, quality
- **Bounding Box**: Visualization controls

#### Video Display (Right)
- Live camera feed
- Pose overlay (optional)
- Bounding box overlay (optional)
- Auto-scaling to window size

#### Status Bar (Bottom)
- Current operation status
- Error messages
- Success confirmations

### Control Groups

#### Camera Controls
- Backend selection dropdown
- Camera index/refresh
- FPS, exposure, gain spinboxes
- Crop coordinates
- Rotation selector
- **Start/Stop Preview** buttons

#### DLC Controls
- Model path browser
- Model type selector
- Processor folder/selection
- Additional options (JSON)
- **Start/Stop Inference** buttons
- "Display pose predictions" checkbox
- "Auto-record" checkbox
- Processor status display

#### Recording Controls
- Output directory browser
- Filename input
- Container/codec selectors
- CRF quality slider
- **Start/Stop Recording** buttons

### Visual Feedback

#### Button States
- **Disabled**: Gray, not clickable
- **Enabled**: Default color, clickable
- **Active**:
  - Preview running: Stop button enabled
  - Inference initializing: Blue "Initializing DLCLive!"
  - Inference ready: Green "DLCLive running!"

#### Status Indicators
- Camera FPS (last 5 seconds)
- DLC performance metrics
- Recording statistics
- Processor connection status

---

## Performance Monitoring

### Real-Time Metrics

#### Camera Metrics
- **Throughput**: FPS over last 5 seconds
- **Formula**: `(frame_count - 1) / time_elapsed`
- **Display**: "45.2 fps (last 5 s)"

#### DLC Metrics
- **Inference FPS**: Poses processed per second
- **Latency**:
  - Last frame latency (ms)
  - Average latency over session (ms)
- **Queue**: Number of frames waiting
- **Dropped**: Frames skipped due to queue full
- **Format**: "150/152 frames | inference 42.1 fps | latency 23.5 ms (avg 24.1 ms) | queue 2 | dropped 2"

#### Recording Metrics
- **Write FPS**: Encoding rate
- **Frames**: Written/Enqueued ratio
- **Latency**: Encoding delay (ms)
- **Buffer**: Queue size (~milliseconds)
- **Dropped**: Encoding failures
- **Format**: "1500/1502 frames | write 59.8 fps | latency 12.3 ms (avg 12.5 ms) | queue 5 (~83 ms) | dropped 2"

### Performance Optimization

#### Automatic Adjustments
- Frame display throttling (25 Hz max)
- Queue backpressure handling
- Automatic resolution detection

#### User Adjustments
- Reduce camera FPS
- Enable ROI cropping
- Use hardware encoding
- Increase CRF value
- Disable pose visualization
- Adjust buffer counts

---

## Advanced Features

### Frame Synchronization

All components share frame timestamps:
- Camera controller generates timestamps
- DLC processor preserves timestamps
- Video recorder stores timestamps
- Enables post-hoc alignment

### Error Recovery

#### Camera Connection Loss
- Automatic detection via frame grab failure
- User notification
- Clean resource cleanup
- Restart capability

#### Recording Errors
- Frame size mismatch detection
- Automatic recovery with new settings
- Warning display
- No data loss

### Thread Safety

Multi-threaded architecture:
- **Main Thread**: GUI event loop
- **Camera Thread**: Frame acquisition
- **DLC Thread**: Pose inference
- **Recording Thread**: Video encoding

Qt signals/slots ensure thread-safe communication.

### Resource Management

#### Automatic Cleanup
- Camera release on stop/error
- DLC model unload on stop
- Recording finalization
- Thread termination

#### Memory Management
- Bounded queues prevent memory leaks
- Frame copy-on-write
- Efficient numpy array handling

### Extensibility

#### Custom Backends
Implement `CameraBackend` abstract class:
```python
class MyBackend(CameraBackend):
    def open(self): ...
    def read(self) -> Tuple[np.ndarray, float]: ...
    def close(self): ...

    @classmethod
    def get_device_count(cls) -> int: ...
```

Register in `factory.py`:
```python
_BACKENDS = {
    "mybackend": ("module.path", "MyBackend")
}
```

#### Custom Processors
Place in `processors/` directory:
```python
class MyProcessor:
    def __init__(self, **kwargs):
        # Initialize
        pass

    def process(self, pose, timestamp):
        # Process pose
        pass
```

### Debugging Features

#### Logging
- Console output for errors
- Frame acquisition logging
- Performance warnings
- Connection status

#### Development Mode
- Syntax validation: `python -m compileall dlclivegui`
- Type checking: `mypy dlclivegui`
- Test files included

---

## Use Case Examples

### High-Speed Behavior Tracking

**Setup**:
- Camera: GenTL industrial camera @ 120 FPS
- Codec: h264_nvenc (GPU encoding)
- Crop: Region of interest only
- DLC: PyTorch model on GPU

**Settings**:
```json
{
  "camera": {"fps": 120, "crop_x0": 200, "crop_y0": 100, "crop_x1": 800, "crop_y1": 600},
  "recording": {"codec": "h264_nvenc", "crf": 28},
  "dlc": {"additional_options": {"processor": "gpu", "resize": 0.5}}
}
```

### Event-Triggered Recording

**Setup**:
- Processor: Socket processor with auto-record
- Trigger: Remote computer sends START/STOP commands
- Session naming: Unique per trial

**Workflow**:
1. Enable "Auto-record video on processor command"
2. Start preview and inference
3. Remote system connects via socket
4. Sends `START_RECORDING:trial_001` → recording starts
5. Sends `STOP_RECORDING` → recording stops
6. Files saved as `trial_001.mp4`

### Multi-Camera Synchronization

**Setup**:
- Multiple GUI instances
- Shared trigger signal
- Synchronized filenames

**Configuration**:
Each instance with different camera index but same settings template.

---

## Keyboard Shortcuts

- **Ctrl+O**: Load configuration
- **Ctrl+S**: Save configuration
- **Ctrl+Shift+S**: Save configuration as
- **Ctrl+Q**: Quit application

---

## Platform-Specific Notes

### Windows
- Best GenTL support (vendor CTI files)
- NVENC highly recommended
- DirectShow backend for webcams

### Linux
- Best Aravis support (native GigE)
- V4L2 backend for webcams
- NVENC available with proprietary drivers

### macOS
- Limited industrial camera support
- Aravis via Homebrew
- Software encoding recommended

### NVIDIA Jetson
- Optimized for edge deployment
- Hardware encoding available
- Some Aravis compatibility issues
