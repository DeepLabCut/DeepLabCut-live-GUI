# DeepLabCut-live-GUI User Guide

Complete walkthrough for using the DeepLabCut-live-GUI application.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Camera Setup](#camera-setup)
3. [DLCLive Configuration](#dlclive-configuration)
4. [Recording Videos](#recording-videos)
5. [Working with Configurations](#working-with-configurations)
6. [Common Workflows](#common-workflows)
7. [Tips and Best Practices](#tips-and-best-practices)

---

## Getting Started

### First Launch

1. Open a terminal/command prompt
2. Run the application:
   ```bash
   dlclivegui
   ```
3. The main window will appear with three control panels and a video display area

### Interface Overview

```
┌─────────────────────────────────────────────────────┐
│ File  Help                                          │
├─────────────┬───────────────────────────────────────┤
│ Camera      │                                       │
│ Settings    │                                       │
│             │                                       │
│ ─────────── │        Video Display                  │
│ DLCLive     │                                       │
│ Settings    │                                       │
│             │                                       │
│ ─────────── │                                       │
│ Recording   │                                       │
│ Settings    │                                       │
│             │                                       │
│ ─────────── │                                       │
│ Bounding    │                                       │
│ Box         │                                       │
│             │                                       │
│ ─────────── │                                       │
│ [Preview]   │                                       │
│ [Stop]      │                                       │
└─────────────┴───────────────────────────────────────┘
│ Status: Ready                                       │
└─────────────────────────────────────────────────────┘
```

---

## Camera Setup

### Step 1: Select Camera Backend

The **Backend** dropdown shows available camera drivers:

| Backend | When to Use |
|---------|-------------|
| **opencv** | Webcams, USB cameras (universal) |
| **gentl** | Industrial cameras (Windows/Linux) |
| **aravis** | GenICam/GigE cameras (Linux/macOS) |
| **basler** | Basler cameras specifically |

**Note**: Unavailable backends appear grayed out. Install required drivers to enable them.

### Step 2: Select Camera

1. Click **Refresh** next to the camera dropdown
2. Wait for camera detection (1-3 seconds)
3. Select your camera from the dropdown

The list shows camera details:
```
0:DMK 37BUX287 (26320523)
│ │             └─ Serial Number
│ └─ Model Name
└─ Index
```

### Step 3: Configure Camera Parameters

#### Frame Rate
- **Range**: 1-240 FPS (hardware dependent)
- **Recommendation**: Start with 30 FPS, increase as needed
- **Note**: Higher FPS = more processing load

#### Exposure Time
- **Auto**: Set to 0 (default)
- **Manual**: Microseconds (e.g., 10000 = 10ms)
- **Tips**:
  - Shorter exposure = less motion blur
  - Longer exposure = better low-light performance
  - Typical range: 5,000-30,000 μs

#### Gain
- **Auto**: Set to 0.0 (default)
- **Manual**: 0.0-100.0
- **Tips**:
  - Higher gain = brighter image but more noise
  - Start low (5-10) and increase if needed
  - Auto mode works well for most cases

#### Cropping (Optional)
Reduce frame size for faster processing:

1. Set crop region: (x0, y0, x1, y1)
   - x0, y0: Top-left corner
   - x1, y1: Bottom-right corner
2. Use Bounding Box visualization to preview
3. Set all to 0 to disable cropping

**Example**: Crop to center 640x480 region of 1280x720 camera:
```
x0: 320
y0: 120
x1: 960
y1: 600
```

#### Rotation
Select if camera is mounted at an angle:
- 0° (default)
- 90° (rotated right)
- 180° (upside down)
- 270° (rotated left)

### Step 4: Start Camera Preview

1. Click **Start Preview**
2. Video feed should appear in the display area
3. Check the **Throughput** metric below camera settings
4. Verify frame rate matches expected value

**Troubleshooting**:
- **No preview**: Check camera connection and permissions
- **Low FPS**: Reduce resolution or increase exposure time
- **Black screen**: Check exposure settings
- **Distorted image**: Verify backend compatibility

---

## DLCLive Configuration

### Prerequisites

1. Exported DLCLive model (see DLC documentation)
2. DeepLabCut-live installed (`pip install deeplabcut-live`)
3. Camera preview running

### Step 1: Select Model

1. Click **Browse** next to "Model directory"
2. Navigate to your exported DLCLive model folder
3. Select the folder containing:
   - `pose_cfg.yaml`
   - Model weights (`.pb`, `.pth`, etc.)

### Step 2: Choose Model Type

Select from dropdown:
- **Base (TensorFlow)**: Standard DLC models
- **PyTorch**: PyTorch-based models (requires PyTorch)

### Step 3: Configure Options (Optional)

Click in "Additional options" field and enter JSON:

```json
{
  "processor": "gpu",
  "resize": 0.5,
  "pcutoff": 0.6
}
```

**Common options**:
- `processor`: "cpu" or "gpu"
- `resize`: Scale factor (0.5 = half size)
- `pcutoff`: Likelihood threshold
- `cropping`: Crop before inference

### Step 4: Select Processor (Optional)

If using custom pose processors:

1. Click **Browse** next to "Processor folder" (or use default)
2. Click **Refresh** to scan for processors
3. Select processor from dropdown
4. Processor will activate when inference starts

### Step 5: Start Inference

1. Ensure camera preview is running
2. Click **Start pose inference**
3. Button changes to "Initializing DLCLive!" (blue)
4. Wait for model loading (5-30 seconds)
5. Button changes to "DLCLive running!" (green)
6. Check **Performance** metrics

**Performance Metrics**:
```
150/152 frames | inference 42.1 fps | latency 23.5 ms (avg 24.1 ms) | queue 2 | dropped 2
```
- **150/152**: Processed/Total frames
- **inference 42.1 fps**: Processing rate
- **latency 23.5 ms**: Current processing delay
- **queue 2**: Frames waiting
- **dropped 2**: Skipped frames (due to full queue)

### Step 6: Enable Visualization (Optional)

Check **"Display pose predictions"** to overlay keypoints on video.

- Keypoints appear as green circles
- Updates in real-time with video
- Can be toggled during inference

---

## Recording Videos

### Basic Recording

1. **Configure output path**:
   - Click **Browse** next to "Output directory"
   - Select or create destination folder

2. **Set filename**:
   - Enter base filename (e.g., "session_001")
   - Extension added automatically based on container

3. **Select format**:
   - **Container**: mp4 (recommended), avi, mov
   - **Codec**:
     - `h264_nvenc` (NVIDIA GPU - fastest)
     - `libx264` (CPU - universal)
     - `hevc_nvenc` (NVIDIA H.265)

4. **Set quality** (CRF slider):
   - 0-17: Very high quality, large files
   - 18-23: High quality (recommended)
   - 24-28: Medium quality, smaller files
   - 29-51: Lower quality, smallest files

5. **Start recording**:
   - Ensure camera preview is running
   - Click **Start recording**
   - **Stop recording** button becomes enabled

6. **Monitor performance**:
   - Check "Performance" metrics
   - Watch for dropped frames
   - Verify write FPS matches camera FPS

### Advanced Recording Options

#### High-Speed Recording (60+ FPS)

**Settings**:
- Codec: `h264_nvenc` (requires NVIDIA GPU)
- CRF: 28 (higher compression)
- Crop region: Reduce frame size
- Close other applications

#### High-Quality Recording

**Settings**:
- Codec: `libx264` or `h264_nvenc`
- CRF: 18-20
- Full resolution
- Sufficient disk space

#### Long Duration Recording

**Tips**:
- Use CRF 23-25 to balance quality/size
- Monitor disk space
- Consider splitting into multiple files
- Use fast SSD storage

### Auto-Recording

Enable automatic recording triggered by processor events:

1. **Select a processor** that supports auto-recording
2. **Enable**: Check "Auto-record video on processor command"
3. **Start inference**: Processor will control recording
4. **Session management**: Files named by processor

**Use cases**:
- Trial-based experiments
- Event-triggered recording
- Remote control via socket processor
- Conditional data capture

---

## Working with Configurations

### Saving Current Settings

**Save** (overwrites existing file):
1. File → Save configuration (or Ctrl+S)
2. If no file loaded, prompts for location

**Save As** (create new file):
1. File → Save configuration as… (or Ctrl+Shift+S)
2. Choose location and filename
3. Enter name (e.g., `mouse_experiment.json`)
4. Click Save

### Loading Saved Settings

1. File → Load configuration… (or Ctrl+O)
2. Navigate to configuration file
3. Select `.json` file
4. Click Open
5. All GUI fields update automatically

### Managing Multiple Configurations

**Recommended structure**:
```
configs/
├── default.json          # Base settings
├── mouse_arena1.json     # Arena-specific
├── mouse_arena2.json
├── rat_setup.json
└── high_speed.json       # Performance-specific
```

**Workflow**:
1. Create base configuration with common settings
2. Save variants for different:
   - Animals/subjects
   - Experimental setups
   - Camera positions
   - Recording quality levels

### Configuration Templates

#### Webcam + CPU Processing
```json
{
  "camera": {
    "backend": "opencv",
    "index": 0,
    "fps": 30.0
  },
  "dlc": {
    "model_type": "base",
    "additional_options": {"processor": "cpu"}
  },
  "recording": {
    "codec": "libx264",
    "crf": 23
  }
}
```

#### Industrial Camera + GPU
```json
{
  "camera": {
    "backend": "gentl",
    "index": 0,
    "fps": 60.0,
    "exposure": 10000,
    "gain": 8.0
  },
  "dlc": {
    "model_type": "pytorch",
    "additional_options": {
      "processor": "gpu",
      "resize": 0.5
    }
  },
  "recording": {
    "codec": "h264_nvenc",
    "crf": 23
  }
}
```

---

## Common Workflows

### Workflow 1: Simple Webcam Tracking

**Goal**: Track mouse behavior with webcam

1. **Camera Setup**:
   - Backend: opencv
   - Camera: Built-in webcam (index 0)
   - FPS: 30

2. **Start Preview**: Verify mouse is visible

3. **Load DLC Model**: Browse to mouse tracking model

4. **Start Inference**: Enable pose estimation

5. **Verify Tracking**: Enable pose visualization

6. **Record Trial**: Start/stop recording as needed

### Workflow 2: High-Speed Industrial Camera

**Goal**: Track fast movements at 120 FPS

1. **Camera Setup**:
   - Backend: gentl or aravis
   - Refresh and select camera
   - FPS: 120
   - Exposure: 4000 μs (short exposure)
   - Crop: Region of interest only

2. **Start Preview**: Check FPS is stable

3. **Configure Recording**:
   - Codec: h264_nvenc
   - CRF: 28
   - Output: Fast SSD

4. **Load DLC Model** (if needed):
   - PyTorch model
   - GPU processor
   - Resize: 0.5 (reduce load)

5. **Start Recording**: Begin data capture

6. **Monitor Performance**: Watch for dropped frames

### Workflow 3: Event-Triggered Recording

**Goal**: Record only during specific events

1. **Camera Setup**: Configure as normal

2. **Processor Setup**:
   - Select socket processor
   - Enable "Auto-record video on processor command"

3. **Start Preview**: Camera running

4. **Start Inference**: DLC + processor active

5. **Remote Control**:
   - Connect to socket (default port 5000)
   - Send `START_RECORDING:trial_001`
   - Recording starts automatically
   - Send `STOP_RECORDING`
   - Recording stops, file saved

### Workflow 4: Multi-Subject Tracking

**Goal**: Track multiple animals simultaneously

**Option A: Single Camera, Multiple Keypoints**
1. Use DLC model trained for multiple subjects
2. Single GUI instance
3. Processor distinguishes subjects

**Option B: Multiple Cameras**
1. Launch multiple GUI instances
2. Each with different camera index
3. Synchronized configurations
4. Coordinated filenames

---

## Tips and Best Practices

### Camera Tips

1. **Lighting**:
   - Consistent, diffuse lighting
   - Avoid shadows and reflections
   - IR lighting for night vision

2. **Positioning**:
   - Stable mount (minimize vibration)
   - Appropriate angle for markers
   - Sufficient field of view

3. **Settings**:
   - Start with auto exposure/gain
   - Adjust manually if needed
   - Test different FPS rates
   - Use cropping to reduce load

### Recording Tips

1. **File Management**:
   - Use descriptive filenames
   - Include date/subject/trial info
   - Organize by experiment/session
   - Regular backups

2. **Performance**:
   - Close unnecessary applications
   - Monitor disk space
   - Use SSD for high-speed recording
   - Enable GPU encoding if available

3. **Quality**:
   - Test CRF values beforehand
   - Balance quality vs. file size
   - Consider post-processing needs
   - Verify recordings occasionally

### DLCLive Tips

1. **Model Selection**:
   - Use model trained on similar conditions
   - Test offline before live use
   - Consider resize for speed
   - GPU highly recommended

2. **Performance**:
   - Monitor inference FPS
   - Check latency values
   - Watch queue depth
   - Reduce resolution if needed

3. **Validation**:
   - Enable visualization initially
   - Verify tracking quality
   - Check all keypoints
   - Test edge cases

### General Best Practices

1. **Configuration Management**:
   - Save configurations frequently
   - Version control config files
   - Document custom settings
   - Share team configurations

2. **Testing**:
   - Test setup before experiments
   - Run trial recordings
   - Verify all components
   - Check file outputs

3. **Troubleshooting**:
   - Check status messages
   - Monitor performance metrics
   - Review error dialogs carefully
   - Restart if issues persist

4. **Data Organization**:
   - Consistent naming scheme
   - Separate folders per session
   - Include metadata files
   - Regular data validation

---

## Troubleshooting Guide

### Camera Issues

**Problem**: Camera not detected
- **Solution**: Click Refresh, check connections, verify drivers

**Problem**: Low frame rate
- **Solution**: Reduce resolution, increase exposure, check CPU usage

**Problem**: Image too dark/bright
- **Solution**: Adjust exposure and gain settings

### DLCLive Issues

**Problem**: Model fails to load
- **Solution**: Verify path, check model type, install dependencies

**Problem**: Slow inference
- **Solution**: Enable GPU, reduce resolution, use resize option

**Problem**: Poor tracking
- **Solution**: Check lighting, enable visualization, verify model quality

### Recording Issues

**Problem**: Dropped frames
- **Solution**: Use GPU encoding, increase CRF, reduce FPS

**Problem**: Large file sizes
- **Solution**: Increase CRF value, use better codec

**Problem**: Recording won't start
- **Solution**: Check disk space, verify path permissions

---

## Keyboard Reference

| Action | Shortcut |
|--------|----------|
| Load configuration | Ctrl+O |
| Save configuration | Ctrl+S |
| Save configuration as | Ctrl+Shift+S |
| Quit application | Ctrl+Q |

---

## Next Steps

- Explore [Features Documentation](features.md) for detailed capabilities
- Review [Camera Backend Guide](camera_support.md) for advanced setup
- Check [Processor System](PLUGIN_SYSTEM.md) for custom processing
- See [Aravis Backend](aravis_backend.md) for Linux industrial cameras

---

## Getting Help

If you encounter issues:
1. Check status messages in GUI
2. Review this user guide
3. Consult technical documentation
4. Check GitHub issues
5. Contact support team
