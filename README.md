# DeepLabCut-Live-GUI <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1596193544929-NHMVMXPVEYZ6R4I45DSR/ke17ZwdGBToddI8pDm48kOHwsIwndRGCkvek0XAcW4ZZw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVH0wqgmu6zkAOZ3crWCtkmLcPIuzHaxU8QRzZwtrVtHupu3E9Ef3XsXP1C_826c-iU/DLCLIVEGUI_LOGO.png?format=750w" width="350" title="DLC-live GUI" alt="DLC LIVE! GUI" align="right" vspace="100"/>


![PyPI - Package Version](https://img.shields.io/pypi/v/deeplabcut-live-gui)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deeplabcut-live-gui?color=purple)
![Python versions](https://img.shields.io/pypi/pyversions/deeplabcut-live-gui)
<!-- ![Python package](https://github.com/DeepLabCut/DeepLabCut-live/workflows/Python%20package/badge.svg) -->

[![License](https://img.shields.io/github/license/DeepLabCut/DeepLabCut-live-GUI?label=license)](https://github.com/DeepLabCut/DeepLabCut-live-GUI/blob/main/LICENSE)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut)
[![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)
[![Code style:Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

GUI to run [DeepLabCut-Live](https://github.com/DeepLabCut/DeepLabCut-live) on a video feed, **preview and record from one or multiple cameras**, and optionally record external timestamps and processor outputs.

## Update

The GUI has been modernized and is now built with **PySide6 (Qt)** (replacing tkinter).

The new interface supports **multi-camera preview** with a **tiled display**, PyTorch models, and features improved interactive workflows for experimental use.

## Documentation

> Find the full documentation at the [DeepLabCut docs website](https://deeplabcut.github.io/DeepLabCut/docs/dlc-live/dlc-live-gui/index.html)

## System requirements (quick summary)

- **Python 3.10, 3.11 or 3.12**
- One inference backend (choose at least one):
  - **PyTorch** *(recommended for best performance & compatibility)*
  - **TensorFlow** *(for backwards compatibility with existing models; Windows installs are no longer available for Python > 3.10)*
- A supported camera backend (OpenCV webcams by default; additional backends supported)

## Installation

`deeplabcut-live-gui` 2.0 is now available on PyPI.

To get the **latest version**, please follow the **instructions below**.


### Option A — Install with `uv`

#### 1) Create & activate a new virtual environment

```bash
uv venv -p 3.12 dlclivegui

# Linux/macOS:
source dlclivegui/bin/activate

# Windows (Command Prompt):
.\dlclivegui\Scripts\activate.bat

# Windows (PowerShell):
.\dlclivegui\Scripts\Activate.ps1
```

#### 2) Choose inference backend and install

You may install **PyTorch** or **TensorFlow** extras (or both), but you must choose at least one to run inference.

- **PyTorch (recommended):**

```bash
uv pip install --pre deeplabcut-live-gui[pytorch]
```

- **TensorFlow (backwards compatibility):**

```bash
uv pip install --pre deeplabcut-live-gui[tf]
```

---

### Option B — Install with `mamba` / `conda`

#### 1) Create & activate a new conda environment

```bash
conda create -n dlclivegui python=3.12 # or mamba
conda activate dlclivegui
```

#### 2) Install with your inference backend

- **PyTorch (recommended):**

```bash
pip install --pre deeplabcut-live-gui[pytorch]
```

- **TensorFlow:**

```bash
pip install --pre deeplabcut-live-gui[tf]
```

## Run the application

> [!TIP]
> For GPU/CUDA support specifics and framework compatibility, please follow the **official PyTorch/TensorFlow install guidance** for your OS and drivers.

After installation, start the application with:

```bash
dlclivegui # in conda/mamba

# OR:
uv run dlclivegui
```

> [!IMPORTANT]
> Activate your venv/conda environment before launching so the GUI can access installed dependencies.

## Typical workflow

The new GUI supports **one or more cameras**.

Typical workflow:

1. **Configure Cameras** (choose backend and devices)
   - Adjust camera settings (serial, exposure, ROI/cropping, etc.)
2. **Start Preview**
   - Adjust visualization settings (keypoint color map, bounding boxes, etc.)
3. **Start inference**
   - Choose a DeepLabCut Live model
   - Choose which camera to run inference on (currently one at a time)
4. **Start recording**
   - Adjust recording settings (codec, output format, etc.)
   - Record video and timestamps to organized session folders

> [!NOTE]
> OpenCV-compatible cameras (USB webcams, OBS virtual camera) work out of the box.
> For additional camera ecosystems (Basler, GenTL, Aravis, etc.), see the relevant documentation.
<!-- TODO @C-Achard add link to docs website once available -->

## Current limitations

- Pose inference runs on **one selected camera at a time** (even in multi-camera mode)
- Camera features support and availability depends on backend capabilities and hardware
  - OpenCV controls for resolution/FPS are best-effort and device-driver dependent
- DeepLabCut-Live models must be exported and compatible with the chosen backend
- Performance depends on resolution, frame rate, GPU availability, and codec choice

---

## References

If you use this code, we kindly ask you to please cite:

- **[Kane et al., eLife 2020](https://elifesciences.org/articles/61909)**
  - If preferred, see the **[Preprint](https://www.biorxiv.org/content/10.1101/2020.08.04.236422v2)**

---

## Contributing / Feedback

This project is under active development — feedback from real experimental use is highly valued.

Please report issues, suggest features, or contribute here on [GitHub](https://github.com/DeepLabCut/DeepLabCut-live-GUI/issues).
