## Installation Instructions

### Windows or Linux Desktop

We recommend that you install DeepLabCut-live in a conda environment. First, please install Anaconda:
- [Windows](https://docs.anaconda.com/anaconda/install/windows/)
- [Linux](https://docs.anaconda.com/anaconda/install/linux/)

Create a conda environment with python 3.7 and tensorflow:
```
conda create -n dlc-live python=3.7 tensorflow-gpu==1.13.1 # if using GPU
conda create -n dlc-live python=3.7 tensorflow==1.13.1 # if not using GPU
```

Activate the conda environment and install the DeepLabCut-live package:
```
conda activate dlc-live
pip install deeplabcut-live-gui
```

### NVIDIA Jetson Development Kit

First, please refer to our complete instructions for [installing DeepLabCut-Live! on a NVIDIA Jetson Development Kit](https://github.com/DeepLabCut/DeepLabCut-live/blob/master/docs/install_jetson.md).

Next, install the DeepLabCut-live-GUI: `pip install deeplabcut-live-gui`.