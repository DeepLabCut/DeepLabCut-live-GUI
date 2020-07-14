# DeepLabCutLive
GUI to run DeepLabCut on live video feed

## Install

Launch **console**:
```python
conda create -n dlc_live
conda activate dlc_live
conda install python=3.7 tensorflow-GPU==1.13.1
```
**Install** dlc live GUI **packages**:
```python
pip install https://github.com/DeepLabCut/DeepLabCut-live.git
pip install https://github.com/DeepLabCut/DeepLabCut-live-GUI.git
```
**Run dlc live gui**:
```python
dlclivegui
```
## Configure GUI

- **Config**: scroll down menu: Create new config
- **Camera**: scroll down menu: Add new camera or video from browser
  - **Edit Camera Settings**: 
    - Select the camera by picking its **serial number**.
    - Rotate/Crop the desired portion of the image
    - **display_resize**: to use less resources while displaying
  - **Init Camera**

<p align="center">
<img src= https://imagizer.imageshack.com/img923/2419/QQKyMH.png>
<img src= https://imagizer.imageshack.com/img924/626/acJhWD.png>
</p>

- **Processor settings**:
  - Chose your processor directory using **Processor Dir**
  - Pick the processor class your wrote (see appendix 1)
  - **Edit** or **Set Proc**

- **DeepLabCut**:
  - From scroll down menu: **Add DLC** or chose your model, these settings can be modified using **Edit DLC Settings**
    - Specify Name, browse to extracted DLC model folder
    - Specify resizing similar to display_resize to make the inference quicker.
  - **Init DLC**
  - **Display DLC Keypoints**, keypoints can be edited but need to be 'undisplayed' first to have an effect.

<p align="center">
<img src= https://imagizer.imageshack.com/img923/9730/MNzr1J.png>
</p>

## Record Session

- **Record**:
  - **Set Up Session -> Ready -> On -> Off -> Save Video**
  
# Apendices
1. **Processor**:
Use the processor folder as a template to create your own processor
The default processor should contain:
- init
- process: takes in a pose, performs operations, and returns a pose
- save: saves any internal data generated by the processor (such as timestamps for commands to external hardware)

```python
class Processor(object):

    def __init__(self, **kwargs):
        pass

    def process(self, pose, **kwargs):
        return pose

    def save(self, file=''):
        return 0
```
