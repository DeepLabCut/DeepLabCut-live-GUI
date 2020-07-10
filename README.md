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
<img src= https://bc3-production-blob-previews-us-east-2.s3.us-east-2.amazonaws.com/04b78c1a-c2b7-11ea-a1bf-a0369f6bea8a?response-content-disposition=inline%3B%20filename%3D%22preview-lightbox-image.png%22%3B%20filename%2A%3DUTF-8%27%27preview-lightbox-image.png&response-content-type=image%2Fpng&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAS5PME4CT5QW2PJJU%2F20200710%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20200710T140950Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=62ebe4865a99d4a90ac3a66d0b898e6b3e3e2cdce1fbc63d97bca4f306bb60b2>
<img src= https://bc3-production-blob-previews-us-east-2.s3.us-east-2.amazonaws.com/b55cf2da-c2b7-11ea-8d78-c81f66d3f0a2?response-content-disposition=inline%3B%20filename%3D%22preview-lightbox-image.png%22%3B%20filename%2A%3DUTF-8%27%27preview-lightbox-image.png&response-content-type=image%2Fpng&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAS5PME4CT5QW2PJJU%2F20200710%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20200710T141446Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=cf3ba83bfc5af64e9b510722a25b210a71402c170adaaa4c0028d585c6252613>
</p>

- **Processor settings**: TODO
- **DeepLabCut**:
  - From scroll down menu: **Add DLC** or chose your model, these settings can be modified using **Edit DLC Settings**
    - Specify Name, browse to extracted DLC model folder
    - Specify resizing similar to display_resize to make the inference quicker.
  - **Init DLC**
  - **Display DLC Keypoints**, keypoints can be edited but need to be 'undisplayed' first to have an effect.

<p align="center">
<img src= https://bc3-production-blob-previews-us-east-2.s3.us-east-2.amazonaws.com/3ffebfc2-c2b8-11ea-b280-a0369f6bed60?response-content-disposition=inline%3B%20filename%3D%22preview-lightbox-image.png%22%3B%20filename%2A%3DUTF-8%27%27preview-lightbox-image.png&response-content-type=image%2Fpng&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAS5PME4CT5QW2PJJU%2F20200710%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20200710T141839Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=dacc90e60f6dab4f282e25f4b468a42bc3e79ed9529c3fa99915df0440c47420>
</p>

## Record Session

- **Record**:
  - **Set Up Session -> Ready -> On -> Off -> Save Video**
