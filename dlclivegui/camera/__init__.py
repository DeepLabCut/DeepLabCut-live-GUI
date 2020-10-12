"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import platform

from dlclivegui.camera.camera import Camera, CameraError
from dlclivegui.camera.opencv import OpenCVCam

if platform.system() == "Windows":
    try:
        from dlclivegui.camera.tiscamera_windows import TISCam
    except Exception as e:
        pass

if platform.system() == "Linux":
    try:
        from dlclivegui.camera.tiscamera_linux import TISCam
    except Exception as e:
        pass
        # print(f"Error importing TISCam on Linux: {e}")

if platform.system() in ["Darwin", "Linux"]:
    try:
        from dlclivegui.camera.aravis import AravisCam
    except Exception as e:
        pass
        # print(f"Error importing AravisCam: f{e}")

if platform.system() == "Darwin":
    try:
        from dlclivegui.camera.pseye import PSEyeCam
    except Exception as e:
        pass
