import platform

from dlclivegui.camera.camera import Camera, DLCLiveCameraError
from dlclivegui.camera.opencv import OpenCVCam

if platform.system() == "Windows":
    from dlclivegui.camera.tiscamera_windows import TISCam

if platform.system() == "Linux":
    from dlclivegui.camera.tiscamera_linux import TISCam

if platform.system() in ["Darwin", "Linux"]:
    from dlclivegui.camera.aravis import AravisCam

if platform.system() == "Darwin":
    from dlclivegui.camera.pseye import PSEyeCam
