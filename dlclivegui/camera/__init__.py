from dlclivegui.camera.camera import Camera, DLCLiveCameraError
from dlclivegui.camera.opencv import OpenCVCam

import platform
if platform.system() == "Windows":
    from dlclivegui.camera.tiscamera import TISCam
