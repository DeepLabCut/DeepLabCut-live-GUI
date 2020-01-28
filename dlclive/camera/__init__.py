from .camera import Camera
from .video_feed import VideoFeed

import platform
if platform.system() == "Windows":
    from .ic_camera import ICCam
