"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import cv2
from imutils import rotate_bound
import numpy as np
import pseyepy

from dlclivegui.camera import Camera, CameraError


class PSEyeCam(Camera):
    @staticmethod
    def arg_restrictions():

        return {
            "device": [i for i in range(pseyepy.cam_count())],
            "resolution": [[320, 240], [640, 480]],
            "fps": [30, 40, 50, 60, 75, 100, 125],
            "colour": [True, False],
            "auto_whitebalance": [True, False],
        }

    def __init__(
        self,
        device=0,
        resolution=[320, 240],
        exposure=100,
        gain=20,
        rotate=0,
        crop=None,
        fps=60,
        colour=False,
        auto_whitebalance=False,
        red_balance=125,
        blue_balance=125,
        green_balance=125,
        display=True,
        display_resize=1.0,
    ):

        super().__init__(
            device,
            resolution=resolution,
            exposure=exposure,
            gain=gain,
            rotate=rotate,
            crop=crop,
            fps=fps,
            use_tk_display=display,
            display_resize=display_resize,
        )
        self.colour = colour
        self.auto_whitebalance = auto_whitebalance
        self.red_balance = red_balance
        self.blue_balance = blue_balance
        self.green_balance = green_balance

    def set_capture_device(self):

        if self.im_size[0] == 320:
            res = pseyepy.Camera.RES_SMALL
        elif self.im_size[0] == 640:
            res = pseyepy.Camera.RES_LARGE
        else:
            raise CameraError(f"pseye resolution {self.im_size} not supported")

        self.cap = pseyepy.Camera(
            self.id,
            fps=self.fps,
            resolution=res,
            exposure=self.exposure,
            gain=self.gain,
            colour=self.colour,
            auto_whitebalance=self.auto_whitebalance,
            red_balance=self.red_balance,
            blue_balance=self.blue_balance,
            green_balance=self.green_balance,
        )

        return True

    def get_image_on_time(self):

        frame, _ = self.cap.read()

        if self.rotate != 0:
            frame = rotate_bound(frame, self.rotate)
        if self.crop:
            frame = frame[self.crop[2] : self.crop[3], self.crop[0] : self.crop[1]]

        return frame

    def close_capture_device(self):

        self.cap.end()
