"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import cv2
from imutils import rotate_bound
import numpy as np
import pseyepy

from dlclivegui.camera import Camera


class PSEyeCam(Camera):


    @staticmethod
    def arg_restrictions():

        return {'device' : np.arange(pseyepy.cam_count()),
                'colour' : [True, False]}


    def __init__(self, device=0, resolution=[320, 240], exposure=100, gain=20, rotate=0, crop=None, fps=60, colour=False, display=True, display_resize=1.0):

        super().__init__(device, resolution=resolution, exposure=exposure, gain=gain, rotate=rotate, crop=crop, fps=fps, use_tk_display=display, display_resize=display_resize)
        self.colour = colour


    def set_capture_device(self):

        self.cap = pseyepy.Camera(self.id, fps=self.fps, exposure=self.exposure, gain=self.gain, colour=self.colour)


    def get_image(self):

        frame, _ = self.cap.read()

        if self.colour:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.rotate != 0:
            frame = rotate_bound(frame, self.rotate)
        if self.crop:
            frame = frame[self.crop[2]:self.crop[3], self.crop[0]:self.crop[1]]

        return frame


    def close_capture_device(self):

        self.cap.end()
