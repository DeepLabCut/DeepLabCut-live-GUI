"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import time
import cv2

from dlclivegui.camera import Camera, DLCLiveCameraError
from dlclivegui.camera.tisgrabber_windows import TIS_CAM


class TISCam(Camera):


    @staticmethod
    def arg_restrictions():

        return {'serial_number' : TIS_CAM().GetDevices()}


    def __init__(self, serial_number='', exposure=.005, rotate=0, crop=[], fps=100, display=True):
        '''
        Params
        ------
        serial_number = string; serial number for imaging source camera
        crop = dict; contains ints named top, left, height, width for cropping
            default = None, uses default parameters specific to camera
        '''

        self.cam = TIS_CAM()

        self.serial_number = serial_number
        self.im_size = (720, 540)

        self.crop_filter = None
        self.rotation_filter = None
        self.display = display
        self.next_frame = None

        super().__init__(serial_number, exposure=exposure, rotate=rotate, crop=crop, fps=fps)


    def set_exposure(self, val):

        val = 1 if val > 1 else val
        val = 0 if val < 0 else val
        self.cam.SetPropertyAbsoluteValue("Exposure", "Value", val)


    def get_exposure(self):

        exposure = [0]
        self.cam.GetPropertyAbsoluteValue("Exposure", "Value", exposure)
        return round(exposure[0], 3)


    def set_crop(self, crop):

        if crop:
            left = crop[0]
            width = crop[1]-left
            top = crop[3]
            height = top-crop[2]

            if not self.crop_filter:
                self.crop_filter = self.cam.CreateFrameFilter(b'ROI')
                self.cam.AddFrameFilter(self.crop_filter)

            self.cam.FilterSetParameter(self.crop_filter, b'Top', top)
            self.cam.FilterSetParameter(self.crop_filter, b'Left', left)
            self.cam.FilterSetParameter(self.crop_filter, b'Height', height)
            self.cam.FilterSetParameter(self.crop_filter, b'Width', width)
            self.im_size = (width, height)


    def set_rotation(self, rotate):

        if not self.rotation_filter:
            self.rotation_filter = self.cam.CreateFrameFilter(b'Rotate Flip')
            self.cam.AddFrameFilter(self.rotation_filter)
        self.cam.FilterSetParameter(self.rotation_filter, b'Rotation Angle', rotate)


    def set_fps(self, fps):

        self.fps = fps
        self.cam.SetFrameRate(fps)


    def get_image(self):

        self.cam.SnapImage()
        frame = self.cam.GetImageEx()
        frame = cv2.flip(frame, 0)
        return frame


    def open(self):

        super().open()

        self.cam.open(self.serial_number)
        self.cam.SetContinuousMode(0)
        self.cam.StartLive(int(self.display))


    def close(self):

        super().close()
        self.cam.StopLive()
