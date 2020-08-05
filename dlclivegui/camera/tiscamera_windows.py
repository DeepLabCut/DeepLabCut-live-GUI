"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import time
import cv2

from dlclivegui.camera import Camera, CameraError
from dlclivegui.camera.tisgrabber_windows import TIS_CAM


class TISCam(Camera):
    @staticmethod
    def arg_restrictions():

        return {"serial_number": TIS_CAM().GetDevices(), "rotate": [0, 90, 180, 270]}

    def __init__(
        self,
        serial_number="",
        resolution=[720, 540],
        exposure=0.005,
        rotate=0,
        crop=None,
        fps=100,
        display=True,
        display_resize=1.0,
    ):
        """
        Params
        ------
        serial_number = string; serial number for imaging source camera
        crop = dict; contains ints named top, left, height, width for cropping
            default = None, uses default parameters specific to camera
        """

        if (rotate == 90) or (rotate == 270):
            resolution = [resolution[1], resolution[0]]

        super().__init__(
            serial_number,
            resolution=resolution,
            exposure=exposure,
            rotate=rotate,
            crop=crop,
            fps=fps,
            use_tk_display=display,
            display_resize=display_resize,
        )
        self.display = display

    def set_exposure(self):

        val = self.exposure
        val = 1 if val > 1 else val
        val = 0 if val < 0 else val
        self.cam.SetPropertyAbsoluteValue("Exposure", "Value", val)

    def get_exposure(self):

        exposure = [0]
        self.cam.GetPropertyAbsoluteValue("Exposure", "Value", exposure)
        return round(exposure[0], 3)

    # def set_crop(self):

    #     crop = self.crop

    #     if crop:
    #         top = int(crop[0])
    #         left = int(crop[2])
    #         height = int(crop[1]-top)
    #         width = int(crop[3]-left)

    #         if not self.crop_filter:
    #             self.crop_filter = self.cam.CreateFrameFilter(b'ROI')
    #             self.cam.AddFrameFilter(self.crop_filter)

    #         self.cam.FilterSetParameter(self.crop_filter, b'Top', top)
    #         self.cam.FilterSetParameter(self.crop_filter, b'Left', left)
    #         self.cam.FilterSetParameter(self.crop_filter, b'Height', height)
    #         self.cam.FilterSetParameter(self.crop_filter, b'Width', width)

    def set_rotation(self):

        if not self.rotation_filter:
            self.rotation_filter = self.cam.CreateFrameFilter(b"Rotate Flip")
            self.cam.AddFrameFilter(self.rotation_filter)
        self.cam.FilterSetParameter(
            self.rotation_filter, b"Rotation Angle", self.rotate
        )

    def set_fps(self):

        self.cam.SetFrameRate(self.fps)

    def set_capture_device(self):

        self.cam = TIS_CAM()
        self.crop_filter = None
        self.rotation_filter = None
        self.set_rotation()
        # self.set_crop()
        self.set_fps()
        self.next_frame = time.time()

        self.cam.open(self.id)
        self.cam.SetContinuousMode(0)
        self.cam.StartLive(0)

        self.set_exposure()

        return True

    def get_image(self):

        self.cam.SnapImage()
        frame = self.cam.GetImageEx()
        frame = cv2.flip(frame, 0)
        if self.crop is not None:
            frame = frame[self.crop[0] : self.crop[1], self.crop[2] : self.crop[3]]
        return frame

    def close_capture_device(self):

        self.cam.StopLive()
