"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import pypylon as pylon
from imutils import rotate_bound
import time

from dlclivegui.camera import Camera, CameraError


class BaslerCam(Camera):
    @staticmethod
    def arg_restrictions():
        """ Returns a dictionary of arguments restrictions for DLCLiveGUI
        """

        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        return {"device": devices, "display": [True, False]}

    def __init__(
        self,
        device="",
        resolution=[640, 480],
        exposure=0,
        rotate=0,
        crop=None,
        fps=30,
        display=True,
        display_resize=1.0,
    ):

        super().__init__(
            device,
            resolution=resolution,
            exposure=exposure,
            rotate=rotate,
            crop=crop,
            fps=fps,
            use_tk_display=display,
            display_resize=display_resize,
        )

        self.display = display

    def set_capture_device(self):

        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateDevice(self.id))
        self.cam.Open()

        self.cam.Gain.SetValue(self.gain)
        self.cam.Exposure.SetValue(self.exposure)
        self.cam.Width.SetValue(self.im_size[0])
        self.cam.Height.SetValue(self.im_size[1])

        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        return True

    def get_image(self):

        grabResult = self.cam.RetrieveResult(
            1, pylon.TimeoutHandling_ThrowException)

        frame = None

        if grabResult.GrabSucceeded():

            image = self.converter.Convert(grabResult)
            frame = image.GetArray()

            if self.rotate:
                frame = rotate_bound(frame, self.rotate)
            if self.crop:
                frame = frame[self.crop[2]: self.crop[3],
                              self.crop[0]: self.crop[1]]

        else:

            raise CameraError("Basler Camera did not return an image!")

        grabResult.Release()

        return frame

    def close_capture_device(self):

        self.cam.StopGrabbing()
