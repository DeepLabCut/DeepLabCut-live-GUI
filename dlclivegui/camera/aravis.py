"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import ctypes
import numpy as np
import time

import gi

gi.require_version("Aravis", "0.6")
from gi.repository import Aravis
import cv2

from dlclivegui.camera import Camera


class AravisCam(Camera):
    @staticmethod
    def arg_restrictions():

        Aravis.update_device_list()
        n_cams = Aravis.get_n_devices()
        ids = [Aravis.get_device_id(i) for i in range(n_cams)]
        return {"id": ids}

    def __init__(
        self,
        id="",
        resolution=[720, 540],
        exposure=0.005,
        gain=0,
        rotate=0,
        crop=None,
        fps=100,
        display=True,
        display_resize=1.0,
    ):

        super().__init__(
            id,
            resolution=resolution,
            exposure=exposure,
            gain=gain,
            rotate=rotate,
            crop=crop,
            fps=fps,
            use_tk_display=display,
            display_resize=display_resize,
        )

    def set_capture_device(self):

        self.cam = Aravis.Camera.new(self.id)
        self.no_auto()
        self.set_exposure(self.exposure)
        self.set_crop(self.crop)
        self.cam.set_frame_rate(self.fps)

        self.stream = self.cam.create_stream()
        self.stream.push_buffer(Aravis.Buffer.new_allocate(self.cam.get_payload()))
        self.cam.start_acquisition()

        return True

    def no_auto(self):

        self.cam.set_exposure_time_auto(0)
        self.cam.set_gain_auto(0)

    def set_exposure(self, val):

        val = 1 if val > 1 else val
        val = 0 if val < 0 else val
        self.cam.set_exposure_time(val * 1e6)

    def set_crop(self, crop):

        if crop:
            left = crop[0]
            width = crop[1] - left
            top = crop[3]
            height = top - crop[2]
            self.cam.set_region(left, top, width, height)
            self.im_size = (width, height)

    def get_image_on_time(self):

        buffer = None
        while buffer is None:
            buffer = self.stream.try_pop_buffer()

        frame = self._convert_image_to_numpy(buffer)
        self.stream.push_buffer(buffer)

        return frame, time.time()

    def _convert_image_to_numpy(self, buffer):
        """ from https://github.com/SintefManufacturing/python-aravis """

        pixel_format = buffer.get_image_pixel_format()
        bits_per_pixel = pixel_format >> 16 & 0xFF

        if bits_per_pixel == 8:
            INTP = ctypes.POINTER(ctypes.c_uint8)
        else:
            INTP = ctypes.POINTER(ctypes.c_uint16)

        addr = buffer.get_data()
        ptr = ctypes.cast(addr, INTP)

        frame = np.ctypeslib.as_array(
            ptr, (buffer.get_image_height(), buffer.get_image_width())
        )
        frame = frame.copy()

        if frame.ndim < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        return frame

    def close_capture_device():

        self.cam.stop_acquisition()
