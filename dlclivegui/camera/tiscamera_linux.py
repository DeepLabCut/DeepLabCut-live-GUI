"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import warnings
import numpy as np

import gi
gi.require_version("Tcam", "0.1")
gi.require_version("Gst", "1.0")
from gi.repository import Tcam, Gst, GLib, GObject

from dlclivegui.camera import Camera


class TISCam(Camera):


    FRAME_RATE_OPTIONS = [15, 30, 60, 120, 240, 480]
    FRAME_RATE_FRACTIONS = ['15/1', '30/1', '60/1', '120/1', '5000000/20833', '480/1']
    IM_FORMAT = (720, 540)
    ROTATE_OPTIONS = ['identity', '90r', '180', '90l']


    @staticmethod
    def arg_restrictions():

        if not Gst.is_initialized():
            Gst.init()

        source = Gst.ElementFactory.make("tcambin")
        return {'serial_number' : source.get_device_serials(),
                'fps' : TISCam.FRAME_RATE_OPTIONS,
                'rotate' : TISCam.ROTATE_OPTIONS}


    def __init__(self, serial_number='', resolution=[720, 540], exposure=.005, gain=0, rotate='identity', crop=None, fps=120, display=True):

        super().__init__(serial_number, resolution=resolution, exposure=exposure, gain=gain, rotate=rotate, crop=crop, fps=fps)
        self.display = display
        self.sample_locked = False
        self.new_sample = False



    def set_capture_device(self):

        self.setup_gst(self.id, self.fps)

        if self.display:
            self.gst_pipeline.set_state(Gst.State.PLAYING)


    def no_auto(self):

        self.cam.set_tcam_property("Exposure Auto", GObject.Value(bool, False))


    def setup_gst(self, serial_number, fps):

        if not Gst.is_initialized():
            Gst.init()

        fps_index = np.where([int(fps) == int(opt) for opt in TISCam.FRAME_RATE_OPTIONS])[0][0]
        fps_frac = TISCam.FRAME_RATE_FRACTIONS[fps_index]

        pipeline = "tcambin name=cam " \
                   "! videocrop name=crop " \
                   "! videoflip name=rotate " \
                   "! video/x-raw,framerate={} ".format(fps_frac)

        if self.display:
            pipe_sink = "! tee name=t " \
                        "t. ! queue ! videoconvert ! ximagesink " \
                        "t. ! queue ! appsink name=sink"
        else:
            pipe_sink = "! appsink name=sink"

        pipeline += pipe_sink

        self.gst_pipeline = Gst.parse_launch(pipeline)

        self.cam = self.gst_pipeline.get_by_name("cam")
        self.cam.set_property("serial", serial_number)
        
        self.set_exposure(self.exposure)
        self.set_crop(self.crop)
        self.set_rotation(self.rotate)
        self.set_sink()


    def set_exposure(self, val):

        val = 1 if val > 1 else val
        val = 0 if val < 0 else val
        self.cam.set_tcam_property("Exposure", val*1e6)


    def set_crop(self, crop):

        if crop:
            self.gst_crop = self.gst_pipeline.get_by_name("crop")
            self.gst_crop.set_property("left", crop[0])
            self.gst_crop.set_property("right", TISCam.IM_FORMAT[0] - crop[1])
            self.gst_crop.set_property("top", crop[2])
            self.gst_crop.set_property("bottom", TISCam.IM_FORMAT[1] - crop[3])
            self.im_size = (crop[3]-crop[2], crop[1]-crop[0])


    def set_rotation(self, val):

        if val:
            self.gst_rotate = self.gst_pipeline.get_by_name("rotate")
            self.gst_rotate.set_property("video-direction", val)

    
    def set_sink(self):

        self.gst_sink = self.gst_pipeline.get_by_name("sink")
        self.gst_sink.set_property("max-buffers", 1)
        self.gst_sink.set_property("drop", 1)
        self.gst_sink.set_property("emit-signals", 1)
        self.gst_sink.connect('new-sample', self.get_image)


    def get_image(self, sink):

        # wait for sample to unlock
        while self.sample_locked:
            pass

        try:

            self.sample = sink.get_property('last-sample')
            self._convert_image_to_numpy()

        except GLib.Error as e:

            warnings.warn("Error reading image :: {}".format(e))

        finally:

            return 0


    def _convert_image_to_numpy(self):

        self.sample_locked = True

        buffer = self.sample.get_buffer()
        struct = self.sample.get_caps().get_structure(0)
        format = struct.get_value('format')

        dtype = np.uint8 if format is "GRAY8" else np.uint16
        np_buff = buffer.extract_dup(0, buffer.get_size())
        self.frame = np.ndarray((struct.get_value('height'), struct.get_value('width'), 1),
                                buffer=np_buff, dtype=dtype)

        self.new_image = True
        self.sample_locked = False


    def get_image_on_time(self):

        # wait for new sample
        while not self.new_sample:
            pass

        return self.frame, time.time()


    def close_capture_device(self):
        
        self.gst_pipeline.set_state(Gst.State.NULL)
