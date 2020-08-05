"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import cv2
from tkinter import filedialog
from imutils import rotate_bound
import time
import platform

from dlclivegui.camera import Camera, CameraError


class OpenCVCam(Camera):
    @staticmethod
    def arg_restrictions():
        """ Returns a dictionary of arguments restrictions for DLCLiveGUI
        """

        cap = cv2.VideoCapture()
        devs = [-1]
        avail = True
        while avail:
            cur_index = devs[-1] + 1
            avail = cap.open(cur_index)
            if avail:
                devs.append(cur_index)
                cap.release()

        return {"device": devs, "display": [True, False]}

    def __init__(
        self,
        device=-1,
        file="",
        resolution=[640, 480],
        auto_exposure=0,
        exposure=0,
        gain=0,
        rotate=0,
        crop=None,
        fps=30,
        display=True,
        display_resize=1.0,
    ):

        if device != -1:
            if file:
                raise DLCLiveCameraError(
                    "A device and file were provided to OpenCVCam. Must initialize an OpenCVCam with either a device id or a video file."
                )

            self.video = False
            id = int(device)

        else:
            if not file:
                file = filedialog.askopenfilename(
                    title="Select video file for DLC-live-GUI"
                )
                if not file:
                    raise DLCLiveCameraError(
                        "Neither a device nor file were provided to OpenCVCam. Must initialize an OpenCVCam with either a device id or a video file."
                    )

            self.video = True
            cap = cv2.VideoCapture(file)
            resolution = (
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            )
            fps = cap.get(cv2.CAP_PROP_FPS)
            del cap
            id = file

        super().__init__(
            id,
            resolution=resolution,
            exposure=exposure,
            rotate=rotate,
            crop=crop,
            fps=fps,
            use_tk_display=display,
            display_resize=display_resize,
        )
        self.auto_exposure = auto_exposure
        self.gain = gain

    def set_capture_device(self):

        if not self.video:

            self.cap = (
                cv2.VideoCapture(self.id, cv2.CAP_V4L)
                if platform.system() == "Linux"
                else cv2.VideoCapture(self.id)
            )
            ret, frame = self.cap.read()

            if self.im_size:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.im_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.im_size[1])
            if self.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)
            if self.exposure:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            if self.gain:
                self.cap.set(cv2.CAP_PROP_GAIN, self.gain)
            if self.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        else:

            self.cap = cv2.VideoCapture(self.id)

            # self.im_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.last_cap_read = 0

        self.cv2_color = self.cap.get(cv2.CAP_PROP_MODE)

        return True

    def get_image_on_time(self):

        # if video, wait...
        if self.video:
            while time.time() - self.last_cap_read < (1.0 / self.fps):
                pass

        ret, frame = self.cap.read()

        if ret:
            if self.rotate:
                frame = rotate_bound(frame, self.rotate)
            if self.crop:
                frame = frame[self.crop[2] : self.crop[3], self.crop[0] : self.crop[1]]

            if frame.ndim == 3:
                if self.cv2_color == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.last_cap_read = time.time()

            return frame, self.last_cap_read
        else:
            raise CameraError("OpenCV VideoCapture.read did not return an image!")

    def close_capture_device(self):

        self.cap.release()
