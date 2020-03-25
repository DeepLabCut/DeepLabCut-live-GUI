"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

"""
To use an Open CV webcam or video file
"""


import cv2
from tkinter import filedialog
from imutils import rotate_bound

from dlclivegui.camera import Camera, DLCLiveCameraError


class OpenCVCam(Camera):


    @staticmethod
    def arg_restrictions():

        return {'device' : [0, 1]}


    def __init__(self, device=-1, file='', exposure=0, rotate=0, crop=[], fps=100, display=True):

        if device != -1:
            if file:
                raise DLCLiveCameraError("A device and file were provided to OpenCVCam. Must initialize an OpenCVCam with either a device id or a video file.")

            self.cap = cv2.VideoCapture(int(device))
            id = int(device)

        else:
            if not file:
                raise DLCLiveCameraError("Neither a device nor file were provided to OpenCVCam. Must initialize an OpenCVCam with either a device id or a video file.")

            self.cap = cv2.VideoCapture(file)
            id = file

        self.im_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.crop = None
        self.rotate = None
        self.display = display

        fps = fps if fps is not None else self.cap.get(cv2.CAP_PROP_FPS)
        super().__init__(int(device), exposure=exposure, rotate=rotate, crop=crop, fps=fps)


    def set_exposure(self, val):

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, val)


    def get_exposure(self):

        return self.cap.get(cv2.CAP_PROP_EXPOSURE)


    def set_crop(self, crop):

        if crop:
            self.crop = crop
            self.im_size = (crop[3]-crop[2], crop[1]-crop[0])

    def set_rotation(self, rotate):

        if rotate:
            self.rotate = rotate


    def set_fps(self, fps):

        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)


    def get_image(self):

        ret, frame = self.cap.read()

        if ret:
            if self.rotate:
                frame = rotate_bound(frame, self.rotate)
            if self.crop:
                frame = frame[self.crop[2]:self.crop[3], self.crop[0]:self.crop[1]]

        if self.display:
            self.display_frame(frame)

        return frame


    def display_frame(self, frame):

        cv2.imshow('OpenCVCam', frame)
        cv2.waitKey(1)


    def destroy_display(self):

        cv2.destroyWindow('OpenCVCam')


    def open(self):

        if not self.cap.isOpened():
            self.cap.open()


    def close(self):

        super().close()
        if self.cap.isOpened():
            self.cap.release()
