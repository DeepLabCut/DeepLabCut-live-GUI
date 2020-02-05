"""
reads video frame by frame to test live DLC

"""

from . import Camera
import cv2
from tkinter import filedialog

class VideoFeed(Camera):

    def __init__(self, file='', display=False):

        file = file if file else filedialog.askopenfilename(title="Select Video File for DLC Live")
        self.cap = cv2.VideoCapture(file)
        super().__init__(fps=self.cap.get(cv2.CAP_PROP_FPS))
        self.im_size = (int(self.cap.get(3)), int(self.cap.get(4)))
        self.display = display
        if self.display:
            cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video Feed', self.im_size[0], self.im_size[1])

    def get_image(self):
        ret, frame = self.cap.read()

        if ret:
            if self.display:
                cv2.imshow('Video Feed', frame)
                cv2.waitKey(1)
            return frame
        else:
            raise Exception('no more frames to read!')

    def open(self):
        pass

    def close(self):
        super().close()
        if self.display:
            cv2.destroyAllWindows()
