"""
Camera Control
Copyright M. Mathis Lab
Written by  Gary Kane - https://github.com/gkane26
post-doctoral fellow @ the Adaptive Motor Control Lab
https://github.com/AdaptiveMotorControlLab

Default video writer class
"""

import time
import threading
import cv2

class CV2Video(object):

    def __init__(self, camera, file_name):

        self.camera = camera
        self.file_name = file_name
        self.record = False

        ### create video writer ###
        self.writer = cv2.VideoWriter(self.file_name, cv2.VideoWriter_fourcc(*'DIVX'), int(self.cam.fps)), self.cam.get_image_dimensions())

    def create_writer(self, file_nam)
        self.capture = False
        self.record = False
        self.frame = None
        self.new_frame = False
        self.frame_time = []
        self.frames_to_write = []
        self.fps = 100

    def set_exposure(self, val):
        pass

    def set_crop(self, val):
        pass

    def set_rotation(self, val):
        pass

    def set_fps(self, val):
        self.fps = val

    def get_image(self):
        pass

    def get_image_dimensions(self):
        raise NotImplementedError

    def open(self):
        pass

    def display_on_thread(self):
        while self.display:
            if self.new_frame:
                cv2.imshow('DLC Live Video', self.frame)
                cv2.waitKey(1)
                self.new_frame = False

    def start_display(self):
        self.display = True
        cv2.namedWindow('DLC Live Video', cv2.WINDOW_NORMAL)
        im_size = self.get_image_dimensions()
        cv2.resizeWindow('DLC Live Video', im_size[0], im_size[1])
        threading.Thread(target=self.display_on_thread).start()

    def stop_display(self):
        self.display = False
        cv2.destroyWindow('DLC Live Video')

    def capture_on_thread(self):
        next_frame = time.time()
        while self.capture:
            cur_time = time.time()
            if cur_time >= next_frame:
                self.frame_time.append(cur_time)
                self.frame = self.get_image()
                self.new_frame = True
                if self.record:
                    self.frames_to_write.append(self.frame)
                next_frame = max(next_frame + 1.0/self.fps, self.frame_time[-1] + .5/self.fps)

    def start_capture(self):
        self.capture = True
        threading.Thread(target=self.capture_on_thread).start()

    def start_record(self):
        self.record = True

    def stop_capture(self):
        self.capture = False

    def stop_record(self):
        self.record = False

    def close(self):
        if self.record:
            self.stop_record()
        if self.capture:
            self.stop_capture()
        if self.display:
            self.stop_display()
