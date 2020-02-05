"""
Camera Control
Copyright M. Mathis Lab
Written by  Gary Kane - https://github.com/gkane26
post-doctoral fellow @ the Adaptive Motor Control Lab
https://github.com/AdaptiveMotorControlLab

Base camera class. Performs recording, pose estimation, and display.
Template with additional methods to be implemented by child camera objects.
"""

import time
import threading
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue


class Camera:
    def __init__(self, exposure=None, rotate=None, crop=None, fps=100):

        self.capture = False
        self.record = False
        self.estimate_pose = False
        self.dlc_live = None
        self.reset()

        self.set_fps(fps)
        if exposure:
            self.set_exposure(exposure)
        if rotate:
            self.set_rotation(rotate)
        if crop:
            self.set_crop(crop['top'], crop['left'], crop['height'], crop['width'])

    def reset(self):
        self.video_writer = None
        self.frame = None
        self.new_frame = False

        self.frame_time = 0
        self.frame_times_record = []
        self.frames_to_write = Queue()

        self.pose_times = []
        self.pose_frame_time = 0
        self.pose_frame_times_record = []
        self.pose = None
        if hasattr(self, 'poses'):
            self.poses = np.zeros((0, len(self.dlc_live.cfg['bodyparts']), 3))

    def set_exposure(self, val):
        raise NotImplementedError

    def set_crop(self, val):
        raise NotImplementedError

    def set_rotation(self, val):
        raise NotImplementedError

    def set_fps(self, val):
        self.fps = val

    def get_image(self):
        raise NotImplementedError

    def open(self):
        raise NotImplementedError

    def display_on_thread(self):
        while self.display:
            if self.new_frame and self.estimate_pose and self.pose is not None:
                self.canvas.restore_region(self.background)
                self.im.set_data(self.frame)
                pose = self.pose.copy()
                mask = pose[:, 2] > self.dlc_live.cfg["pcutoff"]
                pose[~mask] = np.nan
                self.scat.set_offsets(pose[:, :2])
                self.canvas.blit(self.ax.bbox)
                self.new_frame = False

    def start_display(self):
        self.display = True
        dpi = 100
        w = self.im_size[0] / dpi
        h = self.im_size[1] / dpi
        self.fig = plt.Figure(frameon=False, figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        img = np.empty((self.im_size[0], self.im_size[1], 3))
        self.im = self.ax.imshow(img)
        self.scat = self.ax.scatter([], [], s=self.dlc_live.cfg["dotsize"] ** 2, alpha=0.7)
        self.scat.set_color(self.colors)
        self.canvas = FigureCanvasTkAgg(self.fig)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        thr = threading.Thread(target=self.display_on_thread)
        thr.daemon = True
        thr.start()

    def stop_display(self):
        self.display = False
        cv2.destroyWindow('DLC Live Video')

    def capture_on_thread(self):
        next_frame = time.time()
        while self.capture:
            cur_time = time.time()
            if cur_time > next_frame:
                self.frame_time = cur_time
                self.frame = self.get_image()
                self.new_frame = True
                if self.record:
                    self.frame_times_record.append(self.frame_time)
                    self.frames_to_write.put(self.frame)
                next_frame = max(next_frame+1.0/self.fps, cur_time+.5/self.fps)

    def start_capture(self):
        self.capture = True
        thr = threading.Thread(target=self.capture_on_thread)
        thr.daemon = True
        thr.start()

    def stop_capture(self):
        self.capture = False

    def create_video_writer(self, filename):
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.im_size)

    def close_video(self):
        self.stop_record()
        self.video_writer.release()
        self.reset()

    def record_on_thread(self):
        while self.record:
            self.video_writer.write(self.frames_to_write.get())

    def start_record(self):
        if self.video_writer:
            self.record = True
            thr = threading.Thread(target=self.record_on_thread)
            thr.daemon = True
            thr.start()
        return self.record

    def stop_record(self):
        self.record = False

    def pose_on_thread(self):
        last_pose = time.time()
        while self.estimate_pose:
            if self.new_frame:
                print(time.time()-last_pose)
                last_pose = time.time()
                self.new_frame = False
                self.pose_frame_time = self.frame_time
                self.pose = self.dlc_live.get_pose(self.frame)
                if self.record:
                    self.poses = np.vstack((self.poses, self.pose.reshape((1,)+self.pose.shape)))
                    self.pose_frame_times_record.append(self.pose_frame_time)
                    self.pose_times.append(time.time())

    def start_pose_estimation(self, dlc_live):
        self.dlc_live = dlc_live

        # initialize pose array
        self.poses = np.zeros((0, len(self.dlc_live.cfg['bodyparts']), 3))

        # init colors for display
        colorclass = plt.cm.ScalarMappable(cmap=self.dlc_live.cfg["colormap"])
        C = colorclass.to_rgba(np.linspace(0,1,len(self.dlc_live.cfg['bodyparts'])))
        self.colors = (C[:,:3]*255).astype(np.uint8)

        # start pose estimation
        self.estimate_pose = True
        thr = threading.Thread(target=self.pose_on_thread)
        thr.daemon = True
        thr.start()

    def stop_pose_estimation(self):
        self.estimate_pose = False

    def get_pose_df(self):
        pose_np = self.poses.reshape((self.poses.shape[0], self.poses.shape[1]*self.poses.shape[2]))
        pdindex = pd.MultiIndex.from_product([self.dlc_live.cfg['bodyparts'], ['x', 'y', 'likelihood'], ['frame_times', 'pose_times']], names=['bodyparts', 'coords'])
        pose_df = pd.DataFrame(pose_np, columns=pdindex)
        pose_df['frame_time'] = self.pose_frame_times_record
        pose_df['pose_time'] = self.pose_times
        return pose_df

    def close(self):
        if self.estimate_pose:
            self.stop_pose_estimation()
        if self.record:
            self.close_video()
        if self.capture:
            self.stop_capture()
        # if self.display:
        #     self.stop_display()
