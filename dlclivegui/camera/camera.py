"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

"""
Base camera class. Performs recording, pose estimation, and display.
Template with additional methods to be implemented by child camera objects.
"""

import time
import threading
import cv2
from skimage.draw import circle
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from queue import Queue


class DLCLiveCameraError(Exception):
    ''' Exception for incorrect use of DLC-live-GUI Cameras '''
    pass


class Camera(object):


    @staticmethod
    def arg_restrictions():

        return {}


    def __init__(self, id, exposure=0, rotate=0, crop=[], fps=100):

        self.id = id
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
            self.set_crop(crop)

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

    # def display_on_thread(self):
    #     while self.display:
    #         if self.new_frame:
    #             this_frame = np.copy(self.frame)
    #             if self.estimate_pose:
    #                 if self.pose is not None:
    #                     try:
    #                         for i in range(self.pose.shape[0]):
    #                             if self.pose[i,2] > self.dlc_live.cfg["pcutoff"]:
    #                                 rr, cc = circle(self.pose[i,1], self.pose[i,0], self.dlc_live.cfg["dotsize"], shape=self.im_size)
    #                                 rr[rr > self.im_size[0]] = self.im_size[0]
    #                                 cc[cc > self.im_size[1]] = self.im_size[1]
    #                                 this_frame[rr, cc, :] = self.colors[i]
    #                     except Exception as e:
    #                         pass
    #             cv2.imshow('DLC Live Video', this_frame)
    #             cv2.waitKey(1)
    #             self.new_frame = False

    def start_display(self):
        pass
        # self.display = True
        # cv2.namedWindow('DLC Live Video', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('DLC Live Video', self.im_size[0], self.im_size[1])
        # thr = threading.Thread(target=self.display_on_thread)
        # thr.daemon = True
        # thr.start()

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

        # # init colors for display
        # colorclass = plt.cm.ScalarMappable(cmap=self.dlc_live.cfg["colormap"])
        # C = colorclass.to_rgba(np.linspace(0,1,len(self.dlc_live.cfg['bodyparts'])))
        # self.colors = (C[:,:3]*255).astype(np.uint8)

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
