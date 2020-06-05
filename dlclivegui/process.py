"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import multiprocess as mp
import threading
import time
import pandas as pd
import numpy as np

from cameracontrol import CameraProcess
from cameracontrol.queue import ClearableQueue, ClearableMPQueue
from dlclive import DLCLive


class DLCLiveProcessError(Exception):
    """
    Exception for incorrect use of DLC-live-GUI Process Manager
    """
    
    pass


class CameraPoseProcess(CameraProcess):
    """ Camera Process Manager class. Controls image capture, pose estimation and writing images to a video file in a background process. 
    
    Parameters
    ----------
    device : :class:`cameracontrol.Camera`
        a camera object
    q_to_process : :class:`cameracontrol.queue.ClearableQueue`
        a queue for receiving commands from the :class:`cameracontrol.cameracontrolgui`
    q_from_process : :class:`cameracontrol.queue.ClearableQueue`
        a queue for sending commands to an the :class:`cameracontrol.cameracontrolgui`
    ctx : :class:`multiprocess.Context`
        multiprocessing context
    """

    def __init__(self,
                 device,
                 q_to_process,
                 q_from_process,
                 ctx=mp.get_context("spawn")):
        """ Constructor method
        """

        super().__init__(device, q_to_process, q_from_process, ctx)
        self.display_pose_queue = ClearableMPQueue(2, ctx=ctx)
        self.display_pose = None


    def _open_dlc_live(self, dlc_params):

        self.dlc = DLCLive(**dlc_params)
        if self.frame is not None:
            self.dlc.init_inference(self.frame)
            return True
        else:
            return False


    def _pose_loop(self):
        """ Conduct pose estimation using deeplabcut-live in loop
        """

        ltime = time.time()

        while self.pose_capture:

            if self.frame_time > self.pose_frame_time:

                stime = time.time()

                self.pose = self.dlc.get_pose(self.frame)
                self.pose_time = time.time()
                self.pose_frame_time = self.frame_time

                ptime = time.time()

                if self.device.use_tk_display:
                    self.display_pose_queue.write(self.pose)

                if self.dev_write:
                    self.write_poses.append(self.pose)
                    self.write_pose_times.append(self.pose_time)
                    self.write_pose_frame_times.append(self.pose_frame_time)

                wtime = time.time()

                print(f"POSE RATE = {int(1/(wtime-ltime))}")

                ltime = wtime


    def _start_pose_estimation(self):
        """ opens pose estimation thread on background process
        """

        self.pose_capture = True
        self.pose_frame_time = 0

        self.write_poses = []
        self.write_pose_times = []
        self.write_pose_frame_times = []

        self.pose_thread = threading.Thread(target=self._pose_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()


    def _stop_pose_estimation(self):

        self.pose_capture = False
        self.pose_thread.join(1)


    def _save_pose(self, filename=None):
        """ Saves a pandas data frame with pose data collected while recording video
        
        Returns
        -------
        bool
            a logical flag indicating whether save was successful
        """

        if filename is not None:
            dlc_file = f"{filename}.hdf5"
            proc_file = filename
        else:
            dlc_file = f"{self.basename}_DLC.hdf5"
            proc_file = f"{self.basename}_PROC"

        bodyparts = self.dlc.cfg['all_joints_names']
        poses = np.array(self.write_poses)
        poses = poses.reshape((poses.shape[0], poses.shape[1]*poses.shape[2]))
        pdindex = pd.MultiIndex.from_product([bodyparts, ['x', 'y', 'likelihood']], names=['bodyparts', 'coords'])
        pose_df = pd.DataFrame(poses, columns=pdindex)
        pose_df['frame_time'] = self.write_pose_frame_times
        pose_df['pose_time'] = self.write_pose_times

        pose_df.to_hdf(dlc_file, key='df_with_missing', mode='w')
        if self.dlc.processor is not None:
            self.dlc.processor.save(proc_file)

        return True


    def _run_capture(self, q_to_process, q_from_process):

        run = True
        self.dev_open = False
        self.dev_capture = False
        self.write_open = False
        self.dev_write = False
        self.pose_open = False
        self.pose_capture = False

        while run:

            cmd = q_to_process.read()

            if cmd is not None:

                if cmd[0] == "run":

                    if cmd[1] == "close":
                        run = False
                        q_from_process.write(cmd + (True,))

                if cmd[0] == "capture":

                    if cmd[1] == "open":
                        ret = False
                        if not self.dev_open:
                            ret = self.device.set_capture_device()
                        if ret:
                            self.dev_open = True
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                    if cmd[1] == "start":
                        if (self.dev_open) and (not self.dev_capture):
                            self._start_capture()
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                    elif cmd[1] == "stop":
                        if self.dev_capture:
                            self._stop_capture()
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                    elif cmd[1] == "close":
                        if self.dev_open:
                            self.device.close_capture_device()
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                elif cmd[0] == "write":

                    if cmd[1] == "open":
                        if not self.write_open:
                            self._create_writer(cmd[2])
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))
                    
                    if cmd[1] == "start":
                        if (self.write_open) and (not self.dev_write):
                            self._start_write()
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                    elif cmd[1] == "stop":
                        if self.dev_write:
                            self._stop_write()
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                    elif cmd[1] == "save":
                        ret = self._save_video()
                        if (ret) and (self.pose_open):
                            ret = self._save_pose()
                        if ret:
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                    elif cmd[1] == "delete":
                        ret = self._save_video(delete=True)
                        if ret:
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                elif cmd[0] == "pose":

                    if cmd[1] == "open":
                        if not self.pose_capture:
                            ret = self._open_dlc_live(cmd[2])
                            if ret:
                                self.pose_open = True
                                q_from_process.write(cmd + (True,))
                            else:
                                q_from_process.write(cmd + (False,))

                    if cmd[1] == "start":
                        if self.pose_open:
                            self._start_pose_estimation()
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

                    if cmd[1] == "stop":
                        if self.pose_capture:
                            self._stop_pose_estimation()
                            q_from_process.write(cmd + (True,))
                        else:
                            q_from_process.write(cmd + (False,))

        if self.dev_write:
            self._stop_write()
        if self.write_open:
            self._save_video(delete=True)
        if self.pose_capture:
            self._stop_pose_estimation()
        if self.dev_capture:
            self._stop_capture()
        if self.dev_open:
            self.device.close_capture_device()  


    def get_display_pose(self):

        pose = self.display_pose_queue.read(clear=True)
        if pose is not None:
            self.display_pose = pose
            if self.device.display_resize != 1:
                self.display_pose[:, :2] *= self.device.display_resize

        return self.display_pose
