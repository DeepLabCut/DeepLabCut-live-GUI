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
    ctx : :class:`multiprocess.Context`
        multiprocessing context
    """

    def __init__(self,
                 device,
                 ctx=mp.get_context("spawn")):
        """ Constructor method
        """

        super().__init__(device, ctx)

        self.display_pose_queue = ClearableMPQueue(2, ctx=ctx)

        self.pose_process = ctx.Process(target=self._run_pose,
                                        args=(self.frame_shared, self.frame_time),
                                        daemon=True)

                                    
    def start_procs(self):

        super().start_procs()
        self.pose_process.start()


    def stop_procs(self, force=False, timeout=5):

        cap, write = super().stop_procs(force, timeout)

        if not force:
            pose_ret = self.execute_command(("pose", "close"))

            if pose_ret:
                self.pose_process.join(timeout)
        
        else:

            self.pose_process.terminate()

        return cap, write, self.pose_process.exitcode


    def _open_dlc_live(self, dlc_params):

        ret = False

        if not self.pose_open:
            self.dlc = DLCLive(**dlc_params)
            if self.frame is not None:
                self.dlc.init_inference(self.frame)
                self.pose_open = True

        return self.pose_open


    def _pose_loop(self):
        """ Conduct pose estimation using deeplabcut-live in loop
        """

        ltime = time.time()

        while self.pose_on:

            if self.frame_time.value > self.pose_frame_time:

                stime = time.time()

                self.pose = self.dlc.get_pose(self.frame)
                self.pose_time = time.time()
                self.pose_frame_time = self.frame_time.value

                ptime = time.time()

                if self.device.use_tk_display:
                    self.display_pose_queue.write(self.pose)

                if self.pose_write:
                    self.write_poses.append(self.pose)
                    self.write_pose_times.append(self.pose_time)
                    self.write_pose_frame_times.append(self.pose_frame_time)

                wtime = time.time()

                print(f"POSE RATE = {int(1/(wtime-ltime))}")

                ltime = wtime


    def _start_pose_estimation(self):
        """ opens pose estimation thread on background process
        """

        if self.pose_open and (not self.pose_on):

            self.pose_on = True
            self.pose_frame_time = 0

            self.write_poses = []
            self.write_pose_times = []
            self.write_pose_frame_times = []

            self.pose_thread = threading.Thread(target=self._pose_loop)
            self.pose_thread.daemon = True
            self.pose_thread.start()

        return self.pose_on

    
    def _start_pose_write(self):

        if (self.pose_on) and (not self.pose_write):
            self.pose_write = True
        return self.pose_write


    def _stop_pose_write(self):

        if self.pose_write:
            self.pose_write = False
        return self.pose_write


    def _stop_pose_estimation(self):

        if self.pose_on:
            self.pose_on = False
            self.pose_thread.join(5)

        return not self.pose_on


    def _save_pose(self, filename):
        """ Saves a pandas data frame with pose data collected while recording video
        
        Returns
        -------
        bool
            a logical flag indicating whether save was successful
        """
        
        ret = False

        if (self.pose_on) and (not self.pose_write):

            if len(self.write_pose_times) > 0:

                dlc_file = f"{filename}_DLC.hdf5"
                proc_file = f"{filename}_PROC"

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

                ret = True

        return ret


    def _run_pose(self, frame_shared, frame_time):

        res = self.device.im_size
        self.frame = np.frombuffer(frame_shared.get_obj(), dtype='uint8').reshape(res[1], res[0], 3)
        self.frame_time = frame_time

        run = True
        self.pose_open = False
        self.pose_on = False
        self.pose_write = False

        while run:

            cmd = self.q_to_process.read()

            if cmd is not None:

                if cmd[0] == "pose":

                    if cmd[1] == "open":
                        ret = self._open_dlc_live(cmd[2])
                        self.q_from_process.write(cmd + (ret,))

                    if cmd[1] == "start":
                        ret = self._start_pose_estimation()
                        self.q_from_process.write(cmd + (ret,))

                    if cmd[1] == "write":
                        ret = self._start_pose_write()
                        self.q_from_process.write(cmd + (ret,))

                    if cmd[1] == "stop":
                        ret = self._stop_pose_write()
                        self.q_from_process.write(cmd + (ret,))

                    if cmd[1] == "end":
                        ret = self._stop_pose_estimation()
                        self.q_from_process.write(cmd + (ret,))

                    if cmd[1] == "save":
                        ret = self._save_pose(cmd[2])
                        self.q_from_process.write(cmd + (ret,))

                    if cmd[1] == "close":
                        if not self.pose_on:
                            run = False
                        self.q_from_process.write(cmd + (not run,))
                
                else:

                    self.q_to_process.write(cmd)


    def get_display_pose(self):

        pose = self.display_pose_queue.read(clear=True)
        if pose is not None:
            self.display_pose = pose
            if self.device.display_resize != 1:
                self.display_pose[:, :2] *= self.device.display_resize

        return self.display_pose
