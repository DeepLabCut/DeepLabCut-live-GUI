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

        self.write_pose_queue = ClearableMPQueue(ctx=self.ctx)
        self.display_pose_queue = ClearableMPQueue(2, ctx=self.ctx)
        self.pose_process = None


    def start_pose_process(self, dlc_params, timeout=300):

        self.pose_process = self.ctx.Process(target=self._run_pose,
                                             args=(self.frame_shared, self.frame_time, dlc_params),
                                             daemon=True)
        self.pose_process.start()

        stime = time.time()
        while time.time()-stime < timeout:
            cmd = self.q_from_process.read()
            if cmd is not None:
                if (cmd[0] == "pose") and (cmd[1] == "start"):
                    return cmd[2]
                else:
                    self.q_to_process.write(cmd)


    def _run_pose(self, frame_shared, frame_time, dlc_params):

        res = self.device.im_size
        self.frame = np.frombuffer(frame_shared.get_obj(), dtype='uint8').reshape(res[1], res[0], 3)
        self.frame_time = frame_time

        ret = self._open_dlc_live(dlc_params)
        self.q_from_process.write(("pose", "start", ret))

        self._pose_loop()
        self.q_from_process.write(("pose", "end"))


    def _open_dlc_live(self, dlc_params):

        ret = False
        self.dlc = DLCLive(**dlc_params)
        if self.frame is not None:
            self.dlc.init_inference(self.frame)
            ret = True

        return ret


    def _pose_loop(self):
        """ Conduct pose estimation using deeplabcut-live in loop
        """

        run = True
        write = False
        pose_frame_time = 0
        ltime = time.time()

        while run:

            if self.frame_time.value > pose_frame_time:

                stime = time.time()

                pose = self.dlc.get_pose(self.frame)
                pose_time = time.time()
                pose_frame_time = self.frame_time.value

                ptime = time.time()

                if self.device.use_tk_display:
                    self.display_pose_queue.write(pose)

                if write:
                    self.write_pose_queue((pose, pose_time, pose_frame_time))

                wtime = time.time()

                print(f"POSE RATE = {int(1/(wtime-ltime))} / GET POSE = {ptime-stime:0.6f} / WRITE TIME = {wtime-ptime:0.6f}")

                ltime = wtime

                cmd = self.q_to_process.read()
                if cmd is not None:
                    if cmd[0] == "pose":
                        if cmd[1] == "write":
                            write = cmd[2]
                        elif cmd[1] == "end":
                            run = False
                    else:
                        self.q_to_process.write(cmd)

    
    def start_record(self):

        ret = super().start_record()

        if self.pose_process.is_alive() and self.writer_process.is_alive():
            self.q_to_process.write(("pose", "write", True))
            
            stime = time.time()
            while time.time()-stime < timeout:
                cmd = self.q_from_process.read()
                if cmd is not None:
                    if (cmd[0] == "pose") and (cmd[1] == "write"):
                        ret = cmd[2]
                        break
                    else:
                        self.q_from_process.write(cmd)

        return ret


    def stop_record(self):

        ret = super().start_record()

        if self.pose_process.is_alive() and self.writer_process.is_alive():
            self.q_to_process.write(("pose", "write", False)) 

            stime = time.time()
            while time.time()-stime < timeout:
                cmd = self.q_from_process.read()
                if cmd is not None:
                    if (cmd[0] == "pose") and (cmd[1] == "write"):
                        ret = not cmd[2]
                        break
                    else:
                        self.q_from_process.write(cmd)
                        
        return ret


    def stop_pose_process(self):

        ret = True
        if self.pose_process is not None:
            if self.pose_process.is_alive():
                self.q_to_process.write(("pose", "end"))

                while True:
                    cmd = self.q_from_process.read()
                    if cmd is not None:
                        if cmd[0] == "pose":
                            if cmd[1] == "end":
                                break
                        else:
                            self.q_from_process.write(cmd)
                    
                self.pose_process.join(5)
                if self.pose_process.is_alive():
                    self.pose_process.terminate()

        return True


    # def _start_pose_estimation(self):
    #     """ opens pose estimation thread on background process
    #     """

    #     if self.pose_open and (not self.pose_on):

    #         self.pose_on = True
    #         self.pose_frame_time = 0

    #         self.write_poses = []
    #         self.write_pose_times = []
    #         self.write_pose_frame_times = []

    #         self.pose_thread = threading.Thread(target=self._pose_loop)
    #         self.pose_thread.daemon = True
    #         self.pose_thread.start()

    #     return self.pose_on

    
    # def _start_pose_write(self):

    #     if (self.pose_on) and (not self.pose_write):
    #         self.pose_write = True
    #     return self.pose_write


    # def _stop_pose_write(self):

    #     if self.pose_write:
    #         self.pose_write = False
    #     return self.pose_write


    # def _stop_pose_estimation(self):

    #     if self.pose_on:
    #         self.pose_on = False
    #         self.pose_thread.join(5)

    #     return not self.pose_on

    
    def _save_video(self):

        ret = super()._save_video()
        if ret:
            ret = self._save_pose()
        return ret


    def _save_pose(self):
        """ Saves a pandas data frame with pose data collected while recording video
        
        Returns
        -------
        bool
            a logical flag indicating whether save was successful
        """
        
        ret = False

        poses = []
        pose_times = []
        pose_frame_times = []
        pq = self.write_pose_queue.read(clear=True, position='all')
        if pq is not None:
            for i in pq:
                poses.append(pq[0])
                pose_times.append(pq[1])
                pose_frame_times.append(pq[2])
            
        if len(pose_times) > 0:

            dlc_file = f"{self.filename}_DLC.hdf5"
            proc_file = f"{self.filename}_PROC"

            bodyparts = self.dlc.cfg['all_joints_names']
            poses = np.array(poses)
            poses = poses.reshape((poses.shape[0], poses.shape[1]*poses.shape[2]))
            pdindex = pd.MultiIndex.from_product([bodyparts, ['x', 'y', 'likelihood']], names=['bodyparts', 'coords'])
            pose_df = pd.DataFrame(poses, columns=pdindex)
            pose_df['frame_time'] = pose_frame_times
            pose_df['pose_time'] = pose_times

            pose_df.to_hdf(dlc_file, key='df_with_missing', mode='w')
            if self.dlc.processor is not None:
                self.dlc.processor.save(proc_file)

            ret = True

        return ret


    def get_display_pose(self):

        pose = self.display_pose_queue.read(clear=True)
        if pose is not None:
            self.display_pose = pose
            if self.device.display_resize != 1:
                self.display_pose[:, :2] *= self.device.display_resize

        return self.display_pose
