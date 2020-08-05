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

from dlclivegui import CameraProcess
from dlclivegui.queue import ClearableQueue, ClearableMPQueue


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

    def __init__(self, device, ctx=mp.get_context("spawn")):
        """ Constructor method
        """

        super().__init__(device, ctx)
        self.display_pose = None
        self.display_pose_queue = ClearableMPQueue(2, ctx=self.ctx)
        self.pose_process = None

    def start_pose_process(self, dlc_params, timeout=300):

        self.pose_process = self.ctx.Process(
            target=self._run_pose,
            args=(self.frame_shared, self.frame_time_shared, dlc_params),
            daemon=True,
        )
        self.pose_process.start()

        stime = time.time()
        while time.time() - stime < timeout:
            cmd = self.q_from_process.read()
            if cmd is not None:
                if (cmd[0] == "pose") and (cmd[1] == "start"):
                    return cmd[2]
                else:
                    self.q_to_process.write(cmd)

    def _run_pose(self, frame_shared, frame_time, dlc_params):

        res = self.device.im_size
        self.frame = np.frombuffer(frame_shared.get_obj(), dtype="uint8").reshape(
            res[1], res[0], 3
        )
        self.frame_time = np.frombuffer(frame_time.get_obj(), dtype="d")

        ret = self._open_dlc_live(dlc_params)
        self.q_from_process.write(("pose", "start", ret))

        self._pose_loop()
        self.q_from_process.write(("pose", "end"))

    def _open_dlc_live(self, dlc_params):

        from dlclive import DLCLive

        ret = False

        self.opt_rate = True if dlc_params.pop("mode") == "Optimize Rate" else False

        proc_params = dlc_params.pop("processor")
        if proc_params is not None:
            proc_obj = proc_params.pop("object", None)
            if proc_obj is not None:
                dlc_params["processor"] = proc_obj(**proc_params)

        self.dlc = DLCLive(**dlc_params)
        if self.frame is not None:
            self.dlc.init_inference(
                self.frame, frame_time=self.frame_time[0], record=False
            )
            self.poses = []
            self.pose_times = []
            self.pose_frame_times = []
            ret = True

        return ret

    def _pose_loop(self):
        """ Conduct pose estimation using deeplabcut-live in loop
        """

        run = True
        write = False
        frame_time = 0
        pose_time = 0
        end_time = time.time()

        while run:

            ref_time = frame_time if self.opt_rate else end_time

            if self.frame_time[0] > ref_time:

                frame = self.frame
                frame_time = self.frame_time[0]
                pose = self.dlc.get_pose(frame, frame_time=frame_time, record=write)
                pose_time = time.time()

                self.display_pose_queue.write(pose, clear=True)

                if write:
                    self.poses.append(pose)
                    self.pose_times.append(pose_time)
                    self.pose_frame_times.append(frame_time)

                cmd = self.q_to_process.read()
                if cmd is not None:
                    if cmd[0] == "pose":
                        if cmd[1] == "write":
                            write = cmd[2]
                            self.q_from_process.write(cmd)
                        elif cmd[1] == "save":
                            ret = self._save_pose(cmd[2])
                            self.q_from_process.write(cmd + (ret,))
                        elif cmd[1] == "end":
                            run = False
                    else:
                        self.q_to_process.write(cmd)

    def start_record(self, timeout=5):

        ret = super().start_record(timeout=timeout)

        if (self.pose_process is not None) and (self.writer_process is not None):
            if (self.pose_process.is_alive()) and (self.writer_process.is_alive()):
                self.q_to_process.write(("pose", "write", True))

                stime = time.time()
                while time.time() - stime < timeout:
                    cmd = self.q_from_process.read()
                    if cmd is not None:
                        if (cmd[0] == "pose") and (cmd[1] == "write"):
                            ret = cmd[2]
                            break
                        else:
                            self.q_from_process.write(cmd)

        return ret

    def stop_record(self, timeout=5):

        ret = super().stop_record(timeout=timeout)

        if (self.pose_process is not None) and (self.writer_process is not None):
            if (self.pose_process.is_alive()) and (self.writer_process.is_alive()):
                self.q_to_process.write(("pose", "write", False))

                stime = time.time()
                while time.time() - stime < timeout:
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

    def save_pose(self, filename, timeout=60):

        ret = False
        if self.pose_process is not None:
            if self.pose_process.is_alive():
                self.q_to_process.write(("pose", "save", filename))

                stime = time.time()
                while time.time() - stime < timeout:
                    cmd = self.q_from_process.read()
                    if cmd is not None:
                        if (cmd[0] == "pose") and (cmd[1] == "save"):
                            ret = cmd[3]
                            break
                        else:
                            self.q_from_process.write(cmd)
        return ret

    def _save_pose(self, filename):
        """ Saves a pandas data frame with pose data collected while recording video

        Returns
        -------
        bool
            a logical flag indicating whether save was successful
        """

        ret = False

        if len(self.pose_times) > 0:

            dlc_file = f"{filename}_DLC.hdf5"
            proc_file = f"{filename}_PROC"

            bodyparts = self.dlc.cfg["all_joints_names"]
            poses = np.array(self.poses)
            poses = poses.reshape((poses.shape[0], poses.shape[1] * poses.shape[2]))
            pdindex = pd.MultiIndex.from_product(
                [bodyparts, ["x", "y", "likelihood"]], names=["bodyparts", "coords"]
            )
            pose_df = pd.DataFrame(poses, columns=pdindex)
            pose_df["frame_time"] = self.pose_frame_times
            pose_df["pose_time"] = self.pose_times

            pose_df.to_hdf(dlc_file, key="df_with_missing", mode="w")
            if self.dlc.processor is not None:
                self.dlc.processor.save(proc_file)

            self.poses = []
            self.pose_times = []
            self.pose_frame_times = []

            ret = True

        return ret

    def get_display_pose(self):

        pose = self.display_pose_queue.read(clear=True)
        if pose is not None:
            self.display_pose = pose
            if self.device.display_resize != 1:
                self.display_pose[:, :2] *= self.device.display_resize

        return self.display_pose
