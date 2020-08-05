"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import time
import multiprocess as mp
import ctypes
from dlclivegui.queue import ClearableQueue, ClearableMPQueue
import threading
import cv2
import numpy as np
import os


class CameraProcessError(Exception):
    """
    Exception for incorrect use of Cameras
    """

    pass


class CameraProcess(object):
    """ Camera Process Manager class. Controls image capture and writing images to a video file in a background process. 
    
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

        self.device = device
        self.ctx = ctx

        res = self.device.im_size
        self.frame_shared = mp.Array(ctypes.c_uint8, res[1] * res[0] * 3)
        self.frame = np.frombuffer(self.frame_shared.get_obj(), dtype="uint8").reshape(
            res[1], res[0], 3
        )
        self.frame_time_shared = mp.Array(ctypes.c_double, 1)
        self.frame_time = np.frombuffer(self.frame_time_shared.get_obj(), dtype="d")

        self.q_to_process = ClearableMPQueue(ctx=self.ctx)
        self.q_from_process = ClearableMPQueue(ctx=self.ctx)
        self.write_frame_queue = ClearableMPQueue(ctx=self.ctx)

        self.capture_process = None
        self.writer_process = None

    def start_capture_process(self, timeout=60):

        cmds = self.q_to_process.read(clear=True, position="all")
        if cmds is not None:
            for c in cmds:
                if c[1] != "capture":
                    self.q_to_process.write(c)

        self.capture_process = self.ctx.Process(
            target=self._run_capture,
            args=(self.frame_shared, self.frame_time_shared),
            daemon=True,
        )
        self.capture_process.start()

        stime = time.time()
        while time.time() - stime < timeout:
            cmd = self.q_from_process.read()
            if cmd is not None:
                if (cmd[0] == "capture") and (cmd[1] == "start"):
                    return cmd[2]
                else:
                    self.q_to_process.write(cmd)

        return True

    def _run_capture(self, frame_shared, frame_time):

        res = self.device.im_size
        self.frame = np.frombuffer(frame_shared.get_obj(), dtype="uint8").reshape(
            res[1], res[0], 3
        )
        self.frame_time = np.frombuffer(frame_time.get_obj(), dtype="d")

        ret = self.device.set_capture_device()
        if not ret:
            raise CameraProcessError("Could not start capture device.")
        self.q_from_process.write(("capture", "start", ret))

        self._capture_loop()

        self.device.close_capture_device()
        self.q_from_process.write(("capture", "end", True))

    def _capture_loop(self):
        """ Acquires frames from frame capture device in a loop
        """

        run = True
        write = False
        last_frame_time = time.time()

        while run:

            start_capture = time.time()

            frame, frame_time = self.device.get_image_on_time()

            write_capture = time.time()

            np.copyto(self.frame, frame)
            self.frame_time[0] = frame_time

            if write:
                ret = self.write_frame_queue.write((frame, frame_time))

            end_capture = time.time()

            # print("read frame = %0.6f // write to queues = %0.6f" % (write_capture-start_capture, end_capture-write_capture))
            # print("capture rate = %d" % (int(1 / (time.time()-last_frame_time))))
            # print("\n")

            last_frame_time = time.time()

            ### read commands
            cmd = self.q_to_process.read()
            if cmd is not None:
                if cmd[0] == "capture":
                    if cmd[1] == "write":
                        write = cmd[2]
                        self.q_from_process.write(cmd)
                    elif cmd[1] == "end":
                        run = False
                else:
                    self.q_to_process.write(cmd)

    def stop_capture_process(self):

        ret = True
        if self.capture_process is not None:
            if self.capture_process.is_alive():
                self.q_to_process.write(("capture", "end"))

                while True:
                    cmd = self.q_from_process.read()
                    if cmd is not None:
                        if cmd[0] == "capture":
                            if cmd[1] == "end":
                                break
                        else:
                            self.q_from_process.write(cmd)

                self.capture_process.join(5)
                if self.capture_process.is_alive():
                    self.capture_process.terminate()

        return True

    def start_writer_process(self, filename, timeout=60):

        cmds = self.q_to_process.read(clear=True, position="all")
        if cmds is not None:
            for c in cmds:
                if c[1] != "writer":
                    self.q_to_process.write(c)

        self.writer_process = self.ctx.Process(
            target=self._run_writer, args=(filename,), daemon=True
        )
        self.writer_process.start()

        stime = time.time()
        while time.time() - stime < timeout:
            cmd = self.q_from_process.read()
            if cmd is not None:
                if (cmd[0] == "writer") and (cmd[1] == "start"):
                    return cmd[2]
                else:
                    self.q_to_process.write(cmd)

        return True

    def _run_writer(self, filename):

        ret = self._create_writer(filename)
        self.q_from_process.write(("writer", "start", ret))

        save = self._write_loop()

        ret = self._save_video(not save)
        self.q_from_process.write(("writer", "end", ret))

    def _create_writer(self, filename):

        self.filename = filename
        self.video_file = f"{self.filename}_VIDEO.avi"
        self.timestamp_file = f"{self.filename}_TS.npy"

        self.video_writer = cv2.VideoWriter(
            self.video_file,
            cv2.VideoWriter_fourcc(*"DIVX"),
            self.device.fps,
            self.device.im_size,
        )
        self.write_frame_ts = []

        return True

    def _write_loop(self):
        """ read frames from write_frame_queue and write to file
        """

        run = True
        new_frame = None

        while run or (new_frame is not None):

            new_frame = self.write_frame_queue.read()
            if new_frame is not None:
                frame, ts = new_frame
                if frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                self.video_writer.write(frame)
                self.write_frame_ts.append(ts)

            cmd = self.q_to_process.read()
            if cmd is not None:
                if cmd[0] == "writer":
                    if cmd[1] == "end":
                        run = False
                        save = cmd[2]
                else:
                    self.q_to_process.write(cmd)

        return save

    def _save_video(self, delete=False):

        ret = False

        self.video_writer.release()

        if (not delete) and (len(self.write_frame_ts) > 0):
            np.save(self.timestamp_file, self.write_frame_ts)
            ret = True
        else:
            os.remove(self.video_file)
            if os.path.isfile(self.timestamp_file):
                os.remove(self.timestamp_file)

        return ret

    def stop_writer_process(self, save=True):

        ret = False
        if self.writer_process is not None:
            if self.writer_process.is_alive():
                self.q_to_process.write(("writer", "end", save))

                while True:
                    cmd = self.q_from_process.read()
                    if cmd is not None:
                        if cmd[0] == "writer":
                            if cmd[1] == "end":
                                ret = cmd[2]
                                break
                        else:
                            self.q_from_process.write(cmd)

                self.writer_process.join(5)
                if self.writer_process.is_alive():
                    self.writer_process.terminate()

        return ret

    def start_record(self, timeout=5):

        ret = False

        if (self.capture_process is not None) and (self.writer_process is not None):
            if self.capture_process.is_alive() and self.writer_process.is_alive():
                self.q_to_process.write(("capture", "write", True))

                stime = time.time()
                while time.time() - stime < timeout:
                    cmd = self.q_from_process.read()
                    if cmd is not None:
                        if (cmd[0] == "capture") and (cmd[1] == "write"):
                            ret = cmd[2]
                            break
                        else:
                            self.q_from_process.write(cmd)

        return ret

    def stop_record(self, timeout=5):

        ret = False

        if (self.capture_process is not None) and (self.writer_process is not None):
            if (self.capture_process.is_alive()) and (self.writer_process.is_alive()):
                self.q_to_process.write(("capture", "write", False))

                stime = time.time()
                while time.time() - stime < timeout:
                    cmd = self.q_from_process.read()
                    if cmd is not None:
                        if (cmd[0] == "capture") and (cmd[1] == "write"):
                            ret = not cmd[2]
                            break
                        else:
                            self.q_from_process.write(cmd)

        return ret

    def get_display_frame(self):

        frame = self.frame.copy()
        if frame is not None:
            if self.device.display_resize != 1:
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1] * self.device.display_resize),
                        int(frame.shape[0] * self.device.display_resize),
                    ),
                )

        return frame
