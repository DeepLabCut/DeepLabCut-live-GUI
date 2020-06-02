"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import time
import multiprocess as mp
from queue import Full
try:
    from _queue import Empty
except ModuleNotFoundError:
    from queue import Empty

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from skimage.draw import circle

from dlclive import DLCLive


class DLCLiveCameraError(Exception):
    """
    Exception for incorrect use of DLC-live-GUI Cameras
    """
    
    pass


def qwrite(queue, lock, obj, clear=False, max_attempt=10):
    """ Puts an object in the queue under a lock, with an option to clear queue before writing.

    
    Parameters
    ----------
    queue : :class:`mp.Queue`
        
    lock : :class:mp.Lock`
        
    obj : [type]
        An object to put in the queue
    clear : bool, optional
        flag to clear queue before putting, by default False
    
    Returns
    -------
    bool
        if write was sucessful, returns True
    """

    # lock.acquire()

    # if obj is None:
    #     print("WRITING NONE TO QUEUE")
    #     return 

    if clear:
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass

    try:
        queue.put_nowait(obj)
        success = True
    except Full:
        success = False
    
    # lock.release()

    return success


def qread(queue, lock):
    """ Gets an object in the queue under a lock
    
    Parameters
    ----------
    queue : :class:`multiprocess.Queue`
        a queue object to read from
    lock : :class:`multiprocess.Lock`
        a lock object associated with the queue
    
    Returns
    -------
    object
        object retrieved from the queue
    """

    # lock.acquire()

    try:
        obj = queue.get_nowait()
    except Empty:
        obj = None

    # lock.release()

    return obj


class Camera(object):
    """ Base camera class. Controls image capture, writing images to video, pose estimation and image display. 
    
    Parameters
    ----------
    id : [type]
        camera id
    exposure : int, optional
        exposure time in microseconds, by default None
    gain : int, optional
        gain value, by default None
    rotate : [type], optional
        [description], by default None
    crop : list, optional
        camera cropping parameters: [left, right, top, bottom], by default None
    fps : float, optional
        frame rate in frames per second, by default None
    cv_display : bool, optional
        flag to use opencv image display, by default False
    cv_resize : float, optional
        factor to resize images if using opencv display (display is very slow for large images), by default None
    """


    @staticmethod
    def arg_restrictions():
        """ Returns a dictionary of arguments restrictions for DLCLiveGUI
        """

        return {}


    def __init__(self, id, resolution=None, exposure=None, gain=None, rotate=None, crop=None, fps=None, use_tk_display=False, display_resize=1.0):
        """ Constructor method
        """

        self.id = id
        self.exposure = exposure
        self.gain = gain
        self.rotate = rotate
        self.crop = [int(c) for c in crop] if crop else None
        self.set_im_size(resolution)
        self.fps = fps
        self.next_frame = 0

        self.display_resize = display_resize if display_resize else 1.0
        self.use_tk_display = use_tk_display
        self.dlc_display_only = not use_tk_display
        self.display_keypoints = False
        self.display_colors = None
        self.cmap = "winter"
        
        self.dlc_params = None
        self.dlc_display_only = False
        self.display_pose = None
        self.vid_file = None

        self.capture_process = None
        self.writer_process = None
        self.pose_process = None

        self.reset()


    def set_im_size(self, res):
        """[summary]
        
        Parameters
        ----------
        default : [, optional
            [description], by default None
        
        Raises
        ------
        DLCLiveCameraError
            throws error if resolution is not set
        """

        if not res:
            raise DLCLiveCameraError("Resolution is not set!")

        self.im_size = (int(res[0]), int(res[1])) if self.crop is None else (self.crop[3]-self.crop[2], self.crop[1]-self.crop[0])


    def reset(self):
        """ Set queues and locks for data collection and communication between frame grabber, video writer, and pose estimator processes
        """

        self.video_writer = None

        self.display_frame_queue = mp.Queue(2)
        self.pose_frame_queue = mp.Queue(2)
        self.write_frame_queue = mp.Queue()
        self.display_frame_lock = mp.Lock()
        self.pose_frame_lock = mp.Lock()
        self.write_frame_lock = mp.Lock()

        self.pose_display_queue = mp.Queue(2)
        self.pose_write_queue = mp.Queue()
        self.pose_display_lock = mp.Lock()
        self.pose_write_lock = mp.Lock()

        self.frame_times_queue = mp.Queue()
        self.frame_times_lock = mp.Lock()
  
        self.instruct_queue = mp.Queue()
        self.instruct_lock = mp.Lock()


    def open_capture(self):
        """ Initialize frame capture process
        """

        self.capture_process = mp.Process(target=self.run_capture)
        self.capture_process.daemon = True
        self.capture_process.start()

        ### wait for signal that capture device is initialialized to return
        cap_init = False
        while not cap_init:
            msg = qread(self.instruct_queue, self.instruct_lock)
            if msg is not None:
                if msg == ("Main", "Capture", True):
                    cap_init = True
                else:
                    qwrite(self.instruct_queue, self.instruct_lock, msg)
        
        qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Capture", True))

        return True

    
    def run_capture(self):
        """ Controls frame capture process:
            - sets frame capture device with desired properties
            - captures frames in a loop, writes frames and timestamps to queues for further processing
        """

        try:
            self.set_capture_device()

            qwrite(self.instruct_queue, self.instruct_lock, ("Main", "Capture", True))
            
            self.capture_on_thread()
            self.close_capture_device()

            qwrite(self.instruct_queue, self.instruct_lock, ("Main", "Capture", False))

        except Exception as e:
            qwrite(self.instruct_queue, self.instruct_lock, ("ERROR", "Capture", e))


    def set_capture_device(self):
        """ Sets frame capture device with desired properties
        """

        raise NotImplementedError()


    def capture_on_thread(self):
        """ Acquires frames from frame capture device in a loop and writes to queues for further processing 
        """

        last_frame_time = time.time()

        continue_capture = True
        write_frames = False

        while continue_capture:

            start_capture = time.time()

            frame, frame_time = self.get_image_on_time()

            write_capture = time.time()

            # write frame to display and pose estimation queue
            qwrite(self.display_frame_queue, self.display_frame_lock, (frame, frame_time), clear=True)

            qwrite(self.pose_frame_queue, self.pose_frame_lock, (frame, frame_time), clear=True)

            # if recording, write frames to write queue
            if write_frames:
                qwrite(self.write_frame_queue, self.write_frame_lock, (frame, frame_time))

            commands_capture = time.time()

            cmd = qread(self.instruct_queue, self.instruct_lock)
            if cmd is not None:
                if cmd[0] == "Capture":
                    if cmd[1] == "stop":
                        continue_capture = False
                    elif cmd[1] == "write":
                        write_frames = cmd[2]
                else:
                    qwrite(self.instruct_queue, self.instruct_lock, cmd)

            end_capture = time.time()            

            #print("read frame = %0.3f // write to queues = %0.3f // read commands = %0.3f" % (write_capture-start_capture, commands_capture-write_capture, end_capture-commands_capture))
            #print("capture rate = %d" % (int(1 / (time.time()-last_frame_time))))

            last_frame_time = time.time()
                    
    
    def get_image_on_time(self):
        """ Gets an image from frame capture device at the appropriate time (according to fps).
        
        Returns
        -------
        `np.ndarray`
            image as a numpy array
        float
            timestamp at which frame was taken, obtained from :func:`time.time`
        """

        frame = None
        while frame is None:
            cur_time = time.time()
            if cur_time > self.next_frame:
                frame = self.get_image()
                timestamp = cur_time
                self.next_frame = max(self.next_frame + 1.0/self.fps, cur_time + 0.5/self.fps)

        return frame, timestamp


    def get_image(self):
        """ Gets image from frame capture device
        """

        raise NotImplementedError()


    def close_capture_device(self):
        """ Closes frame capture device
        """

        raise NotImplementedError()


    def close_capture(self):
        """ Ends frame capture process
        """

        if self.capture_process:

            ret = qwrite(self.instruct_queue, self.instruct_lock, ("Capture", "stop"))

            ### wait for capture process to terminate
            capture_done = False
            while not capture_done:
                msg = qread(self.instruct_queue, self.instruct_lock)
                if msg is not None:
                    if msg == ("Main", "Capture", False):
                        capture_done = True
                    else:
                        qwrite(self.instruct_queue, self.instruct_lock, msg)

            # send message and terminate
            qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Capture", False))
            self.capture_process.terminate()
            self.capture_process = None


    def set_dlc_params(self, dlc_params):

        self.dlc_params = dlc_params


    def open_pose(self):
        """ Initiate pose process
        
        Raises
        ------
        DLCLiveCameraError
            Throws error if DLC params have not been set
        """

        if not self.dlc_params:
            raise DLCLiveCameraError("Cannot open pose process before setting DLC Live parameters. Please call set_dlc_params before open_pose.")

        self.pose_process = mp.Process(target=self.run_pose)
        self.pose_process.daemon = True
        self.pose_process.start()

        # ### wait for signal that dlc live is initialialized to return
        # pose_init = False
        # while not pose_init:
        #     msg = qread(self.instruct_queue, self.instruct_lock)
        #     if msg is not None:
        #         if msg == ("Main", "Pose", True):
        #             pose_init = True
        #         else:
        #             qwrite(self.instruct_queue, self.instruct_lock, msg)

        # qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Pose", True))

        return True

    
    def run_pose(self):
        """ Controls pose process
        """

        try:

            # set DLC Live object
            self.set_dlc_live()

            # get first frame and initialize DLC network
            dlc_init = False
            if self.dlc_live is not None:
                while not dlc_init:
                    frame_read = qread(self.pose_frame_queue, self.pose_frame_lock)
                    if frame_read is not None:
                        frame, _ = frame_read
                        pose = self.dlc_live.init_inference(frame)
                        if pose is not None:
                            dlc_init = True

            qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Pose", True))

            # start pose estimation
            self.pose_on_thread()

            qwrite(self.instruct_queue, self.instruct_lock, ("Main", "Pose", False))

        except Exception as e:
            qwrite(self.instruct_queue, self.instruct_lock, ("ERROR", "Pose", e))


    def set_dlc_live(self):
        """ Instantiates dlc live object
        """

        self.dlc_live = DLCLive(**self.dlc_params)
        qwrite(self.pose_write_queue, self.pose_write_lock, self.dlc_live.cfg['all_joints_names'])


    def pose_on_thread(self):
        """ Conducts pose estimation and writes poses to queues for further processing, in a loop
        """

        continue_pose = True
        write_pose = False

        last_pose_time = time.time()

        while continue_pose:

            frame_read = qread(self.pose_frame_queue, self.pose_frame_lock)
            
            if frame_read is not None:

                frame, frame_ts = frame_read

                frame_read_time = time.time()

                pose = self.dlc_live.get_pose(frame)
                pose_time = time.time()

                get_pose_time = time.time()

                # write pose for display_queue
                qwrite(self.pose_display_queue, self.pose_display_lock, (pose), clear=True)

                # if recording, write pose to write queue
                if write_pose:
                    qwrite(self.pose_write_queue, self.pose_write_lock, (pose, frame_ts, pose_time))

                queue_write_time = time.time()

                # print("frame read = %0.6f // get pose = %0.6f // write queue = %0.6f // total = %0.6f" % (frame_read_time-last_pose_time, get_pose_time-frame_read_time, queue_write_time-get_pose_time, queue_write_time-last_pose_time))
                # print("POSE RATE = %d" % int(1/(get_pose_time-frame_read_time)))

                last_pose_time = time.time()
                             
            cmd = qread(self.instruct_queue, self.instruct_lock)
            if cmd is not None:
                if cmd[0] == "Pose":
                    if cmd[1] == "stop":
                        continue_pose = False
                    elif cmd[1] == "write":
                        write_pose = cmd[2]
                else:
                    qwrite(self.instruct_queue, self.instruct_lock, cmd)


    def close_pose(self):
        """ Ends pose process
        """

        if self.pose_process:
            qwrite(self.instruct_queue, self.instruct_lock, ("Pose", "stop"))

            ### wait for pose process to terminate
            pose_done = False
            while not pose_done:
                msg = qread(self.instruct_queue, self.instruct_lock)
                if msg is not None:
                    if msg == ("Main", "Pose", False):
                        pose_done = True
                    else:
                        qwrite(self.instruct_queue, self.instruct_lock, msg)

            ### send message and terminate
            qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Pose", False))
            self.pose_process.terminate()
            self.pose_process = None


    def open_writer(self):
        """ Initialize video writer process
        """

        if not self.vid_file:
            return False

        self.writer_process = mp.Process(target=self.run_writer)
        self.writer_process.daemon = True
        self.writer_process.start()

        # ### wait for signal that dlc live is initialialized to return
        # writer_init = False
        # while not writer_init:
        #     msg = qread(self.instruct_queue, self.instruct_lock)
        #     if msg is not None:
        #         if msg == ("Main", "Writer", True):
        #             writer_init = True
        #         else:
        #             qwrite(self.instruct_queue, self.instruct_lock, msg)

        # qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Writer", True))

        return True


    def set_video_file(self, filename):
        """ Set name of video file
        
        Parameters
        ----------
        filename : str
            name of avi video file
        """

        self.vid_file = filename        

        
    def run_writer(self):
        """ Controls video writer process
        """

        try:

            ### init video writer
            self.video_writer = cv2.VideoWriter(self.vid_file,
                                                cv2.VideoWriter_fourcc(*'DIVX'),
                                                self.fps,
                                                self.im_size)

            ### send signal that writer is initialized
            qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Writer", True))

            ### run writer loop until video is closed
            self.write_on_thread()

            ### save video to disk
            self.video_writer.release()

            ### send signal that writer is released
            qwrite(self.instruct_queue, self.instruct_lock, ("Main", "Writer", False))

        except Exception as e:
            qwrite(self.instruct_queue, self.instruct_lock, ("ERROR", "Writer", e))


    def write_on_thread(self):
        """ In loop: reads frames from frames to write queue, writes to video file
        """

        continue_write = True
        new_frame = False

        while continue_write or new_frame:

            frame_read = qread(self.write_frame_queue, self.write_frame_lock)
            if frame_read is not None:
                new_frame = True
                frame, frame_time = frame_read
                self.video_writer.write(frame)
                qwrite(self.frame_times_queue, self.frame_times_lock, frame_time)
            else:
                new_frame = False

            cmd = qread(self.instruct_queue, self.instruct_lock)
            if cmd is not None:
                if cmd[0] == "Writer":
                    if cmd[1] == "stop":
                        continue_write = False
                else:
                    qwrite(self.instruct_queue, self.instruct_lock, cmd)


    def start_write(self):
        """ start recording frames and poses
        """

        start_write = False

        if self.writer_process is not None:

            if self.capture_process is not None:
                qwrite(self.instruct_queue, self.instruct_lock, ("Capture", "write", True))
                start_write = True
        
            if self.pose_process is not None:
                qwrite(self.instruct_queue, self.instruct_lock, ("Pose", "write", True))

        return start_write

    
    def stop_write(self):
        """ stop recording frames and poses
        """

        if self.capture_process is not None:
            qwrite(self.instruct_queue, self.instruct_lock, ("Capture", "write", False))
        
        if self.pose_process is not None:
            qwrite(self.instruct_queue, self.instruct_lock, ("Pose", "write", False))


    def close_writer(self):
        """ Issues command to stop video writer process and save file
        """

        self.stop_write()

        if self.writer_process is not None:
            qwrite(self.instruct_queue, self.instruct_lock, ("Writer", "stop"))

            ### wait for process to close

            writer_done = False
            while not writer_done:
                msg = qread(self.instruct_queue, self.instruct_lock)
                if msg is not None:
                    if msg == ("Main", "Writer", False):
                        writer_done = True
                    else:
                        qwrite(self.instruct_queue, self.instruct_lock, msg)
            
            ### send message and terminate process
            qwrite(self.instruct_queue, self.instruct_lock, ("MESSAGE", "Writer", False))
            self.writer_process.terminate()
            self.writer_process = None
            self.vid_file = None


    def get_display_frame(self):
        """ Get latest frame for display
        
        Returns
        -------
        :class:`numpy.ndarray`
            the lastest frame
        """

        frame_read = qread(self.display_frame_queue, self.display_frame_lock)
        if frame_read is not None:
            frame, _ = frame_read
            if self.display_resize != 1:
                frame = cv2.resize(frame, (int(frame.shape[1]*self.display_resize), int(frame.shape[0]*self.display_resize)))
        else:
            frame = None

        return frame


    def get_display_frame2(self):
        """ Return frame to be displayed
        
        Returns
        -------
        :class:`numpy.ndarray`
            an image as a numpy array to be displayed 
        """

        frame_read = qread(self.display_frame_queue, self.display_frame_lock)

        if frame_read is not None:

            frame, _ = frame_read

            if self.display_resize != 1:
                frame = cv2.resize(frame, (int(frame.shape[1]*self.display_resize), int(frame.shape[0]*self.display_resize)))

            if self.display_keypoints:
                new_pose = qread(self.pose_display_queue, self.pose_display_lock)
                if new_pose is not None:
                    self.display_pose = new_pose
                    if self.display_resize != 1:
                        self.display_pose[:, :2] *= self.display_resize

                if self.display_pose is not None:

                    if self.display_colors is None:
                        self.set_display_colors(self.display_pose.shape[0])

                    im_size = (frame.shape[1], frame.shape[0])
                    for i in range(self.display_pose.shape[0]):
                        if self.display_pose[i,2] > 0:
                            rr, cc = circle(self.display_pose[i,1], self.display_pose[i,0], im_size[0]*.01, shape=im_size)
                            rr[rr > im_size[0]] = im_size[0]
                            cc[cc > im_size[1]] = im_size[1]
                            try:
                                frame[rr, cc, :] = self.display_colors[i]
                            except:
                                pass

        else:

            frame = None
            
        return frame


    def get_display_pose(self):
        """ Get latest pose for display
        
        Returns
        -------
        :class:`numpy.ndarray`
            the latest pose as a numpy array
        """

        # if self.display_keypoints:
        new_pose = qread(self.pose_display_queue, self.pose_display_lock)
        if new_pose is not None:
            if self.display_resize != 1:
                new_pose[:, :2] *= self.display_resize
            self.display_pose = new_pose

        return self.display_pose
        


    def set_display_colors(self, n_colors):
        """ Set the colors for keypoints

        Parameters
        -----------
        n_colors : int
            The number of colors (or number of points)
        """

        colorclass = plt.cm.ScalarMappable(cmap=self.cmap)
        C = colorclass.to_rgba(np.linspace(0, 1, n_colors))
        self.display_colors = (C[:,:3]*255).astype(np.uint8)


    def change_display_keypoints(self, display_keypoints=None):

        self.display_keypoints = display_keypoints if (display_keypoints is not None) else (not self.display_keypoints)

        if self.display_keypoints:
            self.use_tk_display = True
        elif self.dlc_display_only:
            self.use_tk_display = False


    def monitor_processes(self):
        """ Monitor camera processes for messages or errors
        
        Returns
        -------
        str
            message returned from process
        
        Raises
        ------
        DLCLiveCameraError
            throws error on main thread if error occurs in process
        """

        cmd = qread(self.instruct_queue, self.instruct_lock)
        if cmd is not None:
            if cmd[0] == "ERROR":
                raise DLCLiveCameraError("Error on {} process :: {}".format(cmd[1], cmd[2]))
            elif cmd[0] == "MESSAGE":
                return cmd[1:]
            else:
                qwrite(self.instruct_queue, self.instruct_lock, cmd)

        return None


    def get_frame_times(self):
        """Return timestamps of frames saved to video
        
        Returns
        -------
        list
            list of timestamps of frames that were saved to video
        """

        frame_times = []
        while not self.frame_times_queue.empty():
            frame_time = self.frame_times_queue.get()
            frame_times.append(frame_time)
        
        return frame_times


    def get_pose_df(self):
        """ Return a pandas data frame with pose data collected while recording video
        
        Returns
        -------
        :class:`pandas.DataFrame`
            a data frame with DeepLabCut keypoints, time of frame acquisition, and time pose was measured
        """

        bodyparts = None
        poses = []
        frame_times = []
        pose_times = []

        try:
            bodyparts = self.pose_write_queue.get_nowait()
            while True:
                pose_read = self.pose_write_queue.get_nowait()
                pose, frame_time, pose_time = pose_read
                poses.append(pose)
                frame_times.append(frame_time)
                pose_times.append(pose_time)

        except Empty:
            pass

        finally:
            if (bodyparts is None) or (len(poses) == 0):
                return None
            else:
                poses = np.array(poses)
                poses = poses.reshape((poses.shape[0], poses.shape[1]*poses.shape[2]))
                pdindex = pd.MultiIndex.from_product([bodyparts, ['x', 'y', 'likelihood']], names=['bodyparts', 'coords'])
                pose_df = pd.DataFrame(poses, columns=pdindex)
                pose_df['frame_time'] = frame_times
                pose_df['pose_time'] = pose_times

                return pose_df


    def close(self):
        """ Closes all capture, pose and writer processes
        """

        if self.writer_process:
            self.close_writer()
        if self.pose_process:
            self.close_pose()
        if self.capture_process:
            self.close_capture()
