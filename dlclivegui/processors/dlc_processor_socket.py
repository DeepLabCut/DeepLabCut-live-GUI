"""Example of socket-based DLC Processor with multi-client support and optional One-Euro filtering."""

# dlclivegui/processors/dlc_processor_socket.py
import logging
import pickle
import socket
import sys
import time
from collections import deque
from math import acos, atan2, copysign, degrees, pi, sqrt
from multiprocessing.connection import Client, Listener
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pandas as pd
from dlclive import Processor  # type: ignore

logger = logging.getLogger("dlc_processor_socket")

# Avoid duplicate handlers if module is imported multiple times
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)

# Registry for GUI discovery
PROCESSOR_REGISTRY = {}


def register_processor(cls):
    registry_key = getattr(cls, "PROCESSOR_ID", cls.__name__)
    if registry_key in PROCESSOR_REGISTRY:
        raise ValueError(
            f"Duplicate processor registration key '{registry_key}': "
            f"{PROCESSOR_REGISTRY[registry_key].__name__} vs {cls.__name__}"
        )
    PROCESSOR_REGISTRY[registry_key] = cls
    return cls


class OneEuroFilter:  # pragma: no cover
    def __init__(self, t0, x0, dx0=None, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        if dx0 is None:
            dx0 = np.zeros_like(x0)
        self.dx_prev = dx0
        self.t_prev = t0

    @staticmethod
    def smoothing_factor(t_e, cutoff):
        r = 2 * pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def exponential_smoothing(alpha, x, x_prev):
        return alpha * x + (1 - alpha) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


# pragma: cover
class BaseProcessorSocket(Processor):
    """
    Base processor class that implements a socket server to help remote control recording in experiments.
    Clients can connect to start/stop recording and receive real-time pose data.
    - Socket server is OPTIONAL: you can instantiate without bind/authkey.
    - Call start_server(...) to enable networking.
    """

    PROCESSOR_NAME = "Base Socket Processor"
    PROCESSOR_DESCRIPTION = "Base class for socket-based processors with multi-client support"
    PROCESSOR_PARAMS = {}

    def __init__(
        self,
        bind=None,
        authkey=None,
        use_perf_counter=False,
        save_original=False,
        *,
        start_server: bool = True,
        socket_timeout: float = 1.0,
    ):
        """
        Args:
            bind: Optional (host, port) tuple. If None, no server is started.
            authkey: Optional auth key bytes. If None and bind is set, defaults to b"secret password".
            use_perf_counter: If True, uses time.perf_counter; else time.time.
            save_original: If True, stores raw pose arrays.
            start_server: If True and bind is not None, starts the socket server in __init__.
            socket_timeout: Socket poll/accept timeout.
        """
        super().__init__()
        self.dlc_cfg = None  # DeepLabCut config for saving original pose data

        # Timing
        self.timing_func = time.perf_counter if use_perf_counter else time.time
        self.start_time = self.timing_func()

        # Recording buffers
        self.time_stamp = deque()
        self.step = deque()
        self.frame_time = deque()
        self.pose_time = deque()
        self.original_pose = deque() if save_original else None

        self._session_name = "test_session"
        self.filename = None

        self._recording = Event()
        self._vid_recording = Event()
        self.curr_step = 0
        self.save_original = save_original

        # Networking (optional)
        self.address = bind
        self.authkey = authkey if authkey is not None else (b"secret password" if bind is not None else None)
        self.listener = None
        self._accept_thread = None
        self._rx_thread = None
        self._rx_threads = set()
        self._socket_timeout = float(socket_timeout)

        self._stop = Event()
        self.conns = set()

        if start_server and self.address is not None:
            self.start_server(self.address, self.authkey, timeout=self._socket_timeout)

    # --------------------------------------------------------------------------------------
    # PROPERTIES
    # --------------------------------------------------------------------------------------

    @property
    def recording(self):
        return self._recording.is_set()

    @property
    def video_recording(self):
        return self._vid_recording.is_set()

    @property
    def session_name(self):
        return self._session_name

    @session_name.setter
    def session_name(self, name):
        self._session_name = name
        self.filename = f"{name}_dlc_processor_data.pkl"

    # --------------------------------------------------------------------------------------
    # SERVER CONTROL
    # --------------------------------------------------------------------------------------
    def start_server(self, bind, authkey=b"secret password", *, timeout: float = 1.0):
        """
        Start the socket server if not already running.
        Safe to call multiple times.
        """
        if self.listener is not None:
            return

        if self._stop.is_set():
            self._stop.clear()

        self.address = bind
        self.authkey = authkey

        self.listener = Listener(bind, authkey=authkey)
        try:
            # Underlying socket timeout
            self.listener._listener.settimeout(timeout)
        except Exception:
            pass

        self._accept_thread = Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        logger.info(f"Processor server started on {bind[0]}:{bind[1]}")

    # --------------------------------------------------------------------------------------
    # ACCEPT LOOP
    # --------------------------------------------------------------------------------------

    def _accept_loop(self):
        if self.listener is None:
            return

        logger.debug(f"DLC Processor listening on {self.address[0]}:{self.address[1]}")
        while not self._stop.is_set():
            try:
                conn = self.listener.accept()

                # Apply safe timeout to client socket
                try:
                    conn._socket.settimeout(self._socket_timeout)
                except Exception:
                    pass

                logger.debug(f"Client connected from {self.listener.last_accepted}")
                self.conns.add(conn)

                self._rx_thread = Thread(target=self._rx_loop, args=(conn,), daemon=True)
                self._rx_threads.add(self._rx_thread)
                self._rx_thread.start()

            except (TimeoutError, OSError, EOFError):
                if self._stop.is_set():
                    break

    # --------------------------------------------------------------------------------------
    # RECEIVE LOOP
    # --------------------------------------------------------------------------------------

    def _rx_loop(self, conn):
        while not self._stop.is_set():
            try:
                if conn.poll(0.1):
                    msg = conn.recv()
                    self._handle_client_message(msg)
                    continue

                if getattr(conn._socket, "_closed", False):
                    raise EOFError

            except (EOFError, OSError, ConnectionError, BrokenPipeError):
                break

        self._close_conn(conn)
        logger.info("Client disconnected")

    # --------------------------------------------------------------------------------------
    # SOCKET CLOSE HELPERS
    # --------------------------------------------------------------------------------------

    def _close_conn(self, conn):
        """Force-close client connection."""
        try:
            conn._socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        self.conns.discard(conn)

    def _close_listener(self):
        """Close both outer and inner listener sockets."""
        if self.listener is None:
            return
        try:
            self.listener._listener.close()  # Raw OS socket
        except Exception:
            pass
        try:
            self.listener.close()  # Python wrapper
        except Exception:
            pass
        self.listener = None

    # --------------------------------------------------------------------------------------
    # HANDLE MESSAGES
    # --------------------------------------------------------------------------------------

    def _handle_client_message(self, msg):
        if not isinstance(msg, dict):
            return

        cmd = msg.get("cmd")
        if cmd == "set_session_name":
            self.session_name = msg.get("session_name", "default_session")

        elif cmd == "start_recording":
            self._recording.set()
            self._vid_recording.set()
            self._clear_data_queues()
            self.curr_step = 0
            logger.info("Recording started")

        elif cmd == "stop_recording":
            self._recording.clear()
            self._vid_recording.clear()
            logger.info("Recording stopped")

        elif cmd == "save":
            file = msg.get("filename", self.filename)
            self.save(file)

    # Optional public helpers (nice for non-socket usage)
    def start_recording(self):
        self._recording.set()
        self._vid_recording.set()
        self._clear_data_queues()
        self.curr_step = 0

    def stop_recording(self):
        self._recording.clear()
        self._vid_recording.clear()

    # --------------------------------------------------------------------------------------
    # STOP / SHUTDOWN
    # --------------------------------------------------------------------------------------

    def stop(self):
        """Gracefully stop listener and clients."""
        if self._stop.is_set():
            return

        logger.info("Stopping processor...")
        self._stop.set()

        # Wake accept() so the accept loop exits quickly (especially helpful on Windows)
        # This is safe even if no clients are connected.
        try:
            if self.address is not None and self.authkey is not None:
                c = Client(self.address, authkey=self.authkey)
                c.close()
        except Exception:
            pass

        for conn in list(self.conns):
            self._close_conn(conn)

        self._close_listener()

        # Join accept thread to avoid race conditions on restart
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            if self._accept_thread.is_alive():
                logger.warning("Accept thread did not terminate cleanly")
            self._accept_thread = None

        # Join rx threads briefly (best effort)
        for t in list(self._rx_threads):
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self._rx_threads.clear()
        self._rx_thread = None

        # Small Windows delay to help TIME_WAIT cleanup
        if sys.platform.startswith("win"):
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                time.sleep(0.3)

        logger.info("Processor stopped")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # --------------------------------------------------------------------------------------
    # BROADCAST
    # --------------------------------------------------------------------------------------

    def broadcast(self, payload):
        """Send payload to all connected clients. No-op if server isn't running."""
        if not self.conns:
            return

        dead = []
        for conn in list(self.conns):
            try:
                conn.send(payload)
            except Exception:
                dead.append(conn)

        for conn in dead:
            self._close_conn(conn)

    # --------------------------------------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------------------------------------

    def process(self, pose, **kwargs):
        curr_time = self.timing_func()

        if self.save_original:
            self.original_pose.append(pose.copy())

        self.curr_step += 1

        if self.recording:
            self.time_stamp.append(curr_time)
            self.step.append(self.curr_step)
            self.frame_time.append(kwargs.get("frame_time", -1))
            if "pose_time" in kwargs:
                self.pose_time.append(kwargs["pose_time"])

        payload = [curr_time, pose]
        self.broadcast(payload)
        return pose

    # --------------------------------------------------------------------------------------
    # UTILITIES
    # --------------------------------------------------------------------------------------

    def _clear_data_queues(self):
        self.time_stamp.clear()
        self.step.clear()
        self.frame_time.clear()
        self.pose_time.clear()
        if self.save_original and self.original_pose is not None:
            self.original_pose.clear()

    def save(self, file=None):
        if not file:
            return 0
        try:
            save_dict = self.get_data()
            path2save = Path(__file__).parent.parent.parent / "data" / file
            path2save.parent.mkdir(parents=True, exist_ok=True)
            if self.save_original:
                original_pose = save_dict.pop("original_pose")
                self.save_original_pose(original_pose, save_dict["frame_time"], save_dict["time_stamp"], path2save)
            with open(path2save, "wb") as f:
                pickle.dump(save_dict, f)
            logger.info(f"Saved data to {path2save}")
            return 1
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return -1

    def save_original_pose(
        self,
        original_pose: np.ndarray,
        pose_frame_times: np.ndarray,
        pose_times: np.ndarray,
        filepath2save: Path,
    ):
        filepath2save = filepath2save.parent / (filepath2save.stem + "_DLC.hdf5")
        if isinstance(self.dlc_cfg, dict):
            bodyparts = self.dlc_cfg.get("metadata", {}).get("bodyparts", [])
        else:
            bodyparts = None
        poses = np.array(original_pose)
        poses = poses.reshape((poses.shape[0], poses.shape[1] * poses.shape[2]))
        if bodyparts and len(bodyparts) * 3 == poses.shape[1]:
            pdindex = pd.MultiIndex.from_product([bodyparts, ["x", "y", "likelihood"]], names=["bodyparts", "coords"])
            pose_df = pd.DataFrame(poses, columns=pdindex)
        else:
            logger.warning("Bodyparts information not found in dlc_cfg; saving without column labels.")
            pose_df = pd.DataFrame(poses)
        pose_df["frame_time"] = pose_frame_times
        pose_df["pose_time"] = pose_times

        pose_df.to_hdf(filepath2save, key="df_with_missing", mode="w")

    def set_dlc_cfg(self, dlc_cfg):
        """Set DLC configuration for saving original pose data."""
        self.dlc_cfg = dlc_cfg

    def get_data(self):
        save_dict = {
            "start_time": self.start_time,
            "time_stamp": np.array(self.time_stamp),
            "step": np.array(self.step),
            "frame_time": np.array(self.frame_time),
            "pose_time": np.array(self.pose_time) if self.pose_time else None,
            "use_perf_counter": self.timing_func == time.perf_counter,
            "original_pose": np.array(self.original_pose) if self.save_original else None,
        }
        if self.dlc_cfg is not None:
            save_dict["dlc_cfg"] = self.dlc_cfg
        return save_dict


@register_processor
class ExampleProcessorSocketCalculateMousePose(BaseProcessorSocket):  # pragma: no cover
    """
    DLC Processor with pose calculations (center, heading, head angle) and optional filtering.

    Calculates:
    - center: Weighted average of head keypoints
    - heading: Body orientation (degrees)
    - head_angle: Head rotation relative to body (radians)

    Broadcasts: [timestamp, center_x, center_y, heading, head_angle]
    """

    PROCESSOR_NAME = "Example Experiment Pose Processor"
    PROCESSOR_DESCRIPTION = "Calculates mouse center, heading, and head angle with optional One-Euro filtering"
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("127.0.0.1", 6000),
            "description": "Server address (host, port)",
        },
        "authkey": {
            "type": "bytes",
            "default": b"secret password",
            "description": "Authentication key for clients",
        },
        "use_perf_counter": {
            "type": "bool",
            "default": False,
            "description": "Use time.perf_counter() instead of time.time()",
        },
        "use_filter": {
            "type": "bool",
            "default": False,
            "description": "Apply One-Euro filter to calculated values",
        },
        "filter_kwargs": {
            "type": "dict",
            "default": {"min_cutoff": 1.0, "beta": 0.02, "d_cutoff": 1.0},
            "description": "One-Euro filter parameters (min_cutoff, beta, d_cutoff)",
        },
        "save_original": {
            "type": "bool",
            "default": False,
            "description": "Save raw pose arrays for analysis",
        },
    }

    def __init__(
        self,
        bind=("127.0.0.1", 6000),
        authkey=b"secret password",
        use_perf_counter=False,
        use_filter=False,
        filter_kwargs: dict | None = None,
        save_original=False,
    ):
        super().__init__(
            bind=bind,
            authkey=authkey,
            use_perf_counter=use_perf_counter,
            save_original=save_original,
        )

        self.center_x = deque()
        self.center_y = deque()
        self.heading_direction = deque()
        self.head_angle = deque()

        self.use_filter = use_filter
        self.filter_kwargs = filter_kwargs if filter_kwargs is not None else {}
        self.filters = None

    def _clear_data_queues(self):
        super()._clear_data_queues()
        self.center_x.clear()
        self.center_y.clear()
        self.heading_direction.clear()
        self.head_angle.clear()

    def _initialize_filters(self, vals):
        t0 = self.timing_func()
        self.filters = {
            "center_x": OneEuroFilter(t0, vals[0], **self.filter_kwargs),
            "center_y": OneEuroFilter(t0, vals[1], **self.filter_kwargs),
            "heading": OneEuroFilter(t0, vals[2], **self.filter_kwargs),
            "head_angle": OneEuroFilter(t0, vals[3], **self.filter_kwargs),
        }
        logger.debug(f"Initialized One-Euro filters with parameters: {self.filter_kwargs}")

    def process(self, pose, **kwargs):
        if self.save_original:
            self.original_pose.append(pose.copy())

        # Extract keypoints and confidence
        xy = pose[:, :2]
        conf = pose[:, 2]

        # Calculate weighted center from head keypoints
        head_xy = xy[[0, 1, 2, 3, 4, 5, 6, 26], :]
        head_conf = conf[[0, 1, 2, 3, 4, 5, 6, 26]]
        center = np.average(head_xy, axis=0, weights=head_conf)

        # Calculate body axis (tail_base -> neck)
        body_axis = xy[7] - xy[13]
        body_axis /= sqrt(np.sum(body_axis**2))

        # Calculate head axis (neck -> nose)
        head_axis = xy[0] - xy[7]
        head_axis /= sqrt(np.sum(head_axis**2))

        # Calculate head angle relative to body
        cross = body_axis[0] * head_axis[1] - head_axis[0] * body_axis[1]
        sign = copysign(1, cross)  # Positive when looking left
        sign = copysign(1, cross)
        try:
            head_angle = acos(body_axis @ head_axis) * sign
        except ValueError:
            head_angle = 0

        # Calculate heading (body orientation)
        heading = degrees(atan2(body_axis[1], body_axis[0]))

        # Raw values (heading unwrapped for filtering)
        vals = [center[0], center[1], heading, head_angle]

        # Apply filtering if enabled
        curr_time = self.timing_func()
        if self.use_filter:
            if self.filters is None:
                self._initialize_filters(vals)

            vals = [
                self.filters["center_x"](curr_time, vals[0]),
                self.filters["center_y"](curr_time, vals[1]),
                self.filters["heading"](curr_time, vals[2]),
                self.filters["head_angle"](curr_time, vals[3]),
            ]

        # Wrap heading to [0, 360) after filtering
        vals[2] = vals[2] % 360
        # Update step counter
        self.curr_step = self.curr_step + 1

        # Store processed data (only if recording)
        if self.recording:
            self.center_x.append(vals[0])
            self.center_y.append(vals[1])
            self.heading_direction.append(vals[2])
            self.head_angle.append(vals[3])
            self.time_stamp.append(curr_time)
            self.step.append(self.curr_step)
            self.frame_time.append(kwargs.get("frame_time", -1))
            if "pose_time" in kwargs:
                self.pose_time.append(kwargs["pose_time"])

        payload = [curr_time, vals[0], vals[1], vals[2], vals[3]]
        self.broadcast(payload)
        return pose

    def get_data(self):
        save_dict = super().get_data()
        save_dict["x_pos"] = np.array(self.center_x)
        save_dict["y_pos"] = np.array(self.center_y)
        save_dict["heading_direction"] = np.array(self.heading_direction)
        save_dict["head_angle"] = np.array(self.head_angle)
        save_dict["use_filter"] = self.use_filter
        save_dict["filter_kwargs"] = self.filter_kwargs
        return save_dict


@register_processor
class ExampleProcessorSocketFilterKeypoints(BaseProcessorSocket):  # pragma: no cover
    PROCESSOR_NAME = "Mouse Pose with less keypoints"
    PROCESSOR_DESCRIPTION = "Calculates mouse center, heading, and head angle with optional One-Euro filtering"
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("127.0.0.1", 6000),
            "description": "Server address (host, port)",
        },
        "authkey": {
            "type": "bytes",
            "default": b"secret password",
            "description": "Authentication key for clients",
        },
        "use_perf_counter": {
            "type": "bool",
            "default": False,
            "description": "Use time.perf_counter() instead of time.time()",
        },
        "use_filter": {
            "type": "bool",
            "default": False,
            "description": "Apply One-Euro filter to calculated values",
        },
        "filter_kwargs": {
            "type": "dict",
            "default": {"min_cutoff": 1.0, "beta": 0.02, "d_cutoff": 1.0},
            "description": "One-Euro filter parameters (min_cutoff, beta, d_cutoff)",
        },
        "save_original": {
            "type": "bool",
            "default": False,
            "description": "Save raw pose arrays for analysis",
        },
    }

    def __init__(
        self,
        bind=("127.0.0.1", 6000),
        authkey=b"secret password",
        use_perf_counter=False,
        use_filter=False,
        filter_kwargs: dict | None = None,
        save_original=False,
        p_cutoff=0.4,
    ):
        super().__init__(
            bind=bind,
            authkey=authkey,
            use_perf_counter=use_perf_counter,
            save_original=save_original,
        )

        self.center_x = deque()
        self.center_y = deque()
        self.heading_direction = deque()
        self.head_angle = deque()

        self.p_cutoff = p_cutoff

        self.use_filter = use_filter
        self.filter_kwargs = filter_kwargs if filter_kwargs is not None else {}
        self.filters = None

    def _clear_data_queues(self):
        super()._clear_data_queues()
        self.center_x.clear()
        self.center_y.clear()
        self.heading_direction.clear()
        self.head_angle.clear()

    def _initialize_filters(self, vals):
        t0 = self.timing_func()
        self.filters = {
            "center_x": OneEuroFilter(t0, vals[0], **self.filter_kwargs),
            "center_y": OneEuroFilter(t0, vals[1], **self.filter_kwargs),
            "heading": OneEuroFilter(t0, vals[2], **self.filter_kwargs),
            "head_angle": OneEuroFilter(t0, vals[3], **self.filter_kwargs),
        }
        logger.debug(f"Initialized One-Euro filters with parameters: {self.filter_kwargs}")

    def process(self, pose, **kwargs):
        if self.save_original:
            self.original_pose.append(pose.copy())

        # Extract keypoints and confidence
        xy = pose[:, :2]
        conf = pose[:, 2]

        # Calculate weighted center from head keypoints
        head_xy = xy[[0, 1, 2, 3, 5, 6, 7], :]
        head_conf = conf[[0, 1, 2, 3, 5, 6, 7]]
        # set low confidence keypoints to zero weight
        head_conf = np.where(head_conf < self.p_cutoff, 0, head_conf)
        try:
            center = np.average(head_xy, axis=0, weights=head_conf)
        except ZeroDivisionError:
            # If all keypoints have zero weight, return without processing
            return pose

        neck = np.average(xy[[2, 3, 6, 7], :], axis=0, weights=conf[[2, 3, 6, 7]])

        # Calculate body axis (tail_base -> neck)
        body_axis = neck - xy[9]
        body_axis /= sqrt(np.sum(body_axis**2))

        # Calculate head axis (neck -> nose)
        head_axis = xy[0] - neck
        head_axis /= sqrt(np.sum(head_axis**2))

        # Calculate head angle relative to body
        cross = body_axis[0] * head_axis[1] - head_axis[0] * body_axis[1]
        sign = copysign(1, cross)  # Positive when looking left
        sign = copysign(1, cross)
        try:
            head_angle = acos(body_axis @ head_axis) * sign
        except ValueError:
            head_angle = 0

        # Calculate heading (body orientation)
        heading = degrees(atan2(body_axis[1], body_axis[0]))
        vals = [center[0], center[1], heading, head_angle]

        curr_time = self.timing_func()
        if self.use_filter:
            if self.filters is None:
                self._initialize_filters(vals)

            vals = [
                self.filters["center_x"](curr_time, vals[0]),
                self.filters["center_y"](curr_time, vals[1]),
                self.filters["heading"](curr_time, vals[2]),
                self.filters["head_angle"](curr_time, vals[3]),
            ]

        # Wrap heading to [0, 360) after filtering
        vals[2] = vals[2] % 360
        # Update step counter
        self.curr_step = self.curr_step + 1

        # Store processed data (only if recording)
        if self.recording:
            self.center_x.append(vals[0])
            self.center_y.append(vals[1])
            self.heading_direction.append(vals[2])
            self.head_angle.append(vals[3])
            self.time_stamp.append(curr_time)
            self.step.append(self.curr_step)
            self.frame_time.append(kwargs.get("frame_time", -1))
            if "pose_time" in kwargs:
                self.pose_time.append(kwargs["pose_time"])

        payload = [curr_time, vals[0], vals[1], vals[2], vals[3]]
        self.broadcast(payload)
        return pose

    def get_data(self):
        save_dict = super().get_data()
        save_dict["x_pos"] = np.array(self.center_x)
        save_dict["y_pos"] = np.array(self.center_y)
        save_dict["heading_direction"] = np.array(self.heading_direction)
        save_dict["head_angle"] = np.array(self.head_angle)
        save_dict["use_filter"] = self.use_filter
        save_dict["filter_kwargs"] = self.filter_kwargs
        return save_dict


def get_available_processors():
    """
    Get list of available processor classes.

    Returns:
        dict: Dictionary mapping registry keys to processor info.
    """
    return {
        name: {
            "class": cls,
            "name": getattr(cls, "PROCESSOR_NAME", name),
            "description": getattr(cls, "PROCESSOR_DESCRIPTION", ""),
            "params": getattr(cls, "PROCESSOR_PARAMS", {}),
        }
        for name, cls in PROCESSOR_REGISTRY.items()
    }


def instantiate_processor(class_name, **kwargs):
    """
    Instantiate a processor by class name with given parameters.

    Args:
        class_name: Registry key (e.g., "MyProcessorSocket")
        **kwargs: Constructor kwargs

    Raises:
        ValueError: If class_name is not in registry
    """
    if class_name not in PROCESSOR_REGISTRY:
        available = ", ".join(PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown processor '{class_name}'. Available: {available}")
    return PROCESSOR_REGISTRY[class_name](**kwargs)
