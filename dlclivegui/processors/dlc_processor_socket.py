import logging
import pickle
import socket
import time
from collections import deque
from math import acos, atan2, copysign, degrees, pi, sqrt
from multiprocessing.connection import Listener
from threading import Event, Thread

import numpy as np
from dlclive import Processor  # type: ignore

LOG = logging.getLogger("dlc_processor_socket")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_handler)


# Registry for GUI discovery
PROCESSOR_REGISTRY = {}


class OneEuroFilter:
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


class BaseProcessor_socket(Processor):
    """
    Base DLC Processor with multi-client broadcasting support.

    Handles network connections, timing, and data logging.
    Subclasses should implement custom pose processing logic.
    """

    # Metadata for GUI discovery
    PROCESSOR_NAME = "Base Socket Processor"
    PROCESSOR_DESCRIPTION = "Base class for socket-based processors with multi-client support"
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("0.0.0.0", 6000),
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
        "save_original": {
            "type": "bool",
            "default": False,
            "description": "Save raw pose arrays for analysis",
        },
    }

    def __init__(
        self,
        bind=("0.0.0.0", 6000),
        authkey=b"secret password",
        use_perf_counter=False,
        save_original=False,
    ):
        """
        Initialize base processor with socket server.

        Args:
            bind: (host, port) tuple for server binding
            authkey: Authentication key for client connections
            use_perf_counter: If True, use time.perf_counter() instead of time.time()
            save_original: If True, save raw pose arrays for analysis
        """
        super().__init__()

        # Network setup
        self.address = bind
        self.authkey = authkey
        self.listener = Listener(bind, authkey=authkey)
        self._stop = Event()
        self.conns = set()

        # Start accept loop in background
        Thread(target=self._accept_loop, name="DLCAccept", daemon=True).start()

        # Timing function
        self.timing_func = time.perf_counter if use_perf_counter else time.time
        self.start_time = self.timing_func()

        # Data storage
        self.time_stamp = deque()
        self.step = deque()
        self.frame_time = deque()
        self.pose_time = deque()
        self.original_pose = deque()

        self._session_name = "test_session"
        self.filename = None
        self._recording = Event()  # Thread-safe recording flag
        self._vid_recording = Event()  # Thread-safe video recording flag

        # State
        self.curr_step = 0
        self.save_original = save_original

    @property
    def recording(self):
        """Thread-safe recording flag."""
        return self._recording.is_set()

    @property
    def video_recording(self):
        """Thread-safe video recording flag."""
        return self._vid_recording.is_set()

    @property
    def session_name(self):
        return self._session_name

    @session_name.setter
    def session_name(self, name):
        self._session_name = name
        self.filename = f"{name}_dlc_processor_data.pkl"

    def _accept_loop(self):
        """Background thread to accept new client connections."""
        LOG.debug(f"DLC Processor listening on {self.address[0]}:{self.address[1]}")
        while not self._stop.is_set():
            try:
                c = self.listener.accept()
                LOG.debug(f"Client connected from {self.listener.last_accepted}")
                self.conns.add(c)
                # Start RX loop for this connection (in case clients send data)
                Thread(target=self._rx_loop, args=(c,), name="DLCRX", daemon=True).start()
            except (OSError, EOFError):
                break

    def _rx_loop(self, c):
        """Background thread to handle receive from a client (detects disconnects)."""
        while not self._stop.is_set():
            try:
                if c.poll(0.05):
                    msg = c.recv()
                    # Handle control messages from client
                    self._handle_client_message(msg)
            except (EOFError, OSError, BrokenPipeError):
                break
        try:
            c.close()
        except Exception:
            pass
        self.conns.discard(c)
        LOG.info("Client disconnected")

    def _handle_client_message(self, msg):
        """Handle control messages from clients."""
        if not isinstance(msg, dict):
            return

        cmd = msg.get("cmd")
        if cmd == "set_session_name":
            session_name = msg.get("session_name", "default_session")
            self.session_name = session_name
            LOG.info(f"Session name set to: {session_name}")

        elif cmd == "start_recording":
            self._vid_recording.set()
            self._recording.set()
            # Clear all data queues
            self._clear_data_queues()
            self.curr_step = 0
            LOG.info("Recording started, data queues cleared")

        elif cmd == "stop_recording":
            self._recording.clear()
            self._vid_recording.clear()
            LOG.info("Recording stopped")

        elif cmd == "save":
            filename = msg.get("filename", self.filename)
            save_code = self.save(filename)
            LOG.info(f"Save {'successful' if save_code == 1 else 'failed'}: {filename}")

        elif cmd == "start_video":
            # Placeholder for video recording start
            self._vid_recording.set()
            LOG.info("Start video recording command received")

        elif cmd == "set_filter":
            # Handle filter enable/disable (subclasses override if they support filtering)
            use_filter = msg.get("use_filter", False)
            if hasattr(self, 'use_filter'):
                self.use_filter = bool(use_filter)
                # Reset filters to reinitialize with new setting
                if hasattr(self, 'filters'):
                    self.filters = None
                LOG.info(f"Filtering {'enabled' if use_filter else 'disabled'}")
            else:
                LOG.warning("set_filter command not supported by this processor")

        elif cmd == "set_filter_params":
            # Handle filter parameter updates (subclasses override if they support filtering)
            filter_kwargs = msg.get("filter_kwargs", {})
            if hasattr(self, 'filter_kwargs'):
                # Update filter parameters
                self.filter_kwargs.update(filter_kwargs)
                # Reset filters to reinitialize with new parameters
                if hasattr(self, 'filters'):
                    self.filters = None
                LOG.info(f"Filter parameters updated: {filter_kwargs}")
            else:
                LOG.warning("set_filter_params command not supported by this processor")

    def _clear_data_queues(self):
        """Clear all data storage queues. Override in subclasses to clear additional queues."""
        self.time_stamp.clear()
        self.step.clear()
        self.frame_time.clear()
        self.pose_time.clear()
        if self.save_original:
            self.original_pose.clear()

    def broadcast(self, payload):
        """Send payload to all connected clients."""
        dead = []
        for c in list(self.conns):
            try:
                c.send(payload)
            except (EOFError, OSError, BrokenPipeError):
                dead.append(c)
        for c in dead:
            try:
                c.close()
            except Exception:
                pass
            self.conns.discard(c)

    def process(self, pose, **kwargs):
        """
        Process pose and broadcast to clients.

        This base implementation just saves original pose and broadcasts it.
        Subclasses should override to add custom processing.

        Args:
            pose: DLC pose array (N_keypoints x 3) with [x, y, confidence]
            **kwargs: Additional metadata (frame_time, pose_time, etc.)

        Returns:
            pose: Unmodified pose array
        """
        curr_time = self.timing_func()

        # Save original pose if requested
        if self.save_original:
            self.original_pose.append(pose.copy())

        # Update step counter
        self.curr_step = self.curr_step + 1

        # Store metadata (only if recording)
        if self.recording:
            self.time_stamp.append(curr_time)
            self.step.append(self.curr_step)
            self.frame_time.append(kwargs.get("frame_time", -1))
            if "pose_time" in kwargs:
                self.pose_time.append(kwargs["pose_time"])

        # Broadcast raw pose to all connected clients
        payload = [curr_time, pose]
        self.broadcast(payload)

        return pose

    def stop(self):
        """Stop the processor and close all connections."""
        LOG.info("Stopping processor...")
        
        # Signal stop to all threads
        self._stop.set()
        
        # Close all client connections first
        for c in list(self.conns):
            try:
                c.close()
            except Exception:
                pass
            self.conns.discard(c)
        
        # Close the listener socket
        if hasattr(self, 'listener') and self.listener:
            try:
                self.listener.close()
            except Exception as e:
                LOG.debug(f"Error closing listener: {e}")
        
        # Give the OS time to release the socket on Windows
        # This prevents WinError 10048 when restarting
        time.sleep(0.1)
        
        LOG.info("Processor stopped, all connections closed")

    def save(self, file=None):
        """Save logged data to file."""
        save_code = 0
        if file:
            LOG.info(f"Saving data to {file}")
            try:
                save_dict = self.get_data()
                pickle.dump(save_dict, open(file, "wb"))                              
                save_code = 1
            except Exception as e:
                LOG.error(f"Save failed: {e}")
                save_code = -1
        return save_code

    def get_data(self):
        """Get logged data as dictionary."""
        save_dict = dict()
        if self.save_original:
            save_dict["original_pose"] = np.array(self.original_pose)
        save_dict["start_time"] = self.start_time
        save_dict["time_stamp"] = np.array(self.time_stamp)
        save_dict["step"] = np.array(self.step)
        save_dict["frame_time"] = np.array(self.frame_time)
        save_dict["pose_time"] = np.array(self.pose_time) if self.pose_time else None
        save_dict["use_perf_counter"] = self.timing_func == time.perf_counter
        return save_dict


class MyProcessor_socket(BaseProcessor_socket):
    """
    DLC Processor with pose calculations (center, heading, head angle) and optional filtering.

    Calculates:
    - center: Weighted average of head keypoints
    - heading: Body orientation (degrees)
    - head_angle: Head rotation relative to body (radians)

    Broadcasts: [timestamp, center_x, center_y, heading, head_angle]
    """

    # Metadata for GUI discovery
    PROCESSOR_NAME = "Mouse Pose Processor"
    PROCESSOR_DESCRIPTION = (
        "Calculates mouse center, heading, and head angle with optional One-Euro filtering"
    )
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("0.0.0.0", 6000),
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
        bind=("0.0.0.0", 6000),
        authkey=b"secret password",
        use_perf_counter=False,
        use_filter=False,
        filter_kwargs={},
        save_original=False,
    ):
        """
        DLC Processor with multi-client broadcasting support.

        Args:
            bind: (host, port) tuple for server binding
            authkey: Authentication key for client connections
            use_perf_counter: If True, use time.perf_counter() instead of time.time()
            use_filter: If True, apply One-Euro filter to pose data
            filter_kwargs: Dict with OneEuroFilter parameters (min_cutoff, beta, d_cutoff)
            save_original: If True, save raw pose arrays
        """
        super().__init__(
            bind=bind,
            authkey=authkey,
            use_perf_counter=use_perf_counter,
            save_original=save_original,
        )

        # Additional data storage for processed values
        self.center_x = deque()
        self.center_y = deque()
        self.heading_direction = deque()
        self.head_angle = deque()

        # Filtering
        self.use_filter = use_filter
        self.filter_kwargs = filter_kwargs
        self.filters = None  # Will be initialized on first pose

    def _clear_data_queues(self):
        """Clear all data storage queues including pose-specific ones."""
        super()._clear_data_queues()
        self.center_x.clear()
        self.center_y.clear()
        self.heading_direction.clear()
        self.head_angle.clear()

    def _initialize_filters(self, vals):
        """Initialize One-Euro filters for each output variable."""
        t0 = self.timing_func()
        self.filters = {
            "center_x": OneEuroFilter(t0, vals[0], **self.filter_kwargs),
            "center_y": OneEuroFilter(t0, vals[1], **self.filter_kwargs),
            "heading": OneEuroFilter(t0, vals[2], **self.filter_kwargs),
            "head_angle": OneEuroFilter(t0, vals[3], **self.filter_kwargs),
        }
        LOG.debug(f"Initialized One-Euro filters with parameters: {self.filter_kwargs}")

    def process(self, pose, **kwargs):
        """
        Process pose: calculate center/heading/head_angle, optionally filter, and broadcast.

        Args:
            pose: DLC pose array (N_keypoints x 3) with [x, y, confidence]
            **kwargs: Additional metadata (frame_time, pose_time, etc.)

        Returns:
            pose: Unmodified pose array
        """
        # Save original pose if requested (from base class)
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
        try:
            head_angle = acos(body_axis @ head_axis) * sign
        except ValueError:
            head_angle = 0

        # Calculate heading (body orientation)
        heading = atan2(body_axis[1], body_axis[0])
        heading = degrees(heading)

        # Raw values (heading unwrapped for filtering)
        vals = [center[0], center[1], heading, head_angle]

        # Apply filtering if enabled
        curr_time = self.timing_func()
        if self.use_filter:
            if self.filters is None:
                self._initialize_filters(vals)

            # Filter each value (heading is filtered in unwrapped space)
            filtered_vals = [
                self.filters["center_x"](curr_time, vals[0]),
                self.filters["center_y"](curr_time, vals[1]),
                self.filters["heading"](curr_time, vals[2]),
                self.filters["head_angle"](curr_time, vals[3]),
            ]
            vals = filtered_vals

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

        # Broadcast processed values to all connected clients
        payload = [curr_time, vals[0], vals[1], vals[2], vals[3]]
        self.broadcast(payload)

        return pose

    def get_data(self):
        """Get logged data including base class data and processed values."""
        # Get base class data
        save_dict = super().get_data()

        # Add processed values
        save_dict["x_pos"] = np.array(self.center_x)
        save_dict["y_pos"] = np.array(self.center_y)
        save_dict["heading_direction"] = np.array(self.heading_direction)
        save_dict["head_angle"] = np.array(self.head_angle)
        save_dict["use_filter"] = self.use_filter
        save_dict["filter_kwargs"] = self.filter_kwargs

        return save_dict


# Register processors for GUI discovery
PROCESSOR_REGISTRY["BaseProcessor_socket"] = BaseProcessor_socket
PROCESSOR_REGISTRY["MyProcessor_socket"] = MyProcessor_socket


def get_available_processors():
    """
    Get list of available processor classes.

    Returns:
        dict: Dictionary mapping class names to processor info:
            {
                "ClassName": {
                    "class": ProcessorClass,
                    "name": "Display Name",
                    "description": "Description text",
                    "params": {...}
                }
            }
    """
    processors = {}
    for class_name, processor_class in PROCESSOR_REGISTRY.items():
        processors[class_name] = {
            "class": processor_class,
            "name": getattr(processor_class, "PROCESSOR_NAME", class_name),
            "description": getattr(processor_class, "PROCESSOR_DESCRIPTION", ""),
            "params": getattr(processor_class, "PROCESSOR_PARAMS", {}),
        }
    return processors


def instantiate_processor(class_name, **kwargs):
    """
    Instantiate a processor by class name with given parameters.

    Args:
        class_name: Name of the processor class (e.g., "MyProcessor_socket")
        **kwargs: Parameters to pass to the processor constructor

    Returns:
        Processor instance

    Raises:
        ValueError: If class_name is not in registry
    """
    if class_name not in PROCESSOR_REGISTRY:
        available = ", ".join(PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown processor '{class_name}'. Available: {available}")

    processor_class = PROCESSOR_REGISTRY[class_name]
    return processor_class(**kwargs)
