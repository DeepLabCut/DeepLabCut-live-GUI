"""Example of socket-based DLC Processor with multi-client support and optional One-Euro filtering."""

# dlclivegui/processors/dlc_processor_socket.py
import logging
import pickle
import socket
import sys
import time
from collections import deque
from multiprocessing.connection import Client, Listener
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pandas as pd

try:
    from dlclive.processor import Processor  # type: ignore
except ImportError:
    Processor = object  # Fallback for type checking if dlclive is not installed

logger = logging.getLogger("dlc_processor_socket")

# Avoid duplicate handlers if module is imported multiple times
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)


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

    # Experimental:
    # Socket/Teensy/Unity processors often start threads, sockets, serial ports, etc.
    # Build them inside the DLCLive worker to match legacy Tk GUI behavior.
    PROCESSOR_BUILD_IN_WORKER = False

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

                if conn.closed:
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

        self.curr_step += 1

        if self.recording:
            if self.save_original and self.original_pose is not None:
                self.original_pose.append(pose.copy())
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
