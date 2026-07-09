"""Standalone mock DLC processor plugin with socket listener support.

This fixture intentionally avoids importing dlclive, dlclivegui, Teensy, serial,
NumPy, or project-specific processor base classes.

It is designed to test two separate concerns without mixing them:

1. GUI processor plugin discovery/configuration
   - Exposes PROCESSOR_* metadata.
   - Exposes PROCESSOR_BUILD_IN_WORKER = True.
   - Exposes get_available_processors(), which your loader can consume without
     requiring this class to inherit from dlclive.processor.Processor.

2. Runtime socket listener behavior
   - Starts a multiprocessing.connection.Listener.
   - Accepts one or more clients on a background thread.
   - Receives simple command dictionaries from clients.
   - Broadcasts mock pose payloads to connected clients from process().

It does NOT mock Teensy serial acquisition. For the listener send/receive tests,
Teensy is not required: the Teensy path is a separate serial-reader concern.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from collections import deque
from multiprocessing.connection import Client, Listener
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)

IP_ADDRESS = "127.0.0.1"
PORT = 6000


class MockSocketProcessor:
    """Standalone socket-based mock processor for tests.

    This intentionally reimplements the core listener/thread/control behavior
    instead of inheriting from project classes. It is suitable for fixture usage
    where external dependencies such as Teensy, serial, dlclive, or dlclivegui
    should not be imported.

    Expected test usage:

        proc = MockSocketProcessor(bind=("127.0.0.1", free_port()))
        conn = Client(proc.address, authkey=proc.authkey)
        conn.send({"cmd": "ping"})
        assert conn.recv()["type"] == "pong"
        proc.process([[1, 2, 0.9]])
        assert conn.recv()["type"] == "pose"
        proc.stop()

    Notes:
        - `process()` accepts any pose-like Python object. It does not validate
          DLC shape because this mock is for socket lifecycle tests, not pose
          validation tests.
        - Payloads are sent through multiprocessing.connection, matching the
          style used by the legacy socket processors.
    """

    PROCESSOR_NAME = "Mock Socket Processor"
    PROCESSOR_DESCRIPTION = "Standalone mock socket processor without Teensy or DLCLive imports."
    PROCESSOR_BUILD_IN_WORKER = False
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": (IP_ADDRESS, PORT),
            "description": "Server bind address. Use port 0 to request an ephemeral port.",
        },
        "authkey": {
            "type": "bytes",
            "default": b"secret password",
            "description": "Authentication key for multiprocessing.connection clients.",
        },
        "start_server": {
            "type": "bool",
            "default": True,
            "description": "Whether to start the listener in __init__.",
        },
        "socket_timeout": {
            "type": "float",
            "default": 0.05,
            "description": "Accept-loop timeout in seconds.",
        },
        "save_original": {
            "type": "bool",
            "default": False,
            "description": "Whether to store raw pose payloads while recording.",
        },
    }

    def __init__(
        self,
        bind: tuple[str, int] = (IP_ADDRESS, PORT),
        authkey: bytes = b"secret password",
        *,
        start_server: bool = True,
        socket_timeout: float = 0.05,
        save_original: bool = False,
    ) -> None:
        self.address = bind
        self.authkey = authkey
        self._socket_timeout = float(socket_timeout)
        self.save_original = bool(save_original)

        # Runtime listener/client state.
        self.listener: Listener | None = None
        self.conns: set[Any] = set()
        self._conns_lock = Lock()
        self._stop = Event()
        self._accept_thread: Thread | None = None
        self._rx_threads: set[Thread] = set()

        # Recording/control state compatible with socket-processor expectations.
        self._recording = Event()
        self._vid_recording = Event()
        self._session_name = "test_session"
        self.filename: str | None = None

        # Minimal data buffers for save/get_data tests.
        self.start_time = time.time()
        self.time_stamp = deque()
        self.step = deque()
        self.frame_time = deque()
        self.pose_time = deque()
        self.original_pose = deque() if self.save_original else None
        self.received_commands = deque()
        self.broadcast_count = 0
        self.curr_step = 0

        if start_server:
            self.start_server(bind, authkey=authkey, timeout=self._socket_timeout)

    # ------------------------------------------------------------------
    # Properties matching the real socket processors
    # ------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._recording.is_set()

    @property
    def video_recording(self) -> bool:
        return self._vid_recording.is_set()

    @property
    def session_name(self) -> str:
        return self._session_name

    @session_name.setter
    def session_name(self, name: str) -> None:
        self._session_name = str(name)
        self.filename = f"{self._session_name}_mock_processor_data.pkl"

    # ------------------------------------------------------------------
    # Listener lifecycle
    # ------------------------------------------------------------------
    def start_server(
        self,
        bind: tuple[str, int] | None = None,
        authkey: bytes | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        """Start the socket listener if it is not already running."""
        if self.listener is not None:
            return

        if bind is not None:
            self.address = bind
        if authkey is not None:
            self.authkey = authkey
        if timeout is not None:
            self._socket_timeout = float(timeout)

        self._stop.clear()
        self.listener = Listener(self.address, authkey=self.authkey)

        # If bind used port 0, update address to the actual ephemeral port.
        self.address = self._actual_listener_address(self.listener, fallback=self.address)

        self._set_listener_timeout(self.listener, self._socket_timeout)

        self._accept_thread = Thread(target=self._accept_loop, name="MockSocketProcessorAccept", daemon=True)
        self._accept_thread.start()
        logger.info("MockSocketProcessor listening on %s:%s", self.address[0], self.address[1])

    @staticmethod
    def _actual_listener_address(listener: Listener, fallback: tuple[str, int]) -> tuple[str, int]:
        """Best-effort extraction of the actual listener address."""
        try:
            raw = getattr(listener, "_listener", None)
            sock = getattr(raw, "_socket", None)
            if sock is not None:
                addr = sock.getsockname()
                return (str(addr[0]), int(addr[1]))
        except Exception:
            pass
        try:
            addr = listener.address
            return (str(addr[0]), int(addr[1]))
        except Exception:
            return fallback

    @staticmethod
    def _set_listener_timeout(listener: Listener, timeout: float) -> None:
        """Set accept timeout on CPython listener internals, best effort."""
        raw = getattr(listener, "_listener", None)
        for candidate in (raw, getattr(raw, "_socket", None)):
            try:
                if candidate is not None and hasattr(candidate, "settimeout"):
                    candidate.settimeout(timeout)
                    return
            except Exception:
                pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.listener is None:
                    return
                conn = self.listener.accept()
            except TimeoutError:
                continue
            except (OSError, EOFError):
                if self._stop.is_set():
                    break
                continue
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("Unexpected accept-loop error")
                continue

            with self._conns_lock:
                self.conns.add(conn)

            rx = Thread(target=self._rx_loop, args=(conn,), name="MockSocketProcessorRx", daemon=True)
            self._rx_threads.add(rx)
            rx.start()
            logger.info("MockSocketProcessor client connected")

    def _rx_loop(self, conn: Any) -> None:
        while not self._stop.is_set():
            try:
                if conn.poll(0.05):
                    msg = conn.recv()
                    self._handle_client_message(msg, conn=conn)
                    continue

                if getattr(conn, "closed", False):
                    break

            except (EOFError, OSError, ConnectionError, BrokenPipeError):
                break
            except Exception:
                logger.exception("Unexpected receive-loop error")
                break

        self._close_conn(conn)

    def _close_conn(self, conn: Any) -> None:
        try:
            conn.close()
        except Exception:
            pass
        with self._conns_lock:
            self.conns.discard(conn)

    def stop(self) -> None:
        """Stop listener, close clients, and join background threads best-effort."""
        if self._stop.is_set():
            return

        self._stop.set()

        # Wake accept() if needed.
        try:
            Client(self.address, authkey=self.authkey).close()
        except Exception:
            pass

        with self._conns_lock:
            conns = list(self.conns)
        for conn in conns:
            self._close_conn(conn)

        try:
            if self.listener is not None:
                self.listener.close()
        except Exception:
            pass
        self.listener = None

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None

        for thread in list(self._rx_threads):
            try:
                thread.join(timeout=0.5)
            except Exception:
                pass
        self._rx_threads.clear()

        if sys.platform.startswith("win"):
            time.sleep(0.05)

    close = stop

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Client command handling
    # ------------------------------------------------------------------
    def _handle_client_message(self, msg: Any, *, conn: Any | None = None) -> None:
        self.received_commands.append(msg)

        if not isinstance(msg, dict):
            self._send_to(conn, {"type": "error", "error": "message must be a dict"})
            return

        cmd = msg.get("cmd")

        if cmd == "ping":
            self._send_to(
                conn,
                {
                    "type": "pong",
                    "timestamp": time.time(),
                    "session_name": self.session_name,
                    "recording": self.recording,
                    "video_recording": self.video_recording,
                    "clients": self.client_count(),
                },
            )

        elif cmd == "status":
            self._send_to(conn, self.status_payload())

        elif cmd == "set_session_name":
            self.session_name = msg.get("session_name", "default_session")
            self._send_to(conn, {"type": "ack", "cmd": cmd, "session_name": self.session_name})

        elif cmd == "start_recording":
            self.start_recording()
            self._send_to(conn, {"type": "ack", "cmd": cmd, "recording": True})

        elif cmd == "stop_recording":
            self.stop_recording()
            self._send_to(conn, {"type": "ack", "cmd": cmd, "recording": False})

        elif cmd == "save":
            file = msg.get("filename", self.filename)
            result = self.save(file)
            self._send_to(conn, {"type": "ack", "cmd": cmd, "result": result, "filename": file})

        elif cmd == "close":
            self._send_to(conn, {"type": "ack", "cmd": cmd})
            if conn is not None:
                self._close_conn(conn)

        else:
            self._send_to(conn, {"type": "error", "error": f"unknown cmd: {cmd!r}"})

    @staticmethod
    def _send_to(conn: Any | None, payload: Any) -> bool:
        if conn is None:
            return False
        try:
            conn.send(payload)
            return True
        except Exception:
            return False

    def client_count(self) -> int:
        with self._conns_lock:
            return len(self.conns)

    def status_payload(self) -> dict[str, Any]:
        return {
            "type": "status",
            "session_name": self.session_name,
            "recording": self.recording,
            "video_recording": self.video_recording,
            "clients": self.client_count(),
            "steps": self.curr_step,
            "broadcast_count": self.broadcast_count,
            "address": self.address,
        }

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def start_recording(self) -> None:
        self._recording.set()
        self._vid_recording.set()
        self._clear_data_queues()
        self.curr_step = 0

    def stop_recording(self) -> None:
        self._recording.clear()
        self._vid_recording.clear()

    def _clear_data_queues(self) -> None:
        self.time_stamp.clear()
        self.step.clear()
        self.frame_time.clear()
        self.pose_time.clear()
        if self.original_pose is not None:
            self.original_pose.clear()

    # ------------------------------------------------------------------
    # Process/broadcast path
    # ------------------------------------------------------------------
    def process(self, pose: Any, **kwargs: Any) -> Any:
        """Mock DLCLive processor callback.

        Records minimal metadata when recording is active and broadcasts a simple
        pose payload to all connected clients.
        """
        now = time.time()
        self.curr_step += 1

        if self.recording:
            self.time_stamp.append(now)
            self.step.append(self.curr_step)
            self.frame_time.append(kwargs.get("frame_time", -1))
            if "pose_time" in kwargs:
                self.pose_time.append(kwargs["pose_time"])
            if self.original_pose is not None:
                self.original_pose.append(pose)

        payload = {
            "type": "pose",
            "timestamp": now,
            "step": self.curr_step,
            "pose": self._make_pickle_safe_pose(pose),
            "frame_time": kwargs.get("frame_time", None),
            "pose_time": kwargs.get("pose_time", None),
            "recording": self.recording,
        }
        self.broadcast(payload)
        return pose

    @staticmethod
    def _make_pickle_safe_pose(pose: Any) -> Any:
        """Convert common array-likes to socket-safe Python types."""
        tolist = getattr(pose, "tolist", None)
        if callable(tolist):
            try:
                return tolist()
            except Exception:
                pass
        return pose

    def broadcast(self, payload: Any) -> None:
        with self._conns_lock:
            conns = list(self.conns)

        dead = []
        for conn in conns:
            try:
                conn.send(payload)
                self.broadcast_count += 1
            except Exception:
                dead.append(conn)

        for conn in dead:
            self._close_conn(conn)

    # ------------------------------------------------------------------
    # Save/get_data helpers
    # ------------------------------------------------------------------
    def get_data(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "session_name": self.session_name,
            "time_stamp": list(self.time_stamp),
            "step": list(self.step),
            "frame_time": list(self.frame_time),
            "pose_time": list(self.pose_time),
            "recording": self.recording,
            "video_recording": self.video_recording,
            "received_commands": list(self.received_commands),
            "broadcast_count": self.broadcast_count,
        }

    def save(self, file: str | Path | None = None) -> int:
        if not file:
            return 0
        try:
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump(self.get_data(), fh)
            return 1
        except Exception:
            logger.exception("MockSocketProcessor save failed")
            return -1


# Optional aliases useful in different test styles.
MockPDSocketProcessor = MockSocketProcessor
MockUnitySocketProcessor = MockSocketProcessor


def get_available_processors() -> dict[str, dict[str, Any]]:
    """Plugin-discovery entrypoint used by dlclivegui.processor_utils.

    This avoids requiring the class to inherit from dlclive.processor.Processor
    during tests. The loader path that prefers get_available_processors() can
    still discover this processor as a GUI plugin fixture.
    """
    return {
        "MockSocketProcessor": {
            "class": MockSocketProcessor,
            "name": getattr(MockSocketProcessor, "PROCESSOR_NAME", "MockSocketProcessor"),
            "description": getattr(MockSocketProcessor, "PROCESSOR_DESCRIPTION", ""),
            "params": getattr(MockSocketProcessor, "PROCESSOR_PARAMS", {}),
        }
    }
