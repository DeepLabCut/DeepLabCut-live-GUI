"""Low-priority, on-demand JPEG web view of the GUI's raw tiled frame.

Not encoding unless ``/frame.jpg`` is requested, and
encoding is capped at ``fps`` regardless of client count.
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from dlclivegui.config import WebStreamSettings

logger = logging.getLogger(__name__)

# User-facing defaults (port, fps, size, quality) live in ``WebStreamSettings``;
# the values below are server internals.
LOOPBACK_HOST = "127.0.0.1"
NETWORK_HOST = "0.0.0.0"  # all interfaces; no auth
SHUTDOWN_TIMEOUT = 2.0  # max wait for a server to release its port, wherever a caller waits
_POLL_INTERVAL = 0.1  # serve_forever shutdown latency
_CLIENT_TIMEOUT = 5.0  # drop idle or stalled connections
_WILDCARD_HOSTS = {"", "0.0.0.0"}


def lan_ip() -> str | None:
    """Best-effort IPv4 address other machines can reach, or ``None`` (offline, no route).

    A UDP ``connect`` sends nothing; it only makes the OS pick the outgoing interface.
    ``gethostbyname(gethostname())`` is avoided: it returns 127.0.1.1 on Debian/Ubuntu.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1 (RFC 5737), never routed to a real host
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") or ip in _WILDCARD_HOSTS else ip


def local_url(host: str, port: int) -> str:
    """URL to open on this machine for a server bound to ``host`` (a wildcard address is not browsable)."""
    return f"http://{'localhost' if host in _WILDCARD_HOSTS else host}:{port}/"


def network_url(port: int) -> str | None:
    """URL for other machines to reach a server bound to all interfaces, or ``None`` if no LAN address."""
    ip = lan_ip()
    return f"http://{ip}:{port}/" if ip else None


class _StreamServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` with bind fail when port is already taken, cross-platform.

    POSIX: keep ``SO_REUSEADDR``; it only permits rebinding over ``TIME_WAIT``
    and still refuses a second listener (macOS/BSD may allow a more specific
    address next to a wildcard bind).
    Windows: ``SO_REUSEADDR`` lets a second listener bind the same port, so
    use a plain bind. ``SO_EXCLUSIVEADDRUSE`` is not used: it can block
    rebinding while closed connections sit in ``TIME_WAIT``.
    """

    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>DLC Live stream</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img id="v" alt="waiting for frames">
<script>
const v = document.getElementById("v");
const period = __PERIOD_MS__;
function next() {
  const img = new Image();
  img.onload = () => { v.src = img.src; setTimeout(next, period); };
  img.onerror = () => setTimeout(next, 1000);
  img.src = "/frame.jpg?t=" + Date.now();
}
next();
</script></body></html>
"""


class WebStreamer:
    """Serve the latest frame from ``frame_source`` as JPEG over HTTP."""

    def __init__(
        self,
        frame_source: Callable[[], np.ndarray | None],
        *,
        host: str,
        port: int,
        fps: float,
        max_size: tuple[int, int],
        jpeg_quality: int,
    ):
        self._frame_source = frame_source
        self._host = host
        self._port = port
        self._min_interval = 1.0 / fps if fps > 0 else 0.0
        self._max_size = max_size
        self._jpeg_quality = int(jpeg_quality)
        self._index = _INDEX_HTML.replace("__PERIOD_MS__", str(int(self._min_interval * 1000))).encode()

        self._server: _StreamServer | None = None
        self._thread: threading.Thread | None = None

        self._cache_lock = threading.Lock()
        self._cached_src: np.ndarray | None = None
        self._cached_jpeg: bytes | None = None
        self._last_encode = 0.0

    @classmethod
    def from_settings(
        cls, frame_source: Callable[[], np.ndarray | None], settings: WebStreamSettings, *, host: str
    ) -> WebStreamer:
        return cls(
            frame_source,
            host=host,
            port=settings.port,
            fps=settings.fps,
            max_size=(settings.max_width, settings.max_height),
            jpeg_quality=settings.jpeg_quality,
        )

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self._port

    @property
    def url(self) -> str:
        return local_url(self._host, self.port)

    def is_running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        """Bind and serve in a daemon thread. Raises ``OSError`` if the port is unavailable."""
        if self._server is not None:
            return
        server = _StreamServer((self._host, self._port), self._make_handler())
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": _POLL_INTERVAL}, name="WebStreamer", daemon=True
        )
        self._thread.start()
        logger.info("Web stream serving at %s", self.url)

    def stop(self, wait: bool = False) -> threading.Thread | None:
        """Shut down on a short-lived daemon thread.

        ``wait=False`` returns immediately (GUI toggle). ``wait=True`` blocks
        until the port is released, for an immediate restart on the same port.
        Returns the shutdown thread (``None`` if not running) so a caller can
        wait for the port later.
        """
        server, self._server = self._server, None
        self._thread = None
        if server is None:
            return None

        def _shutdown() -> None:
            try:
                server.shutdown()
            finally:
                server.server_close()

        shutdown_thread = threading.Thread(target=_shutdown, name="WebStreamerShutdown", daemon=True)
        shutdown_thread.start()
        with self._cache_lock:
            self._cached_src = None
            self._cached_jpeg = None
        if wait:
            shutdown_thread.join(timeout=SHUTDOWN_TIMEOUT)
        return shutdown_thread

    def latest_jpeg(self) -> bytes | None:
        """Return JPEG bytes for the current frame, encoding at most once per ``1/fps``."""
        frame = self._frame_source()
        if frame is None:
            return None
        with self._cache_lock:
            now = time.monotonic()
            if self._cached_jpeg is not None and (
                frame is self._cached_src or now - self._last_encode < self._min_interval
            ):
                return self._cached_jpeg
            jpeg = self._encode(frame)
            if jpeg is not None:
                self._cached_src = frame
                self._cached_jpeg = jpeg
                self._last_encode = now
            return self._cached_jpeg

    def _encode(self, frame: np.ndarray) -> bytes | None:
        h, w = frame.shape[:2]
        max_w, max_h = self._max_size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return buf.tobytes() if ok else None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            timeout = _CLIENT_TIMEOUT

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                path = self.path.split("?", 1)[0]
                try:
                    if path == "/":
                        self._send(200, "text/html; charset=utf-8", streamer._index)
                    elif path == "/frame.jpg":
                        try:
                            jpeg = streamer.latest_jpeg()
                        except Exception:
                            logger.exception("Web stream frame encoding failed")
                            jpeg = None
                        if jpeg is None:
                            self._send(503, "text/plain", b"no frame")
                        else:
                            self._send(200, "image/jpeg", jpeg)
                    else:
                        self._send(404, "text/plain", b"not found")
                except (ConnectionError, TimeoutError):
                    pass  # client went away or stalled past ``timeout``

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A002 - http.server API
                pass

        return _Handler
