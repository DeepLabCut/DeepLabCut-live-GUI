from __future__ import annotations

import socket
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dlclivegui.config import WebStreamSettings
from dlclivegui.services import web_streamer as ws
from dlclivegui.services.web_streamer import LOOPBACK_HOST, NETWORK_HOST, WebStreamer

# Upper bound for joins that only matters if the code under test is broken
# (passing runs return as soon as the event happens); not a timing assertion.
_FAILURE_CAP = 30.0


def _make(frame_box, *, port: int = 0, host: str = LOOPBACK_HOST, **overrides) -> WebStreamer:
    """Streamer with config defaults; port 0 lets the OS pick a free port atomically."""
    d = WebStreamSettings()
    params = {"fps": d.fps, "max_size": (d.max_width, d.max_height), "jpeg_quality": d.jpeg_quality, **overrides}
    return WebStreamer(lambda: frame_box["frame"], host=host, port=port, **params)


def _get(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


@pytest.fixture
def frame_box():
    return {"frame": np.full((960, 1280, 3), 127, dtype=np.uint8)}


@pytest.fixture
def streamer(frame_box):
    s = _make(frame_box, port=0, fps=5.0, max_size=(640, 480))
    s.start()
    yield s
    s.stop()


@pytest.mark.unit
def test_index_serves_html(streamer):
    status, body, ctype = _get(streamer.url)
    assert status == 200
    assert ctype.startswith("text/html")
    assert b"/frame.jpg" in body


@pytest.mark.unit
def test_frame_is_downscaled_jpeg(streamer):
    status, body, ctype = _get(streamer.url + "frame.jpg")
    assert status == 200
    assert ctype == "image/jpeg"
    assert body[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (480, 640)


@pytest.mark.unit
def test_no_frame_returns_503(frame_box, streamer):
    frame_box["frame"] = None
    status, _, _ = _get(streamer.url + "frame.jpg")
    assert status == 503


@pytest.mark.unit
def test_unknown_path_returns_404(streamer):
    status, _, _ = _get(streamer.url + "nope")
    assert status == 404


@pytest.mark.unit
def test_encoding_is_cached_and_rate_limited(monkeypatch, frame_box):
    calls = {"n": 0}
    real_imencode = cv2.imencode

    def counting_imencode(*args, **kwargs):
        calls["n"] += 1
        return real_imencode(*args, **kwargs)

    monkeypatch.setattr(ws.cv2, "imencode", counting_imencode)
    clock = {"now": 100.0}
    monkeypatch.setattr(ws, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    s = _make(frame_box, port=0, fps=5.0)  # min interval 0.2 s

    s.latest_jpeg()
    s.latest_jpeg()  # same source object -> cached
    assert calls["n"] == 1

    frame_box["frame"] = frame_box["frame"].copy()
    clock["now"] += 0.1
    s.latest_jpeg()  # new object but within 1/fps -> cached
    assert calls["n"] == 1

    clock["now"] += 0.2
    s.latest_jpeg()  # new object, interval elapsed -> re-encode
    assert calls["n"] == 2

    clock["now"] += 10.0
    s.latest_jpeg()  # interval elapsed but same object -> cached
    assert calls["n"] == 2


@pytest.mark.unit
def test_stop_returns_before_shutdown_completes(frame_box):
    s = _make(frame_box, port=0)
    s.start()
    server = s._server
    release = threading.Event()
    real_shutdown = server.shutdown

    def blocking_shutdown():
        release.wait()
        real_shutdown()

    server.shutdown = blocking_shutdown

    t = s.stop()  # must not wait for shutdown, which cannot finish until released
    try:
        assert t is not None and t.is_alive()
        assert not s.is_running()
    finally:
        release.set()
        t.join(timeout=_FAILURE_CAP)
    assert not t.is_alive()


@pytest.mark.unit
def test_shutdown_frees_port(frame_box):
    s = _make(frame_box, port=0)
    s.start()
    server = s._server
    port = server.server_address[1]

    t = s.stop()
    t.join(timeout=_FAILURE_CAP)
    assert not t.is_alive()
    assert server.socket.fileno() == -1  # listening socket closed

    again = _make(frame_box, port=port)
    again.start()  # raises OSError if the port were still held
    again.stop(wait=True)


@pytest.mark.unit
def test_second_instance_on_same_port_is_refused(frame_box):
    # Windows SO_REUSEADDR would let this bind succeed silently; POSIX refuses either way.
    first = _make(frame_box, port=0)
    first.start()
    try:
        second = _make(frame_box, port=first._server.server_address[1])
        with pytest.raises(OSError):
            second.start()
        assert not second.is_running()
    finally:
        first.stop(wait=True)


@pytest.mark.unit
def test_restart_on_same_port_after_serving_requests(frame_box):
    # Served requests leave server-side TIME_WAIT connections; rebinding must still work on every OS.
    s = _make(frame_box, port=0)
    s.start()
    port = s._server.server_address[1]
    for _ in range(3):
        assert _get(s.url + "frame.jpg")[0] == 200
    s.stop(wait=True)

    again = _make(frame_box, port=port)
    again.start()
    try:
        assert _get(again.url + "frame.jpg")[0] == 200
    finally:
        again.stop(wait=True)


@pytest.mark.unit
def test_stop_returns_shutdown_thread(frame_box):
    s = _make(frame_box, port=0)
    assert s.stop() is None  # not running
    s.start()
    t = s.stop()
    assert t is not None
    t.join(timeout=_FAILURE_CAP)
    assert not t.is_alive()


@pytest.mark.unit
def test_stalled_client_is_dropped(monkeypatch, frame_box):
    monkeypatch.setattr(ws, "_CLIENT_TIMEOUT", 0.2)
    s = _make(frame_box, port=0)
    s.start()
    try:
        with socket.create_connection(("127.0.0.1", s._server.server_address[1]), timeout=3.0) as sock:
            # Send nothing: the server must close the connection after its timeout.
            try:
                assert sock.recv(1024) == b""
            except ConnectionResetError:
                pass  # some platforms reset instead of closing cleanly
    finally:
        s.stop(wait=True)


@pytest.mark.unit
def test_from_settings_maps_every_field():
    settings = WebStreamSettings(port=9001, fps=2.0, max_width=320, max_height=200, jpeg_quality=55)
    s = WebStreamer.from_settings(lambda: None, settings, host=LOOPBACK_HOST)
    assert (s.port, s._min_interval, s._max_size, s._jpeg_quality) == (9001, 0.5, (320, 200), 55)


@pytest.mark.unit
def test_urls_for_loopback_bind(frame_box):
    s = _make(frame_box, port=8123)
    assert s.url == "http://127.0.0.1:8123/"
    assert s.network_url is None


@pytest.mark.unit
def test_urls_for_wildcard_bind(monkeypatch, frame_box):
    # No bind: only URL formatting. A wildcard address is not browsable, so show localhost.
    s = _make(frame_box, port=8123, host=NETWORK_HOST)
    monkeypatch.setattr(ws, "lan_ip", lambda: "192.168.1.20")
    assert s.url == "http://localhost:8123/"
    assert s.network_url == "http://192.168.1.20:8123/"
    monkeypatch.setattr(ws, "lan_ip", lambda: None)
    assert s.network_url is None


@pytest.mark.unit
def test_lan_ip_is_none_without_route(monkeypatch):
    class _NoRoute:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def connect(self, addr):
            raise OSError("Network is unreachable")

    monkeypatch.setattr(
        ws, "socket", SimpleNamespace(socket=_NoRoute, AF_INET=socket.AF_INET, SOCK_DGRAM=socket.SOCK_DGRAM)
    )
    assert ws.lan_ip() is None


@pytest.mark.unit
def test_lan_ip_is_never_loopback():
    ip = ws.lan_ip()  # environment-dependent value, invariant only
    assert ip is None or not ip.startswith("127.")
