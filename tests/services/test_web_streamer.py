from __future__ import annotations

import socket
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dlclivegui.services import web_streamer as ws
from dlclivegui.services.web_streamer import WebStreamer

# Upper bound for joins that only matters if the code under test is broken
# (passing runs return as soon as the event happens); not a timing assertion.
_FAILURE_CAP = 30.0


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
    s = WebStreamer(lambda: frame_box["frame"], port=0, fps=5.0, max_size=(640, 480))
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
    s = WebStreamer(lambda: frame_box["frame"], port=0, fps=5.0)  # min interval 0.2 s

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
    s = WebStreamer(lambda: frame_box["frame"], port=0)
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
    s = WebStreamer(lambda: frame_box["frame"], port=0)
    s.start()
    server = s._server
    port = server.server_address[1]

    t = s.stop()
    t.join(timeout=_FAILURE_CAP)
    assert not t.is_alive()
    assert server.socket.fileno() == -1  # listening socket closed

    again = WebStreamer(lambda: frame_box["frame"], port=port)
    again.start()  # raises OSError if the port were still held
    again.stop(wait=True)


@pytest.mark.unit
def test_second_instance_on_same_port_is_refused(frame_box):
    # Windows SO_REUSEADDR would let this bind succeed silently; POSIX refuses either way.
    first = WebStreamer(lambda: frame_box["frame"], port=0)
    first.start()
    try:
        second = WebStreamer(lambda: frame_box["frame"], port=first._server.server_address[1])
        with pytest.raises(OSError):
            second.start()
        assert not second.is_running()
    finally:
        first.stop(wait=True)


@pytest.mark.unit
def test_restart_on_same_port_after_serving_requests(frame_box):
    # Served requests leave server-side TIME_WAIT connections; rebinding must still work on every OS.
    s = WebStreamer(lambda: frame_box["frame"], port=0)
    s.start()
    port = s._server.server_address[1]
    for _ in range(3):
        assert _get(s.url + "frame.jpg")[0] == 200
    s.stop(wait=True)

    again = WebStreamer(lambda: frame_box["frame"], port=port)
    again.start()
    try:
        assert _get(again.url + "frame.jpg")[0] == 200
    finally:
        again.stop(wait=True)


@pytest.mark.unit
def test_stop_returns_shutdown_thread(frame_box):
    s = WebStreamer(lambda: frame_box["frame"], port=0)
    assert s.stop() is None  # not running
    s.start()
    t = s.stop()
    assert t is not None
    t.join(timeout=_FAILURE_CAP)
    assert not t.is_alive()


@pytest.mark.unit
def test_stalled_client_is_dropped(monkeypatch, frame_box):
    monkeypatch.setattr(ws, "_CLIENT_TIMEOUT", 0.2)
    s = WebStreamer(lambda: frame_box["frame"], port=0)
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
