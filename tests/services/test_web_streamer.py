from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from dlclivegui.services import web_streamer as ws
from dlclivegui.services.web_streamer import WebStreamer


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
    s = WebStreamer(lambda: frame_box["frame"], port=0, fps=5.0)

    s.latest_jpeg()
    s.latest_jpeg()  # same source object -> cached
    assert calls["n"] == 1

    frame_box["frame"] = frame_box["frame"].copy()
    s.latest_jpeg()  # new object but within 1/fps -> cached
    assert calls["n"] == 1

    s._last_encode -= 1.0
    s.latest_jpeg()  # interval elapsed -> re-encode
    assert calls["n"] == 2


@pytest.mark.unit
def test_stop_is_non_blocking_and_frees_port(frame_box):
    s = WebStreamer(lambda: frame_box["frame"], port=0)
    s.start()
    port = s._server.server_address[1]

    t0 = time.monotonic()
    s.stop()
    assert time.monotonic() - t0 < 0.1
    assert not s.is_running()

    deadline = time.monotonic() + 3.0
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                pass
        except OSError:
            break
        assert time.monotonic() < deadline, "server still accepting connections after stop()"
        time.sleep(0.05)
