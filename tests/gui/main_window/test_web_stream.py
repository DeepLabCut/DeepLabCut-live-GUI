# tests/gui/main_window/test_web_stream.py
from __future__ import annotations

import socket

import pytest

from dlclivegui.config import WebStreamSettings


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ws_window(window):
    window._apply_web_stream_settings(WebStreamSettings(port=_free_port()))
    yield window
    window._stop_web_stream(wait=True)


@pytest.mark.gui
class TestWebStream:
    def test_off_by_default(self, ws_window):
        w = ws_window
        assert not w.action_web_stream.isChecked()
        assert w._web_streamer is None
        assert w._settings_store.get_web_stream_enabled() is False

    def test_user_toggle_starts_stops_and_persists(self, ws_window):
        w = ws_window
        w.action_web_stream.setChecked(True)
        assert w._web_streamer is not None and w._web_streamer.is_running()
        assert w._settings_store.get_web_stream_enabled() is True

        w.action_web_stream.setChecked(False)
        assert w._web_streamer is None
        assert w._settings_store.get_web_stream_enabled() is False

    def test_quick_retoggle_rebinds_same_port(self, ws_window):
        w = ws_window
        for _ in range(3):
            w.action_web_stream.setChecked(True)
            assert w._web_streamer is not None, "re-enable failed to rebind the port"
            w.action_web_stream.setChecked(False)

    def test_label_tracks_settings(self, ws_window):
        w = ws_window
        w._apply_web_stream_settings(WebStreamSettings(host="0.0.0.0", port=9123))
        assert w.action_web_stream.text() == "Web stream (0.0.0.0:9123)"

    def test_param_change_restarts_only_when_running_and_changed(self, ws_window):
        w = ws_window
        w.action_web_stream.setChecked(True)
        first = w._web_streamer

        w._apply_web_stream_settings(w._web_stream_settings.model_copy())
        assert w._web_streamer is first  # unchanged params: no restart

        w._apply_web_stream_settings(w._web_stream_settings.model_copy(update={"fps": 1.0}))
        assert w._web_streamer is not first and w._web_streamer.is_running()
        assert not first.is_running()
        assert w.action_web_stream.isChecked()

    def test_config_load_never_starts_stream(self, ws_window):
        w = ws_window
        w._settings_store.set_web_stream_enabled(True)
        w._apply_web_stream_settings(WebStreamSettings(port=_free_port()))  # restore_enabled defaults False
        assert w._web_streamer is None

    def test_startup_restores_persisted_preference(self, ws_window):
        w = ws_window
        w._settings_store.set_web_stream_enabled(True)
        w._apply_web_stream_settings(w._web_stream_settings.model_copy(), restore_enabled=True)
        assert w._web_streamer is not None
        assert w.action_web_stream.isChecked()

    def test_current_config_carries_params_only(self, ws_window):
        w = ws_window
        w._apply_web_stream_settings(w._web_stream_settings.model_copy(update={"jpeg_quality": 40}))
        cfg = w._current_config(allow_empty_model_path=True).web_stream
        assert cfg.jpeg_quality == 40
        assert "enabled" not in cfg.model_dump()

    def test_user_toggle_bind_failure_unchecks_and_persists_off(self, ws_window, monkeypatch):
        w = ws_window

        def _fail(self):
            raise OSError("address in use")

        monkeypatch.setattr("dlclivegui.services.web_streamer.WebStreamer.start", _fail)
        w.action_web_stream.setChecked(True)

        assert not w.action_web_stream.isChecked()
        assert w._web_streamer is None
        assert w._settings_store.get_web_stream_enabled() is False

    def test_startup_bind_failure_keeps_preference(self, ws_window, monkeypatch):
        # e.g. a second instance: the port is taken now, the user's intent should survive.
        w = ws_window
        w._settings_store.set_web_stream_enabled(True)

        def _fail(self):
            raise OSError("address in use")

        monkeypatch.setattr("dlclivegui.services.web_streamer.WebStreamer.start", _fail)
        w._apply_web_stream_settings(w._web_stream_settings.model_copy(), restore_enabled=True)

        assert not w.action_web_stream.isChecked()
        assert w._settings_store.get_web_stream_enabled() is True


@pytest.mark.gui
def test_close_event_stops_stream_without_blocking(window, monkeypatch):
    w = window
    w._apply_web_stream_settings(WebStreamSettings(port=_free_port()))
    w.action_web_stream.setChecked(True)
    streamer = w._web_streamer
    calls = []
    real_stop = streamer.stop
    monkeypatch.setattr(streamer, "stop", lambda wait=False: calls.append(wait) or real_stop(wait=wait))

    w.close()

    assert calls == [False]
    assert w._web_streamer is None
