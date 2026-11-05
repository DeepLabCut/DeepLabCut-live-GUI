# tests/gui/main_window/test_web_stream.py
from __future__ import annotations

import pytest

from dlclivegui.config import WebStreamSettings


@pytest.fixture
def ws_window(window):
    yield window
    window._apply_web_stream_settings(WebStreamSettings(enabled=False))


@pytest.mark.gui
class TestWebStreamConfig:
    def test_disabled_by_default(self, ws_window):
        w = ws_window
        assert not w.action_web_stream.isChecked()
        assert w._web_streamer is None

    def test_apply_enabled_starts_stream_and_updates_label(self, ws_window):
        w = ws_window
        w._apply_web_stream_settings(WebStreamSettings(enabled=True, port=0, fps=2.0))

        assert w.action_web_stream.isChecked()
        assert w._web_streamer is not None and w._web_streamer.is_running()
        assert w.action_web_stream.text() == "Web stream (127.0.0.1:0)"

    def test_current_config_reflects_toggle_and_params(self, ws_window):
        w = ws_window
        w._apply_web_stream_settings(WebStreamSettings(enabled=False, port=0, jpeg_quality=40))
        assert w._current_config(allow_empty_model_path=True).web_stream.enabled is False

        w.action_web_stream.setChecked(True)
        cfg = w._current_config(allow_empty_model_path=True).web_stream
        assert cfg.enabled is True
        assert cfg.jpeg_quality == 40

    def test_param_change_restarts_running_stream(self, ws_window):
        w = ws_window
        w._apply_web_stream_settings(WebStreamSettings(enabled=True, port=0))
        first = w._web_streamer

        w._apply_web_stream_settings(WebStreamSettings(enabled=True, port=0))
        assert w._web_streamer is first  # unchanged params: no restart

        w._apply_web_stream_settings(WebStreamSettings(enabled=True, port=0, fps=1.0))
        assert w._web_streamer is not first
        assert not first.is_running()

    def test_apply_disabled_stops_stream(self, ws_window):
        w = ws_window
        w._apply_web_stream_settings(WebStreamSettings(enabled=True, port=0))
        w._apply_web_stream_settings(WebStreamSettings(enabled=False, port=0))

        assert not w.action_web_stream.isChecked()
        assert w._web_streamer is None

    def test_bind_failure_unchecks_action(self, ws_window, monkeypatch):
        w = ws_window

        def _fail(self):
            raise OSError("address in use")

        monkeypatch.setattr("dlclivegui.services.web_streamer.WebStreamer.start", _fail)
        w._apply_web_stream_settings(WebStreamSettings(enabled=True, port=0))

        assert not w.action_web_stream.isChecked()
        assert w._web_streamer is None
