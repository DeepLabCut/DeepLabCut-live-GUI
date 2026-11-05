# tests/gui/main_window/test_web_stream.py
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QGuiApplication

from dlclivegui.config import ApplicationSettings, WebStreamSettings
from dlclivegui.gui import web_stream as web_stream_mod
from dlclivegui.gui.main_window import DLCLiveMainWindow
from dlclivegui.services.web_streamer import LOOPBACK_HOST, NETWORK_HOST, WebStreamer
from dlclivegui.utils.settings_store import DLCLiveGUISettingsStore


@pytest.fixture(autouse=True)
def requested_hosts(monkeypatch):
    """Bind loopback on port 0 whatever was requested, and record the requested host.

    Port 0: the OS assigns a free port atomically (no race with other processes).
    Loopback: never bind all interfaces in tests (firewall prompts, exposure).
    Same-port rebinding is covered by tests/services/test_web_streamer.py.
    """
    hosts: list[str] = []

    class _TestStreamer(WebStreamer):
        def __init__(self, *args, **kwargs):
            hosts.append(kwargs["host"])
            kwargs.update(host=LOOPBACK_HOST, port=0)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(web_stream_mod, "WebStreamer", _TestStreamer)
    return hosts


@pytest.fixture
def ctl(window):
    """The window's web stream controller."""
    yield window._web_stream
    window._web_stream._stop(wait=True)


def _fail_bind(monkeypatch):
    def _fail(self):
        raise OSError("address in use")

    monkeypatch.setattr(WebStreamer, "start", _fail)


def _answer_network_prompt(monkeypatch, ctl, answer: bool) -> list[bool]:
    asked: list[bool] = []
    monkeypatch.setattr(ctl, "_confirm_network", lambda: asked.append(True) or answer)
    return asked


@pytest.mark.gui
class TestWebStream:
    def test_actions_grouped_under_view_web_stream_submenu(self, window, ctl):
        view_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
        submenu = next(a.menu() for a in view_menu.actions() if a.menu() and a.text() == "Web stream")
        items = submenu.actions()
        assert [a.isSeparator() for a in items] == [False, False, True, False, False]
        assert [a for a in items if not a.isSeparator()] == [
            ctl.action_enable,
            ctl.action_network,
            ctl.action_open,
            ctl.action_copy,
        ]

    def test_off_and_local_by_default(self, ctl):
        assert not ctl.action_enable.isChecked()
        assert not ctl.action_network.isChecked()
        assert ctl.streamer is None
        assert ctl._store.get_web_stream_enabled() is False
        assert ctl._store.get_web_stream_allow_network() is False

    def test_user_toggle_starts_stops_and_persists(self, ctl, requested_hosts):
        ctl.action_enable.setChecked(True)
        assert ctl.streamer is not None and ctl.streamer.is_running()
        assert ctl.streamer._port == 0  # requested_hosts override is in effect
        assert requested_hosts == [LOOPBACK_HOST]
        assert ctl._store.get_web_stream_enabled() is True

        ctl.action_enable.setChecked(False)
        assert ctl.streamer is None
        assert ctl._store.get_web_stream_enabled() is False

    def test_quick_retoggle_restarts(self, ctl):
        for _ in range(3):
            ctl.action_enable.setChecked(True)
            assert ctl.streamer is not None and ctl.streamer.is_running()
            ctl.action_enable.setChecked(False)
            assert ctl.streamer is None

    def test_open_item_shows_url_and_is_clickable_only_while_running(self, ctl, monkeypatch):
        opened = []
        monkeypatch.setattr(web_stream_mod.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))

        ctl.apply_settings(WebStreamSettings(port=9123))
        assert ctl.action_open.text() == "Open http://127.0.0.1:9123/"  # configured port while stopped
        assert not ctl.action_open.isEnabled()

        ctl.action_enable.setChecked(True)
        url = f"http://127.0.0.1:{ctl.streamer.port}/"  # actual bound port
        assert ctl.action_open.text() == f"Open {url}"
        assert ctl.action_open.isEnabled()
        ctl.action_open.trigger()
        assert opened == [url]

        ctl.action_enable.setChecked(False)
        assert not ctl.action_open.isEnabled()

    def test_copy_link_only_for_other_devices_while_running(self, ctl, monkeypatch):
        monkeypatch.setattr("dlclivegui.services.web_streamer.lan_ip", lambda: "192.168.1.20")
        ctl.action_enable.setChecked(True)
        assert not ctl.action_copy.isEnabled()  # localhost only
        assert ctl.action_copy.text() == "Copy link for other devices"

        _answer_network_prompt(monkeypatch, ctl, True)
        ctl.action_network.setChecked(True)
        url = f"http://192.168.1.20:{ctl.streamer.port}/"
        assert ctl.action_copy.isEnabled()
        assert ctl.action_copy.text() == f"Copy link for other devices: {url}"
        assert ctl.action_open.text() == f"Open http://localhost:{ctl.streamer.port}/"

        ctl.action_copy.trigger()
        assert QGuiApplication.clipboard().text() == url

        ctl.action_enable.setChecked(False)
        assert not ctl.action_copy.isEnabled()

    def test_copy_link_disabled_without_lan_address(self, ctl, monkeypatch):
        monkeypatch.setattr("dlclivegui.services.web_streamer.lan_ip", lambda: None)
        _answer_network_prompt(monkeypatch, ctl, True)
        ctl.action_network.setChecked(True)
        ctl.action_enable.setChecked(True)
        assert not ctl.action_copy.isEnabled()
        assert ctl.action_copy.text() == "Copy link for other devices"

    def test_param_change_restarts_only_when_running_and_changed(self, ctl):
        ctl.action_enable.setChecked(True)
        first = ctl.streamer

        ctl.apply_settings(ctl.settings)
        assert ctl.streamer is first  # unchanged params: no restart

        ctl.apply_settings(ctl.settings.model_copy(update={"fps": 1.0}))
        assert ctl.streamer is not first and ctl.streamer.is_running()
        assert not first.is_running()
        assert ctl.action_enable.isChecked()

    def test_config_load_never_starts_stream(self, window, ctl):
        ctl._store.set_web_stream_enabled(True)
        window._apply_config(window._config)  # File > Load path: restore_local_prefs=False
        assert ctl.streamer is None

    def test_config_load_cannot_expose_stream_on_network(self, window, ctl, requested_hosts):
        ctl.action_enable.setChecked(True)
        data = window._config.to_dict()
        data["web_stream"] = {"host": NETWORK_HOST, "enabled": True, "port": 9000}
        window._apply_config(ApplicationSettings.from_dict(data))

        assert ctl.streamer is not None  # restarted for the new port...
        assert requested_hosts == [LOOPBACK_HOST, LOOPBACK_HOST]  # ...but never on the network
        assert ctl._store.get_web_stream_allow_network() is False

    def test_startup_restores_enabled_preference(self, ctl):
        ctl._store.set_web_stream_enabled(True)
        ctl.apply_settings(ctl.settings, restore_enabled=True)
        assert ctl.streamer is not None
        assert ctl.action_enable.isChecked()

    def test_current_config_carries_params_only(self, window, ctl):
        ctl.apply_settings(ctl.settings.model_copy(update={"jpeg_quality": 40}))
        cfg = window._current_config(allow_empty_model_path=True).web_stream
        assert cfg.jpeg_quality == 40
        assert set(cfg.model_dump()) == {"port", "fps", "max_width", "max_height", "jpeg_quality"}

    def test_user_toggle_bind_failure_unchecks_and_persists_off(self, ctl, monkeypatch):
        _fail_bind(monkeypatch)
        ctl.action_enable.setChecked(True)

        assert not ctl.action_enable.isChecked()
        assert ctl.streamer is None
        assert ctl._store.get_web_stream_enabled() is False

    def test_startup_bind_failure_keeps_preference(self, ctl, monkeypatch):
        # e.g. a second instance: the port is taken now, the user's intent should survive.
        ctl._store.set_web_stream_enabled(True)
        _fail_bind(monkeypatch)
        ctl.apply_settings(ctl.settings, restore_enabled=True)

        assert not ctl.action_enable.isChecked()
        assert ctl._store.get_web_stream_enabled() is True


@pytest.mark.gui
class TestWebStreamNetwork:
    def test_declined_prompt_keeps_localhost(self, ctl, monkeypatch, requested_hosts):
        ctl.action_enable.setChecked(True)
        asked = _answer_network_prompt(monkeypatch, ctl, False)
        ctl.action_network.setChecked(True)

        assert asked == [True]
        assert not ctl.action_network.isChecked()
        assert ctl._store.get_web_stream_allow_network() is False
        assert requested_hosts == [LOOPBACK_HOST]  # no restart

    def test_accepted_prompt_restarts_on_network_and_persists(self, ctl, monkeypatch, requested_hosts):
        ctl.action_enable.setChecked(True)
        _answer_network_prompt(monkeypatch, ctl, True)
        ctl.action_network.setChecked(True)

        assert ctl._store.get_web_stream_allow_network() is True
        assert ctl.streamer is not None
        assert requested_hosts == [LOOPBACK_HOST, NETWORK_HOST]

    def test_disabling_network_needs_no_prompt_and_restarts_local(self, ctl, monkeypatch, requested_hosts):
        _answer_network_prompt(monkeypatch, ctl, True)
        ctl.action_network.setChecked(True)
        ctl.action_enable.setChecked(True)
        asked = _answer_network_prompt(monkeypatch, ctl, False)
        ctl.action_network.setChecked(False)

        assert asked == []
        assert ctl._store.get_web_stream_allow_network() is False
        assert requested_hosts == [NETWORK_HOST, LOOPBACK_HOST]

    def test_toggle_while_stopped_does_not_start(self, ctl, monkeypatch):
        _answer_network_prompt(monkeypatch, ctl, True)
        ctl.action_network.setChecked(True)
        assert ctl.streamer is None

    def test_startup_restores_network_preference_without_prompt(self, qtbot, app_config_two_cams, requested_hosts):
        store = DLCLiveGUISettingsStore(QSettings("DeepLabCut", "DLCLiveGUI"))
        store.set_web_stream_enabled(True)
        store.set_web_stream_allow_network(True)

        def _no_prompt(self):
            raise AssertionError("network prompt shown at startup")

        w = DLCLiveMainWindow(config=app_config_two_cams)
        qtbot.addWidget(w)
        w.setAttribute(Qt.WA_DontShowOnScreen, True)
        try:
            ctl = w._web_stream
            ctl._confirm_network = _no_prompt.__get__(ctl)
            assert ctl.action_network.isChecked()
            assert ctl.streamer is not None
            assert requested_hosts == [NETWORK_HOST]
        finally:
            w._web_stream._stop(wait=True)
            w.close()


@pytest.mark.gui
def test_close_event_stops_stream_without_blocking(window, monkeypatch):
    ctl = window._web_stream
    ctl.action_enable.setChecked(True)
    streamer = ctl.streamer
    calls = []
    real_stop = streamer.stop
    monkeypatch.setattr(streamer, "stop", lambda wait=False: calls.append(wait) or real_stop(wait=wait))

    window.close()

    assert calls == [False]
    assert ctl.streamer is None
